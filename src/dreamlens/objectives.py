import itertools
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .layers import LayerCapture, as_list, first_tensor_output, resolve_module


def cosine_similarity(tensor_a, tensor_b):
    """Return one cosine similarity for each pair of batched tensors."""

    tensor_a = torch.as_tensor(tensor_a)
    tensor_b = torch.as_tensor(
        tensor_b,
        dtype=tensor_a.dtype,
        device=tensor_a.device,
    )
    if tensor_a.ndim < 2 or tensor_b.ndim < 2:
        raise ValueError("cosine_similarity expects tensors with a batch dimension")

    dims_a = tuple(range(1, tensor_a.ndim))
    dims_b = tuple(range(1, tensor_b.ndim))
    norm_a = torch.sqrt(torch.sum(tensor_a * tensor_a, dim=dims_a, keepdim=True))
    norm_b = torch.sqrt(torch.sum(tensor_b * tensor_b, dim=dims_b, keepdim=True))
    eps_a = torch.as_tensor(1e-12, dtype=norm_a.dtype, device=norm_a.device)
    eps_b = torch.as_tensor(1e-12, dtype=norm_b.dtype, device=norm_b.device)
    tensor_a = tensor_a / torch.sqrt(torch.maximum(norm_a * norm_a, eps_a))
    tensor_b = tensor_b / torch.sqrt(torch.maximum(norm_b * norm_b, eps_b))
    return torch.sum(tensor_a * tensor_b, dim=dims_a)


def dot_cossim(tensor_a, tensor_b, cossim_pow=2.0):
    """Return the cosine-weighted dot-product direction objective."""

    tensor_a = torch.as_tensor(tensor_a)
    tensor_b = torch.as_tensor(
        tensor_b,
        dtype=tensor_a.dtype,
        device=tensor_a.device,
    )
    floor = torch.as_tensor(1e-1, dtype=tensor_a.dtype, device=tensor_a.device)
    cosim = torch.maximum(cosine_similarity(tensor_a, tensor_b), floor) ** cossim_pow
    dot = torch.sum(tensor_a * tensor_b)
    return dot * cosim


def _image_dims(images):
    if images.ndim != 4:
        raise ValueError("regularizers expect an NCHW image batch")
    return (1, 2, 3)


def l1_reg(factor=1.0):
    """Build a per-image mean L1 regularizer."""

    def reg(images):
        return float(factor) * torch.mean(torch.abs(images), dim=_image_dims(images))

    return reg


def l2_reg(factor=1.0):
    """Build a per-image root-mean-square L2 regularizer."""

    def reg(images):
        return float(factor) * torch.sqrt(torch.mean(images**2, dim=_image_dims(images)))

    return reg


def l_inf_reg(factor=1.0):
    """Build a per-image L-infinity regularizer."""

    def reg(images):
        return float(factor) * torch.amax(torch.abs(images), dim=_image_dims(images))

    return reg


def total_variation_reg(factor=1.0):
    """Build anisotropic total-variation regularization."""

    def tv_reg(images):
        _image_dims(images)
        vertical = torch.abs(images[:, :, 1:, :] - images[:, :, :-1, :]).sum(
            dim=(1, 2, 3)
        )
        horizontal = torch.abs(images[:, :, :, 1:] - images[:, :, :, :-1]).sum(
            dim=(1, 2, 3)
        )
        return float(factor) * (vertical + horizontal)

    return tv_reg


@dataclass(frozen=True)
class FeatureTarget:
    """A layer feature to maximize during visualization.

    Select exactly one of ``channel``, flattened ``neuron``, or ``direction``.
    Leave all three unset to target the complete layer. For convolutional
    channels, ``position`` may be ``(y, x)``; for sequence channels, it may be
    an integer token index. The same target is accepted by classical maximize
    and MaCo.
    """

    layer: object
    channel: Optional[int] = None
    neuron: Optional[int] = None
    position: object = None
    direction: object = None
    reduction: str = "mean"
    weight: float = 1.0
    sign: float = 1.0
    cossim_power: float = 2.0

    def __post_init__(self):
        selected = sum(
            value is not None for value in (self.channel, self.neuron, self.direction)
        )
        if selected > 1:
            raise ValueError(
                "FeatureTarget accepts only one of channel, neuron, or direction"
            )
        if self.position is not None and self.channel is None:
            raise ValueError("position is only valid with a channel target")
        if self.reduction not in {"mean", "sum", "max", "norm"}:
            raise ValueError("reduction must be 'mean', 'sum', 'max', or 'norm'")
        if self.cossim_power <= 0:
            raise ValueError("cossim_power must be positive")

    @classmethod
    def for_layer(cls, layer, reduction="norm", weight=1.0, sign=1.0):
        return cls(layer=layer, reduction=reduction, weight=weight, sign=sign)

    @classmethod
    def for_channel(
        cls,
        layer,
        channel,
        position=None,
        reduction="mean",
        weight=1.0,
        sign=1.0,
    ):
        return cls(
            layer=layer,
            channel=int(channel),
            position=position,
            reduction=reduction,
            weight=weight,
            sign=sign,
        )

    @classmethod
    def for_neuron(
        cls,
        layer,
        neuron,
        reduction="mean",
        weight=1.0,
        sign=1.0,
    ):
        return cls(
            layer=layer,
            neuron=int(neuron),
            reduction=reduction,
            weight=weight,
            sign=sign,
        )

    @classmethod
    def for_class(cls, class_id, layer=-1, weight=1.0, sign=1.0):
        return cls(
            layer=layer,
            neuron=int(class_id),
            weight=weight,
            sign=sign,
        )

    @classmethod
    def for_direction(
        cls,
        layer,
        direction,
        cossim_power=2.0,
        weight=1.0,
        sign=1.0,
    ):
        return cls(
            layer=layer,
            direction=direction,
            cossim_power=cossim_power,
            weight=weight,
            sign=sign,
        )


@dataclass(frozen=True)
class _LayerMask:
    pass


@dataclass(frozen=True)
class _ChannelMask:
    index: int


@dataclass(frozen=True)
class _NeuronMask:
    index: int


class Objective:
    """Composable objective retained as a root-level compatibility adapter.

    New code normally uses :class:`FeatureTarget`. This class preserves the
    Cartesian objective API while sharing the same root hooks, target
    selection, losses, and optimization kernels.
    """

    def __init__(
        self,
        model,
        layers,
        masks,
        funcs,
        multipliers,
        names,
        input_shape=None,
    ):
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        lengths = {len(layers), len(masks), len(funcs), len(multipliers), len(names)}
        if len(lengths) != 1 or not layers:
            raise ValueError(
                "layers, masks, funcs, multipliers, and names must be non-empty and aligned"
            )
        self.model = model
        self.layers = list(layers)
        self.masks = [_as_choices(mask) for mask in masks]
        self.funcs = list(funcs)
        self.multipliers = [float(multiplier) for multiplier in multipliers]
        self.names = [_as_name_choices(name) for name in names]
        self.input_shape = _normalize_input_shape(input_shape, allow_none=True)

        for choices, name_choices in zip(self.masks, self.names):
            if len(choices) != len(name_choices):
                raise ValueError("each mask choice must have a corresponding name")

    def __add__(self, term):
        if not isinstance(term, Objective):
            raise ValueError(f"{term} is not an objective.")
        input_shape = _merge_input_shapes(self.input_shape, term.input_shape)
        return Objective(
            self.model,
            layers=self.layers + term.layers,
            masks=self.masks + term.masks,
            funcs=self.funcs + term.funcs,
            multipliers=self.multipliers + term.multipliers,
            names=self.names + term.names,
            input_shape=input_shape,
        )

    def __sub__(self, term):
        if not isinstance(term, Objective):
            raise ValueError(f"{term} is not an objective.")
        term.multipliers = [-multiplier for multiplier in term.multipliers]
        return self + term

    def __mul__(self, factor):
        if not isinstance(factor, (int, float)):
            raise ValueError(f"{factor} is not a number.")
        self.multipliers = [multiplier * factor for multiplier in self.multipliers]
        return self

    def __rmul__(self, factor):
        return self * factor

    @property
    def num_combinations(self):
        count = 1
        for choices in self.masks:
            count *= len(choices)
        return count

    def compile(self, input_shape=None):
        """Compile hooks, an ascent score, names, and the NCHW input shape."""

        resolved_shape = _resolve_input_shape(
            self.model,
            input_shape if input_shape is not None else self.input_shape,
        )
        combinations = list(itertools.product(*self.masks))
        compiled_masks = [
            [combination[index] for combination in combinations]
            for index in range(len(self.masks))
        ]
        objective_names = [
            " & ".join(name_parts) for name_parts in itertools.product(*self.names)
        ]
        multipliers = tuple(self.multipliers)

        def objective_function(model_outputs):
            if isinstance(model_outputs, torch.Tensor):
                model_outputs = [model_outputs]
            if len(model_outputs) != len(self.funcs):
                raise ValueError(
                    "Expected {} model outputs, received {}.".format(
                        len(self.funcs), len(model_outputs)
                    )
                )
            loss = None
            for output_index, output in enumerate(model_outputs):
                output = first_tensor(output)
                mask = _materialize_masks(compiled_masks[output_index], output)
                term = self.funcs[output_index](output, mask)
                loss = term if loss is None else loss + term
                loss = loss * multipliers[output_index]
            return loss

        layers = [resolve_module(self.model, layer) for layer in self.layers]
        hooked_model = _HookedModel(self.model, layers)
        return (
            hooked_model,
            objective_function,
            objective_names,
            (len(combinations), *resolved_shape),
        )

    @staticmethod
    def from_target(model, target, input_shape=None):
        """Adapt one canonical :class:`FeatureTarget` for root optimizers."""

        target = coerce_target(target)
        target_objective = TargetObjective([target])

        def maximize_target(output, _mask):
            return -target_objective([output])

        module = resolve_module(model, target.layer)
        return Objective(
            model=model,
            layers=[module],
            masks=[[_LayerMask()]],
            funcs=[maximize_target],
            multipliers=[1.0],
            names=[[f"FeatureTarget#{target.layer}"]],
            input_shape=input_shape,
        )

    @staticmethod
    def layer(
        model,
        layer,
        reducer="magnitude",
        multiplier=1.0,
        name=None,
        input_shape=None,
    ):
        resolved = resolve_module(model, layer)
        power = 2.0 if reducer == "magnitude" else 1.0

        def optim_func(model_output, mask):
            return torch.mean((model_output * mask) ** power)

        display = _layer_display_name(model, resolved, layer)
        name = f"Layer#{display}" if name is None else name
        return Objective(
            model,
            [resolved],
            [[_LayerMask()]],
            [optim_func],
            [multiplier],
            [[name]],
            input_shape=input_shape,
        )

    @staticmethod
    def direction(
        model,
        layer,
        vectors,
        multiplier=1.0,
        cossim_pow=2.0,
        names=None,
        input_shape=None,
    ):
        resolved = resolve_module(model, layer)
        masks = list(vectors) if isinstance(vectors, list) else [vectors]
        display = _layer_display_name(model, resolved, layer)
        names = (
            [f"Direction#{display}_{index}" for index in range(len(masks))]
            if names is None
            else _normalize_names_argument(names, len(masks))
        )

        def optim_func(model_output, mask):
            return dot_cossim(model_output, mask, cossim_pow)

        return Objective(
            model,
            [resolved],
            [masks],
            [optim_func],
            [multiplier],
            [names],
            input_shape=input_shape,
        )

    @staticmethod
    def channel(
        model,
        layer,
        channel_ids,
        multiplier=1.0,
        names=None,
        input_shape=None,
    ):
        resolved = resolve_module(model, layer)
        channel_ids = channel_ids if isinstance(channel_ids, list) else [channel_ids]
        channel_ids = [int(channel_id) for channel_id in channel_ids]
        masks = [_ChannelMask(channel_id) for channel_id in channel_ids]
        display = _layer_display_name(model, resolved, layer)
        names = (
            [f"Channel#{display}_{channel_id}" for channel_id in channel_ids]
            if names is None
            else _normalize_names_argument(names, len(masks))
        )

        def optim_func(output, target):
            return torch.mean(output * target, dim=tuple(range(1, output.ndim)))

        return Objective(
            model,
            [resolved],
            [masks],
            [optim_func],
            [multiplier],
            [names],
            input_shape=input_shape,
        )

    @staticmethod
    def neuron(
        model,
        layer,
        neurons_ids,
        multiplier=1.0,
        names=None,
        input_shape=None,
    ):
        resolved = resolve_module(model, layer)
        neurons_ids = neurons_ids if isinstance(neurons_ids, list) else [neurons_ids]
        neurons_ids = [int(neuron_id) for neuron_id in neurons_ids]
        masks = [_NeuronMask(neuron_id) for neuron_id in neurons_ids]
        display = _layer_display_name(model, resolved, layer)
        names = (
            [f"Neuron#{display}_{neuron_id}" for neuron_id in neurons_ids]
            if names is None
            else _normalize_names_argument(names, len(masks))
        )

        def optim_func(output, target):
            return torch.mean(output * target, dim=tuple(range(1, output.ndim)))

        return Objective(
            model,
            [resolved],
            [masks],
            [optim_func],
            [multiplier],
            [names],
            input_shape=input_shape,
        )


def first_tensor(output):
    return first_tensor_output(output)


def mean_activation_objective(layer_outputs):
    """Default objective: minimize negative activation mean."""

    loss = None
    for output in layer_outputs:
        output = first_tensor(output)
        value = -output.mean()
        loss = value if loss is None else loss + value
    if loss is None:
        raise ValueError("layer_outputs must contain at least one tensor")
    return loss


class PerSampleObjective:
    """Apply one objective function per batch item."""

    def __init__(self, objectives):
        if not objectives:
            raise ValueError("objectives must contain at least one callable")
        self.objectives = list(objectives)

    def __call__(self, layer_outputs):
        layer_outputs = [first_tensor(output) for output in layer_outputs]
        batch_size = layer_outputs[0].shape[0]
        if batch_size != len(self.objectives):
            raise ValueError(
                "Batch size {} does not match {} objectives.".format(
                    batch_size, len(self.objectives)
                )
            )

        loss = layer_outputs[0].new_tensor(0.0)
        for batch_index, objective in enumerate(self.objectives):
            per_item_outputs = [
                output[batch_index].unsqueeze(0) for output in layer_outputs
            ]
            loss = loss + objective(per_item_outputs)
        return loss


class ChannelObjective:
    """Objective for a channel/unit in one captured layer output."""

    def __init__(
        self,
        channel,
        layer_index=0,
        position=None,
        reduction="mean",
        sign=1.0,
    ):
        self.channel = int(channel)
        self.layer_index = int(layer_index)
        self.position = position
        self.reduction = reduction
        self.sign = float(sign)

    def __call__(self, layer_outputs):
        target = FeatureTarget.for_channel(
            layer=self.layer_index,
            channel=self.channel,
            position=self.position,
            reduction=self.reduction,
            sign=self.sign,
        )
        return TargetObjective([target])([layer_outputs[self.layer_index]])


class TargetObjective:
    """Objective generated from one or more ``FeatureTarget`` items."""

    def __init__(self, targets):
        self.targets = [coerce_target(target) for target in as_list(targets)]
        if not self.targets:
            raise ValueError("targets must contain at least one FeatureTarget")

    def __call__(self, layer_outputs):
        if len(layer_outputs) != len(self.targets):
            raise ValueError(
                "Expected {} layer outputs, received {}.".format(
                    len(self.targets),
                    len(layer_outputs),
                )
            )
        loss = None
        for output, target in zip(layer_outputs, self.targets):
            tensor = first_tensor(output)
            value = _target_value(tensor, target)
            term = -float(target.weight) * float(target.sign) * value
            loss = term if loss is None else loss + term
        return loss


class FeatureAmplificationObjective:
    """Loss that amplifies activations in the direction of reference features."""

    def __init__(self, targets, strength=1.0, cosine_floor=0.1, eps=1e-6):
        self.targets = [target.detach() for target in targets]
        self.strength = strength
        self.cosine_floor = cosine_floor
        self.eps = eps

    def __call__(self, layer_outputs):
        layer_outputs = [first_tensor(output) for output in layer_outputs]
        if len(layer_outputs) != len(self.targets):
            raise ValueError("layer_outputs and targets must have the same length")

        loss = layer_outputs[0].new_tensor(0.0)
        for current, target in zip(layer_outputs, self.targets):
            target = resize_target_like(
                target.to(device=current.device, dtype=current.dtype),
                current,
            ).detach()
            dot = torch.sum(current * target)
            target_norm = torch.sqrt(torch.sum(target * target)).clamp_min(self.eps)
            cosine = (dot / target_norm).clamp_min(self.cosine_floor)
            magnitude = dot.clamp_min(self.eps).pow(self.strength)
            loss = loss - cosine * magnitude
        return loss


class ReferenceAmplificationObjective:
    """Cosine-weighted dot-product objective used by the reference amplification path."""

    def __init__(self, targets, power=1.0, cosine_floor=0.1, eps=1e-6):
        self.targets = [_remove_single_batch(target).detach() for target in targets]
        self.power = power
        self.cosine_floor = cosine_floor
        self.eps = eps

    def __call__(self, layer_outputs):
        layer_outputs = [_remove_single_batch(first_tensor(output)) for output in layer_outputs]
        if len(layer_outputs) != len(self.targets):
            raise ValueError("layer_outputs and targets must have the same length")

        loss = layer_outputs[0].new_tensor(0.0)
        for current, target in zip(layer_outputs, self.targets):
            target = _resize_reference_target(
                target.to(device=current.device, dtype=current.dtype),
                current,
            ).detach()
            numerator = torch.sum(current * target)
            denominator = torch.sqrt(torch.sum(target**2)) + self.eps
            cosine = numerator / denominator
            floor = torch.tensor(
                self.cosine_floor,
                dtype=cosine.dtype,
                device=cosine.device,
            )
            cosine = torch.maximum(floor, cosine)
            magnitude = numerator.clamp_min(self.eps).pow(self.power)
            loss = loss - cosine * magnitude
        return loss


def channel_objective(channel, layer_index=0, position=None, reduction="mean", sign=1.0):
    return ChannelObjective(
        channel=channel,
        layer_index=layer_index,
        position=position,
        reduction=reduction,
        sign=sign,
    )


def feature_target(
    layer,
    channel=None,
    neuron=None,
    position=None,
    direction=None,
    reduction="mean",
    weight=1.0,
    sign=1.0,
    cossim_power=2.0,
):
    return FeatureTarget(
        layer=layer,
        channel=channel,
        neuron=neuron,
        position=position,
        direction=direction,
        reduction=reduction,
        weight=weight,
        sign=sign,
        cossim_power=cossim_power,
    )


def coerce_target(target):
    if isinstance(target, FeatureTarget):
        return target
    return FeatureTarget(layer=target)


def resize_target_like(target, current):
    if target.shape == current.shape:
        return target
    if target.dim() == 4 and current.dim() == 4:
        if target.shape[0] != current.shape[0] or target.shape[1] != current.shape[1]:
            raise ValueError("target and current feature batches/channels do not match")
        return F.interpolate(
            target,
            size=current.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    if target.dim() == 3 and current.dim() == 3:
        if target.shape[0] != current.shape[0] or target.shape[-1] != current.shape[-1]:
            raise ValueError("target and current sequence feature shapes do not match")
        return F.interpolate(
            target.transpose(1, 2),
            size=current.shape[1],
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
    raise ValueError(
        "target feature shape {} cannot be matched to {}".format(
            tuple(target.shape),
            tuple(current.shape),
        )
    )


def _select_channel(output, channel, position=None):
    if output.dim() == 2:
        return output[:, channel]
    if output.dim() == 3:
        if position is None:
            return output[:, :, channel]
        return output[:, int(position), channel]
    if output.dim() == 4:
        if position is None:
            return output[:, channel]
        y, x = position
        return output[:, channel, int(y), int(x)]
    raise ValueError("Expected 2D, 3D, or 4D layer output.")


def _select_neuron(output, neuron):
    if output.dim() < 2:
        raise ValueError("Neuron targets require a batched layer output")
    flattened = output.reshape(output.shape[0], -1)
    neuron = int(neuron)
    if neuron < -flattened.shape[1] or neuron >= flattened.shape[1]:
        raise IndexError("neuron index is out of bounds")
    return flattened[:, neuron]


def _direction_value(output, direction, cossim_power):
    direction = torch.as_tensor(
        direction,
        dtype=output.dtype,
        device=output.device,
    )
    if tuple(direction.shape) == tuple(output.shape[1:]):
        direction = direction.unsqueeze(0)
    if direction.dim() != output.dim():
        raise ValueError(
            "direction must match the layer output with or without a batch dimension"
        )
    if direction.shape[0] == 1 and output.shape[0] != 1:
        direction = direction.expand(output.shape[0], *direction.shape[1:])
    if tuple(direction.shape) != tuple(output.shape):
        raise ValueError(
            "direction shape {} does not match layer output {}".format(
                tuple(direction.shape), tuple(output.shape)
            )
        )
    output_flat = output.reshape(output.shape[0], -1)
    direction_flat = direction.reshape(direction.shape[0], -1)
    cosine = F.cosine_similarity(output_flat, direction_flat, dim=1, eps=1e-12)
    cosine = cosine.clamp_min(0.1).pow(float(cossim_power))
    dot = torch.sum(output_flat * direction_flat, dim=1)
    return torch.mean(dot * cosine)


def _target_value(output, target):
    if target.direction is not None:
        return _direction_value(output, target.direction, target.cossim_power)
    if target.neuron is not None:
        selected = _select_neuron(output, target.neuron)
    elif target.channel is not None:
        selected = _select_channel(output, target.channel, target.position)
    else:
        selected = output
    return _reduce_feature(selected, target.reduction)


def _reduce_feature(selected, reduction):
    if reduction == "mean":
        return selected.mean()
    if reduction == "sum":
        return selected.sum()
    if reduction == "max":
        return selected.max()
    if reduction == "norm":
        return selected.norm()
    raise ValueError("reduction must be 'mean', 'sum', 'max', or 'norm'")


def _remove_single_batch(tensor):
    if tensor.dim() >= 2 and tensor.shape[0] == 1:
        return tensor[0]
    return tensor


def _resize_reference_target(target, current):
    if target.shape == current.shape:
        return target
    if target.dim() == 3 and current.dim() == 3:
        if target.shape[0] != current.shape[0]:
            raise ValueError("target and current feature channels do not match")
        return F.interpolate(
            target.unsqueeze(0),
            size=current.shape[-2:],
            mode="bilinear",
        ).squeeze(0)
    return resize_target_like(target, current)


class _HookedModel(torch.nn.Module):
    """Return selected intermediate outputs while retaining the model graph."""

    def __init__(self, model, layers):
        super().__init__()
        self.model = model
        self._captures = [LayerCapture(layer, clone=True).open() for layer in layers]

    def forward(self, inputs):
        for capture in self._captures:
            capture.clear()
        self.model(inputs)
        return [capture.tensor_output() for capture in self._captures]

    def close(self):
        while self._captures:
            self._captures.pop().close()

    def __del__(self):
        if hasattr(self, "_captures"):
            self.close()


def _materialize_masks(choices, output):
    target_batch = len(choices)
    if target_batch not in (1, output.shape[0]):
        raise ValueError(
            "Objective combinations ({}) do not match model batch size ({}).".format(
                target_batch, output.shape[0]
            )
        )
    shape = (target_batch, *output.shape[1:])
    first = choices[0]

    if isinstance(first, _LayerMask):
        return torch.ones(shape, dtype=output.dtype, device=output.device)
    if isinstance(first, _ChannelMask):
        if output.ndim < 2:
            raise ValueError("channel objectives require an output with a channel axis")
        mask = torch.zeros(shape, dtype=output.dtype, device=output.device)
        for batch_index, choice in enumerate(choices):
            if choice.index < -output.shape[1] or choice.index >= output.shape[1]:
                raise IndexError("channel index is out of bounds")
            mask[batch_index, choice.index] = 1.0
        return mask
    if isinstance(first, _NeuronMask):
        mask = torch.zeros(shape, dtype=output.dtype, device=output.device)
        flat = mask.reshape(target_batch, -1)
        for batch_index, choice in enumerate(choices):
            if choice.index < -flat.shape[1] or choice.index >= flat.shape[1]:
                raise IndexError("neuron index is out of bounds")
            flat[batch_index, choice.index] = 1.0
        return mask

    tensors = [
        torch.as_tensor(choice, dtype=output.dtype, device=output.device)
        for choice in choices
    ]
    try:
        return torch.stack(tensors)
    except RuntimeError as exc:
        raise ValueError("all masks in a sub-objective must have the same shape") from exc


def _as_choices(value):
    if isinstance(value, list):
        if not value:
            raise ValueError("mask choices cannot be empty")
        return value
    if isinstance(value, torch.Tensor) and value.ndim > 0:
        return list(value)
    if isinstance(value, np.ndarray) and value.ndim > 0:
        return list(value)
    return [value]


def _as_name_choices(value):
    if isinstance(value, str):
        return [value]
    values = list(value)
    if not values:
        raise ValueError("objective names cannot be empty")
    return [str(item) for item in values]


def _normalize_names_argument(names, count):
    values = [names] if isinstance(names, str) else list(names)
    if len(values) != count:
        raise ValueError("names must contain one name per objective choice")
    return [str(value) for value in values]


def _layer_display_name(model, resolved, original):
    for name, module in model.named_modules():
        if module is resolved:
            return name or model.__class__.__name__
    return str(original)


def _normalize_input_shape(input_shape, allow_none=False):
    if input_shape is None:
        if allow_none:
            return None
        raise ValueError(
            "PyTorch models do not define input resolution metadata. Pass "
            "input_shape=(channels, height, width)."
        )
    shape = tuple(input_shape)
    if len(shape) == 4:
        shape = shape[1:]
    if len(shape) != 3 or any(value is None for value in shape):
        raise ValueError("input_shape must be (channels, height, width)")
    shape = tuple(int(value) for value in shape)
    if min(shape) < 1:
        raise ValueError("input_shape dimensions must be positive")
    return shape


def _resolve_input_shape(model, explicit):
    if explicit is not None:
        return _normalize_input_shape(explicit)
    declared = getattr(model, "input_shape", None)
    if declared is None:
        declared = getattr(model, "_dreamlens_input_shape", None)
    return _normalize_input_shape(declared)


def _merge_input_shapes(left, right):
    if left is not None and right is not None and left != right:
        raise ValueError("combined objectives must use the same input_shape")
    return left if left is not None else right


def infer_input_channels(model):
    """Infer channels from an actual PyTorch input module, without guessing."""

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            return int(module.in_channels)
    declared = getattr(model, "input_shape", None)
    if declared is not None:
        return _normalize_input_shape(declared)[0]
    raise ValueError(
        "Could not determine image channels. Pass input_shape=(channels, height, width)."
    )
