"""Composable Xplique-style feature visualization objectives for PyTorch."""

import itertools
from dataclasses import dataclass

import torch

from ..layers import resolve_module
from .losses import dot_cossim


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
    """Combine layer objectives and optimize their Cartesian product.

    The public construction and arithmetic semantics mirror Xplique. PyTorch
    does not store a model input shape on ``nn.Module``, so pass
    ``input_shape=(C, H, W)`` to a factory/constructor when the model itself
    does not expose an ``input_shape`` attribute.
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
        # Preserve Xplique's in-place arithmetic behavior.
        term.multipliers = [-1.0 * multiplier for multiplier in term.multipliers]
        return self + term

    def __mul__(self, factor):
        if not isinstance(factor, (int, float)):
            raise ValueError(f"{factor} is not a number.")
        # Preserve Xplique's in-place arithmetic behavior.
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
        """Compile hooks, objective loss, names, and the NCHW input shape."""

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
            " & ".join(name_parts)
            for name_parts in itertools.product(*self.names)
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
                output = _first_tensor(output)
                mask = _materialize_masks(compiled_masks[output_index], output)
                term = self.funcs[output_index](output, mask)
                loss = term if loss is None else loss + term
                # This ordering intentionally matches Xplique's implementation.
                loss = loss * multipliers[output_index]
            return loss

        layers = [resolve_module(self.model, layer) for layer in self.layers]
        reconfigured_model = _HookedModel(self.model, layers)
        return (
            reconfigured_model,
            objective_function,
            objective_names,
            (len(combinations), *resolved_shape),
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
        """Build an objective that maximizes a complete layer."""

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
        """Build one objective per activation direction."""

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
        """Build one objective per NCHW activation channel."""

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
        """Build one objective per flattened activation neuron."""

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


class _HookedModel(torch.nn.Module):
    """Return selected intermediate outputs while retaining the original model."""

    def __init__(self, model, layers):
        super().__init__()
        self.model = model
        self._outputs = [None] * len(layers)
        self._handles = [
            layer.register_forward_hook(self._make_hook(index))
            for index, layer in enumerate(layers)
        ]

    def _make_hook(self, index):
        def hook(module, inputs, output):
            # A following in-place activation must not mutate the selected
            # module's output after the hook has observed it. ``clone`` keeps
            # the autograd edge while preserving the exact layer value.
            self._outputs[index] = _first_tensor(output).clone()

        return hook

    def forward(self, inputs):
        self._outputs = [None] * len(self._outputs)
        self.model(inputs)
        if any(output is None for output in self._outputs):
            raise RuntimeError("At least one targeted layer was not called by the model.")
        return [_first_tensor(output) for output in self._outputs]

    def close(self):
        while self._handles:
            self._handles.pop().remove()

    def __del__(self):
        if hasattr(self, "_handles"):
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


def _first_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        return _first_tensor(output[0])
    raise TypeError("targeted layers must return a tensor or non-empty tensor sequence")


def _as_choices(value):
    if isinstance(value, list):
        if not value:
            raise ValueError("mask choices cannot be empty")
        return value
    if isinstance(value, torch.Tensor) and value.ndim > 0:
        return list(value)
    try:
        import numpy as np

        if isinstance(value, np.ndarray) and value.ndim > 0:
            return list(value)
    except ImportError:
        pass
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
            "input_shape=(channels, height, width) to Objective or compile/optimize/maco."
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
    """Infer channels from an actual PyTorch input module, without guessing size."""

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            return int(module.in_channels)
    declared = getattr(model, "input_shape", None)
    if declared is not None:
        return _normalize_input_shape(declared)[0]
    raise ValueError(
        "Could not determine image channels. Pass input_shape=(channels, height, width)."
    )


__all__ = ["Objective"]
