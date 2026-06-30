from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FeatureTarget:
    """A layer feature to maximize during visualization.

    Use ``channel=None`` to maximize the complete layer activation. For
    convolutional layers, ``position`` may be ``(y, x)``; for sequence outputs,
    it may be an integer token index.
    """

    layer: object
    channel: Optional[int] = None
    position: object = None
    reduction: str = "mean"
    weight: float = 1.0
    sign: float = 1.0


def first_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)) and output:
        return first_tensor(output[0])
    raise TypeError("Layer output must be a tensor or a non-empty tensor sequence")


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
        output = first_tensor(layer_outputs[self.layer_index])
        selected = _select_channel(output, self.channel, self.position)
        value = _reduce_feature(selected, self.reduction)
        return -self.sign * value


class TargetObjective:
    """Objective generated from one or more ``FeatureTarget`` items."""

    def __init__(self, targets):
        self.targets = [coerce_target(target) for target in _as_list(targets)]
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
            selected = (
                tensor
                if target.channel is None
                else _select_channel(tensor, target.channel, target.position)
            )
            value = _reduce_feature(selected, target.reduction)
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
    position=None,
    reduction="mean",
    weight=1.0,
    sign=1.0,
):
    return FeatureTarget(
        layer=layer,
        channel=channel,
        position=position,
        reduction=reduction,
        weight=weight,
        sign=sign,
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


def _as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


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
