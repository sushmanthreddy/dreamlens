from dataclasses import dataclass
from collections.abc import Iterable
from typing import Optional, Tuple

import torch


def as_list(value):
    """Normalize one object or a non-string iterable to a list."""

    if isinstance(value, (str, torch.nn.Module)) or not isinstance(value, Iterable):
        return [value]
    return list(value)


@dataclass(frozen=True)
class LayerInfo:
    name: str
    module_type: str
    output_shape: Optional[Tuple[int, ...]] = None
    channels: Optional[int] = None
    spatial_shape: Optional[Tuple[int, ...]] = None
    supported: bool = False


def resolve_module(model, layer):
    """Resolve a module name, leaf-module index, or module object."""

    if isinstance(layer, torch.nn.Module):
        return layer
    if isinstance(layer, int):
        modules = [
            module
            for name, module in model.named_modules()
            if name and not any(module.children())
        ]
        try:
            return modules[layer]
        except IndexError as exc:
            raise IndexError(
                "Layer index {} is out of range for {} leaf modules.".format(
                    layer, len(modules)
                )
            ) from exc
    if not isinstance(layer, str):
        raise TypeError(
            "layer must be a module name string, leaf-module index, or torch.nn.Module"
        )
    if hasattr(model, "get_submodule"):
        try:
            return model.get_submodule(layer)
        except AttributeError:
            pass
    modules = dict(model.named_modules())
    try:
        return modules[layer]
    except KeyError as exc:
        raise KeyError("Could not find layer {!r} in model.".format(layer)) from exc


def first_tensor_output(output):
    """Return the first tensor from a tensor or nested tuple/list output."""

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        return first_tensor_output(output[0])
    raise TypeError("layer output must be a tensor or non-empty tensor sequence")


class LayerCapture:
    """Reusable forward-hook capture for every DreamLens subsystem."""

    def __init__(self, module, clone=False):
        if not isinstance(module, torch.nn.Module):
            raise TypeError("module must be a torch.nn.Module")
        self.module = module
        self.clone = bool(clone)
        self.output = None
        self._handle = None

    def open(self):
        if self._handle is None:
            self._handle = self.module.register_forward_hook(self._hook)
        return self

    def close(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def clear(self):
        self.output = None

    def tensor_output(self):
        if self.output is None:
            raise RuntimeError("The requested layer was not called by the model.")
        return self.output

    def _hook(self, module, inputs, output):
        output = first_tensor_output(output)
        self.output = output.clone() if self.clone else output

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def list_layers(
    model,
    sample_input=None,
    preprocess=None,
    device=None,
    include_root=False,
    include_containers=False,
    activation_format="NCHW",
):
    """List model layers, optionally probing tensor output shapes.

    A layer is marked supported when its output is a tensor with shape compatible
    with activation collection and feature visualization: ``[N, C]``,
    ``[N, L, C]``, or image-like ``[N, C, H, W]`` / ``[N, H, W, C]``.
    """

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")

    output_shapes = {}
    if sample_input is not None:
        output_shapes = probe_layer_outputs(
            model,
            sample_input=sample_input,
            preprocess=preprocess,
            device=device,
        )

    infos = []
    for name, module in model.named_modules():
        if name == "" and not include_root:
            continue
        if not include_containers and any(module.children()):
            continue
        output_shape = output_shapes.get(name)
        channels, spatial_shape, supported = _layer_shape_info(
            output_shape,
            activation_format=activation_format,
        )
        infos.append(
            LayerInfo(
                name=name,
                module_type=module.__class__.__name__,
                output_shape=output_shape,
                channels=channels,
                spatial_shape=spatial_shape,
                supported=supported,
            )
        )
    return infos


def supported_layers(*args, **kwargs):
    """Return only layers with tensor outputs supported by this package."""

    return [info for info in list_layers(*args, **kwargs) if info.supported]


def probe_layer_outputs(model, sample_input, preprocess=None, device=None):
    """Run one forward pass and collect output shapes for named modules."""

    device = torch.device(device) if device is not None else model_device(model)
    model.to(device)
    model.eval()
    preprocess = (lambda x: x) if preprocess is None else preprocess

    sample_input = sample_input.to(device)
    output_shapes = {}
    handles = []

    def make_hook(name):
        def hook(module, inputs, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            if isinstance(output, torch.Tensor):
                output_shapes[name] = tuple(output.shape)

        return hook

    for name, module in model.named_modules():
        if name == "":
            continue
        handles.append(module.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            model(preprocess(sample_input))
    finally:
        for handle in handles:
            handle.remove()

    return output_shapes


def model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _layer_shape_info(output_shape, activation_format="NCHW"):
    if output_shape is None or len(output_shape) < 2:
        return None, None, False

    if len(output_shape) == 2:
        return output_shape[-1], None, True
    if len(output_shape) == 3:
        return output_shape[-1], (output_shape[1],), True
    if len(output_shape) == 4:
        activation_format = activation_format.upper()
        if activation_format == "NCHW":
            return output_shape[1], output_shape[2:4], True
        if activation_format == "NHWC":
            return output_shape[-1], output_shape[1:3], True
    return None, None, False
