from contextlib import contextmanager
from dataclasses import dataclass, field
import numbers
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .image_parameters import (
    fft_image,
    fft_to_rgb,
    get_fft_scale,
    init_maco_buffer,
    maco_image_parametrization,
    to_valid_grayscale,
    to_valid_rgb,
)
from .layers import model_device
from .objectives import Objective, _normalize_input_shape, infer_input_channels
from .transforms import compose_transformations, generate_standard_transformations


@dataclass(frozen=True)
class TransformConfig:
    """Robustness transforms used during image optimization."""

    rotate_degrees: float = 15.0
    scale_min: float = 0.5
    scale_max: float = 1.2
    translate_x: float = 0.0
    translate_y: float = 0.0
    transforms: object = None


@dataclass(frozen=True)
class RenderConfig:
    """Configuration for feature maximization from noise or an image."""

    width: int = 256
    height: int = 256
    steps: int = 120
    lr: float = 9e-3
    weight_decay: float = 0.0
    grad_clip: Optional[float] = 1.0
    transform: TransformConfig = field(default_factory=TransformConfig)
    preprocess: object = None
    optimizer_cls: object = None
    fft: bool = True
    decorrelate: bool = True
    attempts: int = 1
    noise_std: float = 0.01
    parameterization: str = "lucid"

    @classmethod
    def reference(
        cls,
        width=256,
        height=256,
        steps=120,
        lr=9e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        transform=None,
    ):
        """Reference-parameterized config for deterministic render parity."""

        return cls(
            width=width,
            height=height,
            steps=steps,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            transform=TransformConfig() if transform is None else transform,
            preprocess=None,
            noise_std=0.01,
            parameterization="reference",
        )


@dataclass(frozen=True)
class AmplifyConfig:
    """Configuration for DreamLens feature amplification from an input image."""

    steps: int = 120
    lr: float = 3e-4
    weight_decay: float = 1e-1
    grad_clip: Optional[float] = 0.1
    transform: TransformConfig = field(
        default_factory=lambda: TransformConfig(translate_x=0.1, translate_y=0.1)
    )
    preprocess: object = None
    optimizer_cls: object = None
    start: str = "input"
    target_mode: str = "paired"
    preserve_weight: float = 0.0
    variation_weight: float = 0.0
    noise_std: float = 0.01
    fft: bool = True
    decorrelate: bool = True
    frequency_decay: float = 1.0
    raw_scale: float = 0.25
    fft_norm: object = None
    parameterization: str = "lucid"

    @classmethod
    def dream(cls, steps=220, lr=2e-2):
        """Noise-start config for strong DreamLens amplification."""

        return cls(
            steps=steps,
            lr=lr,
            weight_decay=1e-3,
            grad_clip=1.0,
            start="noise",
            target_mode="paired",
            preserve_weight=0.0,
            variation_weight=0.0,
            noise_std=0.05,
            frequency_decay=1.0,
            raw_scale=0.75,
            fft_norm=None,
            parameterization="lucid",
            transform=TransformConfig(
                rotate_degrees=15,
                scale_min=0.5,
                scale_max=1.2,
                translate_x=0.1,
                translate_y=0.1,
            ),
        )

    @classmethod
    def reference(cls, steps=120, lr=9e-3):
        """Reference-parameterized config for exact parity checks."""

        return cls(
            steps=steps,
            lr=lr,
            weight_decay=1e-3,
            grad_clip=1.0,
            start="noise",
            target_mode="paired",
            preserve_weight=0.0,
            variation_weight=0.0,
            noise_std=0.01,
            parameterization="reference",
            transform=TransformConfig(
                rotate_degrees=15,
                scale_min=0.5,
                scale_max=1.2,
                translate_x=0.1,
                translate_y=0.1,
            ),
        )


@dataclass(frozen=True)
class MacoConfig:
    """Configuration for fixed-magnitude, phase-only MaCo visualization."""

    width: int = 512
    height: int = 512
    input_shape: tuple[int, int, int] = (3, 224, 224)
    steps: int = 256
    lr: float = 1.0
    crops: int = 32
    noise_intensity: object = 0.08
    box_size: object = None
    values_range: tuple[float, float] = (0.0, 1.0)
    preprocess: object = None
    optimizer_cls: object = None

    def __post_init__(self):
        if not isinstance(self.width, int) or not isinstance(self.height, int):
            raise TypeError("width and height must be integers")
        if self.width < 1 or self.height < 1:
            raise ValueError("width and height must be positive")
        if not isinstance(self.input_shape, (tuple, list)):
            raise TypeError("input_shape must be (channels, height, width)")
        if len(self.input_shape) != 3 or min(self.input_shape) < 1:
            raise ValueError("input_shape must be (channels, height, width)")
        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        if self.crops < 0:
            raise ValueError("crops must be >= 0")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if not isinstance(self.values_range, (tuple, list)) or len(self.values_range) != 2:
            raise ValueError("values_range must contain exactly two values")
        if min(self.values_range) == max(self.values_range):
            raise ValueError("values_range endpoints must be different")


@dataclass(frozen=True)
class OptimizationResult:
    """Return object for the project-owned high-level API."""

    image: object
    losses: list[float]
    objective_value: Optional[float] = None
    attempt_index: int = 0
    transparency: object = None
    metadata: object = None

    def save(self, filename):
        if hasattr(self.image, "save"):
            return self.image.save(filename)
        return _save_chw_tensor(self.as_chw(), filename)

    def as_chw(self, device=None):
        if hasattr(self.image, "as_chw"):
            return self.image.as_chw(device=device)
        tensor = torch.as_tensor(self.image)
        if tensor.dim() == 4:
            tensor = tensor[0]
        if tensor.dim() != 3:
            raise ValueError("result image must be CHW or NCHW")
        if device is not None:
            tensor = tensor.to(device)
        return tensor.detach().cpu()

    def as_hwc(self, device=None):
        if hasattr(self.image, "as_hwc"):
            return self.image.as_hwc(device=device)
        return self.as_chw(device=device).permute(1, 2, 0)

    def as_nchw(self, device=None):
        if hasattr(self.image, "as_nchw"):
            return self.image.as_nchw(device=device)
        if hasattr(self.image, "forward"):
            return self.image.forward(device=device).detach().cpu()
        return self.as_chw(device=device).unsqueeze(0)

    def transparency_chw(self, device=None):
        if self.transparency is None:
            return None
        tensor = torch.as_tensor(self.transparency)
        if tensor.dim() == 4:
            tensor = tensor[0]
        if tensor.dim() != 3:
            raise ValueError("transparency must be CHW or NCHW")
        if device is not None:
            tensor = tensor.to(device)
        return tensor.detach().cpu()

    def save_transparency(self, filename):
        transparency = self.transparency_chw()
        if transparency is None:
            raise ValueError("this result has no transparency map")
        heatmap = transparency.abs().mean(dim=0, keepdim=True)
        heatmap = heatmap - heatmap.amin()
        heatmap = heatmap / heatmap.amax().clamp_min(1e-12)
        return _save_chw_tensor(heatmap, filename)

    def __array__(self, dtype=None):
        array = self.as_hwc().numpy()
        if dtype is not None:
            array = array.astype(dtype)
        return array


def _save_chw_tensor(tensor, filename):
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Saving images requires pillow to be installed.") from exc

    tensor = torch.as_tensor(tensor).detach().cpu().float().clamp(0.0, 1.0)
    if tensor.dim() != 3:
        raise ValueError("image tensor must be CHW")
    if tensor.shape[0] == 1:
        array = (tensor[0].numpy() * 255).astype("uint8")
    elif tensor.shape[0] in (3, 4):
        array = (tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
    else:
        raise ValueError("image tensor must have 1, 3, or 4 channels")
    Image.fromarray(array).save(filename)


def optimize(
    objective,
    optimizer=None,
    nb_steps=256,
    use_fft=True,
    fft_decay=0.85,
    std=0.01,
    regularizers=None,
    image_normalizer="sigmoid",
    values_range=(0, 1),
    transformations="standard",
    warmup_steps=False,
    custom_shape=(512, 512),
    save_every=None,
    input_shape=None,
    device=None,
    preprocess=None,
    progress_callback=None,
):
    """Run the exact composable-objective optimizer from the root module."""

    if not isinstance(objective, Objective):
        raise TypeError("objective must be an Objective")
    nb_steps = int(nb_steps)
    if nb_steps < 1:
        raise ValueError("nb_steps must be >= 1")
    if save_every is not None and int(save_every) < 1:
        raise ValueError("save_every must be >= 1")
    low, high = min(values_range), max(values_range)
    values_range = (low, high)

    resolved_input_shape = _optimization_input_shape(
        objective,
        input_shape=input_shape,
        custom_shape=custom_shape,
    )
    model, objective_function, objective_names, compiled_shape = objective.compile(
        input_shape=resolved_input_shape
    )
    combination_count, channels, input_height, input_width = compiled_shape

    if custom_shape is None:
        canvas_height, canvas_width = input_height, input_width
    else:
        if len(custom_shape) != 2:
            model.close()
            raise ValueError("custom_shape must be (height, width) or None")
        canvas_height, canvas_width = (int(value) for value in custom_shape)
        if min(canvas_height, canvas_width) < 1:
            model.close()
            raise ValueError("custom_shape dimensions must be positive")
    image_shape = (combination_count, channels, canvas_height, canvas_width)

    if channels not in (1, 3):
        model.close()
        raise ValueError("feature visualization supports 1 or 3 input channels")
    if transformations == "standard":
        transformations = generate_standard_transformations(
            size=canvas_height,
            channels=channels,
        )
    elif isinstance(transformations, (list, tuple)):
        transformations = compose_transformations(transformations)
    elif transformations is not None and not callable(transformations):
        model.close()
        raise TypeError("transformations must be 'standard', a callable, a list, or None")

    to_valid_image = to_valid_rgb if channels == 3 else to_valid_grayscale
    device = torch.device(device) if device is not None else model_device(objective.model)
    objective.model.to(device)

    if use_fft:
        inputs = torch.nn.Parameter(fft_image(image_shape, std=std, device=device))
        fft_scale = get_fft_scale(
            canvas_height,
            canvas_width,
            decay_power=fft_decay,
            device=device,
        )

        def image_param(buffer):
            return to_valid_image(
                fft_to_rgb(image_shape, buffer, fft_scale),
                image_normalizer,
                values_range,
            )

    else:
        inputs = torch.nn.Parameter(
            torch.randn(image_shape, dtype=torch.float32, device=device) * float(std)
        )

        def image_param(buffer):
            return to_valid_image(buffer, image_normalizer, values_range)

    optimizer = _prepare_optimizer(
        optimizer,
        inputs,
        default_cls=torch.optim.Adam,
        default_lr=0.05,
    )
    optimisation_step = _get_optimisation_step(
        objective_function=objective_function,
        image_param=image_param,
        input_shape=compiled_shape,
        transformations=transformations,
        regularizers=regularizers,
        preprocess=preprocess,
    )

    images_optimized = []
    try:
        with _frozen_eval_model(objective.model):
            if warmup_steps:
                with _open_relu_gradients(objective.model):
                    for _ in range(int(warmup_steps)):
                        optimisation_step(model, inputs, optimizer)

            for step_index in range(nb_steps):
                optimisation_step(model, inputs, optimizer)
                if progress_callback is not None:
                    progress_callback(step_index + 1, nb_steps)
                last_iteration = step_index == nb_steps - 1
                should_save = save_every and (step_index + 1) % int(save_every) == 0
                if should_save or last_iteration:
                    images_optimized.append(image_param(inputs).detach().clone())
    finally:
        model.close()

    return images_optimized, objective_names


def _get_optimisation_step(
    objective_function,
    image_param,
    input_shape,
    transformations=None,
    regularizers=None,
    preprocess=None,
):
    target_height, target_width = input_shape[-2:]

    def step(model, inputs, optimizer):
        optimizer.zero_grad(set_to_none=True)
        images = image_param(inputs)
        if transformations:
            images = transformations(images)
        images = F.interpolate(
            images,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
        model_inputs = images if preprocess is None else preprocess(images)
        model_outputs = model(model_inputs)
        loss = objective_function(model_outputs)
        if regularizers:
            for regularizer in regularizers:
                loss = loss - regularizer(images)
        score = loss.sum()
        (-score).backward()
        optimizer.step()
        return inputs.grad

    return step


def _optimization_input_shape(objective, input_shape, custom_shape):
    if input_shape is not None:
        return _normalize_input_shape(input_shape)
    if objective.input_shape is not None:
        return objective.input_shape
    declared = getattr(objective.model, "input_shape", None)
    if declared is None:
        declared = getattr(objective.model, "_dreamlens_input_shape", None)
    if declared is not None:
        return _normalize_input_shape(declared)
    if custom_shape is None:
        return _normalize_input_shape(None)
    if len(custom_shape) != 2:
        raise ValueError("custom_shape must be (height, width) or None")
    channels = infer_input_channels(objective.model)
    return channels, int(custom_shape[0]), int(custom_shape[1])


def _prepare_optimizer(optimizer, parameter, default_cls, default_lr):
    if optimizer is None:
        return default_cls([parameter], lr=default_lr, eps=1e-7)
    if isinstance(optimizer, torch.optim.Optimizer):
        if not optimizer.param_groups:
            raise ValueError("optimizer must have at least one parameter group")
        optimizer.state.clear()
        optimizer.param_groups[0]["params"] = [parameter]
        for group in optimizer.param_groups[1:]:
            group["params"] = []
        return optimizer
    if callable(optimizer):
        try:
            candidate = optimizer([parameter])
        except TypeError as first_error:
            try:
                candidate = optimizer([parameter], lr=0.01)
            except TypeError:
                raise TypeError(
                    "optimizer callables must accept an iterable of parameters"
                ) from first_error
        if not isinstance(candidate, torch.optim.Optimizer):
            raise TypeError("optimizer callable must return torch.optim.Optimizer")
        return candidate
    raise TypeError("optimizer must be a torch optimizer, factory/class, or None")


@contextmanager
def _frozen_eval_model(model):
    training = model.training
    requires_grad = [parameter.requires_grad for parameter in model.parameters()]
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, original in zip(model.parameters(), requires_grad):
            parameter.requires_grad_(original)
        model.train(training)


@contextmanager
def _open_relu_gradients(model):
    handles = []
    inplace_states = []

    def open_backward(module, grad_input, grad_output):
        if not grad_output:
            return grad_input
        return (grad_output[0],)

    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            inplace_states.append((module, module.inplace))
            module.inplace = False
            handles.append(module.register_full_backward_hook(open_backward))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        for module, inplace in inplace_states:
            module.inplace = inplace


def maco(
    objective,
    optimizer=None,
    maco_dataset=None,
    nb_steps=256,
    noise_intensity=0.08,
    box_size=None,
    nb_crops=32,
    values_range=(-1, 1),
    custom_shape=(512, 512),
    input_shape=None,
    device=None,
    preprocess=None,
):
    """Optimize phase with a fixed Fourier magnitude and return importance."""

    if not isinstance(objective, Objective):
        raise TypeError("objective must be an Objective")
    nb_steps = int(nb_steps)
    nb_crops = int(nb_crops)
    if nb_steps < 1:
        raise ValueError("nb_steps must be >= 1")
    if nb_crops < 0:
        raise ValueError("nb_crops must be >= 0")
    values_range = (min(values_range), max(values_range))

    resolved_input_shape = _optimization_input_shape(
        objective,
        input_shape=input_shape,
        custom_shape=custom_shape,
    )
    model, objective_function, _, compiled_shape = objective.compile(
        input_shape=resolved_input_shape
    )
    combinations, channels, input_height, input_width = compiled_shape
    if combinations != 1:
        model.close()
        raise AssertionError("You can only optimize one objective at a time with MaCo.")
    if channels == 1 and maco_dataset is None:
        model.close()
        raise ValueError("For grayscale images, a dataset is required to compute the buffer.")

    get_box_size = _schedule(
        box_size,
        default=torch.as_tensor(np.linspace(0.5, 0.05, nb_steps), dtype=torch.float32),
        argument_name="box_size",
    )
    get_noise_intensity = _schedule(
        noise_intensity,
        default=torch.as_tensor(np.logspace(0, -4, nb_steps), dtype=torch.float32),
        argument_name="noise_intensity",
    )

    if custom_shape is None:
        image_height, image_width = input_height, input_width
    else:
        if len(custom_shape) != 2:
            model.close()
            raise ValueError("custom_shape must be (height, width) or None")
        image_height, image_width = (int(value) for value in custom_shape)
        if min(image_height, image_width) < 1:
            model.close()
            raise ValueError("custom_shape dimensions must be positive")

    device = torch.device(device) if device is not None else model_device(objective.model)
    objective.model.to(device)
    magnitude, phase_value = init_maco_buffer(
        (channels, image_height, image_width),
        dataset=maco_dataset,
        device=device,
        data_format="CHW",
    )
    magnitude = magnitude.to(device=device, dtype=torch.float32)
    phase = torch.nn.Parameter(phase_value.to(device=device, dtype=torch.float32))
    optimizer = _prepare_optimizer(
        optimizer,
        phase,
        default_cls=torch.optim.NAdam,
        default_lr=1.0,
    )
    transparency = torch.zeros(
        (channels, image_height, image_width),
        dtype=torch.float32,
        device=device,
    )

    try:
        with _frozen_eval_model(objective.model):
            for step_index in range(nb_steps):
                grads_phase, grads_image = maco_optimisation_step(
                    model=model,
                    objective_function=objective_function,
                    magnitude=magnitude,
                    phase=phase,
                    box_average_size=get_box_size(step_index),
                    noise_std=get_noise_intensity(step_index),
                    nb_crops=nb_crops,
                    values_range=values_range,
                    input_size=(input_height, input_width),
                    preprocess=preprocess,
                )
                optimizer.zero_grad(set_to_none=True)
                phase.grad = -grads_phase.detach()
                optimizer.step()
                transparency = transparency + torch.abs(
                    _resize_chw(grads_image, (image_height, image_width))
                )
    finally:
        model.close()

    image = maco_image_parametrization(magnitude, phase, values_range)
    image = _resize_chw(image, (image_height, image_width))
    return image.detach(), transparency.detach()


def maco_optimisation_step(
    model,
    objective_function,
    magnitude,
    phase,
    box_average_size,
    noise_std,
    nb_crops,
    values_range,
    input_size=None,
    preprocess=None,
):
    """Compute one exact MaCo phase and image-gradient step."""

    image = maco_image_parametrization(magnitude, phase, values_range)
    if input_size is None:
        input_size = image.shape[-2:]
    if nb_crops == 0:
        crops = F.interpolate(
            image.unsqueeze(0),
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
    else:
        dtype, device = image.dtype, image.device
        center_x = 0.5 + torch.randn(nb_crops, dtype=dtype, device=device) * 0.15
        center_y = 0.5 + torch.randn(nb_crops, dtype=dtype, device=device) * 0.15
        average = torch.as_tensor(box_average_size, dtype=dtype, device=device)
        delta_x = average + torch.randn(nb_crops, dtype=dtype, device=device) * 0.05
        delta_x = torch.clamp(delta_x, 0.05, 1.0)
        boxes = torch.stack(
            [
                center_x - delta_x * 0.5,
                center_y - delta_x * 0.5,
                center_x + delta_x * 0.5,
                center_y + delta_x * 0.5,
            ],
            dim=-1,
        )
        crops = _crop_and_resize(image, boxes, input_size)

    noise_std = torch.as_tensor(noise_std, dtype=crops.dtype, device=crops.device)
    crops = crops + torch.randn_like(crops) * noise_std
    crops = crops + (torch.rand_like(crops) - 0.5) * noise_std

    model_inputs = crops if preprocess is None else preprocess(crops)
    model_outputs = model(model_inputs)
    loss = torch.mean(objective_function(model_outputs))
    grads_phase, grads_image = torch.autograd.grad(loss, (phase, image))
    return grads_phase, grads_image


def _crop_and_resize(image, boxes, output_size):
    output_height, output_width = (int(value) for value in output_size)
    y1, x1, y2, x2 = boxes.unbind(dim=-1)
    y_fraction = _sample_fractions(output_height, image)
    x_fraction = _sample_fractions(output_width, image)
    ys = y1[:, None] + (y2 - y1)[:, None] * y_fraction[None, :]
    xs = x1[:, None] + (x2 - x1)[:, None] * x_fraction[None, :]
    grid_y = ys[:, :, None].expand(-1, output_height, output_width)
    grid_x = xs[:, None, :].expand(-1, output_height, output_width)
    grid = torch.stack((grid_x * 2.0 - 1.0, grid_y * 2.0 - 1.0), dim=-1)
    expanded = image.unsqueeze(0).expand(boxes.shape[0], -1, -1, -1)
    return F.grid_sample(
        expanded,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


def _sample_fractions(size, reference):
    if size == 1:
        return reference.new_tensor([0.5])
    return torch.linspace(
        0.0,
        1.0,
        size,
        dtype=reference.dtype,
        device=reference.device,
    )


def _schedule(value, default, argument_name):
    if value is None:
        values = default

        def from_default(step_index):
            return values[step_index]

        return from_default
    if callable(value):
        return value
    if isinstance(value, numbers.Real):

        def constant(_):
            return float(value)

        return constant
    raise ValueError(f"{argument_name} must be a function, a float, or None.")


def _resize_chw(image, size):
    if tuple(image.shape[-2:]) == tuple(size):
        return image
    return F.interpolate(
        image.unsqueeze(0),
        size=size,
        mode="bilinear",
        align_corners=False,
    )[0]
