from contextlib import contextmanager
from dataclasses import dataclass, field
import numbers
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .image_parameters import (
    FeatureAccentuationCanvas,
    fft_image,
    fft_to_rgb,
    get_fft_scale,
    init_maco_buffer,
    maco_image_parametrization,
    to_valid_grayscale,
    to_valid_rgb,
)
from .layers import LayerCapture, model_device, resolve_module
from .objectives import Objective, _normalize_input_shape, infer_input_channels
from .transforms import (
    compose_transformations,
    feature_accentuation_transforms,
    generate_standard_transformations,
)


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
class FeatureAccentuationConfig:
    """Configuration for Faccent-style feature accentuation of a real image."""

    width: int = 512
    height: int = 512
    input_shape: tuple[int, int, int] = (3, 224, 224)
    steps: int = 99
    lr: float = 5e-2
    crops: int = 16
    crop_min: float = 0.05
    crop_max: float = 0.99
    noise_std: float = 0.02
    regularization_strength: float = 1.0
    parameterization: str = "fourier"
    magnitude_source: str = "image"
    use_magnitude_gate: bool = True
    magnitude_gate_init: float = 5.0
    desaturation: Optional[float] = None
    frequency_decay: float = 1.0
    color_decorrelate: bool = True
    center_crop: bool = True
    resize_mode: str = "nearest"
    preprocess: object = None
    optimizer_cls: object = None
    grad_clip: Optional[float] = None
    regularizer_in_transparency: bool = False
    checkpoint_steps: object = None

    def __post_init__(self):
        if not isinstance(self.width, int) or not isinstance(self.height, int):
            raise TypeError("width and height must be integers")
        if min(self.width, self.height) < 1:
            raise ValueError("width and height must be positive")
        if not isinstance(self.input_shape, (tuple, list)) or len(self.input_shape) != 3:
            raise TypeError("input_shape must be (channels, height, width)")
        if tuple(self.input_shape)[0] != 3 or min(self.input_shape) < 1:
            raise ValueError("feature accentuation input_shape must be RGB (3, H, W)")
        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.crops < 1:
            raise ValueError("crops must be >= 1")
        if not 0 < self.crop_min <= self.crop_max <= 1:
            raise ValueError("crop bounds must satisfy 0 < crop_min <= crop_max <= 1")
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative")
        if self.regularization_strength < 0:
            raise ValueError("regularization_strength must be non-negative")
        if self.parameterization not in {"fourier", "fourier_phase"}:
            raise ValueError("parameterization must be 'fourier' or 'fourier_phase'")
        if self.magnitude_source not in {"image", "imagenet"}:
            raise ValueError("magnitude_source must be 'image' or 'imagenet'")
        if self.parameterization == "fourier" and self.magnitude_source != "image":
            raise ValueError(
                "magnitude_source is only configurable for 'fourier_phase'"
            )
        if self.desaturation is not None and self.desaturation <= 0:
            raise ValueError("desaturation must be positive or None")
        if self.frequency_decay <= 0:
            raise ValueError("frequency_decay must be positive")
        if self.resize_mode not in {"nearest", "bilinear", "bicubic"}:
            raise ValueError("resize_mode must be 'nearest', 'bilinear', or 'bicubic'")
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive or None")
        if self.checkpoint_steps is not None:
            steps = tuple(int(step) for step in self.checkpoint_steps)
            if any(step < 0 or step >= self.steps for step in steps):
                raise ValueError("checkpoint_steps must be between 0 and steps - 1")
            if len(set(steps)) != len(steps):
                raise ValueError("checkpoint_steps must not contain duplicates")
            object.__setattr__(self, "checkpoint_steps", steps)


@dataclass(frozen=True)
class OptimizationResult:
    """Return object for the project-owned high-level API."""

    image: object
    losses: list[float]
    objective_value: Optional[float] = None
    attempt_index: int = 0
    transparency: object = None
    metadata: object = None
    checkpoints: object = None
    transparency_checkpoints: object = None

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

    def as_accentuation_rgba(
        self,
        percentile=20.0,
        blur_sigma=2.0,
        checkpoint=None,
    ):
        """Return Faccent's normalized RGB plus attribution-derived alpha.

        Faccent notebook figures are not raw optimized images. They globally
        normalize contrast and use the accumulated absolute image gradient as
        a blurred opacity mask. ``checkpoint`` is an optimization step captured
        through ``FeatureAccentuationConfig.checkpoint_steps``.
        """

        image, transparency = self._accentuation_tensors(checkpoint)
        percentile = float(percentile)
        blur_sigma = float(blur_sigma)
        if not 0.0 <= percentile <= 100.0:
            raise ValueError("percentile must be between 0 and 100")
        if blur_sigma < 0:
            raise ValueError("blur_sigma must be non-negative")

        image = image.float()
        image = image - image.mean()
        image = image / image.std().clamp_min(1e-12)
        image = image - image.amin()
        image = image / image.amax().clamp_min(1e-12)

        if percentile in (0.0, 100.0):
            alpha = torch.ones_like(image[:1])
        else:
            alpha = transparency.float().mean(dim=0, keepdim=True)
            threshold = torch.quantile(alpha.flatten(), 1.0 - percentile / 100.0)
            alpha = alpha.clamp(max=threshold)
            alpha = alpha / alpha.amax().clamp_min(1e-12)
            alpha = _gaussian_blur_map(alpha, blur_sigma)
        return torch.cat([image.clamp(0.0, 1.0), alpha.clamp(0.0, 1.0)], dim=0)

    def save_accentuation(
        self,
        filename,
        percentile=20.0,
        blur_sigma=2.0,
        checkpoint=None,
        background="white",
    ):
        """Save the Faccent-style masked result, optionally composited."""

        rgba = self.as_accentuation_rgba(
            percentile=percentile,
            blur_sigma=blur_sigma,
            checkpoint=checkpoint,
        )
        if background is None:
            return _save_chw_tensor(rgba, filename)
        if background not in {"white", "black"}:
            raise ValueError("background must be 'white', 'black', or None")
        alpha = rgba[3:4]
        fill = 1.0 if background == "white" else 0.0
        composite = rgba[:3] * alpha + fill * (1.0 - alpha)
        return _save_chw_tensor(composite, filename)

    def _accentuation_tensors(self, checkpoint):
        if checkpoint is None:
            image = self.as_chw()
            transparency = self.transparency_chw()
        else:
            checkpoint = int(checkpoint)
            if not self.checkpoints or checkpoint not in self.checkpoints:
                available = sorted(self.checkpoints or {})
                raise KeyError(
                    "checkpoint {} was not captured; available steps: {}".format(
                        checkpoint,
                        available,
                    )
                )
            image = torch.as_tensor(self.checkpoints[checkpoint]).detach().cpu()
            if image.dim() == 4:
                image = image[0]
            transparency = torch.as_tensor(
                self.transparency_checkpoints[checkpoint]
            ).detach().cpu()
            if transparency.dim() == 4:
                transparency = transparency[0]
        if transparency is None:
            raise ValueError("this result has no transparency map")
        if image.shape != transparency.shape:
            raise ValueError("image and transparency must have matching CHW shapes")
        return image, transparency

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


def _gaussian_blur_map(image, sigma, truncate=4.0):
    if sigma == 0:
        return image
    radius = int(float(truncate) * float(sigma) + 0.5)
    if radius == 0:
        return image
    coordinates = torch.arange(
        -radius,
        radius + 1,
        dtype=image.dtype,
        device=image.device,
    )
    kernel = torch.exp(-0.5 * (coordinates / float(sigma)) ** 2)
    kernel = kernel / kernel.sum()
    padded = F.pad(image.unsqueeze(0), (radius, radius, 0, 0), mode="reflect")
    blurred = F.conv2d(padded, kernel.view(1, 1, 1, -1))
    padded = F.pad(blurred, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(padded, kernel.view(1, 1, -1, 1))[0]


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


def feature_accentuation(
    objective,
    image,
    regularization_layer=None,
    optimizer=None,
    image_parameter=None,
    nb_steps=99,
    nb_crops=16,
    crop_min=0.05,
    crop_max=0.99,
    noise_std=0.02,
    regularization_strength=1.0,
    custom_shape=(512, 512),
    input_shape=(3, 224, 224),
    parameterization="fourier",
    magnitude_source="image",
    use_magnitude_gate=True,
    magnitude_gate_init=5.0,
    desaturation=None,
    frequency_decay=1.0,
    color_decorrelate=True,
    center_crop=True,
    resize_mode="nearest",
    grad_clip=None,
    regularizer_in_transparency=False,
    checkpoint_steps=None,
    device=None,
    preprocess=None,
    learning_rate=5e-2,
):
    """Accentuate a target while preserving a reference feature representation.

    This is a native PyTorch implementation of Faccent's core algorithm. The
    default trains a seed-initialized, frequency-preconditioned complex Fourier
    buffer; the optional phase mode constrains magnitude. Candidate and fixed
    reference receive matched crops/noise. A one-time gradient ratio balances
    target maximization against L2 feature preservation.
    """

    if not isinstance(objective, Objective):
        raise TypeError("objective must be an Objective")
    nb_steps, nb_crops = int(nb_steps), int(nb_crops)
    if nb_steps < 1:
        raise ValueError("nb_steps must be >= 1")
    if nb_crops < 1:
        raise ValueError("nb_crops must be >= 1")
    if regularization_strength < 0:
        raise ValueError("regularization_strength must be non-negative")
    if regularization_strength and regularization_layer is None:
        raise ValueError(
            "regularization_layer is required when regularization_strength is non-zero"
        )
    checkpoint_steps = (
        set() if checkpoint_steps is None else {int(step) for step in checkpoint_steps}
    )
    if any(step < 0 or step >= nb_steps for step in checkpoint_steps):
        raise ValueError("checkpoint_steps must be between 0 and nb_steps - 1")
    if custom_shape is None or len(custom_shape) != 2:
        raise ValueError("custom_shape must be (height, width)")
    canvas_height, canvas_width = (int(value) for value in custom_shape)
    resolved_input_shape = _optimization_input_shape(
        objective,
        input_shape=input_shape,
        custom_shape=custom_shape,
    )
    if resolved_input_shape[0] != 3:
        raise ValueError("feature accentuation currently requires RGB model inputs")

    hooked_model, objective_function, objective_names, compiled_shape = objective.compile(
        input_shape=resolved_input_shape
    )
    combinations = compiled_shape[0]
    if combinations != 1:
        hooked_model.close()
        raise AssertionError(
            "You can only optimize one objective at a time with feature accentuation."
        )

    device = (
        torch.device(device)
        if device is not None
        else model_device(objective.model)
    )
    regularization_capture = None
    try:
        objective.model.to(device)
        if image_parameter is None:
            image_parameter = FeatureAccentuationCanvas(
                image=image,
                height=canvas_height,
                width=canvas_width,
                device=device,
                parameterization=parameterization,
                magnitude_source=magnitude_source,
                use_magnitude_gate=use_magnitude_gate,
                magnitude_gate_init=magnitude_gate_init,
                desaturation=desaturation,
                frequency_decay=frequency_decay,
                color_decorrelate=color_decorrelate,
                center_crop=center_crop,
                resize_mode=resize_mode,
            )
        elif not isinstance(image_parameter, FeatureAccentuationCanvas):
            raise TypeError("image_parameter must be a FeatureAccentuationCanvas")
        else:
            image_parameter = image_parameter.to(device)

        parameters = [
            parameter
            for parameter in image_parameter.parameters()
            if parameter.requires_grad
        ]
        if not parameters:
            raise ValueError("image_parameter must contain trainable phase/gate parameters")
        optimizer = _prepare_parameter_optimizer(
            optimizer,
            parameters,
            default_cls=torch.optim.Adam,
            default_lr=float(learning_rate),
        )
        reference = image_parameter.reference_image.detach()
        transparency = torch.zeros_like(reference[0])
        checkpoints = {}
        transparency_checkpoints = {}
        losses = []
        target_losses = []
        regularization_distances = []
        if regularization_layer is not None:
            regularization_capture = LayerCapture(
                resolve_module(objective.model, regularization_layer),
                clone=True,
            ).open()
    except Exception:
        if regularization_capture is not None:
            regularization_capture.close()
        hooked_model.close()
        raise

    try:
        with _frozen_eval_model(objective.model):
            if regularization_strength:
                candidate = image_parameter()
                target_loss, regularization_distance = _accentuation_forward(
                    hooked_model=hooked_model,
                    objective_function=objective_function,
                    regularization_capture=regularization_capture,
                    candidate=candidate,
                    reference=reference,
                    nb_crops=nb_crops,
                    crop_min=crop_min,
                    crop_max=crop_max,
                    noise_std=noise_std,
                    input_size=resolved_input_shape[-2:],
                    preprocess=preprocess,
                )
                regularization_grads = torch.autograd.grad(
                    regularization_distance,
                    parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                target_grads = torch.autograd.grad(
                    target_loss,
                    parameters,
                    allow_unused=True,
                )
                regularization_gradient = _absolute_gradient_sum(regularization_grads)
                target_gradient = _absolute_gradient_sum(target_grads)
                if regularization_gradient <= 1e-12:
                    raise RuntimeError(
                        "The initial regularization gradient is zero; choose a "
                        "responsive layer or use the gated fourier_phase parameterization."
                    )
                gradient_balance = target_gradient / regularization_gradient
            else:
                gradient_balance = 0.0

            for step in range(nb_steps):
                optimizer.zero_grad(set_to_none=True)
                candidate = image_parameter()
                target_loss, regularization_distance = _accentuation_forward(
                    hooked_model=hooked_model,
                    objective_function=objective_function,
                    regularization_capture=regularization_capture,
                    candidate=candidate,
                    reference=reference,
                    nb_crops=nb_crops,
                    crop_min=crop_min,
                    crop_max=crop_max,
                    noise_std=noise_std,
                    input_size=resolved_input_shape[-2:],
                    preprocess=preprocess,
                )
                full_loss = target_loss + (
                    float(regularization_strength)
                    * float(gradient_balance)
                    * regularization_distance
                )
                transparency_loss = (
                    full_loss if regularizer_in_transparency else target_loss
                )
                image_gradient = torch.autograd.grad(
                    transparency_loss,
                    candidate,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                if image_gradient is None:
                    raise RuntimeError("the target objective does not depend on the image")
                transparency = transparency + image_gradient[0].detach().abs()
                full_loss.backward()
                if step in checkpoint_steps:
                    checkpoints[step] = candidate.detach().cpu().clone()
                    transparency_checkpoints[step] = (
                        transparency.detach().cpu().clone()
                    )
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(parameters, float(grad_clip))
                optimizer.step()
                losses.append(float(full_loss.detach().cpu()))
                target_losses.append(float(target_loss.detach().cpu()))
                regularization_distances.append(
                    float(regularization_distance.detach().cpu())
                )
    finally:
        if regularization_capture is not None:
            regularization_capture.close()
        hooked_model.close()

    return OptimizationResult(
        image=image_parameter,
        transparency=transparency.detach(),
        losses=losses,
        objective_value=losses[-1],
        attempt_index=0,
        metadata={
            "method": "feature_accentuation",
            "objective_names": objective_names,
            "steps": nb_steps,
            "crops": nb_crops,
            "gradient_balance": float(gradient_balance),
            "target_losses": target_losses,
            "regularization_distances": regularization_distances,
            "magnitude_source": image_parameter.magnitude_source,
            "parameterization": image_parameter.parameterization,
            "checkpoint_steps": sorted(checkpoints),
        },
        checkpoints=checkpoints,
        transparency_checkpoints=transparency_checkpoints,
    )


def _accentuation_forward(
    hooked_model,
    objective_function,
    regularization_capture,
    candidate,
    reference,
    nb_crops,
    crop_min,
    crop_max,
    noise_std,
    input_size,
    preprocess,
):
    transformed = feature_accentuation_transforms(
        candidate,
        reference,
        output_size=input_size,
        crops=nb_crops,
        crop_min=crop_min,
        crop_max=crop_max,
        noise_std=noise_std,
    )
    model_inputs = transformed if preprocess is None else preprocess(transformed)
    if regularization_capture is not None:
        regularization_capture.clear()
    target_outputs = hooked_model(model_inputs)
    # ``Objective.compile`` returns an ascent score (the convention shared by
    # classical rendering and MaCo). Faccent minimizes a negative target. Its
    # default objective sees both transformed pair members; the reference half
    # is constant but is retained here for exact gradient scaling/parity.
    target_loss = -torch.mean(objective_function(target_outputs))
    if regularization_capture is None:
        regularization_distance = target_loss.new_zeros(())
    else:
        regularization_output = _split_accentuation_pairs(
            regularization_capture.tensor_output(),
            nb_crops,
        )
        difference = regularization_output[:, 0] - regularization_output[:, 1]
        if difference.ndim < 2:
            raise ValueError("regularization layer output must include a feature axis")
        regularization_distance = torch.linalg.vector_norm(
            difference,
            ord=2,
            dim=1,
        ).mean()
    return target_loss, regularization_distance


def _split_accentuation_pairs(output, nb_crops):
    if output.shape[0] != nb_crops * 2:
        raise ValueError(
            "model output batch does not match the candidate/reference crop layout"
        )
    return output.reshape(nb_crops, 2, *output.shape[1:])


def _absolute_gradient_sum(gradients):
    return sum(
        float(gradient.detach().abs().sum().cpu())
        for gradient in gradients
        if gradient is not None
    )


def _prepare_parameter_optimizer(optimizer, parameters, default_cls, default_lr):
    parameters = list(parameters)
    if optimizer is None:
        return default_cls(parameters, lr=default_lr)
    if isinstance(optimizer, torch.optim.Optimizer):
        if not optimizer.param_groups:
            raise ValueError("optimizer must have at least one parameter group")
        optimizer.state.clear()
        optimizer.param_groups[0]["params"] = parameters
        for group in optimizer.param_groups[1:]:
            group["params"] = []
        return optimizer
    if callable(optimizer):
        try:
            candidate = optimizer(parameters, lr=default_lr)
        except TypeError as first_error:
            try:
                candidate = optimizer(parameters)
            except TypeError:
                raise TypeError(
                    "optimizer callables must accept an iterable of parameters"
                ) from first_error
        if not isinstance(candidate, torch.optim.Optimizer):
            raise TypeError("optimizer callable must return torch.optim.Optimizer")
        return candidate
    raise TypeError("optimizer must be a torch optimizer, factory/class, or None")


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
