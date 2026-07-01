import os
from pathlib import Path
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F

from .preprocessing import IMAGENET_MEAN, IMAGENET_STD, image_to_tensor, rgb_to_logit
from .render import ImageParameterization, fft_2d_freq


IMAGENET_SPECTRUM_URL = (
    "https://storage.googleapis.com/serrelab/loupe/spectrums/imagenet_decorrelated.npy"
)

_XPLIQUE_COLOR_CORRELATION = (
    (0.56282854, 0.58447580, 0.58447580),
    (0.19482528, 0.00000000, -0.19482528),
    (0.04329450, -0.10823626, 0.06494176),
)


def recorrelate_colors(images):
    """Map an uncorrelated CHW/NCHW color basis to RGB."""

    if not isinstance(images, torch.Tensor) or images.ndim not in (3, 4):
        raise ValueError("images must be a CHW or NCHW torch.Tensor")
    channel_dim = 0 if images.ndim == 3 else 1
    if images.shape[channel_dim] != 3:
        raise ValueError("color recorrelation requires exactly three channels")
    matrix = images.new_tensor(_XPLIQUE_COLOR_CORRELATION)
    if images.ndim == 3:
        flat = images.permute(1, 2, 0)
        return torch.matmul(flat, matrix).permute(2, 0, 1)
    flat = images.permute(0, 2, 3, 1)
    return torch.matmul(flat, matrix).permute(0, 3, 1, 2)


def _normalize_to_range(images, values_range):
    low, high = min(values_range), max(values_range)
    minimum = torch.amin(images, dim=(1, 2, 3), keepdim=True)
    images = images - minimum
    images = images / torch.amax(images, dim=(1, 2, 3), keepdim=True)
    return images * (high - low) + low


def _apply_normalizer(images, normalizer, values_range):
    if normalizer == "sigmoid":
        return torch.sigmoid(images)
    if normalizer == "clip":
        low, high = min(values_range), max(values_range)
        return torch.clamp(images, min=low, max=high)
    if callable(normalizer):
        return normalizer(images)
    raise ValueError("Invalid normalizer.")


def to_valid_rgb(images, normalizer="sigmoid", values_range=(0, 1)):
    """Map an NCHW tensor to valid, recorrelated RGB values."""

    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("images must be an NCHW torch.Tensor")
    images = recorrelate_colors(images)
    images = _apply_normalizer(images, normalizer, values_range)
    return _normalize_to_range(images, values_range)


def to_valid_grayscale(images, normalizer="sigmoid", values_range=(0, 1)):
    """Map an NCHW single-channel tensor to the requested range."""

    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[1] != 1:
        raise ValueError("images must be a one-channel NCHW torch.Tensor")
    images = _apply_normalizer(images, normalizer, values_range)
    return _normalize_to_range(images, values_range)


def get_fft_scale(height, width, decay_power=1.0, device=None):
    """Create Lucid's image-size-normalized Fourier energy scale."""

    frequencies = fft_2d_freq(height, width)
    scale = 1.0 / np.maximum(frequencies, 1.0 / max(width, height)) ** decay_power
    scale = scale * np.sqrt(width * height)
    return torch.as_tensor(scale, dtype=torch.complex64, device=device)


def fft_image(shape, std=0.01, device=None, dtype=torch.float32):
    """Initialize a real/imaginary Fourier buffer for an NCHW shape."""

    batch, channels, height, width = _parse_nchw_shape(shape)
    frequencies = fft_2d_freq(height, width)
    return torch.randn(
        (2, batch, channels) + frequencies.shape,
        dtype=dtype,
        device=device,
    ) * float(std)


def fft_to_rgb(shape, buffer, fft_scale):
    """Convert a Lucid Fourier buffer into an NCHW pixel-basis tensor."""

    batch, channels, height, width = _parse_nchw_shape(shape)
    if buffer.ndim != 5 or buffer.shape[0] != 2:
        raise ValueError("buffer must have shape (2, N, C, H, frequency_width)")
    scale = torch.as_tensor(fft_scale, device=buffer.device)
    spectrum = torch.complex(buffer[0], buffer[1]) * scale
    image = torch.fft.irfft2(spectrum, dim=(-2, -1))
    return image[:batch, :channels, :height, :width] / 4.0


def init_maco_buffer(image_shape, dataset=None, std=1.0, device=None, data_format=None):
    """Initialize fixed magnitude and trainable phase for MaCo."""

    channels, height, width = _parse_maco_shape(image_shape, data_format)
    spectrum_shape = (height, width // 2 + 1)

    if dataset is None:
        if channels not in (None, 3):
            raise ValueError(
                "The built-in MaCo spectrum is RGB; provide a dataset for grayscale images."
            )
        phase = np.random.normal(size=(3, *spectrum_shape), scale=std).astype(np.float32)
        magnitude = np.load(_get_imagenet_spectrum_path())
        magnitude = torch.as_tensor(magnitude, dtype=torch.float32).unsqueeze(0)
        magnitude = F.interpolate(
            magnitude,
            size=spectrum_shape,
            mode="bilinear",
            align_corners=False,
        )[0]
        return magnitude.to(device=device), torch.as_tensor(phase, device=device)

    magnitude_sum = None
    count = 0
    dataset_channels = None
    for batch in dataset:
        images = _extract_dataset_images(batch)
        images = torch.as_tensor(images, dtype=torch.float32, device=device)
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4:
            raise ValueError("MaCo datasets must yield NCHW image batches")
        dataset_channels = images.shape[1]
        if channels is not None and channels != dataset_channels:
            raise ValueError("image_shape channels do not match the dataset")
        batch_magnitude = torch.abs(torch.fft.rfft2(images, dim=(-2, -1))).sum(dim=0)
        magnitude_sum = (
            batch_magnitude if magnitude_sum is None else magnitude_sum + batch_magnitude
        )
        count += images.shape[0]
    if count == 0 or magnitude_sum is None:
        raise ValueError("dataset must yield at least one image")

    magnitude = magnitude_sum / count
    magnitude = F.interpolate(
        magnitude.unsqueeze(0),
        size=spectrum_shape,
        mode="bilinear",
        align_corners=False,
    )[0]
    phase = torch.randn(
        (dataset_channels, *spectrum_shape),
        dtype=torch.float32,
        device=magnitude.device,
    ) * float(std)
    return magnitude.to(dtype=torch.float32), phase


def maco_image_parametrization(magnitude, phase, values_range):
    """Reconstruct one CHW image from fixed magnitude and trainable phase."""

    magnitude = torch.as_tensor(magnitude)
    phase = torch.as_tensor(phase, device=magnitude.device, dtype=magnitude.dtype)
    phase = phase - torch.mean(phase)
    phase = phase / (torch.std(phase, unbiased=False) + 1e-5)

    buffer = torch.complex(torch.cos(phase) * magnitude, torch.sin(phase) * magnitude)
    image = torch.fft.irfft2(buffer, dim=(-2, -1))
    image = image - torch.mean(image)
    image = image / (torch.std(image, unbiased=False) + 1e-5)
    if image.shape[0] == 3:
        image = recorrelate_colors(image)
    image = torch.sigmoid(image)
    low, high = min(values_range), max(values_range)
    return image * (high - low) + low


class FourierCanvas(ImageParameterization):
    """Optimizable random image backed by Lucid FFT colorspace."""

    def __init__(
        self,
        height=256,
        width=None,
        device=None,
        standard_deviation=0.01,
        batch_size=1,
        channels=3,
        fft=True,
        decorrelate=True,
        frequency_decay=1.0,
        raw_scale=0.25,
        fft_norm=None,
    ):
        width = height if width is None else width
        super().__init__(
            size=(height, width),
            batch=batch_size,
            channels=channels,
            sd=standard_deviation,
            decorrelate=decorrelate,
            fft=fft,
            frequency_decay=frequency_decay,
            raw_scale=raw_scale,
            fft_norm=fft_norm,
            device=device,
        )


class ReferenceCanvas(torch.nn.Module):
    """FFT image parameterization used by the reference amplification path."""

    def __init__(
        self,
        height=256,
        width=None,
        device=None,
        standard_deviation=0.01,
    ):
        super().__init__()
        self.height = int(height)
        self.width = self.height if width is None else int(width)
        self.sd = standard_deviation
        self.optimizer = None
        param_width = self.width + 1 if self.width % 2 == 1 else self.width
        init = np.random.normal(
            size=(1, 3, self.height, param_width),
            scale=standard_deviation,
        ).astype("float32")
        self.param = torch.nn.Parameter(torch.tensor(init, device=device))
        self.register_buffer(
            "color_matrix",
            _color_correlation_matrix(device=device),
        )
        mean = IMAGENET_MEAN.to(dtype=torch.float32)
        std = IMAGENET_STD.to(dtype=torch.float32)
        if device is not None:
            mean = mean.to(device)
            std = std.to(device)
        self.register_buffer("imagenet_mean", mean.view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", std.view(1, 3, 1, 1))

    def visible_image(self, device=None):
        if device is not None:
            self.to(device)
        image = _reference_fft_to_rgb(
            self.param,
            height=self.height,
            width=self.width,
            device=self.param.device,
        )
        image = _reference_lucid_colorspace_to_rgb(
            image,
            self.color_matrix.to(device=image.device, dtype=image.dtype),
        )
        return torch.sigmoid(image)

    def forward(self, device=None):
        image = self.visible_image(device=device)
        mean = self.imagenet_mean.to(device=image.device, dtype=image.dtype)
        std = self.imagenet_std.to(device=image.device, dtype=image.dtype)
        return (image - mean) / std

    def make_optimizer(self, lr=9e-3, weight_decay=1e-3, optimizer_cls=None):
        optimizer_cls = torch.optim.AdamW if optimizer_cls is None else optimizer_cls
        self.optimizer = optimizer_cls([self.param], lr=lr, weight_decay=weight_decay)
        return self.optimizer

    def clip_gradients(self, grad_clip=1.0):
        torch.nn.utils.clip_grad_norm_(self.param, grad_clip)

    def as_nchw(self, device=None):
        with torch.no_grad():
            return self.visible_image(device=device).detach().cpu()

    def as_chw(self, device=None):
        return self.as_nchw(device=device)[0]

    def as_hwc(self, device=None):
        return self.as_chw(device=device).permute(1, 2, 0)

    def __array__(self, dtype=None):
        array = self.as_hwc().numpy()
        if dtype is not None:
            array = array.astype(dtype)
        return array

    def save(self, filename):
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Saving images requires pillow to be installed.") from exc

        array = np.clip(self.as_hwc().numpy(), 0.0, 1.0)
        Image.fromarray((array * 255).astype("uint8")).save(filename)


class ReferenceImageCanvas(torch.nn.Module):
    """Reference-compatible FFT canvas initialized from an existing image."""

    def __init__(self, image, device=None):
        super().__init__()
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.optimizer = None
        self.register_buffer(
            "color_matrix",
            _color_correlation_matrix(device=self.device),
        )
        mean = IMAGENET_MEAN.to(device=self.device, dtype=torch.float32)
        std = IMAGENET_STD.to(device=self.device, dtype=torch.float32)
        self.register_buffer("imagenet_mean", mean.view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", std.view(1, 3, 1, 1))
        self.reset_image(image)

    def visible_image(self, device=None):
        if device is not None:
            self.to(device)
        image = _reference_custom_fft_to_rgb(
            self.param,
            height=self.height,
            width=self.width,
            device=self.param.device,
        )
        image = _reference_lucid_colorspace_to_rgb(
            image,
            self.color_matrix.to(device=image.device, dtype=image.real.dtype),
        )
        return image.clamp(0.0, 1.0)

    def forward(self, device=None):
        image = self.visible_image(device=device)
        mean = self.imagenet_mean.to(device=image.device, dtype=image.dtype)
        std = self.imagenet_std.to(device=image.device, dtype=image.dtype)
        return ((image - mean) / std).clamp(0.0, 1.0)

    def reset_image(self, image):
        tensor = image_to_tensor(image, device=self.device)
        if tensor.shape[0] != 1:
            raise ValueError("ReferenceImageCanvas expects a single NCHW image")
        self.height, self.width = tensor.shape[-2], tensor.shape[-1]
        spectrum = _reference_chw_rgb_to_fft_param(
            tensor[0],
            color_matrix=self.color_matrix,
            mean=self.imagenet_mean,
            std=self.imagenet_std,
            device=self.device,
        )
        scale = _reference_custom_fft_scale(
            self.height,
            self.width,
            device=self.device,
        )
        self.param = torch.nn.Parameter(spectrum / scale)
        self.optimizer = None

    def make_optimizer(self, lr=3e-4, weight_decay=1e-1, optimizer_cls=None):
        optimizer_cls = torch.optim.AdamW if optimizer_cls is None else optimizer_cls
        self.optimizer = optimizer_cls([self.param], lr=lr, weight_decay=weight_decay)
        return self.optimizer

    def clip_gradients(self, grad_clip=0.1):
        torch.nn.utils.clip_grad_norm_(self.param, grad_clip)

    def as_nchw(self, device=None):
        with torch.no_grad():
            return self.forward(device=device).detach().cpu()

    def as_chw(self, device=None):
        return self.as_nchw(device=device)[0]

    def as_hwc(self, device=None):
        return self.as_chw(device=device).permute(1, 2, 0)

    def __array__(self, dtype=None):
        array = self.as_hwc().numpy()
        if dtype is not None:
            array = array.astype(dtype)
        return array

    def save(self, filename):
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Saving images requires pillow to be installed.") from exc

        array = np.clip(self.as_hwc().numpy(), 0.0, 1.0)
        Image.fromarray((array * 255).astype("uint8")).save(filename)


class PixelCanvas(torch.nn.Module):
    """Optimizable image initialized from an existing image tensor or file."""

    def __init__(self, image, device=None):
        super().__init__()
        self.optimizer = None
        self.reset_image(image, device=device)

    def forward(self, device=None):
        if device is not None:
            self.to(device)
        return torch.sigmoid(self.param)

    def reset_image(self, tensor, device=None):
        tensor = image_to_tensor(tensor, device=device).clamp(1e-6, 1.0 - 1e-6)
        self.height, self.width = tensor.shape[-2], tensor.shape[-1]
        self.param = torch.nn.Parameter(rgb_to_logit(tensor))
        self.optimizer = None

    def make_optimizer(self, lr=3e-4, weight_decay=1e-1, optimizer_cls=None):
        optimizer_cls = torch.optim.AdamW if optimizer_cls is None else optimizer_cls
        self.optimizer = optimizer_cls(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        return self.optimizer

    def clip_gradients(self, grad_clip=0.1):
        torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)

    def as_nchw(self, device=None):
        with torch.no_grad():
            return self.forward(device=device).detach().cpu()

    def as_chw(self, device=None):
        return self.as_nchw(device=device)[0]

    def as_hwc(self, device=None):
        return self.as_chw(device=device).permute(1, 2, 0)

    def __array__(self, dtype=None):
        array = self.as_hwc().numpy()
        if dtype is not None:
            array = array.astype(dtype)
        return array

    def save(self, filename):
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Saving images requires pillow to be installed.") from exc

        array = np.clip(self.as_hwc().numpy(), 0.0, 1.0)
        Image.fromarray((array * 255).astype("uint8")).save(filename)


class MaskedCanvas(PixelCanvas):
    """Optimize an image while preserving pixels outside a mask."""

    def __init__(self, mask_tensor, image=None, device=None):
        mask = image_to_tensor(mask_tensor, device=device)
        if mask.shape[1] == 1:
            mask = mask.repeat(1, 3, 1, 1)
        if image is None:
            image = torch.rand_like(mask[:, :3])
        super().__init__(image=image, device=device)
        self.register_buffer("mask", mask[:, :3].clamp(0.0, 1.0))
        original = image_to_tensor(image, device=device)[:, :3].clamp(0.0, 1.0)
        if original.shape[-2:] != self.mask.shape[-2:]:
            raise ValueError("image and mask must have the same height and width")
        self.register_buffer("original_nchw_image_tensor", original)

    def forward(self, device=None):
        if device is not None:
            self.to(device)
        optimized = torch.sigmoid(self.param)
        return optimized * self.mask + self.original_nchw_image_tensor * (1.0 - self.mask)

    def replace_mask(self, mask):
        self.original_nchw_image_tensor = self.forward().detach()
        mask = image_to_tensor(mask, device=self.mask.device)
        if mask.shape[1] == 1:
            mask = mask.repeat(1, 3, 1, 1)
        if mask.shape[-2:] != self.original_nchw_image_tensor.shape[-2:]:
            raise ValueError("new mask must match the image height and width")
        self.mask = mask[:, :3].clamp(0.0, 1.0)


class ReferenceMaskedCanvas(ReferenceImageCanvas):
    """Reference-compatible masked FFT canvas initialized from an image."""

    def __init__(self, mask_tensor, image=None, device=None):
        mask = image_to_tensor(mask_tensor, device=device)
        self.height, self.width = mask.shape[-2], mask.shape[-1]
        if image is None:
            image = ReferenceCanvas(
                height=self.height,
                width=self.width,
                device=device,
                standard_deviation=0.01,
            ).visible_image(device=device)
        original = image_to_tensor(image, device=device)
        if original.shape[-2:] != mask.shape[-2:]:
            raise ValueError("image and mask must have the same height and width")
        super().__init__(image=original, device=device)
        self.register_buffer("mask", mask[:, :3].clamp(0.0, 1.0))
        self.register_buffer("original_nchw_image_tensor", original[:, :3])

    def forward(self, device=None):
        if device is not None:
            self.to(device)
        mask = self.mask.to(device=self.param.device)
        return super().forward(device=self.param.device).clamp(0.0, 1.0) * mask

    def as_nchw(self, device=None):
        with torch.no_grad():
            foreground = self.forward(device=device).detach().cpu()
            mask = self.mask.detach().cpu()
            original = self.original_nchw_image_tensor.detach().cpu()
            return foreground + original * (1.0 - mask)

    def as_chw(self, device=None):
        return self.as_nchw(device=device)[0]

    def as_hwc(self, device=None):
        return self.as_chw(device=device).permute(1, 2, 0)

    def replace_mask(self, mask):
        self.original_nchw_image_tensor = self.as_nchw(device=self.device).to(
            self.device
        )
        mask = image_to_tensor(mask, device=self.device)
        if mask.shape[-2:] != self.original_nchw_image_tensor.shape[-2:]:
            raise ValueError("new mask must match the image height and width")
        self.mask = mask[:, :3].clamp(0.0, 1.0)


class CanvasBatch(torch.nn.Module):
    """Batch several image parameters behind one optimizer-friendly object."""

    def __init__(self, image_params):
        super().__init__()
        if not image_params:
            raise ValueError("image_params must contain at least one item")
        self.image_params = torch.nn.ModuleList(image_params)
        heights = {param.height for param in self.image_params}
        widths = {param.width for param in self.image_params}
        if len(heights) != 1 or len(widths) != 1:
            raise ValueError("all image parameters in a batch must have the same size")
        self.height = next(iter(heights))
        self.width = next(iter(widths))
        self.optimizer = None

    def forward(self, device=None):
        return torch.cat(
            [call_image_parameter(param, device) for param in self.image_params],
            dim=0,
        )

    def make_optimizer(self, lr=9e-3, weight_decay=0.0, optimizer_cls=None):
        optimizer_cls = torch.optim.AdamW if optimizer_cls is None else optimizer_cls
        self.optimizer = optimizer_cls(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        return self.optimizer

    def clip_gradients(self, grad_clip=1.0):
        torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)

    def __getitem__(self, index):
        return self.image_params[index]

    def __len__(self):
        return len(self.image_params)


class FourierCanvasBatch(CanvasBatch):
    """Batched random image parameter."""

    def __init__(
        self,
        batch_size=1,
        height=256,
        width=None,
        standard_deviation=0.01,
        device=None,
        fft=True,
        decorrelate=True,
    ):
        width = height if width is None else width
        super().__init__(
            [
                FourierCanvas(
                    height=height,
                    width=width,
                    device=device,
                    standard_deviation=standard_deviation,
                    fft=fft,
                    decorrelate=decorrelate,
                )
                for _ in range(batch_size)
            ]
        )


class ReferenceCanvasBatch(CanvasBatch):
    """Batch of reference-compatible FFT canvases."""

    def __init__(
        self,
        batch_size=1,
        height=256,
        width=None,
        standard_deviation=0.01,
        device=None,
        lr=9e-3,
        weight_decay=0.0,
        optimizer_cls=None,
    ):
        width = height if width is None else width
        super().__init__(
            [
                ReferenceCanvas(
                    height=height,
                    width=width,
                    device=device,
                    standard_deviation=standard_deviation,
                )
                for _ in range(batch_size)
            ]
        )
        self.make_optimizer(
            lr=lr,
            weight_decay=weight_decay,
            optimizer_cls=optimizer_cls,
        )

    def make_optimizer(self, lr=9e-3, weight_decay=0.0, optimizer_cls=None):
        optimizers = [
            param.make_optimizer(
                lr=lr,
                weight_decay=weight_decay,
                optimizer_cls=optimizer_cls,
            )
            for param in self.image_params
        ]
        self.optimizer = _GroupedOptimizer(optimizers)
        return self.optimizer

    def clip_gradients(self, grad_clip=1.0):
        for param in self.image_params:
            param.clip_gradients(grad_clip=grad_clip)


class _GroupedOptimizer:
    def __init__(self, optimizers):
        self.optimizers = list(optimizers)

    def clear_gradients(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def advance(self):
        for optimizer in self.optimizers:
            optimizer.step()


def call_image_parameter(image_parameter, device):
    try:
        return image_parameter.forward(device=device)
    except TypeError:
        return image_parameter()


def _color_correlation_matrix(device=None):
    color_correlation_svd_sqrt = np.asarray(
        [[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]],
        dtype="float32",
    )
    max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))
    return torch.tensor(
        color_correlation_svd_sqrt / max_norm_svd_sqrt,
        dtype=torch.float32,
        device=device,
    )


def _reference_fft_scale(height, width, device=None, decay_power=0.75):
    d = 0.5**0.5
    fy = np.fft.fftfreq(height, d=d)[:, None]
    if width % 2 == 1:
        fx = np.fft.rfftfreq(width, d=d)[: (width + 1) // 2]
    else:
        fx = np.fft.rfftfreq(width, d=d)[: width // 2]
    freqs = (fx * fx + fy * fy) ** decay_power
    scale = 1.0 / np.maximum(freqs, 1.0 / (max(width, height) * d))
    return torch.tensor(scale, dtype=torch.float32, device=device)


def _reference_custom_fft_scale(height, width, device=None, decay_power=0.75):
    d = 0.5**0.5
    fy = np.fft.fftfreq(height, d=d)[:, None]
    fx = np.fft.rfftfreq(width, d=d)[: (width // 2) + 1]
    freqs = (fx * fx + fy * fy) ** decay_power
    scale = 1.0 / np.maximum(freqs, 1.0 / (max(width, height) * d))
    return torch.tensor(scale, dtype=torch.float32, device=device)


def _reference_fft_to_rgb(image_parameter, height, width, device=None):
    scale = _reference_fft_scale(height, width, device=device).to(
        image_parameter.device
    )
    if width % 2 == 1:
        shaped = image_parameter.reshape(1, 3, height, (width + 1) // 2, 2)
    else:
        shaped = image_parameter.reshape(1, 3, height, width // 2, 2)
    complex_spectrum = torch.complex(shaped[..., 0], shaped[..., 1])
    scaled = scale * complex_spectrum
    return torch.fft.irfft2(scaled, s=(height, width), norm="ortho")


def _reference_custom_fft_to_rgb(image_parameter, height, width, device=None):
    scale = _reference_custom_fft_scale(height, width, device=device).to(
        image_parameter.device
    )
    scaled = scale * image_parameter
    return torch.fft.irfft2(scaled, s=(height, width), norm="ortho")


def _reference_lucid_colorspace_to_rgb(image, color_matrix):
    flat = image.permute(0, 2, 3, 1)
    flat = torch.matmul(flat, color_matrix.T)
    return flat.permute(0, 3, 1, 2)


def _reference_rgb_to_lucid_colorspace(image, color_matrix, device=None):
    flat = image.permute(0, 2, 3, 1)
    inverse = torch.inverse(color_matrix.T.to(device))
    flat = torch.matmul(flat.to(device), inverse)
    return flat.permute(0, 3, 1, 2)


def _reference_denormalize(image, mean, std):
    return image.float() * std.to(image.device) + mean.to(image.device)


def _reference_chw_rgb_to_fft_param(chw_image, color_matrix, mean, std, device=None):
    if isinstance(chw_image, torch.Tensor):
        image = chw_image.detach().clone().to(device=device).unsqueeze(0).float()
    else:
        image = torch.tensor(chw_image, device=device).unsqueeze(0).float()
    image = _reference_denormalize(image, mean=mean, std=std)
    image = _reference_rgb_to_lucid_colorspace(
        image,
        color_matrix=color_matrix,
        device=device,
    )
    return torch.fft.rfft2(image, s=(image.shape[-2], image.shape[-1]), norm="ortho")


def _parse_nchw_shape(shape):
    if len(shape) != 4:
        raise ValueError("shape must be (batch, channels, height, width)")
    batch, channels, height, width = (int(value) for value in shape)
    if min(batch, channels, height, width) < 1:
        raise ValueError("all shape dimensions must be positive")
    return batch, channels, height, width


def _parse_maco_shape(shape, data_format):
    shape = tuple(int(value) for value in shape)
    if len(shape) == 2:
        return None, shape[0], shape[1]
    if len(shape) != 3:
        raise ValueError("image_shape must be (C,H,W), (H,W,C), or (H,W)")
    if data_format is not None:
        data_format = data_format.upper()
        if data_format == "CHW":
            return shape
        if data_format == "HWC":
            return shape[2], shape[0], shape[1]
        raise ValueError("data_format must be 'CHW' or 'HWC'")
    if shape[-1] in (1, 3) and shape[0] not in (1, 3):
        return shape[-1], shape[0], shape[1]
    return shape


def _extract_dataset_images(batch):
    if isinstance(batch, (list, tuple)):
        if not batch:
            raise ValueError("dataset yielded an empty sequence")
        return batch[0]
    if isinstance(batch, dict):
        for key in ("image", "images", "input", "inputs"):
            if key in batch:
                return batch[key]
        raise ValueError("dictionary batches must contain an image/input key")
    return batch


def _get_imagenet_spectrum_path():
    cache_root = Path(
        os.environ.get("DREAMLENS_CACHE", Path.home() / ".cache" / "dreamlens")
    )
    target = cache_root / "spectrums" / "spectrum_decorrelated.npy"
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".npy.part")
    try:
        urllib.request.urlretrieve(IMAGENET_SPECTRUM_URL, temporary)
        temporary.replace(target)
    except Exception as exc:
        if temporary.exists():
            temporary.unlink()
        raise RuntimeError(
            "Could not download the ImageNet MaCo spectrum. Pass a representative "
            "NCHW dataset or set DREAMLENS_CACHE to a cache containing "
            "spectrums/spectrum_decorrelated.npy."
        ) from exc
    return target
