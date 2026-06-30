"""Fourier and color preconditioners for native PyTorch feature visualization."""

import os
from pathlib import Path
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F


IMAGENET_SPECTRUM_URL = (
    "https://storage.googleapis.com/serrelab/loupe/spectrums/imagenet_decorrelated.npy"
)

_IMAGENET_COLOR_CORRELATION = (
    (0.56282854, 0.58447580, 0.58447580),
    (0.19482528, 0.00000000, -0.19482528),
    (0.04329450, -0.10823626, 0.06494176),
)


def recorrelate_colors(images):
    """Map Xplique's uncorrelated color basis to RGB.

    Both CHW and NCHW tensors are accepted; the layout is preserved.
    """

    if not isinstance(images, torch.Tensor) or images.ndim not in (3, 4):
        raise ValueError("images must be a CHW or NCHW torch.Tensor")
    channel_dim = 0 if images.ndim == 3 else 1
    if images.shape[channel_dim] != 3:
        raise ValueError("color recorrelation requires exactly three channels")
    matrix = images.new_tensor(_IMAGENET_COLOR_CORRELATION)
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
    """Map an NCHW single-channel tensor to the requested value range."""

    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[1] != 1:
        raise ValueError("images must be a one-channel NCHW torch.Tensor")
    images = _apply_normalizer(images, normalizer, values_range)
    return _normalize_to_range(images, values_range)


def fft_2d_freq(height, width):
    """Return real-FFT radial frequencies for an ``(height, width)`` image.

    Xplique/Lucid allocate one additional frequency column for odd widths and
    crop the even inverse-FFT result back to the requested size. That behavior
    is retained for numerical parity.
    """

    height, width = int(height), int(width)
    if height < 1 or width < 1:
        raise ValueError("height and width must be positive")
    freq_y = np.fft.fftfreq(height)[:, np.newaxis]
    odd_extra = int(width % 2 == 1)
    freq_x = np.fft.fftfreq(width)[: width // 2 + 1 + odd_extra]
    return np.sqrt(freq_x**2 + freq_y**2)


def get_fft_scale(height, width, decay_power=1.0, device=None):
    """Create Lucid's image-size-normalized Fourier energy scale."""

    frequencies = fft_2d_freq(height, width)
    scale = 1.0 / np.maximum(frequencies, 1.0 / max(width, height)) ** decay_power
    scale = scale * np.sqrt(width * height)
    return torch.as_tensor(scale, dtype=torch.complex64, device=device)


def fft_image(shape, std=0.01, device=None, dtype=torch.float32):
    """Initialize a Fourier buffer for an NCHW image shape."""

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
    """Initialize the fixed magnitude and trainable phase used by MaCo.

    Native shapes are ``(C, H, W)``. For migration compatibility, Xplique's
    ``(H, W, C)`` shape is also recognized when the last dimension is 1 or 3;
    pass ``data_format='CHW'`` or ``'HWC'`` to remove any ambiguity. A two-item
    ``(H, W)`` shape is accepted when channels come from the dataset or the
    built-in RGB ImageNet spectrum.
    """

    channels, height, width = _parse_maco_shape(image_shape, data_format)
    spectrum_shape = (height, width // 2 + 1)

    if dataset is None:
        if channels not in (None, 3):
            raise ValueError(
                "The built-in MaCo spectrum is RGB; provide a dataset for grayscale images."
            )
        phase = np.random.normal(
            size=(3, *spectrum_shape), scale=std
        ).astype(np.float32)
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
            batch_magnitude
            if magnitude_sum is None
            else magnitude_sum + batch_magnitude
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
    """Reconstruct one CHW image from a fixed magnitude and trainable phase."""

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
        os.environ.get(
            "DREAMLENS_CACHE",
            Path.home() / ".cache" / "dreamlens",
        )
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
            "NCHW dataset to init_maco_buffer/maco or set DREAMLENS_CACHE to a "
            "cache containing spectrums/spectrum_decorrelated.npy."
        ) from exc
    return target


__all__ = [
    "IMAGENET_SPECTRUM_URL",
    "fft_2d_freq",
    "fft_image",
    "fft_to_rgb",
    "get_fft_scale",
    "init_maco_buffer",
    "maco_image_parametrization",
    "recorrelate_colors",
    "to_valid_grayscale",
    "to_valid_rgb",
]
