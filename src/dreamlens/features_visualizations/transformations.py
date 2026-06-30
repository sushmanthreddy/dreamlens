"""Stochastic, differentiable transformations for NCHW image batches."""

import torch
import torch.nn.functional as F


def _validate_images(images):
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("transformations expect an NCHW torch.Tensor")


def _random_blur_for_channels(sigma_range, kernel_size, expected_channels=None):
    kernel_size = int(kernel_size)
    if kernel_size < 1:
        raise ValueError("kernel_size must be >= 1")
    sigma_min = max(float(sigma_range[0]), 0.1)
    sigma_max = max(float(sigma_range[1]), 0.1)
    if sigma_max < sigma_min:
        raise ValueError("sigma_range maximum must be >= its minimum")

    def blur(images):
        _validate_images(images)
        channels = images.shape[1]
        if expected_channels is not None and channels != expected_channels:
            raise ValueError(
                "expected {} image channel(s), received {}".format(
                    expected_channels, channels
                )
            )
        sigma = torch.empty((), device=images.device, dtype=images.dtype).uniform_(
            sigma_min, sigma_max
        )
        uniform = torch.linspace(
            -(kernel_size - 1) / 2.0,
            (kernel_size - 1) / 2.0,
            kernel_size,
            device=images.device,
            dtype=images.dtype,
        )
        yy, xx = torch.meshgrid(uniform, uniform, indexing="ij")
        kernel = torch.exp(-0.5 * (xx**2 + yy**2) / sigma**2)
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.reshape(1, 1, kernel_size, kernel_size).repeat(
            channels, 1, 1, 1
        )
        # SAME padding with stride one is asymmetric for even
        # kernels (4 before and 5 after for Xplique's 10x10 blur).  Make that
        # explicit instead of relying on PyTorch's warning-producing
        # ``padding='same'`` path.
        total_padding = kernel_size - 1
        padding_before = total_padding // 2
        padding_after = total_padding - padding_before
        images = F.pad(
            images,
            (padding_before, padding_after, padding_before, padding_after),
        )
        return F.conv2d(images, kernel, groups=channels)

    return blur


def random_blur(sigma_range=(1.0, 2.0), kernel_size=10):
    """Build a random depthwise Gaussian blur for RGB images."""

    return _random_blur_for_channels(sigma_range, kernel_size, expected_channels=3)


def random_blur_grayscale(sigma_range=(1.0, 2.0), kernel_size=10):
    """Build a random depthwise Gaussian blur for grayscale images."""

    return _random_blur_for_channels(sigma_range, kernel_size, expected_channels=1)


def random_jitter(delta=6):
    """Build a random crop that removes ``delta`` pixels from each dimension."""

    delta = int(delta)
    if delta < 0:
        raise ValueError("delta must be >= 0")

    def jitter(images):
        _validate_images(images)
        if delta == 0:
            return images
        height, width = images.shape[-2:]
        if delta >= height or delta >= width:
            raise ValueError("delta must be smaller than image height and width")
        top = int(torch.randint(delta + 1, (), device=images.device).item())
        left = int(torch.randint(delta + 1, (), device=images.device).item())
        return images[:, :, top : top + height - delta, left : left + width - delta]

    return jitter


def random_scale(scale_range=(0.95, 1.05)):
    """Build a random aspect-ratio-preserving bilinear resize."""

    min_scale = float(scale_range[0])
    max_scale = float(scale_range[1])
    if min_scale <= 0 or max_scale < min_scale:
        raise ValueError("scale_range must contain positive values in ascending order")

    def scale(images):
        _validate_images(images)
        factor = torch.empty((), device=images.device, dtype=images.dtype).uniform_(
            min_scale, max_scale
        )
        height, width = images.shape[-2:]
        size = (max(1, int(height * factor)), max(1, int(width * factor)))
        return F.interpolate(images, size=size, mode="bilinear", align_corners=False)

    return scale


def random_flip(horizontal=True, vertical=False):
    """Build independent random horizontal/vertical flips for each batch item."""

    def flip(images):
        _validate_images(images)
        if horizontal:
            mask = torch.rand(
                (images.shape[0], 1, 1, 1), device=images.device
            ) < 0.5
            images = torch.where(mask, torch.flip(images, dims=(-1,)), images)
        if vertical:
            mask = torch.rand(
                (images.shape[0], 1, 1, 1), device=images.device
            ) < 0.5
            images = torch.where(mask, torch.flip(images, dims=(-2,)), images)
        return images

    return flip


def pad(size=6, pad_value=0.0):
    """Build constant padding of ``size`` pixels on every spatial side."""

    size = int(size)
    if size < 0:
        raise ValueError("size must be >= 0")

    def pad_func(images):
        _validate_images(images)
        return F.pad(images, (size, size, size, size), value=float(pad_value))

    return pad_func


def compose_transformations(transformations):
    """Compose transformations in the order provided."""

    transformations = list(transformations)
    if not all(callable(func) for func in transformations):
        raise TypeError("every transformation must be callable")

    def composed_func(images):
        for func in transformations:
            images = func(images)
        return images

    return composed_func


def generate_standard_transformations(size, channels=3):
    """Return the robust transformation sequence used by Xplique."""

    if channels not in (1, 3):
        raise AssertionError("Only grayscale or RGB images are supported.")
    unit = int(size / 16)
    blur = random_blur_grayscale if channels == 1 else random_blur
    return compose_transformations(
        [
            pad(unit * 4, 0.0),
            random_jitter(unit * 2),
            random_jitter(unit * 2),
            random_jitter(unit * 4),
            random_jitter(unit * 4),
            random_jitter(unit * 4),
            random_scale((0.92, 0.96)),
            blur(sigma_range=(1.0, 1.1)),
            random_jitter(unit),
            random_jitter(unit),
            random_flip(),
        ]
    )


__all__ = [
    "compose_transformations",
    "generate_standard_transformations",
    "pad",
    "random_blur",
    "random_blur_grayscale",
    "random_flip",
    "random_jitter",
    "random_scale",
]
