import math
import random

import torch
import torch.nn.functional as F

from .render import crop_or_pad_to


def random_translate(translate_x, translate_y):
    def inner(image):
        if translate_x == 0 and translate_y == 0:
            return image
        tx = _uniform_like(image, -float(translate_x), float(translate_x))
        ty = _uniform_like(image, -float(translate_y), float(translate_y))
        theta = image.new_tensor([[1.0, 0.0, tx], [0.0, 1.0, ty]])
        theta = theta.unsqueeze(0).repeat(image.shape[0], 1, 1)
        grid = F.affine_grid(theta, image.shape, align_corners=False)
        return F.grid_sample(
            image,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

    return inner


def single_default_transform(
    image,
    height,
    width,
    rotate_degrees,
    scale_min,
    scale_max,
    translate_x,
    translate_y,
):
    scale, angle, tx, ty = sample_transform_params(
        image,
        rotate_degrees=rotate_degrees,
        scale_min=scale_min,
        scale_max=scale_max,
        translate_x=translate_x,
        translate_y=translate_y,
    )
    return apply_transform_params(image, scale, angle, tx, ty, height, width)


def paired_default_transforms(
    image,
    reference,
    height,
    width,
    rotate_degrees,
    scale_min,
    scale_max,
    translate_x,
    translate_y,
):
    scale, angle, tx, ty = sample_transform_params(
        image,
        rotate_degrees=rotate_degrees,
        scale_min=scale_min,
        scale_max=scale_max,
        translate_x=translate_x,
        translate_y=translate_y,
    )
    return (
        apply_transform_params(image, scale, angle, tx, ty, height, width),
        apply_transform_params(reference, scale, angle, tx, ty, height, width),
    )


def feature_accentuation_transforms(
    candidate,
    reference,
    output_size=(224, 224),
    crops=16,
    crop_min=0.05,
    crop_max=0.99,
    noise_std=0.02,
):
    """Create Faccent's transform-matched candidate/reference crop batch.

    Each crop samples one square box and one Gaussian-plus-uniform noise field.
    Both images receive exactly the same crop and noise before resizing. The
    returned batch is ordered ``candidate, reference`` for every crop.
    """

    _validate_images(candidate)
    _validate_images(reference)
    if candidate.shape != reference.shape:
        raise ValueError("candidate and reference must have the same NCHW shape")
    if candidate.shape[0] != 1:
        raise ValueError("feature accentuation expects one candidate/reference pair")
    crops = int(crops)
    if crops < 1:
        raise ValueError("crops must be >= 1")
    crop_min, crop_max = float(crop_min), float(crop_max)
    if not 0 < crop_min <= crop_max <= 1:
        raise ValueError("crop bounds must satisfy 0 < crop_min <= crop_max <= 1")
    noise_std = float(noise_std)
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    if not isinstance(output_size, (tuple, list)) or len(output_size) != 2:
        raise ValueError("output_size must be (height, width)")
    output_size = tuple(int(value) for value in output_size)
    if min(output_size) < 1:
        raise ValueError("output_size dimensions must be positive")

    pair = torch.cat([candidate, reference], dim=0)
    height, width = pair.shape[-2:]
    transformed = []
    for _ in range(crops):
        # Faccent samples crop geometry with the CPU generator even when the
        # image/model live on CUDA. Noise below is sampled on the image device.
        fraction = torch.rand(1) * (crop_max - crop_min) + crop_min
        top_fraction = torch.rand(1) * (1.0 - fraction)
        left_fraction = torch.rand(1) * (1.0 - fraction)
        top = int((top_fraction * height).item())
        bottom = int(((top_fraction + fraction) * height).item())
        left = int((left_fraction * width).item())
        right = int(((left_fraction + fraction) * width).item())
        bottom = max(top + 1, min(height, bottom))
        right = max(left + 1, min(width, right))
        view = pair[..., top:bottom, left:right]
        if noise_std:
            gaussian = torch.randn_like(view[0:1]) * noise_std
            uniform = (torch.rand_like(view[0:1]) - 0.5) * noise_std
            view = view + gaussian.expand_as(view) + uniform.expand_as(view)
        view = F.interpolate(
            view,
            size=output_size,
            mode="bilinear",
            align_corners=True,
            antialias=True,
        )
        transformed.append(view)
    return torch.cat(transformed, dim=0)


def reference_paired_transforms(
    image,
    reference,
    rotate_degrees,
    scale_min,
    scale_max,
    translate_x,
    translate_y,
):
    """Reference-compatible paired transforms for image amplification."""

    try:
        from torchvision import transforms as tv_transforms
        from torchvision.transforms import functional as tv_functional
    except ImportError as exc:
        raise ImportError(
            "Reference-compatible amplification transforms require torchvision."
        ) from exc

    height_factor = random.uniform(a=float(scale_min), b=float(scale_max))
    width_factor = random.uniform(a=float(scale_min), b=float(scale_max))
    image = F.interpolate(
        image,
        scale_factor=(height_factor, width_factor),
        mode="bilinear",
    )
    reference = F.interpolate(
        reference,
        scale_factor=(height_factor, width_factor),
        mode="bilinear",
    )

    params = tv_transforms.RandomAffine.get_params(
        degrees=(-float(rotate_degrees), float(rotate_degrees)),
        translate=(float(translate_x), float(translate_y)),
        scale_ranges=(1, 1),
        shears=(0, 0),
        img_size=(image.shape[-2], image.shape[1]),
    )
    return (
        tv_functional.affine(image, *params),
        tv_functional.affine(reference, *params),
    )


def reference_masked_transform(
    image,
    mask,
    original,
    rotate_degrees,
    scale_min,
    scale_max,
    translate_x,
    translate_y,
):
    """Reference-compatible transform/recombine path for masked canvases."""

    try:
        from torchvision import transforms as tv_transforms
        from torchvision.transforms import functional as tv_functional
    except ImportError as exc:
        raise ImportError(
            "Reference-compatible amplification transforms require torchvision."
        ) from exc

    height_factor = random.uniform(a=float(scale_min), b=float(scale_max))
    width_factor = random.uniform(a=float(scale_min), b=float(scale_max))
    resized = [
        F.interpolate(
            tensor,
            scale_factor=(height_factor, width_factor),
            mode="bilinear",
        )
        for tensor in (image, mask, original)
    ]

    params = tv_transforms.RandomAffine.get_params(
        degrees=(-float(rotate_degrees), float(rotate_degrees)),
        translate=(float(translate_x), float(translate_y)),
        scale_ranges=(1, 1),
        shears=(0, 0),
        img_size=(resized[0].shape[-2], resized[0].shape[1]),
    )
    image_t, mask_t, original_t = [
        tv_functional.affine(tensor, *params) for tensor in resized
    ]
    return image_t * mask_t + original_t.float() * (1.0 - mask_t)


def reference_single_transform(
    image,
    rotate_degrees,
    scale_min,
    scale_max,
    translate_x,
    translate_y,
):
    """Reference-compatible single-image transform for static amplification."""

    try:
        from torchvision import transforms as tv_transforms
    except ImportError as exc:
        raise ImportError(
            "Reference-compatible amplification transforms require torchvision."
        ) from exc

    image = tv_transforms.RandomAffine(
        degrees=float(rotate_degrees),
        translate=(float(translate_x), float(translate_y)),
    )(image)
    height_factor = random.uniform(a=float(scale_min), b=float(scale_max))
    width_factor = random.uniform(a=float(scale_min), b=float(scale_max))
    return F.interpolate(
        image,
        scale_factor=(height_factor, width_factor),
        mode="bilinear",
    )


def sample_transform_params(
    tensor,
    rotate_degrees,
    scale_min,
    scale_max,
    translate_x,
    translate_y,
):
    scale = _uniform_like(tensor, float(scale_min), float(scale_max))
    angle = _uniform_like(tensor, -float(rotate_degrees), float(rotate_degrees))
    tx = _uniform_like(tensor, -float(translate_x), float(translate_x))
    ty = _uniform_like(tensor, -float(translate_y), float(translate_y))
    return scale, math.radians(angle), tx, ty


def apply_transform_params(image, scale, angle, tx, ty, height, width):
    if scale != 1.0:
        image = F.interpolate(
            image,
            scale_factor=(scale, scale),
            mode="bilinear",
            align_corners=False,
        )

    theta = image.new_tensor(
        [
            [math.cos(angle), -math.sin(angle), tx],
            [math.sin(angle), math.cos(angle), ty],
        ]
    )
    theta = theta.unsqueeze(0).repeat(image.shape[0], 1, 1)
    grid = F.affine_grid(theta, image.shape, align_corners=False)
    image = F.grid_sample(
        image,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return crop_or_pad_to(height, width)(image)


def _uniform_like(tensor, low, high):
    return (
        torch.empty((), device=tensor.device, dtype=tensor.dtype)
        .uniform_(low, high)
        .item()
    )


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
    """Build a random crop that removes ``delta`` pixels per dimension."""

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
    """Build independent random horizontal/vertical batch flips."""

    def flip(images):
        _validate_images(images)
        if horizontal:
            mask = torch.rand((images.shape[0], 1, 1, 1), device=images.device) < 0.5
            images = torch.where(mask, torch.flip(images, dims=(-1,)), images)
        if vertical:
            mask = torch.rand((images.shape[0], 1, 1, 1), device=images.device) < 0.5
            images = torch.where(mask, torch.flip(images, dims=(-2,)), images)
        return images

    return flip


def pad(size=6, pad_value=0.0):
    """Build constant padding for every spatial side."""

    size = int(size)
    if size < 0:
        raise ValueError("size must be >= 0")

    def pad_func(images):
        _validate_images(images)
        return F.pad(images, (size, size, size, size), value=float(pad_value))

    return pad_func


def compose_transformations(transformations):
    """Compose transformations in the provided order."""

    transformations = list(transformations)
    if not all(callable(func) for func in transformations):
        raise TypeError("every transformation must be callable")

    def composed_func(images):
        for func in transformations:
            images = func(images)
        return images

    return composed_func


def generate_standard_transformations(size, channels=3):
    """Return the robust Xplique/Lucid transformation sequence."""

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
