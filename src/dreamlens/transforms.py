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
