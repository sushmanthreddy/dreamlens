from pathlib import Path

import numpy as np
import torch


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def identity_preprocess(image):
    return image


def imagenet_normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    mean = mean.to(device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    std = std.to(device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    return (image - mean) / std


def image_to_tensor(image, device=None):
    """Convert a path, PIL image, NumPy array, or tensor to NCHW float RGB."""

    if isinstance(image, (str, Path)):
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Loading image files requires pillow to be installed.") from exc

        image = np.asarray(Image.open(image).convert("RGB")).copy()

    if not isinstance(image, torch.Tensor):
        image = torch.as_tensor(np.asarray(image).copy())

    image = image.detach().clone().float()
    if image.numel() and image.max() > 1.0:
        image = image / 255.0

    if image.dim() == 2:
        image = image.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
    elif image.dim() == 3:
        if image.shape[0] in (1, 3, 4):
            image = image[:3].unsqueeze(0)
        elif image.shape[-1] in (1, 3, 4):
            image = image[..., :3].permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError("3D image tensors must be CHW or HWC")
    elif image.dim() == 4:
        if image.shape[1] in (1, 3, 4):
            image = image[:, :3]
        elif image.shape[-1] in (1, 3, 4):
            image = image[..., :3].permute(0, 3, 1, 2)
        else:
            raise ValueError("4D image tensors must be NCHW or NHWC")
    else:
        raise ValueError("image must be 2D, 3D, or 4D")

    if image.shape[1] == 1:
        image = image.repeat(1, 3, 1, 1)
    if image.shape[1] != 3:
        raise ValueError("image must have one, three, or four channels")

    image = image.clamp(0.0, 1.0)
    if device is not None:
        image = image.to(device)
    return image


def rgb_to_logit(tensor, eps=1e-6):
    tensor = tensor.clamp(eps, 1.0 - eps)
    return torch.log(tensor) - torch.log1p(-tensor)
