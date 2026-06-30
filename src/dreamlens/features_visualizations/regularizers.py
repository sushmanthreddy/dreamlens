"""Image regularizers for NCHW PyTorch image batches."""

import torch


def _image_dims(images):
    if images.ndim != 4:
        raise ValueError("regularizers expect an NCHW image batch")
    return (1, 2, 3)


def l1_reg(factor=1.0):
    """Build a per-image mean L1 regularizer."""

    def reg(images):
        return float(factor) * torch.mean(torch.abs(images), dim=_image_dims(images))

    return reg


def l2_reg(factor=1.0):
    """Build a per-image root-mean-square L2 regularizer."""

    def reg(images):
        return float(factor) * torch.sqrt(torch.mean(images**2, dim=_image_dims(images)))

    return reg


def l_inf_reg(factor=1.0):
    """Build a per-image L-infinity regularizer."""

    def l_inf(images):
        return float(factor) * torch.amax(torch.abs(images), dim=_image_dims(images))

    return l_inf


def total_variation_reg(factor=1.0):
    """Build Xplique-compatible anisotropic total variation regularization."""

    def tv_reg(images):
        _image_dims(images)
        vertical = torch.abs(images[:, :, 1:, :] - images[:, :, :-1, :]).sum(
            dim=(1, 2, 3)
        )
        horizontal = torch.abs(images[:, :, :, 1:] - images[:, :, :, :-1]).sum(
            dim=(1, 2, 3)
        )
        return float(factor) * (vertical + horizontal)

    return tv_reg


__all__ = ["l1_reg", "l2_reg", "l_inf_reg", "total_variation_reg"]
