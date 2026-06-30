"""Losses used by feature visualization objectives.

The reductions intentionally match Xplique's reference implementation: every
dimension except the leading batch dimension is treated as one feature vector.
"""

import torch


def cosine_similarity(tensor_a, tensor_b):
    """Return one cosine similarity for each pair of batched tensors."""

    tensor_a = torch.as_tensor(tensor_a)
    tensor_b = torch.as_tensor(
        tensor_b,
        dtype=tensor_a.dtype,
        device=tensor_a.device,
    )
    if tensor_a.ndim < 2 or tensor_b.ndim < 2:
        raise ValueError("cosine_similarity expects tensors with a batch dimension")

    dims_a = tuple(range(1, tensor_a.ndim))
    dims_b = tuple(range(1, tensor_b.ndim))
    norm_a = torch.sqrt(torch.sum(tensor_a * tensor_a, dim=dims_a, keepdim=True))
    norm_b = torch.sqrt(torch.sum(tensor_b * tensor_b, dim=dims_b, keepdim=True))
    eps_a = torch.as_tensor(1e-12, dtype=norm_a.dtype, device=norm_a.device)
    eps_b = torch.as_tensor(1e-12, dtype=norm_b.dtype, device=norm_b.device)
    tensor_a = tensor_a / torch.sqrt(torch.maximum(norm_a * norm_a, eps_a))
    tensor_b = tensor_b / torch.sqrt(torch.maximum(norm_b * norm_b, eps_b))
    return torch.sum(tensor_a * tensor_b, dim=dims_a)


def dot_cossim(tensor_a, tensor_b, cossim_pow=2.0):
    """Return Xplique's cosine-weighted dot-product objective."""

    tensor_a = torch.as_tensor(tensor_a)
    tensor_b = torch.as_tensor(
        tensor_b,
        dtype=tensor_a.dtype,
        device=tensor_a.device,
    )
    floor = torch.as_tensor(1e-1, dtype=tensor_a.dtype, device=tensor_a.device)
    cosim = torch.maximum(cosine_similarity(tensor_a, tensor_b), floor) ** cossim_pow
    dot = torch.sum(tensor_a * tensor_b)
    return dot * cosim


__all__ = ["cosine_similarity", "dot_cossim"]
