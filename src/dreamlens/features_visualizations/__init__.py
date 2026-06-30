"""Complete native-PyTorch port of Xplique's feature visualization API."""

from .losses import cosine_similarity, dot_cossim
from .maco import maco, maco_optimisation_step
from .objectives import Objective
from .optim import optimize
from .preconditioning import (
    IMAGENET_SPECTRUM_URL,
    fft_2d_freq,
    fft_image,
    fft_to_rgb,
    get_fft_scale,
    init_maco_buffer,
    maco_image_parametrization,
    recorrelate_colors,
    to_valid_grayscale,
    to_valid_rgb,
)
from .regularizers import l1_reg, l2_reg, l_inf_reg, total_variation_reg
from .transformations import (
    compose_transformations,
    generate_standard_transformations,
    pad,
    random_blur,
    random_blur_grayscale,
    random_flip,
    random_jitter,
    random_scale,
)

__all__ = [
    "IMAGENET_SPECTRUM_URL",
    "Objective",
    "compose_transformations",
    "cosine_similarity",
    "dot_cossim",
    "fft_2d_freq",
    "fft_image",
    "fft_to_rgb",
    "generate_standard_transformations",
    "get_fft_scale",
    "init_maco_buffer",
    "l1_reg",
    "l2_reg",
    "l_inf_reg",
    "maco",
    "maco_image_parametrization",
    "maco_optimisation_step",
    "optimize",
    "pad",
    "random_blur",
    "random_blur_grayscale",
    "random_flip",
    "random_jitter",
    "random_scale",
    "recorrelate_colors",
    "to_valid_grayscale",
    "to_valid_rgb",
    "total_variation_reg",
]
