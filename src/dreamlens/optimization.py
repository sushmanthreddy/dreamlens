from dataclasses import dataclass, field
from typing import Optional

import numpy as np


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
class OptimizationResult:
    """Return object for the project-owned high-level API."""

    image: object
    losses: list[float]
    objective_value: Optional[float] = None
    attempt_index: int = 0

    def save(self, filename):
        return self.image.save(filename)

    def as_chw(self, device=None):
        return self.image.as_chw(device=device)

    def as_hwc(self, device=None):
        return self.image.as_hwc(device=device)

    def as_nchw(self, device=None):
        if hasattr(self.image, "as_nchw"):
            return self.image.as_nchw(device=device)
        return self.image.forward(device=device).detach().cpu()

    def __array__(self, dtype=None):
        array = np.asarray(self.image)
        if dtype is not None:
            array = array.astype(dtype)
        return array
