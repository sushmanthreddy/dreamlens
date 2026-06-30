import numpy as np
import torch

from .preprocessing import IMAGENET_MEAN, IMAGENET_STD, image_to_tensor, rgb_to_logit
from .render import ImageParameterization


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
            [_call_image_parameter(param, device) for param in self.image_params],
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


def _call_image_parameter(image_parameter, device):
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
