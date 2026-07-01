from itertools import chain
import math

import numpy as np
import torch
import torch.nn.functional as F

from .layers import LayerCapture, model_device, resolve_module


COLOR_CORRELATION_SVD_SQRT = np.asarray(
    [[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]],
    dtype="float32",
)
MAX_NORM_SVD_SQRT = np.max(np.linalg.norm(COLOR_CORRELATION_SVD_SQRT, axis=0))


def fft_2d_freq(height, width):
    """Return Lucid-compatible real-FFT radial frequencies."""

    height, width = int(height), int(width)
    if height < 1 or width < 1:
        raise ValueError("height and width must be positive")
    freq_y = np.fft.fftfreq(height)[:, np.newaxis]
    odd_extra = int(width % 2 == 1)
    freq_x = np.fft.fftfreq(width)[: width // 2 + 1 + odd_extra]
    return np.sqrt(freq_x**2 + freq_y**2)


class ImageParameterization(torch.nn.Module):
    """Optimizable image parameterization matching Lucid's RGB constraints."""

    def __init__(
        self,
        size,
        batch,
        channels=3,
        sd=None,
        decorrelate=True,
        fft=True,
        frequency_decay=1.0,
        raw_scale=0.25,
        fft_norm=None,
        device=None,
    ):
        super().__init__()
        self.height, self.width = _normalize_image_size(size)
        self.size = self.height if self.height == self.width else (self.height, self.width)
        self.channels = channels
        self.decorrelate = decorrelate
        self.fft = fft
        self.frequency_decay = frequency_decay
        self.raw_scale = raw_scale
        self.fft_norm = fft_norm
        self.sd = 0.01 if sd is None else sd
        self.optimizer = None

        if fft:
            h, w = self.height, self.width
            freqs = self._rfft2d_freqs(h, w).astype("float32")
            init_shape = (2, batch, channels) + freqs.shape
            init_val = np.random.normal(size=init_shape, scale=self.sd).astype(
                "float32"
            )
            self.spectrum_real_imag = torch.nn.Parameter(
                torch.tensor(init_val, device=device)
            )
            scale_freqs = np.maximum(freqs, 1.0 / max(w, h)) ** frequency_decay
            scale = 1.0 / scale_freqs
            scale *= np.sqrt(w * h)
            self.register_buffer("spectrum_scale", torch.tensor(scale, device=device))
        else:
            init_shape = (batch, channels, self.height, self.width)
            init_val = np.random.normal(size=init_shape, scale=self.sd).astype(
                "float32"
            )
            self.pixels = torch.nn.Parameter(torch.tensor(init_val, device=device))

        color_matrix = COLOR_CORRELATION_SVD_SQRT / MAX_NORM_SVD_SQRT
        self.register_buffer(
            "color_matrix",
            torch.tensor(color_matrix, dtype=torch.float32, device=device),
        )

    @staticmethod
    def _rfft2d_freqs(h, w):
        return fft_2d_freq(h, w)

    def _raw_image(self):
        if not self.fft:
            return self.pixels

        real = self.spectrum_real_imag[0]
        imag = self.spectrum_real_imag[1]
        spectrum = torch.complex(real, imag)
        scaled_spectrum = spectrum * self.spectrum_scale
        image = torch.fft.irfft2(
            scaled_spectrum,
            s=(self.height, self.width),
            dim=(-2, -1),
            norm=self.fft_norm,
        )
        return image[:, :, : self.height, : self.width] * self.raw_scale

    def forward(self, device=None):
        if device is not None:
            self.to(device)
        image = self._raw_image()
        rgb = image[:, :3]
        if self.channels == 3:
            return self._to_valid_rgb(rgb)
        alpha = torch.sigmoid(image[:, 3:4])
        return torch.cat([self._to_valid_rgb(rgb), alpha], dim=1)

    def _to_valid_rgb(self, image):
        if self.decorrelate:
            image = torch.einsum("oc,bchw->bohw", self.color_matrix, image)
        return torch.sigmoid(image)

    def make_optimizer(self, lr=0.05, weight_decay=0.0, optimizer_cls=None):
        """Create and store an optimizer for this parameter."""

        optimizer_cls = torch.optim.AdamW if optimizer_cls is None else optimizer_cls
        self.optimizer = optimizer_cls(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        return self.optimizer

    def clip_gradients(self, grad_clip=1.0):
        torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)

    def as_nchw(self, device=None):
        with torch.no_grad():
            return self.forward(device=device)[:, :3].detach().cpu()

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


def render_icons(
    directions,
    model,
    layer,
    size=80,
    n_steps=128,
    verbose=False,
    S=None,
    num_attempts=3,
    cossim=True,
    alpha=False,
    device=None,
    learning_rate=0.05,
    preprocess=None,
    transforms=None,
    fft=True,
    decorrelate=True,
    activation_format="NCHW",
    boundary_penalty_weight=1.0,
    optimizer_cls=None,
    weight_decay=0.0,
    grad_clip=None,
):
    """Render one icon for each activation direction with PyTorch autograd."""

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if num_attempts < 1:
        raise ValueError("num_attempts must be >= 1")

    device = torch.device(device) if device is not None else model_device(model)
    model.to(device)
    model.eval()

    directions_t = _directions_to_tensor(directions, device=device)
    batch = directions_t.shape[0]
    channels = 4 if alpha else 3
    transforms = (
        standard_icon_transforms(size, alpha=alpha) if transforms is None else transforms
    )
    transform_f = compose(transforms)
    preprocess = (lambda image: image) if preprocess is None else preprocess
    feature_module = resolve_module(model, layer)

    if S is not None:
        S_t = torch.as_tensor(S, dtype=directions_t.dtype, device=device)
        directions_t = directions_t.matmul(S_t)

    image_attempts = []
    score_attempts = []

    with LayerCapture(feature_module) as capture:
        for attempt in range(num_attempts):
            param = ImageParameterization(
                size,
                batch=batch,
                channels=channels,
                decorrelate=decorrelate,
                fft=fft,
                device=device,
            )
            optimizer = _make_optimizer(
                param.parameters(),
                optimizer_cls=optimizer_cls,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )
            last_scores = None

            for step in range(n_steps):
                optimizer.zero_grad(set_to_none=True)
                image = param()
                model_input = preprocess(transform_f(image))
                capture.clear()
                model(model_input)
                acts = capture.output
                if acts is None:
                    raise RuntimeError("The requested layer was not called by model.")

                scores = _direction_scores(
                    acts,
                    directions_t,
                    cossim=cossim,
                    activation_format=activation_format,
                )
                objective = scores.sum() + boundary_penalty_weight * penalize_boundary_complexity(
                    image, w=5
                )
                (-objective).backward()
                if grad_clip is not None:
                    param.clip_gradients(grad_clip)
                optimizer.step()
                last_scores = scores.detach()

                if verbose and step % 100 == 0:
                    print(
                        "attempt",
                        attempt,
                        "step",
                        step,
                        "objective",
                        float(objective.detach()),
                    )

            final_image = param().detach()
            if alpha:
                rgb = final_image[:, :3]
                alpha_channel = final_image[:, 3:4]
                k = 0.8
                final_image = rgb * ((1 - k) + k * alpha_channel)
            else:
                final_image = final_image[:, :3]

            image_attempts.append(_to_numpy_nhwc(final_image))
            score_attempts.append(last_scores.cpu().numpy())

    score_attempts = np.asarray(score_attempts)
    image_final = []
    loss_final = []
    for i in range(batch):
        best_attempt = int(np.argmax(score_attempts[:, i]))
        image_final.append(image_attempts[best_attempt][i])
        loss_final.append(float(score_attempts[best_attempt, i]))

    return image_final, loss_final


def render_neurons(
    neurons,
    model,
    layer,
    positions=None,
    size=80,
    n_steps=128,
    verbose=False,
    num_attempts=3,
    alpha=False,
    device=None,
    learning_rate=0.05,
    preprocess=None,
    transforms=None,
    fft=True,
    decorrelate=True,
    activation_format="NCHW",
    boundary_penalty_weight=1.0,
    optimizer_cls=None,
    weight_decay=0.0,
    grad_clip=None,
):
    """Render images that maximize specific neurons/channels in a layer.

    Args:
      neurons: Iterable of channel/unit indices. One icon is rendered per item.
      positions: Optional spatial positions. For 4D conv outputs use ``(y, x)``;
        for 3D sequence outputs use an integer position. A single position is
        reused for all neurons, or pass one position per neuron.
    """

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if num_attempts < 1:
        raise ValueError("num_attempts must be >= 1")

    neurons = [int(neuron) for neuron in neurons]
    if not neurons:
        raise ValueError("neurons must contain at least one index")
    positions = normalize_positions(positions, len(neurons))

    device = torch.device(device) if device is not None else model_device(model)
    model.to(device)
    model.eval()

    batch = len(neurons)
    channels = 4 if alpha else 3
    transforms = (
        standard_icon_transforms(size, alpha=alpha) if transforms is None else transforms
    )
    transform_f = compose(transforms)
    preprocess = (lambda image: image) if preprocess is None else preprocess
    feature_module = resolve_module(model, layer)

    image_attempts = []
    score_attempts = []

    with LayerCapture(feature_module) as capture:
        for attempt in range(num_attempts):
            param = ImageParameterization(
                size,
                batch=batch,
                channels=channels,
                decorrelate=decorrelate,
                fft=fft,
                device=device,
            )
            optimizer = _make_optimizer(
                param.parameters(),
                optimizer_cls=optimizer_cls,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )
            last_scores = None

            for step in range(n_steps):
                optimizer.zero_grad(set_to_none=True)
                image = param()
                model_input = preprocess(transform_f(image))
                capture.clear()
                model(model_input)
                acts = capture.output
                if acts is None:
                    raise RuntimeError("The requested layer was not called by model.")

                scores = _neuron_scores(
                    acts,
                    neurons,
                    positions,
                    activation_format=activation_format,
                )
                objective = scores.sum() + boundary_penalty_weight * penalize_boundary_complexity(
                    image, w=5
                )
                (-objective).backward()
                if grad_clip is not None:
                    param.clip_gradients(grad_clip)
                optimizer.step()
                last_scores = scores.detach()

                if verbose and step % 100 == 0:
                    print(
                        "attempt",
                        attempt,
                        "step",
                        step,
                        "objective",
                        float(objective.detach()),
                    )

            final_image = param().detach()
            if alpha:
                rgb = final_image[:, :3]
                alpha_channel = final_image[:, 3:4]
                k = 0.8
                final_image = rgb * ((1 - k) + k * alpha_channel)
            else:
                final_image = final_image[:, :3]

            image_attempts.append(_to_numpy_nhwc(final_image))
            score_attempts.append(last_scores.cpu().numpy())

    score_attempts = np.asarray(score_attempts)
    image_final = []
    score_final = []
    for i in range(batch):
        best_attempt = int(np.argmax(score_attempts[:, i]))
        image_final.append(image_attempts[best_attempt][i])
        score_final.append(float(score_attempts[best_attempt, i]))

    return image_final, score_final


def render_channels(*args, **kwargs):
    """Alias for ``render_neurons`` for convolutional channel visualizations."""

    return render_neurons(*args, **kwargs)


def standard_icon_transforms(size, alpha=False):
    transforms = [
        pad_image(16, mode="constant", constant_value=0.5),
        jitter(4),
        jitter(4),
        jitter(8),
        jitter(8),
        jitter(8),
        random_scale_from_choices(0.998**n for n in range(20, 40)),
        random_rotate(chain(range(-20, 20), range(-10, 10), range(-5, 5), 5 * [0])),
        jitter(2),
        crop_or_pad_to(size, size),
    ]
    if alpha:
        transforms.append(collapse_alpha_random())
    return transforms


def direction_neuron_score(activation, direction, cossim=True, cossim_pow=2):
    dot = torch.mean(activation * direction)
    if not cossim:
        return dot
    mag = torch.sqrt(torch.sum(activation**2))
    cosine = dot / (1e-4 + mag)
    cosine = torch.clamp(cosine, min=0.1)
    return dot * cosine**cossim_pow


def penalize_boundary_complexity(image, w=5, C=0.5):
    mask = torch.ones_like(image)
    if image.shape[-2] > 2 * w and image.shape[-1] > 2 * w:
        mask[:, :, w:-w, w:-w] = 0

    blur = _blur(image, kernel_size=5)
    diffs = (blur - image) ** 2
    diffs = diffs + 0.8 * (image - C) ** 2
    return -torch.sum(diffs * mask)


def pad_image(width, mode="constant", constant_value=0.5):
    def inner(image):
        if mode.lower() == "constant":
            return F.pad(image, (width, width, width, width), value=constant_value)
        return F.pad(image, (width, width, width, width), mode=mode.lower())

    return inner


def jitter(amount):
    def inner(image):
        if amount <= 0:
            return image
        _, _, h, w = image.shape
        if h <= amount or w <= amount:
            return image
        top = torch.randint(0, amount + 1, (), device=image.device).item()
        left = torch.randint(0, amount + 1, (), device=image.device).item()
        return image[:, :, top : top + h - amount, left : left + w - amount]

    return inner


def random_scale_from_choices(scales):
    scales = list(scales)

    def inner(image):
        if not scales:
            return image
        scale = scales[torch.randint(0, len(scales), (), device=image.device).item()]
        _, _, h, w = image.shape
        target_h = max(1, int(scale * h))
        target_w = max(1, int(scale * w))
        return F.interpolate(
            image, size=(target_h, target_w), mode="bilinear", align_corners=False
        )

    return inner


def random_rotate(angles, units="degrees"):
    angles = list(angles)

    def inner(image):
        if not angles:
            return image
        angle = angles[torch.randint(0, len(angles), (), device=image.device).item()]
        if units.lower() == "degrees":
            angle = math.pi * float(angle) / 180.0
        theta = image.new_tensor(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
            ]
        )
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


def crop_or_pad_to(height, width):
    def inner(image):
        _, _, h, w = image.shape
        if h < height or w < width:
            pad_top = max((height - h) // 2, 0)
            pad_bottom = max(height - h - pad_top, 0)
            pad_left = max((width - w) // 2, 0)
            pad_right = max(width - w - pad_left, 0)
            image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
            _, _, h, w = image.shape
        top = max((h - height) // 2, 0)
        left = max((w - width) // 2, 0)
        return image[:, :, top : top + height, left : left + width]

    return inner


def collapse_alpha_random(sd=0.5):
    def inner(image):
        rgb, alpha = image[:, :3], image[:, 3:4]
        random_bg = torch.sigmoid(torch.randn_like(rgb) * sd)
        return alpha * rgb + (1.0 - alpha) * random_bg

    return inner


def compose(transforms):
    def inner(image):
        for transform in transforms:
            image = transform(image)
        return image

    return inner


def _blur(image, kernel_size=5):
    channels = image.shape[1]
    kernel = torch.full(
        (channels, 1, kernel_size, kernel_size),
        0.5,
        dtype=image.dtype,
        device=image.device,
    )
    if kernel_size > 2:
        kernel[:, :, 1:-1, 1:-1] = 1.0
    padding = kernel_size // 2
    numerator = F.conv2d(image, kernel, padding=padding, groups=channels)
    denominator = F.conv2d(torch.ones_like(image), kernel, padding=padding, groups=channels)
    return numerator / denominator.clamp_min(1e-6)


def _normalize_image_size(size):
    if isinstance(size, int):
        if size <= 0:
            raise ValueError("size must be positive")
        return size, size
    if isinstance(size, (tuple, list)) and len(size) == 2:
        height, width = int(size[0]), int(size[1])
        if height <= 0 or width <= 0:
            raise ValueError("image height and width must be positive")
        return height, width
    raise TypeError("size must be an int or a (height, width) pair")


def _make_optimizer(parameters, optimizer_cls=None, learning_rate=0.05, weight_decay=0.0):
    optimizer_cls = torch.optim.Adam if optimizer_cls is None else optimizer_cls
    try:
        return optimizer_cls(parameters, lr=learning_rate, weight_decay=weight_decay)
    except TypeError:
        if weight_decay != 0.0:
            raise
        return optimizer_cls(parameters, lr=learning_rate)


def _direction_scores(acts, directions, cossim=True, activation_format="NCHW"):
    if acts.shape[0] != directions.shape[0]:
        raise ValueError(
            "Layer batch size {} does not match direction batch size {}.".format(
                acts.shape[0], directions.shape[0]
            )
        )

    selected = _select_center_activations(acts, activation_format=activation_format)
    if selected.shape[-1] != directions.shape[-1]:
        raise ValueError(
            "Layer has {} channels after selection, but directions have {}.".format(
                selected.shape[-1], directions.shape[-1]
            )
        )

    return torch.stack(
        [
            direction_neuron_score(selected[n], directions[n], cossim=cossim)
            for n in range(directions.shape[0])
        ]
    )


def _neuron_scores(acts, neurons, positions, activation_format="NCHW"):
    if acts.shape[0] != len(neurons):
        raise ValueError(
            "Layer batch size {} does not match neuron batch size {}.".format(
                acts.shape[0], len(neurons)
            )
        )

    return torch.stack(
        [
            _select_neuron_activation(
                acts,
                batch_index=n,
                neuron=neurons[n],
                position=positions[n],
                activation_format=activation_format,
            )
            for n in range(len(neurons))
        ]
    )


def _select_center_activations(acts, activation_format="NCHW"):
    if acts.dim() == 2:
        return acts
    if acts.dim() == 3:
        return acts[:, acts.shape[1] // 2, :]
    if acts.dim() != 4:
        raise ValueError("Expected 2D, 3D, or 4D layer activations.")

    activation_format = activation_format.upper()
    if activation_format == "NCHW":
        return acts[:, :, acts.shape[2] // 2, acts.shape[3] // 2]
    if activation_format == "NHWC":
        return acts[:, acts.shape[1] // 2, acts.shape[2] // 2, :]
    raise ValueError("activation_format must be 'NCHW' or 'NHWC'.")


def _select_neuron_activation(
    acts,
    batch_index,
    neuron,
    position=None,
    activation_format="NCHW",
):
    if acts.dim() == 2:
        return acts[batch_index, neuron]
    if acts.dim() == 3:
        pos = acts.shape[1] // 2 if position is None else int(position)
        return acts[batch_index, pos, neuron]
    if acts.dim() != 4:
        raise ValueError("Expected 2D, 3D, or 4D layer activations.")

    activation_format = activation_format.upper()
    if activation_format == "NCHW":
        if position is None:
            y, x = acts.shape[2] // 2, acts.shape[3] // 2
        else:
            y, x = position
        return acts[batch_index, neuron, int(y), int(x)]
    if activation_format == "NHWC":
        if position is None:
            y, x = acts.shape[1] // 2, acts.shape[2] // 2
        else:
            y, x = position
        return acts[batch_index, int(y), int(x), neuron]
    raise ValueError("activation_format must be 'NCHW' or 'NHWC'.")


def normalize_positions(positions, count):
    if positions is None:
        return [None] * count
    if isinstance(positions, tuple) and len(positions) == 2:
        return [positions] * count
    if isinstance(positions, int):
        return [positions] * count

    positions = list(positions)
    if len(positions) != count:
        raise ValueError("positions must be None, a single position, or match targets")
    return positions


def _directions_to_tensor(directions, device):
    directions_array = np.asarray(
        [
            d.detach().cpu().numpy() if isinstance(d, torch.Tensor) else np.asarray(d)
            for d in directions
        ],
        dtype="float32",
    )
    if directions_array.ndim != 2 or directions_array.shape[0] == 0:
        raise ValueError("directions must be a non-empty iterable of 1D vectors.")
    return torch.tensor(directions_array, dtype=torch.float32, device=device)


def _to_numpy_nhwc(image):
    return image.detach().clamp(0.0, 1.0).permute(0, 2, 3, 1).cpu().numpy()
