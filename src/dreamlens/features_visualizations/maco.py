"""MAgnitude Constrained Optimization (MaCo) for PyTorch models."""

import numbers

import numpy as np
import torch
import torch.nn.functional as F

from ..layers import model_device
from .objectives import Objective
from .optim import (
    _frozen_eval_model,
    _optimization_input_shape,
    _prepare_optimizer,
)
from .preconditioning import init_maco_buffer, maco_image_parametrization


def maco(
    objective,
    optimizer=None,
    maco_dataset=None,
    nb_steps=256,
    noise_intensity=0.08,
    box_size=None,
    nb_crops=32,
    values_range=(-1, 1),
    custom_shape=(512, 512),
    input_shape=None,
    device=None,
    preprocess=None,
):
    """Optimize one objective by changing phase while fixing FFT magnitude.

    The return values are CHW PyTorch tensors: the optimized image and its
    accumulated absolute input-gradient transparency map. ``preprocess`` is
    applied to generated crops immediately before model inference.
    """

    if not isinstance(objective, Objective):
        raise TypeError("objective must be an Objective")
    nb_steps = int(nb_steps)
    nb_crops = int(nb_crops)
    if nb_steps < 1:
        raise ValueError("nb_steps must be >= 1")
    if nb_crops < 0:
        raise ValueError("nb_crops must be >= 0")
    values_range = (min(values_range), max(values_range))

    resolved_input_shape = _optimization_input_shape(
        objective,
        input_shape=input_shape,
        custom_shape=custom_shape,
    )
    model, objective_function, _, compiled_shape = objective.compile(
        input_shape=resolved_input_shape
    )
    combinations, channels, input_height, input_width = compiled_shape
    if combinations != 1:
        model.close()
        raise AssertionError("You can only optimize one objective at a time with MaCo.")
    if channels == 1 and maco_dataset is None:
        model.close()
        raise ValueError("For grayscale images, a dataset is required to compute the buffer.")

    get_box_size = _schedule(
        box_size,
        default=torch.as_tensor(np.linspace(0.5, 0.05, nb_steps), dtype=torch.float32),
        argument_name="box_size",
    )
    get_noise_intensity = _schedule(
        noise_intensity,
        default=torch.as_tensor(np.logspace(0, -4, nb_steps), dtype=torch.float32),
        argument_name="noise_intensity",
    )

    if custom_shape is None:
        image_height, image_width = input_height, input_width
    else:
        if len(custom_shape) != 2:
            model.close()
            raise ValueError("custom_shape must be (height, width) or None")
        image_height, image_width = (int(value) for value in custom_shape)
        if min(image_height, image_width) < 1:
            model.close()
            raise ValueError("custom_shape dimensions must be positive")

    device = torch.device(device) if device is not None else model_device(objective.model)
    objective.model.to(device)
    magnitude, phase_value = init_maco_buffer(
        (channels, image_height, image_width),
        dataset=maco_dataset,
        device=device,
        data_format="CHW",
    )
    magnitude = magnitude.to(device=device, dtype=torch.float32)
    phase = torch.nn.Parameter(phase_value.to(device=device, dtype=torch.float32))
    optimizer = _prepare_optimizer(
        optimizer,
        phase,
        default_cls=torch.optim.NAdam,
        default_lr=1.0,
    )
    transparency = torch.zeros(
        (channels, image_height, image_width),
        dtype=torch.float32,
        device=device,
    )

    try:
        with _frozen_eval_model(objective.model):
            for step_index in range(nb_steps):
                box_size_at_step = get_box_size(step_index)
                noise_at_step = get_noise_intensity(step_index)
                grads_phase, grads_image = maco_optimisation_step(
                    model=model,
                    objective_function=objective_function,
                    magnitude=magnitude,
                    phase=phase,
                    box_average_size=box_size_at_step,
                    noise_std=noise_at_step,
                    nb_crops=nb_crops,
                    values_range=values_range,
                    input_size=(input_height, input_width),
                    preprocess=preprocess,
                )
                optimizer.zero_grad(set_to_none=True)
                phase.grad = -grads_phase.detach()
                optimizer.step()
                transparency = transparency + torch.abs(
                    _resize_chw(grads_image, (image_height, image_width))
                )
    finally:
        model.close()

    image = maco_image_parametrization(magnitude, phase, values_range)
    image = _resize_chw(image, (image_height, image_width))
    return image.detach(), transparency.detach()


def maco_optimisation_step(
    model,
    objective_function,
    magnitude,
    phase,
    box_average_size,
    noise_std,
    nb_crops,
    values_range,
    input_size=None,
    preprocess=None,
):
    """Compute MaCo phase and image gradients for one ascent step."""

    image = maco_image_parametrization(magnitude, phase, values_range)
    if input_size is None:
        input_size = image.shape[-2:]
    if nb_crops == 0:
        crops = F.interpolate(
            image.unsqueeze(0),
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
    else:
        dtype, device = image.dtype, image.device
        center_x = 0.5 + torch.randn(nb_crops, dtype=dtype, device=device) * 0.15
        center_y = 0.5 + torch.randn(nb_crops, dtype=dtype, device=device) * 0.15
        average = torch.as_tensor(box_average_size, dtype=dtype, device=device)
        delta_x = average + torch.randn(nb_crops, dtype=dtype, device=device) * 0.05
        delta_x = torch.clamp(delta_x, 0.05, 1.0)
        boxes = torch.stack(
            [
                center_x - delta_x * 0.5,
                center_y - delta_x * 0.5,
                center_x + delta_x * 0.5,
                center_y + delta_x * 0.5,
            ],
            dim=-1,
        )
        crops = _crop_and_resize(image, boxes, input_size)

    noise_std = torch.as_tensor(noise_std, dtype=crops.dtype, device=crops.device)
    crops = crops + torch.randn_like(crops) * noise_std
    crops = crops + (torch.rand_like(crops) - 0.5) * noise_std

    model_inputs = crops if preprocess is None else preprocess(crops)
    model_outputs = model(model_inputs)
    loss = torch.mean(objective_function(model_outputs))
    grads_phase, grads_image = torch.autograd.grad(loss, (phase, image))
    return grads_phase, grads_image


def _crop_and_resize(image, boxes, output_size):
    """Match the reference crop-and-resize operation for one source CHW image."""

    output_height, output_width = (int(value) for value in output_size)
    y1, x1, y2, x2 = boxes.unbind(dim=-1)
    y_fraction = _sample_fractions(output_height, image)
    x_fraction = _sample_fractions(output_width, image)
    ys = y1[:, None] + (y2 - y1)[:, None] * y_fraction[None, :]
    xs = x1[:, None] + (x2 - x1)[:, None] * x_fraction[None, :]
    grid_y = ys[:, :, None].expand(-1, output_height, output_width)
    grid_x = xs[:, None, :].expand(-1, output_height, output_width)
    grid = torch.stack((grid_x * 2.0 - 1.0, grid_y * 2.0 - 1.0), dim=-1)
    expanded = image.unsqueeze(0).expand(boxes.shape[0], -1, -1, -1)
    return F.grid_sample(
        expanded,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


def _sample_fractions(size, reference):
    if size == 1:
        return reference.new_tensor([0.5])
    return torch.linspace(0.0, 1.0, size, dtype=reference.dtype, device=reference.device)


def _schedule(value, default, argument_name):
    if value is None:
        values = default

        def from_default(step_index):
            return values[step_index]

        return from_default
    if callable(value):
        return value
    if isinstance(value, numbers.Real):

        def constant(_):
            return float(value)

        return constant
    raise ValueError(f"{argument_name} must be a function, a float, or None.")


def _resize_chw(image, size):
    if tuple(image.shape[-2:]) == tuple(size):
        return image
    return F.interpolate(
        image.unsqueeze(0),
        size=size,
        mode="bilinear",
        align_corners=False,
    )[0]


__all__ = ["maco", "maco_optimisation_step"]
