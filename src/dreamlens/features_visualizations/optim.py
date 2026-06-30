"""Xplique-compatible feature visualization optimization in native PyTorch."""

from contextlib import contextmanager

import torch
import torch.nn.functional as F

from ..layers import model_device
from .objectives import Objective, infer_input_channels, _normalize_input_shape
from .preconditioning import (
    fft_image,
    fft_to_rgb,
    get_fft_scale,
    to_valid_grayscale,
    to_valid_rgb,
)
from .transformations import compose_transformations, generate_standard_transformations


def optimize(
    objective,
    optimizer=None,
    nb_steps=256,
    use_fft=True,
    fft_decay=0.85,
    std=0.01,
    regularizers=None,
    image_normalizer="sigmoid",
    values_range=(0, 1),
    transformations="standard",
    warmup_steps=False,
    custom_shape=(512, 512),
    save_every=None,
    input_shape=None,
    device=None,
    preprocess=None,
    progress_callback=None,
):
    """Optimize a composable :class:`Objective` with gradient ascent.

    Images and shapes use PyTorch's NCHW/CHW convention. ``input_shape`` is the
    model's ``(C,H,W)`` input. ``custom_shape`` controls the generated canvas,
    which is transformed and resized to the model input before inference.
    ``preprocess`` is applied after that resize and before the model; use it for
    differentiable torchvision normalization. ``progress_callback``, when
    provided, receives ``(completed_steps, total_steps)`` after every update.
    """

    if not isinstance(objective, Objective):
        raise TypeError("objective must be an Objective")
    nb_steps = int(nb_steps)
    if nb_steps < 1:
        raise ValueError("nb_steps must be >= 1")
    if save_every is not None and int(save_every) < 1:
        raise ValueError("save_every must be >= 1")
    low, high = min(values_range), max(values_range)
    values_range = (low, high)

    resolved_input_shape = _optimization_input_shape(
        objective,
        input_shape=input_shape,
        custom_shape=custom_shape,
    )
    model, objective_function, objective_names, compiled_shape = objective.compile(
        input_shape=resolved_input_shape
    )
    combination_count, channels, input_height, input_width = compiled_shape

    if custom_shape is None:
        canvas_height, canvas_width = input_height, input_width
    else:
        if len(custom_shape) != 2:
            model.close()
            raise ValueError("custom_shape must be (height, width) or None")
        canvas_height, canvas_width = (int(value) for value in custom_shape)
        if min(canvas_height, canvas_width) < 1:
            model.close()
            raise ValueError("custom_shape dimensions must be positive")
    image_shape = (combination_count, channels, canvas_height, canvas_width)

    if channels not in (1, 3):
        model.close()
        raise ValueError("Xplique feature visualization supports 1 or 3 input channels")
    if transformations == "standard":
        transformations = generate_standard_transformations(
            size=canvas_height,
            channels=channels,
        )
    elif isinstance(transformations, (list, tuple)):
        transformations = compose_transformations(transformations)
    elif transformations is not None and not callable(transformations):
        model.close()
        raise TypeError("transformations must be 'standard', a callable, a list, or None")

    to_valid_image = to_valid_rgb if channels == 3 else to_valid_grayscale
    device = torch.device(device) if device is not None else model_device(objective.model)
    objective.model.to(device)

    if use_fft:
        inputs = torch.nn.Parameter(fft_image(image_shape, std=std, device=device))
        fft_scale = get_fft_scale(
            canvas_height,
            canvas_width,
            decay_power=fft_decay,
            device=device,
        )

        def image_param(buffer):
            return to_valid_image(
                fft_to_rgb(image_shape, buffer, fft_scale),
                image_normalizer,
                values_range,
            )

    else:
        inputs = torch.nn.Parameter(
            torch.randn(image_shape, dtype=torch.float32, device=device) * float(std)
        )

        def image_param(buffer):
            return to_valid_image(buffer, image_normalizer, values_range)

    optimizer = _prepare_optimizer(
        optimizer,
        inputs,
        default_cls=torch.optim.Adam,
        default_lr=0.05,
    )
    optimisation_step = _get_optimisation_step(
        objective_function=objective_function,
        image_param=image_param,
        input_shape=compiled_shape,
        transformations=transformations,
        regularizers=regularizers,
        preprocess=preprocess,
    )

    images_optimized = []
    try:
        with _frozen_eval_model(objective.model):
            if warmup_steps:
                with _open_relu_gradients(objective.model):
                    for _ in range(int(warmup_steps)):
                        optimisation_step(model, inputs, optimizer)

            for step_index in range(nb_steps):
                optimisation_step(model, inputs, optimizer)
                if progress_callback is not None:
                    progress_callback(step_index + 1, nb_steps)
                last_iteration = step_index == nb_steps - 1
                should_save = save_every and (step_index + 1) % int(save_every) == 0
                if should_save or last_iteration:
                    images_optimized.append(image_param(inputs).detach().clone())
    finally:
        model.close()

    return images_optimized, objective_names


def _get_optimisation_step(
    objective_function,
    image_param,
    input_shape,
    transformations=None,
    regularizers=None,
    preprocess=None,
):
    """Create one differentiable ascent step for the given objective."""

    target_height, target_width = input_shape[-2:]

    def step(model, inputs, optimizer):
        optimizer.zero_grad(set_to_none=True)
        images = image_param(inputs)
        if transformations:
            images = transformations(images)
        images = F.interpolate(
            images,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
        model_inputs = images if preprocess is None else preprocess(images)
        model_outputs = model(model_inputs)
        loss = objective_function(model_outputs)
        if regularizers:
            for regularizer in regularizers:
                loss = loss - regularizer(images)
        score = loss.sum()
        (-score).backward()
        optimizer.step()
        return inputs.grad

    return step


def _optimization_input_shape(objective, input_shape, custom_shape):
    if input_shape is not None:
        return _normalize_input_shape(input_shape)
    if objective.input_shape is not None:
        return objective.input_shape
    declared = getattr(objective.model, "input_shape", None)
    if declared is None:
        declared = getattr(objective.model, "_dreamlens_input_shape", None)
    if declared is not None:
        return _normalize_input_shape(declared)
    if custom_shape is None:
        return _normalize_input_shape(None)
    if len(custom_shape) != 2:
        raise ValueError("custom_shape must be (height, width) or None")
    channels = infer_input_channels(objective.model)
    return channels, int(custom_shape[0]), int(custom_shape[1])


def _prepare_optimizer(optimizer, parameter, default_cls, default_lr):
    if optimizer is None:
        # Keras Adam/Nadam use epsilon=1e-7; retain that default for parity.
        return default_cls([parameter], lr=default_lr, eps=1e-7)
    if isinstance(optimizer, torch.optim.Optimizer):
        if not optimizer.param_groups:
            raise ValueError("optimizer must have at least one parameter group")
        optimizer.state.clear()
        optimizer.param_groups[0]["params"] = [parameter]
        for group in optimizer.param_groups[1:]:
            group["params"] = []
        return optimizer
    if callable(optimizer):
        try:
            candidate = optimizer([parameter])
        except TypeError as first_error:
            try:
                candidate = optimizer([parameter], lr=0.01)
            except TypeError:
                raise TypeError(
                    "optimizer callables must accept an iterable of parameters; use "
                    "functools.partial to configure required options"
                ) from first_error
        if not isinstance(candidate, torch.optim.Optimizer):
            raise TypeError("optimizer callable must return torch.optim.Optimizer")
        return candidate
    raise TypeError("optimizer must be a torch optimizer, optimizer factory/class, or None")


@contextmanager
def _frozen_eval_model(model):
    training = model.training
    requires_grad = [parameter.requires_grad for parameter in model.parameters()]
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, original in zip(model.parameters(), requires_grad):
            parameter.requires_grad_(original)
        model.train(training)


@contextmanager
def _open_relu_gradients(model):
    handles = []
    inplace_states = []

    def open_backward(module, grad_input, grad_output):
        if not grad_output:
            return grad_input
        return (grad_output[0],)

    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            inplace_states.append((module, module.inplace))
            module.inplace = False
            handles.append(module.register_full_backward_hook(open_backward))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        for module, inplace in inplace_states:
            module.inplace = inplace


__all__ = ["optimize"]
