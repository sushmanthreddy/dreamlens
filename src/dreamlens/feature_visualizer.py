from collections.abc import Iterable
from copy import deepcopy

import torch

from .image_parameters import (
    FourierCanvas,
    ReferenceCanvas,
    ReferenceCanvasBatch,
    ReferenceMaskedCanvas,
    PixelCanvas,
    MaskedCanvas,
)
from .layers import model_device, resolve_module
from .objectives import (
    FeatureTarget,
    FeatureAmplificationObjective,
    PerSampleObjective,
    ReferenceAmplificationObjective,
    TargetObjective,
    channel_objective,
    coerce_target,
    mean_activation_objective,
    first_tensor,
)
from .optimization import AmplifyConfig, OptimizationResult, RenderConfig
from .preprocessing import identity_preprocess, image_to_tensor, imagenet_normalize
from .render import compose, crop_or_pad_to, random_rotate, random_scale
from .transforms import (
    paired_default_transforms,
    random_translate,
    reference_masked_transform,
    reference_paired_transforms,
    reference_single_transform,
    single_default_transform,
)


class FeatureVisualizer:
    """High-level feature visualization engine for PyTorch models."""

    def __init__(self, model, quiet=False, device=None, normalize=True, preprocess=None):
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        self.model = model
        self.device = torch.device(device) if device is not None else model_device(model)
        self.model.to(self.device)
        self.model.eval()
        self.quiet = quiet
        self.transforms = None
        self.default_func = mean_activation_objective
        if preprocess is not None:
            self.preprocess = preprocess
        else:
            self.preprocess = imagenet_normalize if normalize else identity_preprocess

    def build_transforms(
        self,
        rotate=15,
        scale_max=1.2,
        scale_min=0.5,
        translate_x=0.0,
        translate_y=0.0,
        height=256,
        width=256,
    ):
        transforms = []
        if scale_min != 1.0 or scale_max != 1.0:
            transforms.append(random_scale(_linspace(scale_min, scale_max, steps=32)))
        if rotate:
            rotate = int(rotate)
            transforms.append(random_rotate(range(-rotate, rotate + 1)))
        if translate_x or translate_y:
            transforms.append(random_translate(translate_x, translate_y))
        transforms.append(crop_or_pad_to(height, width))
        self.transforms = compose(transforms)
        return self.transforms

    def set_preprocess(self, normalization_transform):
        if not callable(normalization_transform):
            raise TypeError("normalization_transform must be callable")
        self.preprocess = normalization_transform

    def set_augmentations(self, transforms):
        self.transforms = _coerce_transforms(transforms)

    def synthesize(
        self,
        layers,
        image_parameter=None,
        width=256,
        height=256,
        iters=120,
        lr=9e-3,
        rotate_degrees=15,
        scale_max=1.2,
        scale_min=0.5,
        translate_x=0.0,
        translate_y=0.0,
        custom_func=None,
        weight_decay=0.0,
        grad_clip=1.0,
        transforms=None,
        preprocess=None,
        optimizer_cls=None,
        fft=True,
        decorrelate=True,
        return_losses=False,
    ):
        """Optimize an image for arbitrary layer objectives.

        ``custom_func`` receives a list of captured layer outputs and returns a
        scalar loss to minimize.
        """

        if iters < 1:
            raise ValueError("iters must be >= 1")

        image_parameter = self._prepare_image_parameter(
            image_parameter=image_parameter,
            height=height,
            width=width,
            fft=fft,
            decorrelate=decorrelate,
        )
        optimizer = self._prepare_optimizer(
            image_parameter=image_parameter,
            lr=lr,
            weight_decay=weight_decay,
            optimizer_cls=optimizer_cls,
        )
        transform_f = self._select_transforms(
            transforms=transforms,
            rotate_degrees=rotate_degrees,
            scale_max=scale_max,
            scale_min=scale_min,
            translate_x=translate_x,
            translate_y=translate_y,
            height=height,
            width=width,
        )
        preprocess_f = self.preprocess if preprocess is None else preprocess
        objective_f = self.default_func if custom_func is None else custom_func
        hooks = [LayerHook(resolve_module(self.model, layer)) for layer in _as_list(layers)]
        losses = []

        try:
            for step in range(iters):
                _zero_grad(optimizer)
                image = _call_image_parameter(image_parameter, self.device)
                if isinstance(image_parameter, ReferenceMaskedCanvas):
                    model_image = reference_masked_transform(
                        image,
                        image_parameter.mask.to(self.device),
                        image_parameter.original_nchw_image_tensor.to(self.device),
                        rotate_degrees=rotate_degrees,
                        scale_min=scale_min,
                        scale_max=scale_max,
                        translate_x=translate_x,
                        translate_y=translate_y,
                    )
                else:
                    model_image = transform_f(image)
                model_input = preprocess_f(model_image)
                layer_outputs = self._capture_layer_outputs(hooks, model_input)
                loss = objective_f(layer_outputs)
                loss.backward()
                if grad_clip is not None:
                    _clip_gradients(image_parameter, grad_clip)
                _step_optimizer(optimizer)
                losses.append(float(loss.detach().cpu()))

                if not self.quiet and (step == 0 or (step + 1) % 100 == 0):
                    print("step", step + 1, "loss", losses[-1])
        finally:
            for hook in hooks:
                hook.close()

        if return_losses:
            return image_parameter, losses
        return image_parameter

    def maximize(
        self,
        target,
        config=None,
        image_parameter=None,
        objective=None,
    ):
        """Optimize one image for explicit feature targets.

        This is the preferred high-level API. ``target`` may be a
        ``FeatureTarget``, a layer name/module, or a list of those values.
        """

        config = RenderConfig() if config is None else config
        if config.attempts < 1:
            raise ValueError("config.attempts must be >= 1")
        if config.parameterization not in {"lucid", "reference"}:
            raise ValueError("config.parameterization must be 'lucid' or 'reference'")

        targets = [coerce_target(item) for item in _target_list(target)]
        layers = [item.layer for item in targets]
        objective_f = TargetObjective(targets) if objective is None else objective

        best_param = None
        best_losses = None
        best_loss = None
        best_index = 0
        for attempt_index in range(config.attempts):
            attempt_image_parameter = image_parameter
            attempt_transforms = config.transform.transforms
            attempt_preprocess = config.preprocess
            if config.parameterization == "reference":
                if attempt_image_parameter is None:
                    attempt_image_parameter = ReferenceCanvas(
                        height=config.height,
                        width=config.width,
                        device=self.device,
                        standard_deviation=config.noise_std,
                    )
                if attempt_transforms is None:
                    attempt_transforms = _reference_transform_callable(config.transform)
                if attempt_preprocess is None:
                    attempt_preprocess = identity_preprocess

            image_param, losses = self.synthesize(
                layers=layers,
                image_parameter=attempt_image_parameter,
                width=config.width,
                height=config.height,
                iters=config.steps,
                lr=config.lr,
                rotate_degrees=config.transform.rotate_degrees,
                scale_max=config.transform.scale_max,
                scale_min=config.transform.scale_min,
                translate_x=config.transform.translate_x,
                translate_y=config.transform.translate_y,
                custom_func=objective_f,
                weight_decay=config.weight_decay,
                grad_clip=config.grad_clip,
                transforms=attempt_transforms,
                preprocess=attempt_preprocess,
                optimizer_cls=config.optimizer_cls,
                fft=config.fft,
                decorrelate=config.decorrelate,
                return_losses=True,
            )
            final_loss = losses[-1]
            if best_loss is None or final_loss < best_loss:
                best_param = image_param
                best_losses = losses
                best_loss = final_loss
                best_index = attempt_index

        return OptimizationResult(
            image=best_param,
            losses=best_losses,
            objective_value=best_loss,
            attempt_index=best_index,
        )

    def synthesize_channel(self, layer, channel, position=None, **kwargs):
        """Convenience wrapper for rendering a single channel/unit."""

        custom_func = kwargs.pop("custom_func", None)
        if custom_func is None:
            custom_func = channel_objective(channel=channel, position=position)
        return self.synthesize(layers=[layer], custom_func=custom_func, **kwargs)

    synthesize_channel = synthesize_channel

    def maximize_channels(
        self,
        layer,
        channels,
        config=None,
        positions=None,
        reduction="mean",
        image_parameter=None,
    ):
        """Optimize one image per channel/unit in a single batched render."""

        config = RenderConfig() if config is None else config
        channels = [int(channel) for channel in channels]
        if not channels:
            raise ValueError("channels must contain at least one channel index")
        if config.attempts != 1:
            raise ValueError("batched channel rendering currently supports attempts=1")
        if config.parameterization not in {"lucid", "reference"}:
            raise ValueError("config.parameterization must be 'lucid' or 'reference'")

        positions = _normalize_positions(positions, len(channels))
        objectives = [
            channel_objective(
                channel=channel,
                position=positions[index],
                reduction=reduction,
            )
            for index, channel in enumerate(channels)
        ]

        transforms = config.transform.transforms
        preprocess = config.preprocess
        if config.parameterization == "reference":
            if image_parameter is None:
                image_parameter = ReferenceCanvasBatch(
                    batch_size=len(channels),
                    height=config.height,
                    width=config.width,
                    device=self.device,
                    standard_deviation=config.noise_std,
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                    optimizer_cls=config.optimizer_cls,
                )
            if transforms is None:
                transforms = _reference_transform_callable(config.transform)
            if preprocess is None:
                preprocess = identity_preprocess
        elif image_parameter is None:
            image_parameter = FourierCanvas(
                height=config.height,
                width=config.width,
                device=self.device,
                standard_deviation=config.noise_std,
                batch_size=len(channels),
                fft=config.fft,
                decorrelate=config.decorrelate,
            )

        image_param, losses = self.synthesize(
            layers=[layer],
            image_parameter=image_parameter,
            width=config.width,
            height=config.height,
            iters=config.steps,
            lr=config.lr,
            rotate_degrees=config.transform.rotate_degrees,
            scale_max=config.transform.scale_max,
            scale_min=config.transform.scale_min,
            translate_x=config.transform.translate_x,
            translate_y=config.transform.translate_y,
            custom_func=PerSampleObjective(objectives),
            weight_decay=config.weight_decay,
            grad_clip=config.grad_clip,
            transforms=transforms,
            preprocess=preprocess,
            optimizer_cls=config.optimizer_cls,
            fft=config.fft,
            decorrelate=config.decorrelate,
            return_losses=True,
        )
        return OptimizationResult(
            image=image_param,
            losses=losses,
            objective_value=losses[-1],
            attempt_index=0,
        )

    def synthesize_from_image(self, image, layers, **kwargs):
        """Optimize a feature visualization initialized from a real image."""

        image_parameter = kwargs.pop("image_parameter", None)
        if image_parameter is None:
            image_parameter = PixelCanvas(image, device=self.device)
        return self.synthesize(layers=layers, image_parameter=image_parameter, **kwargs)

    def caricature(
        self,
        image,
        layers,
        power=1.2,
        config=None,
        image_parameter=None,
    ):
        """Amplify an input image's layer activations into a caricature."""

        config = AmplifyConfig.reference() if config is None else config
        return self.amplify(
            image=image,
            layers=layers,
            strength=power,
            config=config,
            image_parameter=image_parameter,
        )

    def capture_layers(self, layers, input_tensor, preprocess=None, first_batch=True):
        preprocess_f = self.preprocess if preprocess is None else preprocess
        hooks = [LayerHook(resolve_module(self.model, layer)) for layer in _as_list(layers)]
        try:
            with torch.no_grad():
                tensor = image_to_tensor(input_tensor, device=self.device)
                self.model(preprocess_f(tensor.float()))
                outputs = []
                for hook in hooks:
                    output = hook.tensor_output().detach().cpu()
                    outputs.append(output[0] if first_batch else output)
                return outputs
        finally:
            for hook in hooks:
                hook.close()

    def _amplify_lucid(
        self,
        input_tensor,
        layers,
        strength=1.0,
        image_parameter=None,
        start_from_input=True,
        mask=None,
        iters=120,
        lr=3e-4,
        rotate_degrees=15,
        scale_max=1.2,
        scale_min=0.5,
        translate_x=0.1,
        translate_y=0.1,
        weight_decay=1e-1,
        grad_clip=0.1,
        static=False,
        preprocess=None,
        optimizer_cls=None,
        paired_transforms=None,
        preserve_weight=0.0,
        variation_weight=0.0,
        noise_std=0.01,
        fft=True,
        decorrelate=True,
        frequency_decay=1.0,
        raw_scale=0.25,
        fft_norm=None,
        return_losses=False,
    ):
        """Amplify the features a model already sees in an input image."""

        if iters < 1:
            raise ValueError("iters must be >= 1")

        reference = image_to_tensor(input_tensor, device=self.device)
        height, width = reference.shape[-2], reference.shape[-1]
        if image_parameter is None:
            if mask is not None:
                image_parameter = MaskedCanvas(
                    image=reference,
                    mask_tensor=mask,
                    device=self.device,
                )
            elif start_from_input:
                image_parameter = PixelCanvas(reference, device=self.device)
            else:
                image_parameter = FourierCanvas(
                    height=height,
                    width=width,
                    device=self.device,
                    standard_deviation=noise_std,
                    fft=fft,
                    decorrelate=decorrelate,
                    frequency_decay=frequency_decay,
                    raw_scale=raw_scale,
                    fft_norm=fft_norm,
                )

        image_parameter = self._prepare_image_parameter(
            image_parameter=image_parameter,
            height=height,
            width=width,
            fft=fft,
            decorrelate=decorrelate,
        )
        optimizer = self._prepare_optimizer(
            image_parameter=image_parameter,
            lr=lr,
            weight_decay=weight_decay,
            optimizer_cls=optimizer_cls,
        )
        preprocess_f = self.preprocess if preprocess is None else preprocess
        hooks = [LayerHook(resolve_module(self.model, layer)) for layer in _as_list(layers)]
        losses = []

        if static:
            with torch.no_grad():
                static_targets = self._capture_layer_outputs(hooks, preprocess_f(reference))
            static_objective = FeatureAmplificationObjective(
                static_targets,
                strength=strength,
            )

        try:
            for step in range(iters):
                _zero_grad(optimizer)
                image = _call_image_parameter(image_parameter, self.device)

                if static:
                    moving = single_default_transform(
                        image,
                        height=height,
                        width=width,
                        rotate_degrees=rotate_degrees,
                        scale_min=scale_min,
                        scale_max=scale_max,
                        translate_x=translate_x,
                        translate_y=translate_y,
                    )
                    current_outputs = self._capture_layer_outputs(hooks, preprocess_f(moving))
                    loss = static_objective(current_outputs)
                else:
                    if paired_transforms is None:
                        moving, target_image = paired_default_transforms(
                            image,
                            reference,
                            height=height,
                            width=width,
                            rotate_degrees=rotate_degrees,
                            scale_min=scale_min,
                            scale_max=scale_max,
                            translate_x=translate_x,
                            translate_y=translate_y,
                        )
                    else:
                        moving, target_image = paired_transforms(image, reference)

                    with torch.no_grad():
                        target_outputs = self._capture_layer_outputs(
                            hooks,
                            preprocess_f(target_image),
                        )
                    current_outputs = self._capture_layer_outputs(
                        hooks,
                        preprocess_f(moving),
                    )
                    loss = FeatureAmplificationObjective(
                        target_outputs,
                        strength=strength,
                    )(current_outputs)

                if preserve_weight:
                    loss = loss + float(preserve_weight) * torch.mean(
                        (image - reference) ** 2
                    )
                if variation_weight:
                    loss = loss + float(variation_weight) * _total_variation(image)

                loss.backward()
                if grad_clip is not None:
                    _clip_gradients(image_parameter, grad_clip)
                _step_optimizer(optimizer)
                losses.append(float(loss.detach().cpu()))

                if not self.quiet and (step == 0 or (step + 1) % 100 == 0):
                    print("step", step + 1, "amplify_loss", losses[-1])
        finally:
            for hook in hooks:
                hook.close()

        if return_losses:
            return image_parameter, losses
        return image_parameter

    def amplify(
        self,
        image,
        layers,
        config=None,
        strength=1.0,
        mask=None,
        image_parameter=None,
    ):
        """Preferred feature amplification API.

        ``config.target_mode`` is ``"paired"`` for transform-matched targets or
        ``"static"`` for a fixed target snapshot.
        """

        config = AmplifyConfig() if config is None else config
        if config.start not in {"input", "noise"}:
            raise ValueError("config.start must be 'input' or 'noise'")
        if config.target_mode not in {"paired", "static"}:
            raise ValueError("config.target_mode must be 'paired' or 'static'")
        if config.parameterization == "reference":
            return self._amplify_reference(
                image=image,
                layers=layers,
                config=config,
                strength=strength,
                image_parameter=image_parameter,
            )
        if config.parameterization != "lucid":
            raise ValueError("config.parameterization must be 'lucid' or 'reference'")

        image_param, losses = self._amplify_lucid(
            input_tensor=image,
            layers=layers,
            strength=strength,
            image_parameter=image_parameter,
            start_from_input=config.start == "input",
            mask=mask,
            iters=config.steps,
            lr=config.lr,
            rotate_degrees=config.transform.rotate_degrees,
            scale_max=config.transform.scale_max,
            scale_min=config.transform.scale_min,
            translate_x=config.transform.translate_x,
            translate_y=config.transform.translate_y,
            weight_decay=config.weight_decay,
            grad_clip=config.grad_clip,
            static=config.target_mode == "static",
            preprocess=config.preprocess,
            optimizer_cls=config.optimizer_cls,
            preserve_weight=config.preserve_weight,
            variation_weight=config.variation_weight,
            noise_std=config.noise_std,
            fft=config.fft,
            decorrelate=config.decorrelate,
            frequency_decay=config.frequency_decay,
            raw_scale=config.raw_scale,
            fft_norm=config.fft_norm,
            return_losses=True,
        )
        return OptimizationResult(
            image=image_param,
            losses=losses,
            objective_value=losses[-1],
            attempt_index=0,
        )

    def _amplify_reference(
        self,
        image,
        layers,
        config,
        strength,
        image_parameter=None,
    ):
        reference_raw = image_to_tensor(image, device=self.device)
        height, width = reference_raw.shape[-2], reference_raw.shape[-1]
        preprocess_f = self.preprocess if config.preprocess is None else config.preprocess
        reference = preprocess_f(reference_raw)

        if image_parameter is None:
            image_parameter = ReferenceCanvas(
                height=height,
                width=width,
                device=self.device,
                standard_deviation=config.noise_std,
            )
        else:
            image_parameter = deepcopy(image_parameter)
            if isinstance(image_parameter, torch.nn.Module):
                image_parameter.to(self.device)

        optimizer = self._prepare_optimizer(
            image_parameter=image_parameter,
            lr=config.lr,
            weight_decay=config.weight_decay,
            optimizer_cls=config.optimizer_cls,
        )
        hooks = [LayerHook(resolve_module(self.model, layer)) for layer in _as_list(layers)]
        losses = []

        if config.target_mode == "static":
            with torch.no_grad():
                static_targets = self._capture_layer_outputs(hooks, reference)
            static_objective = ReferenceAmplificationObjective(
                static_targets,
                power=strength,
            )

        try:
            for step in range(config.steps):
                _zero_grad(optimizer)
                image_normalized = _call_image_parameter(image_parameter, self.device)

                if config.target_mode == "static":
                    moving = reference_single_transform(
                        image_normalized,
                        rotate_degrees=config.transform.rotate_degrees,
                        scale_min=config.transform.scale_min,
                        scale_max=config.transform.scale_max,
                        translate_x=config.transform.translate_x,
                        translate_y=config.transform.translate_y,
                    )
                    current_outputs = self._capture_layer_outputs(hooks, moving)
                    loss = static_objective(current_outputs)
                else:
                    moving, target_image = reference_paired_transforms(
                        image_normalized,
                        reference,
                        rotate_degrees=config.transform.rotate_degrees,
                        scale_min=config.transform.scale_min,
                        scale_max=config.transform.scale_max,
                        translate_x=config.transform.translate_x,
                        translate_y=config.transform.translate_y,
                    )
                    current_outputs = self._capture_layer_outputs(hooks, moving)
                    with torch.no_grad():
                        target_outputs = self._capture_layer_outputs(
                            hooks,
                            target_image.to(self.device),
                        )
                    loss = ReferenceAmplificationObjective(
                        target_outputs,
                        power=strength,
                    )(current_outputs)

                loss.backward()
                if config.grad_clip is not None:
                    _clip_gradients(image_parameter, config.grad_clip)
                _step_optimizer(optimizer)
                losses.append(float(loss.detach().cpu()))

                if not self.quiet and (step == 0 or (step + 1) % 100 == 0):
                    print("step", step + 1, "reference_amplify_loss", losses[-1])
        finally:
            for hook in hooks:
                hook.close()

        return OptimizationResult(
            image=image_parameter,
            losses=losses,
            objective_value=losses[-1],
            attempt_index=0,
        )

    def _capture_layer_outputs(self, hooks, model_input):
        for hook in hooks:
            hook.clear()
        self.model(model_input)
        return [hook.tensor_output() for hook in hooks]

    def _prepare_image_parameter(
        self,
        image_parameter,
        height,
        width,
        fft,
        decorrelate,
    ):
        if image_parameter is None:
            return FourierCanvas(
                height=height,
                width=width,
                device=self.device,
                fft=fft,
                decorrelate=decorrelate,
            )
        image_parameter = deepcopy(image_parameter)
        if isinstance(image_parameter, torch.nn.Module):
            image_parameter.to(self.device)
        return image_parameter

    def _prepare_optimizer(
        self,
        image_parameter,
        lr,
        weight_decay,
        optimizer_cls,
    ):
        if getattr(image_parameter, "optimizer", None) is not None:
            return image_parameter.optimizer
        if hasattr(image_parameter, "make_optimizer"):
            return image_parameter.make_optimizer(
                lr=lr,
                weight_decay=weight_decay,
                optimizer_cls=optimizer_cls or torch.optim.AdamW,
            )
        optimizer_cls = torch.optim.AdamW if optimizer_cls is None else optimizer_cls
        return optimizer_cls(
            image_parameter.parameters(), lr=lr, weight_decay=weight_decay
        )

    def _select_transforms(
        self,
        transforms,
        rotate_degrees,
        scale_max,
        scale_min,
        translate_x,
        translate_y,
        height,
        width,
    ):
        if transforms is not None:
            return _coerce_transforms(transforms)
        if self.transforms is not None:
            return self.transforms
        return self.build_transforms(
            rotate=rotate_degrees,
            scale_max=scale_max,
            scale_min=scale_min,
            translate_x=translate_x,
            translate_y=translate_y,
            height=height,
            width=width,
        )




class LayerHook:
    def __init__(self, module):
        self.output = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.output = output

    def clear(self):
        self.output = None

    def tensor_output(self):
        if self.output is None:
            raise RuntimeError("The requested layer was not called by the model.")
        return first_tensor(self.output)

    def close(self):
        self.handle.remove()


def _as_list(value):
    if isinstance(value, (str, torch.nn.Module)):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    raise TypeError("layers must be a layer, a layer name, or an iterable of them")


def _target_list(value):
    if isinstance(value, (str, torch.nn.Module, FeatureTarget)):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _normalize_positions(positions, count):
    if positions is None:
        return [None] * count
    if isinstance(positions, tuple) and len(positions) == 2:
        return [positions] * count
    if isinstance(positions, int):
        return [positions] * count
    positions = list(positions)
    if len(positions) != count:
        raise ValueError("positions must be None, a single position, or match channels")
    return positions


def _coerce_transforms(transforms):
    if transforms is None:
        return identity_preprocess
    if isinstance(transforms, (list, tuple)):
        return compose(transforms)
    if callable(transforms):
        return transforms
    raise TypeError("transforms must be callable or a list/tuple of callables")


def _reference_transform_callable(transform_config):
    def transform(image):
        return reference_single_transform(
            image,
            rotate_degrees=transform_config.rotate_degrees,
            scale_min=transform_config.scale_min,
            scale_max=transform_config.scale_max,
            translate_x=transform_config.translate_x,
            translate_y=transform_config.translate_y,
        )

    return transform


def _call_image_parameter(image_parameter, device):
    try:
        return image_parameter.forward(device=device)
    except TypeError:
        return image_parameter()


def _zero_grad(optimizer):
    if hasattr(optimizer, "clear_gradients"):
        optimizer.clear_gradients()
        return
    try:
        optimizer.zero_grad(set_to_none=True)
    except TypeError:
        optimizer.zero_grad()


def _step_optimizer(optimizer):
    if hasattr(optimizer, "advance"):
        optimizer.advance()
        return
    optimizer.step()


def _clip_gradients(image_parameter, grad_clip):
    if hasattr(image_parameter, "clip_gradients"):
        image_parameter.clip_gradients(grad_clip=grad_clip)
    else:
        torch.nn.utils.clip_grad_norm_(image_parameter.parameters(), grad_clip)


def _linspace(start, stop, steps):
    if steps <= 1:
        return [float(stop)]
    return [
        float(start + (stop - start) * index / (steps - 1))
        for index in range(steps)
    ]


def _total_variation(image):
    vertical = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
    horizontal = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]))
    return vertical + horizontal
