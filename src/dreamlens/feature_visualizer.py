from copy import deepcopy

import torch

from .image_parameters import (
    FourierCanvas,
    ReferenceCanvas,
    ReferenceCanvasBatch,
    ReferenceMaskedCanvas,
    PixelCanvas,
    MaskedCanvas,
    call_image_parameter,
)
from .layers import LayerCapture, as_list, model_device, resolve_module
from .objectives import (
    Objective,
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
from .optimization import (
    AmplifyConfig,
    FeatureAccentuationConfig,
    MacoConfig,
    OptimizationResult,
    RenderConfig,
    feature_accentuation as run_feature_accentuation,
    maco as run_maco,
)
from .preprocessing import identity_preprocess, image_to_tensor, imagenet_normalize
from .render import (
    compose,
    crop_or_pad_to,
    normalize_positions,
    random_rotate,
    random_scale_from_choices,
)
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
            transforms.append(
                random_scale_from_choices(_linspace(scale_min, scale_max, steps=32))
            )
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
        hooks = [
            LayerCapture(resolve_module(self.model, layer)).open()
            for layer in as_list(layers)
        ]
        losses = []

        try:
            for step in range(iters):
                _zero_grad(optimizer)
                image = call_image_parameter(image_parameter, self.device)
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

        targets = [coerce_target(item) for item in as_list(target)]
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

    def visualize(
        self,
        target=None,
        method="maximize",
        config=None,
        image_parameter=None,
        objective=None,
        maco_dataset=None,
        image=None,
        layers=None,
        power=1.2,
        regularization_layer=None,
    ):
        """Unified feature-visualization entry point.

        ``method='maximize'`` uses DreamLens's classical FFT/pixel engine.
        ``method='maco'`` keeps Fourier magnitude fixed and optimizes phase.
        ``method='feature_accentuation'`` accentuates ``target`` in ``image``
        while preserving ``regularization_layer`` features.
        ``method='caricature'`` amplifies features captured from ``image`` at
        ``layers``. Algorithm-specific settings live in ``RenderConfig``,
        ``MacoConfig``, ``FeatureAccentuationConfig``, and ``AmplifyConfig``
        respectively.
        """

        method = str(method).lower()
        if method in {"maximize", "fft", "lucid"}:
            if target is None:
                raise ValueError("target is required for classical maximize")
            if image is not None or layers is not None:
                raise ValueError("image and layers are only used by caricature")
            if regularization_layer is not None:
                raise ValueError("regularization_layer is only used by feature accentuation")
            return self.maximize(
                target=target,
                config=config,
                image_parameter=image_parameter,
                objective=objective,
            )
        if method == "maco":
            if target is None:
                raise ValueError("target is required for MaCo")
            if image is not None or layers is not None:
                raise ValueError("image and layers are only used by caricature")
            if regularization_layer is not None:
                raise ValueError("regularization_layer is only used by feature accentuation")
            if image_parameter is not None:
                raise ValueError("MaCo does not accept an image_parameter")
            if objective is not None:
                raise ValueError("Pass a FeatureTarget to MaCo, not a custom objective")
            return self.maco(
                target=target,
                config=config,
                maco_dataset=maco_dataset,
            )
        if method in {"feature_accentuation", "accentuation", "faccent"}:
            if target is None or image is None:
                raise ValueError("target and image are required for feature accentuation")
            if layers is not None:
                raise ValueError("layers is only used by caricature")
            if objective is not None or maco_dataset is not None:
                raise ValueError(
                    "objective and maco_dataset are not used by feature accentuation"
                )
            return self.accentuate(
                target=target,
                image=image,
                regularization_layer=regularization_layer,
                config=config,
                image_parameter=image_parameter,
            )
        if method in {"caricature", "amplify"}:
            if target is not None:
                raise ValueError("caricature uses image and layers, not target")
            if image is None or layers is None:
                raise ValueError("image and layers are required for caricature")
            if objective is not None or maco_dataset is not None:
                raise ValueError("objective and maco_dataset are not used by caricature")
            if regularization_layer is not None:
                raise ValueError("regularization_layer is only used by feature accentuation")
            return self.caricature(
                image=image,
                layers=layers,
                power=power,
                config=config,
                image_parameter=image_parameter,
            )
        raise ValueError(
            "method must be 'maximize', 'maco', 'feature_accentuation', or 'caricature'"
        )

    def maximize_layer(self, layer, config=None, reduction="norm", weight=1.0):
        """Maximize a complete layer with the root DreamLens engine."""

        return self.maximize(
            FeatureTarget.for_layer(layer, reduction=reduction, weight=weight),
            config=config,
        )

    def maximize_channel(
        self,
        layer,
        channel,
        config=None,
        position=None,
        reduction="mean",
        weight=1.0,
    ):
        """Maximize one channel, optionally at one spatial/token position."""

        return self.maximize(
            FeatureTarget.for_channel(
                layer,
                channel,
                position=position,
                reduction=reduction,
                weight=weight,
            ),
            config=config,
        )

    def maximize_neuron(
        self,
        layer,
        neuron,
        config=None,
        reduction="mean",
        weight=1.0,
    ):
        """Maximize one flattened neuron in any batched layer output."""

        return self.maximize(
            FeatureTarget.for_neuron(
                layer,
                neuron,
                reduction=reduction,
                weight=weight,
            ),
            config=config,
        )

    def maximize_class(
        self,
        class_id,
        layer=-1,
        config=None,
        weight=1.0,
    ):
        """Maximize one classifier logit; the final leaf layer is the default."""

        return self.maximize_neuron(
            layer=layer,
            neuron=int(class_id),
            config=config,
            weight=weight,
        )

    def maximize_direction(
        self,
        layer,
        direction,
        config=None,
        cossim_power=2.0,
        weight=1.0,
    ):
        """Maximize a user-provided direction in a layer activation space."""

        return self.maximize(
            FeatureTarget.for_direction(
                layer,
                direction,
                cossim_power=cossim_power,
                weight=weight,
            ),
            config=config,
        )

    def maco(self, target, config=None, maco_dataset=None):
        """Run MaCo through the root API and return image plus importance map."""

        config = MacoConfig() if config is None else config
        if not isinstance(config, MacoConfig):
            raise TypeError("config must be a MacoConfig")
        target = coerce_target(target)
        objective = Objective.from_target(
            self.model,
            target,
            input_shape=config.input_shape,
        )
        preprocess = self.preprocess if config.preprocess is None else config.preprocess

        if config.optimizer_cls is None:
            optimizer = lambda parameters: torch.optim.NAdam(  # noqa: E731
                parameters,
                lr=config.lr,
                eps=1e-7,
            )
        else:
            optimizer = lambda parameters: config.optimizer_cls(  # noqa: E731
                parameters,
                lr=config.lr,
            )
        image, transparency = run_maco(
            objective,
            optimizer=optimizer,
            maco_dataset=maco_dataset,
            nb_steps=config.steps,
            noise_intensity=config.noise_intensity,
            box_size=config.box_size,
            nb_crops=config.crops,
            values_range=config.values_range,
            custom_shape=(config.height, config.width),
            input_shape=config.input_shape,
            device=self.device,
            preprocess=preprocess,
        )
        return OptimizationResult(
            image=image,
            transparency=transparency,
            losses=[],
            objective_value=None,
            attempt_index=0,
            metadata={
                "method": "maco",
                "target": target,
                "steps": config.steps,
                "crops": config.crops,
                "values_range": tuple(config.values_range),
            },
        )

    def accentuate(
        self,
        target,
        image,
        regularization_layer=None,
        config=None,
        image_parameter=None,
    ):
        """Run Faccent-style feature accentuation on one natural image."""

        config = FeatureAccentuationConfig() if config is None else config
        if not isinstance(config, FeatureAccentuationConfig):
            raise TypeError("config must be a FeatureAccentuationConfig")
        target = coerce_target(target)
        if config.regularization_strength and regularization_layer is None:
            raise ValueError(
                "regularization_layer is required when regularization_strength is non-zero"
            )
        objective = Objective.from_target(
            self.model,
            target,
            input_shape=config.input_shape,
        )
        preprocess = self.preprocess if config.preprocess is None else config.preprocess
        result = run_feature_accentuation(
            objective=objective,
            image=image,
            regularization_layer=regularization_layer,
            optimizer=config.optimizer_cls,
            image_parameter=image_parameter,
            nb_steps=config.steps,
            nb_crops=config.crops,
            crop_min=config.crop_min,
            crop_max=config.crop_max,
            noise_std=config.noise_std,
            regularization_strength=config.regularization_strength,
            custom_shape=(config.height, config.width),
            input_shape=config.input_shape,
            parameterization=config.parameterization,
            magnitude_source=config.magnitude_source,
            use_magnitude_gate=config.use_magnitude_gate,
            magnitude_gate_init=config.magnitude_gate_init,
            desaturation=config.desaturation,
            frequency_decay=config.frequency_decay,
            color_decorrelate=config.color_decorrelate,
            center_crop=config.center_crop,
            resize_mode=config.resize_mode,
            grad_clip=config.grad_clip,
            regularizer_in_transparency=config.regularizer_in_transparency,
            checkpoint_steps=config.checkpoint_steps,
            device=self.device,
            preprocess=preprocess,
            learning_rate=config.lr,
        )
        result.metadata.update(
            {
                "target": target,
                "regularization_layer": regularization_layer,
            }
        )
        return result

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

        channels = [int(channel) for channel in channels]
        positions = normalize_positions(positions, len(channels))
        targets = [
            FeatureTarget.for_channel(
                layer,
                channel,
                position=positions[index],
                reduction=reduction,
            )
            for index, channel in enumerate(channels)
        ]
        return self._maximize_target_batch(
            layer=layer,
            targets=targets,
            config=config,
            image_parameter=image_parameter,
        )

    def maximize_neurons(
        self,
        layer,
        neurons,
        config=None,
        reduction="mean",
        image_parameter=None,
    ):
        """Optimize one image per flattened neuron in a single batch."""

        targets = [
            FeatureTarget.for_neuron(layer, neuron, reduction=reduction)
            for neuron in neurons
        ]
        return self._maximize_target_batch(
            layer=layer,
            targets=targets,
            config=config,
            image_parameter=image_parameter,
        )

    def maximize_classes(
        self,
        class_ids,
        layer=-1,
        config=None,
        image_parameter=None,
    ):
        """Optimize one image per classifier logit in a single batch."""

        return self.maximize_neurons(
            layer=layer,
            neurons=class_ids,
            config=config,
            image_parameter=image_parameter,
        )

    def maximize_directions(
        self,
        layer,
        directions,
        config=None,
        cossim_power=2.0,
        image_parameter=None,
    ):
        """Optimize one image per activation direction in a single batch."""

        targets = [
            FeatureTarget.for_direction(
                layer,
                direction,
                cossim_power=cossim_power,
            )
            for direction in directions
        ]
        return self._maximize_target_batch(
            layer=layer,
            targets=targets,
            config=config,
            image_parameter=image_parameter,
        )

    def _maximize_target_batch(
        self,
        layer,
        targets,
        config=None,
        image_parameter=None,
    ):
        config = RenderConfig() if config is None else config
        targets = list(targets)
        if not targets:
            raise ValueError("targets must contain at least one item")
        if config.attempts != 1:
            raise ValueError("batched rendering currently supports attempts=1")
        if config.parameterization not in {"lucid", "reference"}:
            raise ValueError("config.parameterization must be 'lucid' or 'reference'")

        objectives = [TargetObjective([target]) for target in targets]

        transforms = config.transform.transforms
        preprocess = config.preprocess
        if config.parameterization == "reference":
            if image_parameter is None:
                image_parameter = ReferenceCanvasBatch(
                    batch_size=len(targets),
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
                batch_size=len(targets),
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
        hooks = [
            LayerCapture(resolve_module(self.model, layer)).open()
            for layer in as_list(layers)
        ]
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
        hooks = [
            LayerCapture(resolve_module(self.model, layer)).open()
            for layer in as_list(layers)
        ]
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
                image = call_image_parameter(image_parameter, self.device)

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
        hooks = [
            LayerCapture(resolve_module(self.model, layer)).open()
            for layer in as_list(layers)
        ]
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
                image_normalized = call_image_parameter(image_parameter, self.device)

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
