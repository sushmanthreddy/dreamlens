import numpy as np
import pytest
import torch

from dreamlens import atlas
from dreamlens import (
    FourierCanvas,
    AmplifyConfig,
    FourierCanvasBatch,
    PerSampleObjective,
    ChannelObjective,
    ReferenceCanvas,
    ReferenceCanvasBatch,
    ReferenceImageCanvas,
    ReferenceMaskedCanvas,
    FeatureTarget,
    FeatureVisualizer,
    ReferenceAmplificationObjective,
    PixelCanvas,
    MaskedCanvas,
    MacoConfig,
    ModelEnsemble,
    ParameterNoise,
    RenderConfig,
    TransformConfig,
    bin_laid_out_activations,
    list_layers,
    make_canvas,
    render_icons,
    render_neurons,
    supported_layers,
)
from dreamlens import collect_activations


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)

    def forward(self, x):
        return torch.relu(self.conv(x))


class ToyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(4, 5)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def test_render_icons_with_pixel_parameterization():
    model = ToyModel()
    directions = [np.random.randn(4).astype("float32") for _ in range(2)]

    icons, losses = render_icons(
        directions,
        model,
        layer="conv",
        size=16,
        n_steps=2,
        num_attempts=1,
        transforms=[],
        fft=False,
        optimizer_cls=torch.optim.AdamW,
        weight_decay=0.01,
        grad_clip=1.0,
    )

    assert len(icons) == 2
    assert icons[0].shape == (16, 16, 3)
    assert len(losses) == 2
    assert np.isfinite(icons[0]).all()


def test_render_neurons_for_conv_and_linear_layers():
    model = ToyClassifier()

    conv_icons, conv_scores = render_neurons(
        [0, 1],
        model,
        layer="conv",
        size=16,
        n_steps=2,
        num_attempts=1,
        transforms=[],
        fft=False,
    )
    linear_icons, linear_scores = render_neurons(
        [0, 2],
        model,
        layer="fc",
        size=16,
        n_steps=2,
        num_attempts=1,
        transforms=[],
        fft=False,
    )

    assert len(conv_icons) == 2
    assert len(conv_scores) == 2
    assert conv_icons[0].shape == (16, 16, 3)
    assert np.isfinite(conv_icons[0]).all()
    assert len(linear_icons) == 2
    assert len(linear_scores) == 2
    assert linear_icons[0].shape == (16, 16, 3)
    assert np.isfinite(linear_icons[0]).all()


def test_layer_listing_and_supported_layer_probe():
    model = ToyClassifier()
    sample_input = torch.rand(2, 3, 16, 16)

    layers = list_layers(model, sample_input=sample_input)
    supported = supported_layers(model, sample_input=sample_input)
    by_name = {layer.name: layer for layer in layers}

    assert by_name["conv"].output_shape == (2, 4, 16, 16)
    assert by_name["conv"].channels == 4
    assert by_name["conv"].spatial_shape == (16, 16)
    assert by_name["conv"].supported
    assert by_name["fc"].output_shape == (2, 5)
    assert by_name["fc"].channels == 5
    assert by_name["fc"].supported
    assert {"conv", "pool", "fc"}.issubset({layer.name for layer in supported})


def test_bin_and_canvas_helpers():
    layout = np.asarray([[0.1, 0.1], [0.8, 0.8], [0.82, 0.78]])
    activations = np.random.randn(3, 4).astype("float32")
    means, coords, counts = bin_laid_out_activations(
        layout, activations, grid_size=2, threshold=0
    )
    icons = [np.zeros((8, 8, 3), dtype="float32"), np.ones((8, 8, 3), dtype="float32")]
    canvas = make_canvas(icons, [np.asarray([0, 0]), np.asarray([1, 1])], 2)

    assert len(means) == 2
    assert len(coords) == 2
    assert counts == [1, 2]
    assert canvas.shape == (16, 16, 3)


def test_collect_activations_from_model_layer():
    model = ToyModel()
    inputs = torch.rand(3, 3, 8, 8)

    center_activations = collect_activations(
        model,
        layer="conv",
        inputs=inputs,
        spatial="center",
    )
    all_activations = collect_activations(
        model,
        layer="conv",
        inputs=inputs,
        spatial="all",
    )
    random_activations = collect_activations(
        model,
        layer="conv",
        inputs=inputs,
        spatial="random",
        random_seed=0,
    )

    assert center_activations.shape == (3, 4)
    assert all_activations.shape == (3 * 8 * 8, 4)
    assert random_activations.shape == (3, 4)
    assert np.isfinite(center_activations).all()
    assert np.isfinite(all_activations).all()
    assert np.isfinite(random_activations).all()


def test_render_custom_objective_compatibility():
    model = ToyClassifier()
    visualizer = FeatureVisualizer(model, device="cpu", normalize=False, quiet=True)

    def custom_func(layer_outputs):
        return -layer_outputs[0][:, 0].mean()

    image_param, losses = visualizer.synthesize(
        ["conv"],
        width=16,
        height=16,
        iters=2,
        lr=0.01,
        transforms=[],
        custom_func=custom_func,
        return_losses=True,
    )
    image = np.asarray(image_param)

    assert isinstance(image_param, FourierCanvas)
    assert image.shape == (16, 16, 3)
    assert len(losses) == 2
    assert np.isfinite(image).all()


def test_feature_visualizer_synthesize_channel_api():
    model = ToyClassifier()
    visualizer = FeatureVisualizer(model, device="cpu", normalize=False, quiet=True)
    image_param = visualizer.synthesize_channel(
        layer="conv",
        channel=0,
        width=16,
        height=16,
        iters=2,
        lr=0.01,
        transforms=[],
    )

    image = np.asarray(image_param)
    assert image.shape == (16, 16, 3)
    assert np.isfinite(image).all()


def test_feature_visualizer_maximize_target_api():
    model = ToyClassifier()
    visualizer = FeatureVisualizer(model, device="cpu", normalize=False, quiet=True)
    result = visualizer.maximize(
        FeatureTarget(layer="conv", channel=0),
        config=RenderConfig(
            width=16,
            height=16,
            steps=2,
            lr=0.01,
            transform=TransformConfig(transforms=[]),
        ),
    )

    image = np.asarray(result)
    assert image.shape == (16, 16, 3)
    assert len(result.losses) == 2
    assert np.isfinite(image).all()


def test_feature_target_rejects_ambiguous_target_selection():
    with pytest.raises(ValueError, match="only one"):
        FeatureTarget(layer="conv", channel=0, neuron=0)
    with pytest.raises(ValueError, match="position"):
        FeatureTarget(layer="conv", neuron=0, position=(1, 1))
    with pytest.raises(ValueError, match="reduction"):
        FeatureTarget(layer="conv", reduction="median")


def test_feature_target_class_factories_cover_every_target_kind():
    direction = torch.tensor([1.0, 0.0, 0.0])
    targets = [
        FeatureTarget.for_layer("conv"),
        FeatureTarget.for_channel("conv", 2, position=(1, 1)),
        FeatureTarget.for_neuron("conv", 7),
        FeatureTarget.for_class(3, layer="fc"),
        FeatureTarget.for_direction("fc", direction),
    ]

    assert targets[0].reduction == "norm"
    assert targets[1].channel == 2 and targets[1].position == (1, 1)
    assert targets[2].neuron == 7
    assert targets[3].layer == "fc" and targets[3].neuron == 3
    assert torch.equal(targets[4].direction, direction)


def test_feature_visualizer_root_convenience_targets():
    model = ToyClassifier()
    visualizer = FeatureVisualizer(model, device="cpu", normalize=False, quiet=True)
    config = RenderConfig(
        width=16,
        height=16,
        steps=1,
        lr=0.01,
        transform=TransformConfig(transforms=[]),
    )
    direction = torch.nn.functional.one_hot(torch.tensor(2), 5).float()

    results = [
        visualizer.maximize_layer("conv", config=config),
        visualizer.maximize_channel("conv", 1, config=config),
        visualizer.maximize_neuron("conv", 2, config=config),
        visualizer.maximize_class(3, layer="fc", config=config),
        visualizer.maximize_direction("fc", direction, config=config),
        visualizer.visualize(
            FeatureTarget(layer="fc", neuron=1),
            method="maximize",
            config=config,
        ),
    ]

    for result in results:
        assert result.as_chw().shape == (3, 16, 16)
        assert len(result.losses) == 1
        assert torch.isfinite(result.as_chw()).all()


def test_feature_visualizer_root_batched_targets():
    model = ToyClassifier()
    visualizer = FeatureVisualizer(model, device="cpu", normalize=False, quiet=True)
    config = RenderConfig(
        width=16,
        height=16,
        steps=1,
        lr=0.01,
        transform=TransformConfig(transforms=[]),
    )
    directions = [
        torch.nn.functional.one_hot(torch.tensor(index), 5).float()
        for index in (0, 1)
    ]

    neuron_result = visualizer.maximize_neurons("conv", [0, 1], config=config)
    class_result = visualizer.maximize_classes([0, 1], layer="fc", config=config)
    direction_result = visualizer.maximize_directions(
        "fc",
        directions,
        config=config,
    )

    for result in (neuron_result, class_result, direction_result):
        assert result.as_nchw().shape == (2, 3, 16, 16)
        assert len(result.losses) == 1
        assert torch.isfinite(result.as_nchw()).all()


@pytest.mark.parametrize("target_kind", ["layer", "channel", "neuron", "class", "direction"])
def test_feature_visualizer_root_maco_targets(target_kind, tmp_path):
    model = ToyClassifier()
    visualizer = FeatureVisualizer(model, device="cpu", normalize=False, quiet=True)
    direction = torch.nn.functional.one_hot(torch.tensor(2), 5).float()
    targets = {
        "layer": FeatureTarget(layer="conv", reduction="norm"),
        "channel": FeatureTarget(layer="conv", channel=1),
        "neuron": FeatureTarget(layer="conv", neuron=2),
        "class": FeatureTarget(layer="fc", neuron=3),
        "direction": FeatureTarget(layer="fc", direction=direction),
    }
    config = MacoConfig(
        width=16,
        height=16,
        input_shape=(3, 16, 16),
        steps=1,
        crops=1,
        noise_intensity=0.01,
    )

    result = visualizer.visualize(
        targets[target_kind],
        method="maco",
        config=config,
        maco_dataset=[torch.rand(2, 3, 16, 16)],
    )

    assert result.as_chw().shape == (3, 16, 16)
    assert result.transparency_chw().shape == (3, 16, 16)
    assert torch.isfinite(result.as_chw()).all()
    assert torch.isfinite(result.transparency_chw()).all()
    assert result.metadata["method"] == "maco"

    if target_kind == "class":
        image_path = tmp_path / "maco.png"
        importance_path = tmp_path / "maco_importance.png"
        result.save(image_path)
        result.save_transparency(importance_path)
        assert image_path.is_file()
        assert importance_path.is_file()


def test_feature_visualizer_reference_maximize_api():
    model = ToyClassifier()
    visualizer = FeatureVisualizer(model, device="cpu", normalize=False, quiet=True)
    result = visualizer.maximize(
        FeatureTarget(layer="conv"),
        config=RenderConfig(
            width=16,
            height=16,
            steps=2,
            lr=0.01,
            transform=TransformConfig(
                rotate_degrees=0,
                scale_min=1.0,
                scale_max=1.0,
                translate_x=0.0,
                translate_y=0.0,
            ),
            parameterization="reference",
        ),
    )

    assert isinstance(result.image, ReferenceCanvas)
    assert len(result.losses) == 2
    assert np.asarray(result).shape == (16, 16, 3)


def test_feature_visualizer_batched_objective():
    model = ToyClassifier()
    visualizer = FeatureVisualizer(model, device="cpu", normalize=False, quiet=True)
    objective = PerSampleObjective(
        [
            lambda outputs: -outputs[0][:, 0].mean(),
            lambda outputs: -outputs[0][:, 1].mean(),
        ]
    )
    image_param = FourierCanvas(height=16, width=16, batch_size=2, fft=False)

    image_param = visualizer.synthesize(
        "conv",
        image_parameter=image_param,
        iters=1,
        lr=0.01,
        transforms=[],
        custom_func=objective,
    )

    assert image_param.forward().shape == (2, 3, 16, 16)


def test_channel_objective_norm_reduction():
    output = torch.ones(1, 2, 3, 3)
    objective = ChannelObjective(channel=0, reduction="norm")

    loss = objective([output])

    assert torch.isclose(loss, -torch.tensor(3.0))


def test_feature_visualizer_reference_maximize_channels_api():
    model = ToyClassifier()
    visualizer = FeatureVisualizer(model, device="cpu", normalize=False, quiet=True)
    result = visualizer.maximize_channels(
        layer="conv",
        channels=[0, 1],
        reduction="norm",
        config=RenderConfig.reference(
            width=16,
            height=16,
            steps=2,
            lr=0.01,
            transform=TransformConfig(
                rotate_degrees=0,
                scale_min=1.0,
                scale_max=1.0,
                translate_x=0.0,
                translate_y=0.0,
            ),
        ),
    )

    assert isinstance(result.image, ReferenceCanvasBatch)
    assert result.image.forward().shape == (2, 3, 16, 16)
    assert len(result.losses) == 2


def test_image_and_masked_image_params():
    image = torch.rand(1, 3, 12, 12)
    mask = torch.ones(1, 1, 12, 12)
    mask[:, :, :6, :] = 0.0

    param = PixelCanvas(image)
    masked = MaskedCanvas(image=image, mask_tensor=mask)
    masked_output = masked.forward()

    assert param.forward().shape == (1, 3, 12, 12)
    assert np.asarray(param).shape == (12, 12, 3)
    assert torch.allclose(masked_output[:, :, :6, :], image[:, :, :6, :], atol=1e-6)


def test_reference_image_masked_and_batch_params():
    image = torch.rand(1, 3, 12, 12)
    mask = torch.ones(1, 1, 12, 12)
    mask[:, :, :6, :] = 0.0

    image_param = ReferenceImageCanvas(image, device="cpu")
    masked = ReferenceMaskedCanvas(image=image, mask_tensor=mask, device="cpu")
    batched = ReferenceCanvasBatch(batch_size=2, height=12, width=12, device="cpu")

    assert image_param.forward().shape == (1, 3, 12, 12)
    assert masked.forward().shape == (1, 3, 12, 12)
    assert np.asarray(masked).shape == (12, 12, 3)
    assert batched.forward().shape == (2, 3, 12, 12)


def test_feature_visualizer_amplify_static_api():
    model = ToyClassifier()
    visualizer = FeatureVisualizer(model, device="cpu", normalize=False, quiet=True)
    input_tensor = torch.rand(1, 3, 16, 16)

    result = visualizer.amplify(
        image=input_tensor,
        layers=["conv"],
        config=AmplifyConfig(
            steps=2,
            target_mode="static",
            transform=TransformConfig(
                rotate_degrees=0,
                scale_min=1.0,
                scale_max=1.0,
                translate_x=0.0,
                translate_y=0.0,
            ),
        ),
    )

    assert isinstance(result.image, PixelCanvas)
    assert result.image.forward().shape == (1, 3, 16, 16)
    assert len(result.losses) == 2
    assert np.isfinite(np.asarray(result)).all()


def test_feature_visualizer_amplify_api():
    model = ToyClassifier()
    visualizer = FeatureVisualizer(model, device="cpu", normalize=False, quiet=True)
    input_tensor = torch.rand(1, 3, 16, 16)

    result = visualizer.amplify(
        image=input_tensor,
        layers=["conv"],
        config=AmplifyConfig(
            steps=2,
            target_mode="static",
            transform=TransformConfig(
                rotate_degrees=0,
                scale_min=1.0,
                scale_max=1.0,
                translate_x=0.0,
                translate_y=0.0,
            ),
            preserve_weight=0.1,
            variation_weight=0.01,
        ),
    )

    assert isinstance(result.image, PixelCanvas)
    assert len(result.losses) == 2
    assert np.asarray(result).shape == (16, 16, 3)


def test_feature_visualizer_dream_preset_api():
    model = ToyClassifier()
    visualizer = FeatureVisualizer(model, device="cpu", normalize=False, quiet=True)
    input_tensor = torch.rand(1, 3, 16, 16)

    config = AmplifyConfig.dream(steps=2, lr=0.01)
    result = visualizer.amplify(
        image=input_tensor,
        layers=["conv"],
        config=config,
        strength=1.1,
    )

    assert isinstance(result.image, FourierCanvas)
    assert result.image.sd == 0.05
    assert result.image.frequency_decay == 1.0
    assert result.image.raw_scale == 0.75
    assert result.image.fft_norm is None
    assert len(result.losses) == 2
    assert np.asarray(result).shape == (16, 16, 3)


def test_feature_visualizer_reference_amplify_api():
    model = ToyClassifier()
    visualizer = FeatureVisualizer(model, device="cpu", normalize=True, quiet=True)
    input_tensor = torch.rand(1, 3, 16, 16)

    config = AmplifyConfig.reference(steps=2, lr=0.01)
    result = visualizer.amplify(
        image=input_tensor,
        layers=["conv"],
        config=config,
        strength=1.1,
    )

    assert isinstance(result.image, ReferenceCanvas)
    assert result.image.sd == 0.01
    assert len(result.losses) == 2
    assert np.asarray(result).shape == (16, 16, 3)


def test_reference_amplification_fractional_power_is_finite_for_negative_dot():
    objective = ReferenceAmplificationObjective(
        targets=[torch.ones(1, 2, 2, 2)],
        power=1.15,
    )
    current = -torch.ones(1, 2, 2, 2)

    loss = objective([current])

    assert torch.isfinite(loss)


def test_feature_visualizer_caricature_api():
    model = ToyClassifier()
    visualizer = FeatureVisualizer(model, device="cpu", normalize=True, quiet=True)
    input_tensor = torch.rand(1, 3, 16, 16)

    result = visualizer.visualize(
        method="caricature",
        image=input_tensor,
        layers=["conv"],
        power=1.1,
        config=AmplifyConfig.reference(steps=2, lr=0.01),
    )

    assert isinstance(result.image, ReferenceCanvas)
    assert len(result.losses) == 2
    assert np.asarray(result).shape == (16, 16, 3)


def test_model_ensemble_parameter_noise_and_fourier_batch():
    model_a = ToyClassifier()
    model_b = ToyClassifier()
    ensemble = ModelEnsemble({"a": model_a, "b": model_b})
    selected = ModelEnsemble([("a", model_a)], return_format="tuple")
    outputs = ensemble(torch.rand(1, 3, 16, 16))
    selected_outputs = selected(torch.rand(1, 3, 16, 16))
    conv = torch.nn.Conv2d(3, 4, kernel_size=1)
    noisy = ParameterNoise(conv, std=0.0)
    noisy_active = ParameterNoise(conv, std=0.1)
    batched = FourierCanvasBatch(batch_size=2, height=16, width=16, fft=False)
    probe = torch.rand(1, 3, 8, 8)
    original_weight = conv.weight.detach().clone()

    assert ensemble.names() == ("a", "b")
    assert set(outputs.keys()) == {"a", "b"}
    assert len(selected_outputs) == 1
    assert noisy(probe).shape == (1, 4, 8, 8)
    assert noisy_active(probe).shape == (1, 4, 8, 8)
    assert torch.allclose(conv.weight, original_weight)
    assert batched.forward().shape == (2, 3, 16, 16)


def test_activation_atlas_with_injected_layout(monkeypatch):
    def fake_umap(activations, umap_options=None, verbose=False):
        left = np.tile(np.asarray([[0.1, 0.1]], dtype="float32"), (6, 1))
        right = np.tile(np.asarray([[0.8, 0.8]], dtype="float32"), (6, 1))
        return np.vstack([left, right])

    monkeypatch.setattr(atlas, "aligned_umap", fake_umap)
    canvas = atlas.activation_atlas(
        ToyModel(),
        "conv",
        activations=np.random.randn(12, 4).astype("float32"),
        grid_size=2,
        icon_size=8,
        icon_batch_size=2,
        render_kwargs={"n_steps": 1, "transforms": [], "fft": False},
    )

    assert canvas.shape == (16, 16, 3)
    assert np.isfinite(canvas).all()
