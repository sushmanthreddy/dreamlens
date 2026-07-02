from pathlib import Path
import hashlib

import dreamlens
import numpy as np
import pytest
import torch

from dreamlens import (
    FeatureAccentuationCanvas,
    FeatureAccentuationConfig,
    FeatureTarget,
    FeatureVisualizer,
    MacoConfig,
    Objective,
    compose_transformations,
    cosine_similarity,
    decorrelate_colors,
    feature_accentuation,
    feature_accentuation_transforms,
    fft_image,
    fft_to_rgb,
    get_fft_scale,
    init_maco_buffer,
    load_imagenet_spectrum,
    l1_reg,
    l2_reg,
    l_inf_reg,
    maco,
    maco_image_parametrization,
    optimize,
    pad,
    random_blur,
    random_flip,
    random_jitter,
    random_scale,
    total_variation_reg,
)


def test_feature_visualization_implementations_live_in_root_modules():
    assert Objective.__module__ == "dreamlens.objectives"
    assert optimize.__module__ == "dreamlens.optimization"
    assert maco.__module__ == "dreamlens.optimization"
    assert feature_accentuation.__module__ == "dreamlens.optimization"
    assert init_maco_buffer.__module__ == "dreamlens.image_parameters"
    assert random_scale.__module__ == "dreamlens.transforms"
    package_root = Path(dreamlens.__file__).resolve().parent
    assert not (package_root / "features_visualizations").exists()


def test_feature_accentuation_fourier_seed_and_gate_match_reference_equations():
    torch.manual_seed(13)
    seed = torch.rand(1, 3, 17, 15).mul(0.8).add(0.1)
    exact = FeatureAccentuationCanvas(
        seed,
        height=17,
        width=15,
        parameterization="fourier_phase",
        use_magnitude_gate=False,
        center_crop=False,
    )
    gated = FeatureAccentuationCanvas(
        seed,
        height=17,
        width=15,
        parameterization="fourier_phase",
        use_magnitude_gate=True,
        magnitude_gate_init=5.0,
        center_crop=False,
    )

    assert torch.allclose(exact(), seed, atol=3e-6, rtol=1e-5)
    assert torch.allclose(exact.phase, gated.phase)
    assert torch.allclose(exact.magnitude, gated.magnitude)
    assert torch.allclose(
        gated.effective_magnitude(),
        gated.magnitude * torch.sigmoid(torch.tensor(5.0)),
    )
    recovered = decorrelate_colors(dreamlens.recorrelate_colors(seed))
    assert torch.allclose(recovered, seed, atol=2e-6, rtol=1e-5)


def test_feature_accentuation_default_full_fourier_round_trips_seed():
    torch.manual_seed(17)
    seed = torch.rand(1, 3, 16, 18).mul(0.8).add(0.1)
    canvas = FeatureAccentuationCanvas(
        seed,
        height=16,
        width=18,
        center_crop=False,
    )

    assert canvas.parameterization == "fourier"
    assert canvas.fourier_coefficients.shape == (1, 3, 16, 10, 2)
    assert torch.allclose(canvas(), seed, atol=3e-6, rtol=1e-5)


def test_feature_accentuation_and_maco_share_packaged_natural_spectrum():
    package_root = Path(dreamlens.__file__).resolve().parent
    spectrum_path = package_root / "data" / "clean_decorrelated.npy"
    digest = hashlib.sha256(spectrum_path.read_bytes()).hexdigest()
    resized = load_imagenet_spectrum(32, 30, faccent_scale=True)
    maco_magnitude, _ = init_maco_buffer((3, 32, 30))

    assert digest == "a4810ea049ef9a0fe4e3f26660188e53222281879b333e8fd61377f7491aafc8"
    assert resized.shape == maco_magnitude.shape == (3, 32, 16)
    assert torch.isfinite(resized).all()
    assert torch.isfinite(maco_magnitude).all()


def test_feature_accentuation_transforms_share_crop_and_noise():
    torch.manual_seed(19)
    candidate = torch.rand(1, 3, 20, 18, requires_grad=True)
    transformed = feature_accentuation_transforms(
        candidate,
        candidate.detach(),
        output_size=(12, 10),
        crops=4,
        crop_min=0.3,
        crop_max=0.8,
        noise_std=0.03,
    ).reshape(4, 2, 3, 12, 10)

    assert torch.equal(transformed[:, 0], transformed[:, 1])
    transformed[:, 0].sum().backward()
    assert candidate.grad is not None
    assert torch.isfinite(candidate.grad).all()


def test_feature_accentuation_root_api_runs_balanced_paired_optimization(tmp_path):
    torch.manual_seed(23)
    model = FeatureVizModel()
    model.train()
    image = torch.rand(1, 3, 16, 16).mul(0.8).add(0.1)
    target = FeatureTarget.for_class(2, layer="logits")
    config = FeatureAccentuationConfig(
        width=16,
        height=16,
        input_shape=(3, 16, 16),
        steps=4,
        lr=0.02,
        crops=2,
        crop_min=0.6,
        crop_max=0.9,
        noise_std=0.01,
        checkpoint_steps=(0, 3),
    )
    visualizer = FeatureVisualizer(model, device="cpu", normalize=False, quiet=True)
    original_training = model.training
    original_requires_grad = [parameter.requires_grad for parameter in model.parameters()]

    result = visualizer.visualize(
        target,
        method="feature_accentuation",
        image=image,
        regularization_layer="early",
        config=config,
    )

    assert isinstance(result.image, FeatureAccentuationCanvas)
    assert result.as_chw().shape == result.transparency_chw().shape == (3, 16, 16)
    assert len(result.losses) == config.steps
    assert len(result.metadata["target_losses"]) == config.steps
    assert len(result.metadata["regularization_distances"]) == config.steps
    assert result.metadata["method"] == "feature_accentuation"
    assert result.metadata["parameterization"] == "fourier"
    assert result.metadata["target"] == target
    assert result.metadata["regularization_layer"] == "early"
    assert np.isfinite(result.metadata["gradient_balance"])
    assert result.metadata["gradient_balance"] > 0
    assert torch.isfinite(result.as_chw()).all()
    assert torch.isfinite(result.transparency_chw()).all()
    assert float(result.transparency_chw().sum()) > 0
    assert result.metadata["checkpoint_steps"] == [0, 3]
    assert sorted(result.checkpoints) == sorted(result.transparency_checkpoints) == [0, 3]
    rgba = result.as_accentuation_rgba(checkpoint=3)
    assert rgba.shape == (4, 16, 16)
    assert torch.isfinite(rgba).all()
    assert float(rgba.min()) >= 0.0 and float(rgba.max()) <= 1.0
    output_path = tmp_path / "accentuation.png"
    result.save_accentuation(output_path, checkpoint=3)
    assert output_path.exists()
    assert model.training is original_training
    assert [parameter.requires_grad for parameter in model.parameters()] == original_requires_grad


def test_feature_accentuation_requires_explicit_regularization_layer():
    visualizer = FeatureVisualizer(
        FeatureVizModel(),
        device="cpu",
        normalize=False,
        quiet=True,
    )
    config = FeatureAccentuationConfig(
        width=16,
        height=16,
        input_shape=(3, 16, 16),
        steps=1,
        crops=1,
    )
    with pytest.raises(ValueError, match="regularization_layer"):
        visualizer.accentuate(
            FeatureTarget.for_class(0, layer="logits"),
            torch.rand(1, 3, 16, 16),
            config=config,
        )


def test_feature_accentuation_phase_mode_uses_packaged_magnitude_end_to_end():
    torch.manual_seed(29)
    visualizer = FeatureVisualizer(
        FeatureVizModel(),
        device="cpu",
        normalize=False,
        quiet=True,
    )
    result = visualizer.accentuate(
        FeatureTarget.for_class(1, layer="logits"),
        torch.rand(1, 3, 16, 16).mul(0.8).add(0.1),
        config=FeatureAccentuationConfig(
            width=16,
            height=16,
            input_shape=(3, 16, 16),
            steps=1,
            crops=1,
            crop_min=1.0,
            crop_max=1.0,
            noise_std=0.0,
            regularization_strength=0.0,
            parameterization="fourier_phase",
            magnitude_source="imagenet",
        ),
    )

    assert result.image.phase is not None
    assert result.image.magnitude_gate is not None
    assert result.metadata["parameterization"] == "fourier_phase"
    assert result.metadata["magnitude_source"] == "imagenet"
    assert torch.isfinite(result.as_chw()).all()


def test_class_maco_is_exactly_the_root_maco_kernel():
    model = FeatureVizModel()
    target = FeatureTarget.for_class(1, layer="logits")
    dataset = [torch.rand(2, 3, 16, 16)]
    config = MacoConfig(
        width=16,
        height=16,
        input_shape=(3, 16, 16),
        steps=2,
        lr=1.0,
        crops=2,
        noise_intensity=0.01,
    )

    torch.manual_seed(91)
    expected_image, expected_importance = maco(
        Objective.from_target(model, target, input_shape=config.input_shape),
        maco_dataset=dataset,
        nb_steps=config.steps,
        nb_crops=config.crops,
        noise_intensity=config.noise_intensity,
        custom_shape=(config.height, config.width),
        input_shape=config.input_shape,
        values_range=config.values_range,
        preprocess=lambda images: images,
        device="cpu",
    )

    torch.manual_seed(91)
    actual = FeatureVisualizer(
        model,
        device="cpu",
        normalize=False,
        quiet=True,
    ).maco(target, config=config, maco_dataset=dataset)

    assert torch.equal(actual.as_chw(), expected_image)
    assert torch.equal(actual.transparency_chw(), expected_importance)


class FeatureVizModel(torch.nn.Module):
    input_shape = (3, 16, 16)

    def __init__(self):
        super().__init__()
        self.early = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.features = torch.nn.Conv2d(4, 3, 3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.logits = torch.nn.Linear(3, 5)

    def forward(self, inputs):
        outputs = self.relu(self.early(inputs))
        outputs = self.relu(self.features(outputs))
        return self.logits(self.pool(outputs).flatten(1))


class GrayscaleFeatureVizModel(torch.nn.Module):
    input_shape = (1, 8, 8)

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, 3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.logits = torch.nn.Linear(2, 3)

    def forward(self, inputs):
        return self.logits(self.pool(torch.relu(self.conv(inputs))).flatten(1))


def test_cosine_similarity_matches_reference_values():
    vector = torch.tensor([[10.0, 20.0, 30.0]])
    collinear = torch.tensor([[1.0, 2.0, 3.0]])
    orthogonal = torch.zeros_like(vector)
    opposite = torch.tensor([[-0.01, -0.02, -0.03]])

    assert torch.allclose(cosine_similarity(vector, collinear), torch.tensor([1.0]))
    assert torch.allclose(cosine_similarity(vector, orthogonal), torch.tensor([0.0]))
    assert torch.allclose(cosine_similarity(vector, opposite), torch.tensor([-1.0]))


def test_regularizers_match_reference_values():
    vector = torch.tensor([[[[-4.0, 4.0]]]])
    assert torch.allclose(l1_reg(1.0)(vector), torch.tensor([4.0]))
    assert torch.allclose(l2_reg(2.0)(vector), torch.tensor([8.0]))
    assert torch.allclose(l_inf_reg(10.0)(vector), torch.tensor([40.0]))

    image = torch.tensor([[[[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]]])
    assert torch.allclose(total_variation_reg(1.0)(image), torch.tensor([10.0]))


def test_blur_and_pad_match_reference_kernel_and_layout():
    images = torch.zeros(3, 3, 3, 3)
    images[:, 1, 1, 1] = 1.0
    images[:, 2, 2, 2] = 1.0
    blurred = random_blur(kernel_size=3, sigma_range=(1.0, 1.0))(images)

    kernel_sum = np.exp(0) + np.exp(-0.5) * 4 + np.exp(-1) * 4
    c0, c1, c2 = np.exp(0) / kernel_sum, np.exp(-0.5) / kernel_sum, np.exp(-1) / kernel_sum
    expected_green = torch.tensor(
        [[c2, c1, c2], [c1, c0, c1], [c2, c1, c2]], dtype=torch.float32
    )
    assert torch.allclose(blurred[0], blurred[1])
    assert torch.allclose(blurred[0, 1], expected_green, atol=1e-6)

    padded = pad(2, 0.0)(torch.ones(3, 3, 2, 2))
    assert padded.shape == (3, 3, 6, 6)
    assert torch.all(padded[:, :, 2:4, 2:4] == 1)
    assert torch.all(padded[:, :, :2] == 0)


def test_even_blur_kernel_uses_reference_same_padding_alignment():
    images = torch.zeros(1, 3, 3, 3)
    images[:, :, 1, 1] = 1.0
    blurred = random_blur(kernel_size=2, sigma_range=(1.0, 1.0))(images)
    expected = torch.tensor(
        [[0.25, 0.25, 0.0], [0.25, 0.25, 0.0], [0.0, 0.0, 0.0]]
    )
    assert blurred.shape == images.shape
    assert torch.allclose(blurred[0, 0], expected)
    assert torch.allclose(blurred[0, 1], expected)
    assert torch.allclose(blurred[0, 2], expected)


def test_transform_composition_preserves_gradients():
    inputs = torch.rand(2, 3, 16, 16, requires_grad=True)
    transform = compose_transformations(
        [
            pad(2),
            random_jitter(2),
            random_scale((0.95, 1.05)),
            random_flip(),
        ]
    )
    result = transform(inputs)
    result.sum().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


@pytest.mark.parametrize(
    "shape",
    [(2, 3, 8, 8), (2, 1, 64, 64), (2, 3, 15, 17), (1, 1, 32, 24)],
)
def test_fourier_preconditioning_returns_requested_nchw_shape(shape):
    _, _, height, width = shape
    buffer = fft_image(shape)
    scale = get_fft_scale(height, width)
    image = fft_to_rgb(shape, buffer, scale)
    assert image.shape == shape
    assert torch.isfinite(image).all()


def test_objective_factories_and_cartesian_combinations():
    model = FeatureVizModel()
    layer_objective = Objective.layer(model, "logits")
    direction_objective = Objective.direction(model, -1, torch.rand(5))
    channel_objective = Objective.channel(model, "early", [0, 1])
    neuron_objective = Objective.neuron(model, "features", [0, 1, 2])
    combined = layer_objective + direction_objective + channel_objective + neuron_objective

    configured, objective_function, names, input_shape = combined.compile()
    try:
        outputs = configured(torch.rand(6, 3, 16, 16))
        scores = objective_function(outputs)
    finally:
        configured.close()

    assert input_shape == (6, 3, 16, 16)
    assert len(names) == 6
    assert names[0] == (
        "Layer#logits & Direction#logits_0 & Channel#early_0 & Neuron#features_0"
    )
    assert names[-1].endswith("Channel#early_1 & Neuron#features_2")
    assert scores.shape == (6,)
    assert torch.isfinite(scores).all()


def test_objective_requires_explicit_shape_when_model_has_no_metadata():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 2, 3, padding=1),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
    )
    objective = Objective.channel(model, 0, 0)
    with pytest.raises(ValueError, match="input_shape"):
        objective.compile()

    configured, _, _, shape = objective.compile(input_shape=(3, 8, 8))
    configured.close()
    assert shape == (1, 3, 8, 8)


@pytest.mark.parametrize("use_fft", [True, False])
def test_optimize_supports_fft_pixel_regularizers_saves_and_warmup(use_fft):
    model = FeatureVizModel()
    objective = Objective.channel(model, "early", [0, 1])
    dummy = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD([dummy], lr=0.05)
    preprocess_calls = []

    def preprocess(images):
        preprocess_calls.append(tuple(images.shape))
        return images

    images, names = optimize(
        objective,
        optimizer=optimizer,
        nb_steps=4,
        use_fft=use_fft,
        regularizers=[l1_reg(0.01), l2_reg(0.01), total_variation_reg(0.001)],
        warmup_steps=1,
        custom_shape=(20, 18),
        transformations=[],
        save_every=2,
        preprocess=preprocess,
    )

    assert len(images) == 2
    assert images[-1].shape == (2, 3, 20, 18)
    assert names == ["Channel#early_0", "Channel#early_1"]
    assert torch.isfinite(images[-1]).all()
    assert model.relu.inplace is True
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert preprocess_calls == [(2, 3, 16, 16)] * 5


def test_init_maco_buffer_and_image_parameterization_rgb_and_grayscale():
    rgb_dataset = [torch.randn(2, 3, 12, 10), torch.randn(1, 3, 12, 10)]
    gray_dataset = [torch.randn(2, 1, 12, 10), torch.randn(1, 1, 12, 10)]

    rgb_magnitude, rgb_phase = init_maco_buffer((3, 16, 14), dataset=rgb_dataset)
    gray_magnitude, gray_phase = init_maco_buffer((16, 14, 1), dataset=gray_dataset)
    rgb = maco_image_parametrization(rgb_magnitude, rgb_phase, (-1, 1))
    gray = maco_image_parametrization(gray_magnitude, gray_phase, (0, 1))

    assert rgb_magnitude.shape == rgb_phase.shape == (3, 16, 8)
    assert gray_magnitude.shape == gray_phase.shape == (1, 16, 8)
    assert rgb.shape == (3, 16, 14)
    assert gray.shape == (1, 16, 14)
    assert -1 <= float(rgb.min()) <= float(rgb.max()) <= 1
    assert 0 <= float(gray.min()) <= float(gray.max()) <= 1


def test_init_maco_buffer_uses_packaged_reference_spectrum():
    magnitude, phase = init_maco_buffer((3, 12, 10))

    assert magnitude.shape == phase.shape == (3, 12, 6)
    assert torch.isfinite(magnitude).all()
    with pytest.raises(ValueError, match="dataset"):
        init_maco_buffer((1, 12, 10), data_format="CHW")


@pytest.mark.parametrize("kind", ["neuron", "direction", "channel"])
def test_maco_optimizes_all_reference_objective_kinds(kind):
    model = FeatureVizModel()
    if kind == "neuron":
        objective = Objective.neuron(model, "logits", 0)
    elif kind == "direction":
        objective = Objective.direction(model, "logits", torch.nn.functional.one_hot(torch.tensor(1), 5).float())
    else:
        objective = Objective.channel(model, "early", 0)
    dataset = [torch.randn(2, 3, 16, 16), torch.randn(1, 3, 16, 16)]

    image, transparency = maco(
        objective,
        maco_dataset=dataset,
        nb_steps=2,
        nb_crops=2,
        custom_shape=(16, 16),
        noise_intensity=0.01,
        values_range=(-127, 127),
    )

    assert image.shape == (3, 16, 16)
    assert transparency.shape == image.shape
    assert torch.isfinite(image).all()
    assert torch.isfinite(transparency).all()
    assert float(image.min()) >= -127
    assert float(image.max()) <= 127


def test_maco_grayscale_and_single_objective_validation():
    model = GrayscaleFeatureVizModel()
    dataset = [torch.randn(2, 1, 8, 8)]
    image, transparency = maco(
        Objective.neuron(model, "logits", 0),
        maco_dataset=dataset,
        nb_steps=2,
        nb_crops=0,
        custom_shape=(8, 8),
    )
    assert image.shape == transparency.shape == (1, 8, 8)

    rgb_model = FeatureVizModel()
    combined = Objective.channel(rgb_model, "early", [0, 1])
    with pytest.raises(AssertionError, match="one objective"):
        maco(
            combined,
            maco_dataset=[torch.randn(2, 3, 16, 16)],
            nb_steps=1,
            custom_shape=(16, 16),
        )
