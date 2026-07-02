# DreamLens Feature Visualization API

DreamLens contains a native PyTorch implementation of the complete feature-
visualization surface. It does not import Xplique or any second machine-
learning framework, and it no longer maintains a parallel
`dreamlens.features_visualizations` package.

This functional surface is retained for Xplique migration and advanced
composition. New DreamLens code should normally use the single root
`FeatureVisualizer` API with `FeatureTarget`, `RenderConfig`, `MacoConfig`, and
`AmplifyConfig`; `FeatureAccentuationConfig` adds Faccent without replacing
the existing caricature path. Together they cover classical maximize, MaCo,
feature accentuation, and caricature without
switching objective systems.

## Class-first API

```python
from dreamlens import (
    AmplifyConfig,
    FeatureAccentuationConfig,
    FeatureTarget,
    FeatureVisualizer,
    MacoConfig,
    RenderConfig,
)

visualizer = FeatureVisualizer(model, preprocess=imagenet_preprocess)
target = FeatureTarget.for_class(96, layer="fc")

classic = visualizer.visualize(
    target,
    method="maximize",
    config=RenderConfig(width=224, height=224, steps=160),
)
maco_result = visualizer.visualize(
    target,
    method="maco",
    config=MacoConfig(
        width=512,
        height=512,
        input_shape=(3, 224, 224),
        steps=128,
        crops=8,
    ),
)
accentuated = visualizer.visualize(
    FeatureTarget.for_class(258, layer="fc"),
    method="feature_accentuation",
    image=input_image,
    regularization_layer="layer2.1.conv2",
    config=FeatureAccentuationConfig(steps=99, crops=16),
)
caricature = visualizer.visualize(
    method="caricature",
    image=input_image,
    layers=["layer3.1.conv2"],
    config=AmplifyConfig.reference(steps=200),
)
```

Target factories are `for_layer`, `for_channel`, `for_neuron`, `for_class`,
and `for_direction`. Every algorithm parameter remains explicit in its frozen
config dataclass.

## Root imports and advanced functions

Every public feature is available directly from `dreamlens`:

| Xplique area | DreamLens PyTorch API |
| --- | --- |
| Objectives | `Objective.layer`, `direction`, `channel`, `neuron`; arithmetic composition |
| Optimization | `optimize` with FFT/pixel buffers, warmup, transforms, regularizers, checkpoints |
| MaCo | `maco`, `maco_optimisation_step` |
| Faccent feature accentuation | `feature_accentuation`, `FeatureAccentuationCanvas`, `feature_accentuation_transforms` |
| Losses | `cosine_similarity`, `dot_cossim` |
| Preconditioning | `fft_2d_freq`, `get_fft_scale`, `fft_image`, `fft_to_rgb`, `decorrelate_colors`, `recorrelate_colors`, `to_valid_rgb`, `to_valid_grayscale`, `init_maco_buffer`, `load_imagenet_spectrum`, `maco_image_parametrization` |
| Regularizers | `l1_reg`, `l2_reg`, `l_inf_reg`, `total_variation_reg` |
| Transformations | `random_blur`, `random_blur_grayscale`, `random_jitter`, `random_scale`, `random_flip`, `pad`, `compose_transformations`, `generate_standard_transformations` |

## The one deliberate framework convention

Xplique uses NHWC and HWC tensors. DreamLens/PyTorch uses NCHW and CHW tensors
throughout:

```text
model input and optimize checkpoints: [batch, channels, height, width]
MaCo image and transparency:           [channels, height, width]
Faccent image and transparency:        [channels, height, width]
Objective input_shape:                 (channels, height, width)
custom_shape:                          (height, width)
```

Keras models always expose their input shape. Arbitrary `torch.nn.Module`
objects do not. `Objective.compile`, `optimize`, and `maco` therefore use the
following exact order:

1. Explicit `input_shape=(C,H,W)` argument.
2. Shape stored on the objective.
3. `model.input_shape` or `model._dreamlens_input_shape`.
4. For `optimize`/`maco` only, channels read from the first actual `Conv2d`
   input module and spatial size taken from `custom_shape`.

No default model resolution such as 224 is guessed. Models with fixed linear
input sizes should declare `input_shape`.

## Composable objectives

Objective alternatives form a Cartesian product exactly as in Xplique:

```python
from dreamlens import Objective

channels = Objective.channel(
    model, "features.4", [2, 7], input_shape=(3, 224, 224)
)
neurons = Objective.neuron(
    model, "classifier", [0, 1, 2], input_shape=(3, 224, 224)
)
objective = channels + 0.2 * neurons

# compile produces 2 * 3 = 6 objective/image combinations
hooked_model, score, names, shape = objective.compile()
try:
    outputs = hooked_model(torch.rand(*shape))
    values = score(outputs)
finally:
    hooked_model.close()
```

Layer names, leaf-module integer indices (including negative indices), and
module objects are supported.

## Standard optimization

```python
from dreamlens import l1_reg, optimize, total_variation_reg

images, names = optimize(
    objective,
    nb_steps=256,
    use_fft=True,
    fft_decay=0.85,
    std=0.01,
    regularizers=[l1_reg(1e-4), total_variation_reg(1e-5)],
    image_normalizer="sigmoid",
    values_range=(0, 1),
    transformations="standard",
    warmup_steps=0,
    custom_shape=(512, 512),
    save_every=64,
    preprocess=imagenet_normalize,
)
```

For torchvision models, pass differentiable normalization through
`preprocess`; otherwise the optimizer can produce a high target score against
the wrong input distribution and the resulting image can look clipped or
blurred. The preprocessing runs after stochastic transforms and resize, just
before model inference.

The self-contained API notebook uses torchvision ResNet18 and the class-first
root API for preprocessing, explicit targets, classical maximize, MaCo,
caricature, and output tensors. It imports package APIs normally and does not
invoke a helper or runner script.

The source Xplique notebook optimizes a 512×512 canvas but displays each image
in a roughly 224–256 pixel subplot. At native 512-pixel zoom the Fourier canvas
contains much more high-frequency structure than the displayed reference.
Use antialiased downsampling for presentation while retaining the raw canvas
for reproducibility. The ready-to-run implementation is
`examples/Feature_Visualization_Getting_started_PyTorch.ipynb`.

The returned list contains the requested intermediate checkpoints and always
contains the final image batch. A configured optimizer instance or an
optimizer class/factory may be passed. DreamLens rebinds optimizer instances to
the trainable image/phase parameter because PyTorch optimizers bind parameters
at construction time, unlike Keras optimizers.

## Feature accentuation (Faccent)

DreamLens implements Hamblin et al., [“Feature Accentuation: Revealing 'What'
Features Respond to in Natural Images”](https://arxiv.org/abs/2402.10039), in
native PyTorch. It is image-seeded feature visualization, not an alias for
DreamLens caricature or MaCo.

```python
from dreamlens import FeatureAccentuationConfig, FeatureTarget

result = visualizer.accentuate(
    target=FeatureTarget.for_class(258, layer="fc"),
    image="dog.jpg",
    regularization_layer="layer2.1.conv2",
    config=FeatureAccentuationConfig(
        width=512,
        height=512,
        input_shape=(3, 224, 224),
        steps=100,
        lr=0.05,
        crops=16,
        crop_min=0.05,
        crop_max=0.99,
        noise_std=0.02,
        regularization_strength=1.0,
        parameterization="fourier",
        checkpoint_steps=(0, 20, 40, 60, 80, 98),
    ),
)
```

The implementation follows the reference computation:

1. Center-crop/resize the seed and map RGB through safe logit plus ImageNet
   color decorrelation.
2. Initialize Faccent's frequency-preconditioned full complex Fourier buffer.
3. At every step, concatenate candidate and fixed reference and apply the same
   uniformly located crop and shared Gaussian-plus-uniform noise 16 times.
4. Compute the negative target objective and the L2 distance between paired
   features at `regularization_layer`.
5. Once, measure summed absolute parameter gradients and set the balance to
   `target_gradient / regularizer_gradient`.
6. Minimize target loss plus `regularization_strength * balance * distance`.
7. Accumulate absolute image gradients from the target objective alone as the
   returned importance/transparency map.

Two Faccent parameterizations are available:

| `parameterization` | Trainable variables | Magnitude behavior |
| --- | --- | --- |
| `"fourier"` (default/reference) | all frequency-preconditioned real/imaginary coefficients | changes freely |
| `"fourier_phase"` | phase plus optional sigmoid magnitude gate | seed magnitude or packaged ImageNet magnitude |

For `fourier_phase`, use `magnitude_source="image"` for the seeded reference
behavior or `"imagenet"` to reuse `dreamlens/data/clean_decorrelated.npy`.
The bundled asset has shape `(3, 512, 257)`, dtype `float32`, and SHA-256
`a4810ea049ef9a0fe4e3f26660188e53222281879b333e8fd61377f7491aafc8`.

`regularization_layer` is deliberately explicit. DreamLens does not guess a
layer whose feature geometry would materially change the result. Set
`regularization_strength=0` only when an unregularized image-seeded run is
intended.

Faccent's displayed figure is not the raw Fourier canvas. It globally
normalizes RGB contrast and uses the accumulated target-gradient magnitude as
a percentile-clipped, Gaussian-blurred alpha mask. DreamLens exposes both:

```python
result.save("raw_canvas.png")
result.save_accentuation("faccent_view.png", checkpoint=98)
rgba = result.as_accentuation_rgba(checkpoint=98)
```

`checkpoint_steps` captures Faccent-style pre-update images and their
accumulated transparency maps in `result.checkpoints` and
`result.transparency_checkpoints`.

The executed notebook is
`examples/learn_dreamlens_feature_accentuation.ipynb`. It imports DreamLens and
torchvision directly, uses torchvision ResNet18 with locally checked-in iguana
and fox inputs, and has no Faccent-package or converted-model dependency. Its
loggerhead and castle runs produce gradient balances of `9.7570873578` and
`6.8875874332`, respectively.

## MaCo

DreamLens implements MaCo from Fel et al., [“Unlocking Feature Visualization
for Deeper Networks with MAgnitude Constrained
Optimization”](https://arxiv.org/abs/2306.06805), natively in PyTorch. The
Fourier magnitude remains fixed, the phase is optimized, and the accumulated
absolute image gradient is returned as a spatial-importance map.

```python
from dreamlens import Objective, maco

objective = Objective.neuron(
    model, "classifier", 5, input_shape=(3, 224, 224)
)
image, transparency = maco(
    objective,
    nb_steps=256,
    nb_crops=32,
    noise_intensity=0.08,
    custom_shape=(512, 512),
    values_range=(-1, 1),
)
```

With `maco_dataset=None`, DreamLens loads the packaged natural-image magnitude
also used by Faccent's optional ImageNet phase mode. To use a different image
domain, pass an iterable/DataLoader yielding NCHW batches. Grayscale MaCo
always requires such a dataset.

## Validation

The port's tests are derived from Xplique's feature-visualization tests and add
PyTorch-specific checks for NCHW layouts, autograd preservation, model-state
restoration, optimizer rebinding, explicit input metadata, RGB/grayscale MaCo,
Faccent Fourier parity, shared pair transforms, natural-spectrum integrity,
and gradient-balanced feature accentuation:

```bash
PYTHONPATH=src pytest -q tests/test_feature_visualization.py
```

The implementation was also checked directly against the local Faccent
reference on identical tensors and RNG seeds: packaged natural magnitude and
one complete paired crop/noise/resize transform matched exactly; reconstructed
full-Fourier and Fourier-phase images differed by at most `1.79e-7` in
`float32`. These direct parity checks are development evidence; the committed
tests independently cover the equations without making Faccent a runtime or
test dependency.
