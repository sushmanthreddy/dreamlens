# Xplique Feature Visualizations in DreamLens

DreamLens contains a native PyTorch implementation of the complete
`xplique.features_visualizations` package. It does not import Xplique or any
second machine-learning framework.

## Imports

Every public feature is available from both `dreamlens` and
`dreamlens.features_visualizations`:

| Xplique area | DreamLens PyTorch API |
| --- | --- |
| Objectives | `Objective.layer`, `direction`, `channel`, `neuron`; arithmetic composition |
| Optimization | `optimize` with FFT/pixel buffers, warmup, transforms, regularizers, checkpoints |
| MaCo | `maco`, `maco_optimisation_step` |
| Losses | `cosine_similarity`, `dot_cossim` |
| Preconditioning | `fft_2d_freq`, `get_fft_scale`, `fft_image`, `fft_to_rgb`, `recorrelate_colors`, `to_valid_rgb`, `to_valid_grayscale`, `init_maco_buffer`, `maco_image_parametrization` |
| Regularizers | `l1_reg`, `l2_reg`, `l_inf_reg`, `total_variation_reg` |
| Transformations | `random_blur`, `random_blur_grayscale`, `random_jitter`, `random_scale`, `random_flip`, `pad`, `compose_transformations`, `generate_standard_transformations` |

## The one deliberate framework convention

Xplique uses NHWC and HWC tensors. DreamLens/PyTorch uses NCHW and CHW tensors
throughout:

```text
model input and optimize checkpoints: [batch, channels, height, width]
MaCo image and transparency:           [channels, height, width]
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

The self-contained API notebook uses torchvision ResNet18 so every relevant
cell focuses on the DreamLens public API: model preprocessing,
`Objective.neuron`, `optimize`, checkpoints, output tensors, and additional
objective types. It imports package APIs normally and does not invoke a helper
or runner script.

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

With `maco_dataset=None`, DreamLens downloads Xplique's reference ImageNet
magnitude spectrum to
`$DREAMLENS_CACHE/spectrums/spectrum_decorrelated.npy` (default cache root:
`~/.cache/dreamlens`). To avoid a download or use a different image domain,
pass an iterable/DataLoader yielding NCHW batches. Grayscale MaCo always
requires such a dataset.

## Validation

The port's tests are derived from Xplique's feature-visualization tests and add
PyTorch-specific checks for NCHW layouts, autograd preservation, model-state
restoration, optimizer rebinding, explicit input metadata, and RGB/grayscale
MaCo:

```bash
PYTHONPATH=src pytest -q tests/test_features_visualizations.py
```
