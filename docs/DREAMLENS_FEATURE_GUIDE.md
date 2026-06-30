# DreamLens Feature And Activation Atlas Guide

This document is a compact technical guide for humans and LLM agents working on
DreamLens. It explains what the project supports, how the API is structured, and
which files own each feature.

DreamLens is a native PyTorch interpretability toolkit. It supports activation
collection, activation atlas generation, feature visualization, channel/unit
rendering, image amplification, caricature generation, masked optimization, and
multi-model objectives.

## One-Screen Context For LLMs

Use this section as quick context when another LLM needs to work in this repo.

```text
Project: DreamLens
Package path: activation-atlas-pytorch/src/dreamlens
Main public import: from dreamlens import ...
Main class: FeatureVisualizer
Main atlas APIs: collect_activations, activation_atlas, aligned_activation_atlas
Main visualization APIs: maximize, maximize_channels, synthesize, caricature, amplify
Main configs: RenderConfig, TransformConfig, AmplifyConfig
Main result object: OptimizationResult
Model requirement: torch.nn.Module whose selected layer returns a tensor
Supported layer output shapes: [N, C], [N, L, C], [N, C, H, W], [N, H, W, C]
Best model family: CNN image models with NCHW feature maps
Native only: do not import external feature-visualization packages
Xplique feature visualization: fully ported in dreamlens.features_visualizations
MaCo: supported for RGB and dataset-backed grayscale models
Smoke test: PYTHONPATH=src pytest -q tests/test_smoke.py
Notebook: examples/native_dreamlens_results.ipynb
Notebook workflows: batched channel rendering, reference caricature
Notebook outputs: results/native_dreamlens_notebook
```

## Core File Map

| File | Owns |
| --- | --- |
| `src/dreamlens/feature_visualizer.py` | High-level optimization engine: feature maximization, channel rendering, amplification, caricature, layer capture |
| `src/dreamlens/objectives.py` | Target/objective definitions: `FeatureTarget`, `ChannelObjective`, `PerSampleObjective`, amplification objectives |
| `src/dreamlens/optimization.py` | Config and result dataclasses: `RenderConfig`, `TransformConfig`, `AmplifyConfig`, `OptimizationResult` |
| `src/dreamlens/image_parameters.py` | Trainable image/canvas classes: FFT, pixel, masked, image-initialized, batched canvases |
| `src/dreamlens/activations.py` | Activation collection from real inputs |
| `src/dreamlens/atlas.py` | Activation atlas and aligned activation atlas pipelines |
| `src/dreamlens/render.py` | Lower-level icon/channel/neuron rendering utilities |
| `src/dreamlens/layers.py` | Layer resolution, layer listing, supported-layer probing |
| `src/dreamlens/model_wrappers.py` | `ModelEnsemble` and `ParameterNoise` |
| `src/dreamlens/preprocessing.py` | Image conversion and ImageNet normalization helpers |
| `src/dreamlens/transforms.py` | Transform utilities for optimization and paired amplification |
| `src/dreamlens/features_visualizations/` | Native PyTorch Xplique port: objectives, optimize, MaCo, losses, FFT preconditioning, transforms, regularizers |

## Supported Model Types

DreamLens supports any PyTorch model that satisfies these requirements:

1. The model is a `torch.nn.Module`.
2. The model can be called as `model(image_tensor)` or wrapped so it can be.
3. The selected layer can be resolved by module object or name.
4. The selected layer is executed during `model(image_tensor)`.
5. The selected layer returns a tensor, or a tuple/list whose first item is a tensor.
6. The selected layer output has a supported shape.

Supported output shapes:

| Shape | Meaning | Supported Features |
| --- | --- | --- |
| `[N, C]` | Linear/classifier/vector layer | unit rendering, feature targets, capture |
| `[N, L, C]` | Sequence/token layer | token-position targets, channel targets, capture |
| `[N, C, H, W]` | Standard CNN feature map | channels, spatial positions, activation atlas, caricature |
| `[N, H, W, C]` | Channels-last feature map | supported when `activation_format="NHWC"` is used in lower-level APIs |

Models already exercised in this repo:

- ResNet18
- InceptionV3
- GoogLeNet
- Small toy CNN/classifier models in tests

Model families expected to work when their forward pass matches the requirements:

- ResNet, VGG, DenseNet, SqueezeNet, MobileNet, EfficientNet, ConvNeXt
- Inception/GoogLeNet style models
- Custom CNNs
- Vision transformers or sequence models if target outputs are `[N, L, C]`
- Multi-model wrappers built with `ModelEnsemble`

Potentially unsupported without a wrapper:

- Models whose `forward` needs non-image extra arguments.
- Layers returning dictionaries or custom objects instead of tensors.
- Models with non-differentiable operations between input and target layer.
- Models that only accept fixed-size inputs when the optimization config uses a different size.
- Models requiring custom normalization unless `preprocess` is provided.

## Layer Inspection

Use these APIs before visualization to discover valid layer names and shapes.

```python
import torch
from dreamlens import list_layers, supported_layers

sample = torch.rand(1, 3, 224, 224)

for info in supported_layers(model, sample_input=sample):
    print(info.name, info.module_type, info.output_shape, info.channels)
```

Important APIs:

- `list_layers(model, sample_input=...)`: list layers and optional output shapes.
- `supported_layers(model, sample_input=...)`: only layers with supported tensor shapes.
- `probe_layer_outputs(model, sample_input=...)`: raw mapping of layer name to output shape.

Layer names can be passed as strings:

```python
FeatureTarget(layer="layer3.1.conv2", channel=17)
```

or as module objects:

```python
FeatureTarget(layer=model.layer3[1].conv2, channel=17)
```

## Activation Collection

Activation collection converts real input images into feature vectors from one
layer. This is the first step for activation atlases.

Main API:

```python
from dreamlens import collect_activations

acts = collect_activations(
    model,
    layer="layer3.1.conv2",
    inputs=image_batches,
    preprocess=None,
    device="cpu",
    spatial="center",
)
```

Supported `spatial` modes:

| Mode | Behavior |
| --- | --- |
| `"center"` | One activation vector per input image, sampled from the center spatial position |
| `"random"` | One random spatial activation vector per input image |
| `"all"` | One activation vector for every spatial location |

For CNN output `[N, C, H, W]`, collected output is shaped:

```text
center/random: [N, C]
all: [N * H * W, C]
```

For sequence output `[N, L, C]`, collected output is shaped:

```text
center/random: [N, C]
all: [N * L, C]
```

## Activation Atlas

Activation atlas shows a map of what a layer represents across many activation
samples. It groups similar activations and renders an icon for each occupied
grid cell.

Main API:

```python
from dreamlens import activation_atlas

canvas = activation_atlas(
    model,
    layer="layer3.1.conv2",
    activations=acts,
    grid_size=10,
    icon_size=96,
    threshold=5,
    render_kwargs={"n_steps": 128, "num_attempts": 1},
)
```

Pipeline:

```text
real images
-> collect_activations(...)
-> activation vectors [samples, channels]
-> UMAP layout in 2D
-> grid binning
-> average activation direction per occupied grid cell
-> render one icon per average direction
-> compose icons into atlas canvas
```

Important parameters:

| Parameter | Meaning |
| --- | --- |
| `grid_size` | Atlas grid width/height in cells |
| `icon_size` | Rendered icon size in pixels |
| `number_activations` | Max activation samples used |
| `threshold` | Minimum samples in a grid cell before rendering an icon |
| `icon_batch_size` | Number of icons rendered per optimization batch |
| `umap_options` | Optional layout configuration |
| `render_kwargs` | Forwarded to `render_icons(...)` |

Returned value:

```text
NumPy image array in NHWC format
```

### Aligned Activation Atlas

Aligned atlas compares two model/layer activation spaces.

Main API:

```python
from dreamlens import aligned_activation_atlas

iterators = aligned_activation_atlas(
    model1,
    layer1="layer3.1.conv2",
    model2=model2,
    layer2="features.8",
    activations1=acts1,
    activations2=acts2,
)
```

It yields progressive canvas iterators for both models. It can optionally whiten
layer activations via inverse covariance.

Use aligned atlas when you want to compare:

- two architectures
- two checkpoints
- before/after fine-tuning
- source model vs target model

## Feature Visualization From Noise

Feature visualization creates a synthetic image that activates a selected model
feature. The model weights stay frozen; the image/canvas is optimized.

Main API:

```python
from dreamlens import FeatureTarget, FeatureVisualizer, RenderConfig

visualizer = FeatureVisualizer(model, device="cpu", normalize=True, quiet=True)

result = visualizer.maximize(
    FeatureTarget(layer=model.layer3[1].conv2, channel=17),
    config=RenderConfig.reference(width=160, height=160, steps=100),
)

result.save("channel_17.png")
```

Optimization idea:

```text
random trainable image
-> model forward pass
-> capture target layer activation
-> objective says "make selected feature bigger"
-> backpropagate to image parameter
-> update image
-> repeat
```

The model is not trained or changed.

## What Can Be Visualized

DreamLens visualizes more than whole layers.

| Target Type | API Pattern |
| --- | --- |
| Whole layer | `FeatureTarget(layer=...)` |
| CNN channel | `FeatureTarget(layer=..., channel=17)` |
| CNN channel at spatial position | `FeatureTarget(layer=..., channel=17, position=(y, x))` |
| Linear/classifier unit | `FeatureTarget(layer=model.fc, channel=281)` |
| Sequence token/channel | `FeatureTarget(layer=..., channel=32, position=5)` |
| Many channels | `visualizer.maximize_channels(...)` |
| Multiple targets in one image | `visualizer.maximize([FeatureTarget(...), FeatureTarget(...)])` |
| Suppressed/negative feature | `FeatureTarget(..., sign=-1.0)` |
| Custom objective | `visualizer.synthesize(..., custom_func=...)` |
| Real image caricature | `visualizer.caricature(...)` |
| Activation atlas cell directions | `activation_atlas(...)` / `render_icons(...)` |

Supported reductions for target values:

```text
mean, sum, max, norm
```

## Batched Channel Rendering

Use this to produce a channel gallery efficiently.

```python
result = visualizer.maximize_channels(
    layer=model.layer2[1].conv2,
    channels=[3, 17, 41, 64, 89, 121],
    reduction="norm",
    config=RenderConfig.reference(width=160, height=160, steps=42),
)

result.image[0].save("channel_3.png")
result.image[1].save("channel_17.png")
```

Internally this uses:

- `ReferenceCanvasBatch` or `FourierCanvas`
- `PerSampleObjective`
- one objective per batch item

Current limitation:

```text
maximize_channels(...) supports config.attempts == 1
```

## Custom Objectives

Use `synthesize(...)` when built-in targets are not enough.

```python
def objective(layer_outputs):
    acts = layer_outputs[0]
    return -(acts[:, 10].mean() - 0.5 * acts[:, 20].mean())

param = visualizer.synthesize(
    layers=[model.layer3[1].conv2],
    custom_func=objective,
    width=160,
    height=160,
    iters=100,
)

param.save("custom_objective.png")
```

Custom objectives receive a list of captured layer outputs. Return a scalar loss
to minimize. To maximize something, return its negative.

## Image Amplification And Caricature

Caricature starts from a real image reference and exaggerates the features the
model sees at selected layers.

Main API:

```python
from dreamlens import AmplifyConfig

result = visualizer.caricature(
    image=input_image,
    layers=[model.layer3[1].conv2],
    power=1.15,
    config=AmplifyConfig.reference(steps=45, lr=9e-3),
)

result.save("caricature.png")
```

Concept:

```text
input image
-> capture target layer activations
-> optimize a generated image
-> generated image matches and amplifies the input activations
```

`caricature(...)` is a convenience wrapper around `amplify(...)`.

`amplify(...)` supports:

- start from input image or noise
- static target activations
- paired target activations with shared transforms
- multiple target layers
- preservation loss via `preserve_weight`
- total variation smoothing via `variation_weight`
- masks through `mask=...`
- `lucid` and `reference` parameterization modes

Important `AmplifyConfig` fields:

| Field | Meaning |
| --- | --- |
| `steps` | Optimization iterations |
| `lr` | Learning rate |
| `weight_decay` | Optimizer weight decay |
| `grad_clip` | Gradient norm clipping |
| `start` | `"input"` or `"noise"` |
| `target_mode` | `"paired"` or `"static"` |
| `preserve_weight` | Penalize distance from original image |
| `variation_weight` | Penalize noisy local variation |
| `parameterization` | `"lucid"` or `"reference"` |

Presets:

```python
AmplifyConfig.dream(steps=220, lr=2e-2)
AmplifyConfig.reference(steps=120, lr=9e-3)
```

## Image Parameterizations

DreamLens optimizes image parameter objects, not ordinary image files.

| Class | Purpose |
| --- | --- |
| `FourierCanvas` | Native FFT/noise parameterization |
| `ReferenceCanvas` | Reference-style FFT/noise parameterization |
| `PixelCanvas` | Trainable pixels initialized from an input image |
| `MaskedCanvas` | Pixel canvas that preserves pixels outside a mask |
| `ReferenceImageCanvas` | Reference-style FFT canvas initialized from a real image |
| `ReferenceMaskedCanvas` | Reference-style masked FFT canvas |
| `FourierCanvasBatch` | Batch of Fourier canvases |
| `ReferenceCanvasBatch` | Batch of reference canvases |

All high-level results can be saved or converted:

```python
result.save("out.png")
result.as_nchw()
result.as_chw()
result.as_hwc()
result.losses
result.objective_value
```

## Preprocessing And Transforms

By default, `FeatureVisualizer(..., normalize=True)` applies ImageNet
normalization. Use `normalize=False` or pass `preprocess=...` for custom models.

```python
visualizer = FeatureVisualizer(
    model,
    device="cpu",
    normalize=False,
    preprocess=my_preprocess,
)
```

Optimization transforms are controlled by `TransformConfig`:

```python
TransformConfig(
    rotate_degrees=10,
    scale_min=0.7,
    scale_max=1.15,
    translate_x=0.02,
    translate_y=0.02,
)
```

You can also pass a custom transform callable through:

```python
TransformConfig(transforms=my_transform)
```

## Lower-Level Rendering APIs

These are useful for atlas icons or direct neuron/channel experiments.

```python
from dreamlens import render_icons, render_neurons, render_channels
```

`render_icons(...)` renders activation directions, usually average directions
from atlas bins.

`render_neurons(...)` renders specific unit/channel indices.

`render_channels(...)` is an alias for convolutional channel visualizations.

These functions return:

```text
images: list of NHWC NumPy arrays
scores: list of scalar final scores
```

## Multi-Model And Parameter Noise

`ModelEnsemble` lets one visualizer run multiple named models.

```python
from dreamlens import ModelEnsemble

ensemble = ModelEnsemble({"a": model_a, "b": model_b})
```

Use it when one optimized image should satisfy objectives from multiple models.

`ParameterNoise` wraps a module and perturbs parameters transiently during
forward passes without permanently changing the original weights.

```python
from dreamlens import ParameterNoise

model.layer3 = ParameterNoise(model.layer3, mean=1.0, std=0.1)
```

## Typical Workflows

### Find Supported Layers

```python
sample = torch.rand(1, 3, 224, 224)
for info in supported_layers(model, sample_input=sample):
    print(info.name, info.output_shape)
```

### Render One Channel

```python
result = visualizer.maximize(
    FeatureTarget(layer="layer3.1.conv2", channel=17),
    config=RenderConfig.reference(width=160, height=160, steps=100),
)
result.save("layer3_channel17.png")
```

### Render A Channel Gallery

```python
result = visualizer.maximize_channels(
    layer="layer2.1.conv2",
    channels=[3, 17, 41, 64, 89, 121],
    reduction="norm",
    config=RenderConfig.reference(width=160, height=160, steps=42),
)
```

### Caricature An Input Image

```python
result = visualizer.caricature(
    image=input_image,
    layers=["layer3.1.conv2"],
    power=1.15,
    config=AmplifyConfig.reference(steps=45, lr=9e-3),
)
result.save("caricature.png")
```

### Build An Activation Atlas

```python
acts = collect_activations(
    model,
    layer="layer3.1.conv2",
    inputs=image_batches,
    spatial="random",
    random_seed=0,
)

atlas = activation_atlas(
    model,
    layer="layer3.1.conv2",
    activations=acts,
    grid_size=10,
    icon_size=80,
    render_kwargs={"n_steps": 128, "num_attempts": 1},
)
```

## Notebook Tutorial

Notebook:

```text
examples/native_dreamlens_results.ipynb
```

Generated results:

```text
results/native_dreamlens_notebook/
```

The self-contained notebook uses:

- ResNet18
- `layer2.1.conv2` and `layer4.1.conv2` channel galleries
- `FeatureVisualizer.maximize_channels(...)` with reference canvases
- dog features through `layer3.1.conv2`
- shepherd/sheep features through `layer4.1.conv2`
- reference caricature, labeled panels, a contact sheet, and a manifest

## Verification Commands

From `activation-atlas-pytorch/`:

```bash
PYTHONPATH=src pytest -q
```

From the repo root used by Codex in this workspace:

```bash
PYTHONPATH=activation-atlas-pytorch/src pytest -q activation-atlas-pytorch/tests
```

Run `examples/native_dreamlens_results.ipynb` from the repository root or the
`examples/` directory. It locates `src/dreamlens` automatically and does not
depend on a separate runner.

## Known Limits

- MaCo without a supplied dataset needs one initial download of the reference
  ImageNet spectrum; grayscale MaCo always needs a representative dataset.
- Pixel-identical reproduction of any external example image is not guaranteed.
- Very deep/high-channel layers can need more steps and more compute.
- CPU runs work but can be slow for large images, many channels, or high steps.
- `activation_atlas(...)` expects activation arrays shaped `[samples, channels]`.
- `maximize_channels(...)` currently requires `RenderConfig.attempts == 1`.
- Custom models may need custom preprocessing or a wrapper around `forward`.
- Non-tensor layer outputs must be adapted or wrapped before visualization.

## Recommended Defaults

For quick CPU checks:

```python
RenderConfig.reference(width=96, height=96, steps=8)
AmplifyConfig.reference(steps=8, lr=9e-3)
```

For useful small outputs:

```python
RenderConfig.reference(width=160, height=160, steps=42)
AmplifyConfig.reference(steps=45, lr=9e-3)
```

For higher quality:

```python
RenderConfig.reference(width=224, height=224, steps=120)
AmplifyConfig.reference(steps=120, lr=9e-3)
```
