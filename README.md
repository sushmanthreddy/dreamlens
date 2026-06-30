# DreamLens

DreamLens is a native PyTorch toolkit for understanding what neural-network
layers respond to. It keeps the pretrained model fixed and optimizes generated
images using feedback from internal activations.

The three main workflows are:

| Workflow | Question |
| --- | --- |
| `maximize()` | What image makes a selected layer/channel respond strongly? |
| `caricature()` | What features does the model see in an original image, and how can they be amplified? |
| `activation_atlas()` | What feature groups appear across many real images? |

## Setup

From the repository root:

```bash
python -m pip install -e ".[examples,atlas]"
```

The examples use pretrained torchvision models. Their weights may be downloaded
the first time they are used.

## Start with the learning notebooks

| Notebook | What it teaches | Saved output |
| --- | --- | --- |
| [`learn_dreamlens_maximize.ipynb`](examples/learn_dreamlens_maximize.ipynb) | Target, render config, Fourier canvas, optimization, score evaluation | `learning_outputs/dreamlens_maximize_notebook/` |
| [`learn_dreamlens_caricature.ipynb`](examples/learn_dreamlens_caricature.ipynb) | Original/generated paths, paired transforms, feature amplification | `learning_outputs/dreamlens_caricature_notebook/` |
| [`native_dreamlens_results.ipynb`](examples/native_dreamlens_results.ipynb) | Complete reproducible gallery with multiple channels and caricatures | `results/native_dreamlens_notebook/` |

To use a learning notebook:

1. Open it in Jupyter or VS Code.
2. Edit the clearly marked parameter cell.
3. Run all cells from top to bottom.
4. View the image and measurements in the notebook; the image is also saved to
   the output directory shown above.

Both learning notebooks include their latest verified result near the top, so
you can inspect the expected output before rerunning them.

## Verified maximize result

<img src="learning_outputs/dreamlens_maximize_notebook/layer2_1_conv2_channel_17.png" width="320" alt="DreamLens channel 17 maximization result">

| Setting or measurement | Value |
| --- | ---: |
| Model / layer / channel | ResNet18 / `layer2.1.conv2` / `17` |
| Image size | `224 × 224` |
| Steps / learning rate | `400` / `0.012` |
| Final transformed-view score | `39.1118` |
| Clean untransformed norm | `43.8359` |
| Clean size-normalized RMS | `1.5656` |

The RMS score is included because a raw norm naturally increases when a larger
image produces more spatial activation values.

## Verified caricature result

<p>
  <img src="learning_inputs/dog_160.png" width="220" alt="Original dog input">
  <img src="learning_outputs/dreamlens_caricature_notebook/dog_layer3_caricature.png" width="220" alt="DreamLens dog caricature result">
</p>

| Setting or measurement | Value |
| --- | ---: |
| Model / layer | ResNet18 / `layer3.1.conv2` |
| Image size | `224 × 224` |
| Steps / learning rate / power | `200` / `0.009` / `1.20` |
| Clean cosine similarity | `0.7661` |
| Generated/original feature-norm ratio | `4.0172×` |
| Clean target projection | `191.0971` |

The original image is a fixed feature reference. The generated image starts
from Fourier noise and is optimized separately; it is not a normal image filter.

## Minimal API example

```python
from torchvision.models import ResNet18_Weights, resnet18

from dreamlens import FeatureTarget
from dreamlens import FeatureVisualizer
from dreamlens import RenderConfig
from dreamlens import TransformConfig

model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
visualizer = FeatureVisualizer(model, device="cpu", normalize=True)

target = FeatureTarget(
    layer=model.layer2[1].conv2,
    channel=17,
    reduction="norm",
)

result = visualizer.maximize(
    target=target,
    config=RenderConfig.reference(
        width=224,
        height=224,
        steps=400,
        lr=1.2e-2,
        transform=TransformConfig(
            rotate_degrees=6,
            scale_min=0.82,
            scale_max=1.12,
            translate_x=0.01,
            translate_y=0.01,
        ),
    ),
)

result.save("channel_17.png")
```

Use the learning notebooks for the full editable transform configuration,
reproducible seeds, plots, and clean-score evaluation.

## Where Fourier is used

```text
random trainable Fourier coefficients
→ frequency scaling
→ inverse FFT (`torch.fft.irfft2`)
→ color mixing and sigmoid
→ ordinary RGB image
→ frozen neural network
```

The inverse FFT happens before model inference. The model never receives
Fourier coefficients, and a forward FFT is not required when generation starts
directly from coefficients.

## More information

- Public package code: [`src/dreamlens/`](src/dreamlens/)
- Full API and capability guide:
  [`docs/DREAMLENS_FEATURE_GUIDE.md`](docs/DREAMLENS_FEATURE_GUIDE.md)

Run the smoke tests with:

```bash
PYTHONPATH=src pytest -q tests/test_smoke.py
```
