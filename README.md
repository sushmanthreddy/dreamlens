# DreamLens

[![PyPI](https://img.shields.io/pypi/v/dreamlens.svg)](https://pypi.org/project/dreamlens/)
[![Python](https://img.shields.io/pypi/pyversions/dreamlens.svg)](https://pypi.org/project/dreamlens/)
[![License](https://img.shields.io/pypi/l/dreamlens.svg)](LICENSE)

Feature visualization in native PyTorch. DreamLens works with torchvision and
other compatible `torch.nn.Module` models through one `FeatureVisualizer` API.

## Install

```bash
pip install dreamlens
```

Install the notebook dependencies with:

```bash
pip install "dreamlens[examples]"
```

## API

```python
from torchvision.models import ResNet18_Weights, resnet18
from dreamlens import FeatureTarget, FeatureVisualizer

model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
visualizer = FeatureVisualizer(model, normalize=True)
target = FeatureTarget.for_channel("layer2.1.conv2", 17, reduction="norm")

result = visualizer.visualize(target, method="maximize")
result.save("channel_17.png")
```

The same class provides four workflows:

```python
visualizer.visualize(target, method="maximize")
visualizer.visualize(target, method="maco")
visualizer.visualize(
    target,
    method="feature_accentuation",
    image="image.jpg",
    regularization_layer="layer2.1",
)
visualizer.visualize(
    method="caricature",
    image="image.jpg",
    layers=["layer3.1.conv2"],
)
```

The notebooks contain the complete configurations for each method.

## Results

### Activation maximization

<img src="https://raw.githubusercontent.com/sushmanthreddy/dreamlens/main/learning_outputs/dreamlens_maximize_notebook/layer2_1_conv2_channel_17.png" width="320" alt="Activation maximization result">

### MaCo

<img src="https://raw.githubusercontent.com/sushmanthreddy/dreamlens/main/results/feature_visualization_getting_started_pytorch/root_maco_toucan_panel.png" width="100%" alt="MaCo image, spatial importance and overlay">

### Feature accentuation

<img src="https://raw.githubusercontent.com/sushmanthreddy/dreamlens/main/results/dreamlens_feature_accentuation_notebook/torchvision_resnet18_feature_accentuation_examples.png" width="100%" alt="Feature accentuation examples">

### Caricature

<img src="https://raw.githubusercontent.com/sushmanthreddy/dreamlens/main/results/feature_visualization_getting_started_pytorch/root_caricature_dog_panel.png" width="100%" alt="Original image and caricature result">

## Notebooks

- [Getting started](examples/Feature_Visualization_Getting_started_PyTorch.ipynb)
- [Activation maximization](examples/learn_dreamlens_maximize.ipynb)
- [MaCo](examples/learn_dreamlens_maco.ipynb)
- [Feature accentuation](examples/learn_dreamlens_feature_accentuation.ipynb)
- [Caricature](examples/learn_dreamlens_caricature.ipynb)
- [Result gallery](examples/native_dreamlens_results.ipynb)

## Research

DreamLens brings ideas from three papers into one PyTorch API:

- [Feature Visualization, Distill 2017](https://distill.pub/2017/feature-visualization/)
- [MAgnitude Constrained Optimization, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/76d2f8e328e1081c22a77ca0fa330ca5-Abstract-Conference.html)
- [Feature Accentuation, 2024](https://arxiv.org/abs/2402.10039)

This is an independent project for education and research. It is not an
official implementation from the paper authors.

## License

DreamLens code is Apache-2.0 licensed. Papers, model weights, datasets and
example images keep their original rights and licenses. Educational use does
not remove those obligations. See [NOTICE](NOTICE) for sources and attribution.
