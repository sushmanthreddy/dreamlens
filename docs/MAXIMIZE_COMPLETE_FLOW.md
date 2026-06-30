# Native PyTorch Maximize: Complete Flow, Inch by Inch

This document explains only the native PyTorch `maximize` workflow. It does not
use DreamLens.

The corresponding runnable code is:

```text
learn_maximize_torch_only.py
```

Read this document without terminal truncation:

```bash
less -N MAXIMIZE_COMPLETE_FLOW.md
```

Useful controls inside `less`:

```text
Space          next page
b              previous page
/frequency     search for frequency
/inverse FFT   search for inverse FFT
/model input   search for model input
/backpropagate search for backpropagation
n              next search result
q              quit
```

---

# 1. Goal

We want to create an image that makes this internal model feature respond
strongly:

```text
ResNet18 → layer2[1].conv2 → channel 17
```

The model is already trained. We do not change it.

We create adjustable numbers, turn them into an image, show that image to
ResNet18, measure channel 17, and adjust the numbers.

---

# 2. The six important objects

| Object | Meaning | Does it change? |
|---|---|---:|
| `model` | Pretrained ResNet18 | No |
| `fourier_parameter` | Adjustable numbers used to construct an image | Yes |
| `frequency_scale` | Fixed preference for broad waves | No |
| `image` | Temporary RGB image reconstructed from the parameters | Indirectly |
| `score` | Strength of channel 17 | Recalculated |
| `loss` | Negative score used by AdamW | Recalculated |

The most important distinction is:

```text
The optimizer changes fourier_parameter.

The model receives an RGB image.

The model never receives fourier_parameter directly.
```

---

# 3. Complete high-level flow

```text
Create random trainable numbers
        ↓
interpret them as Fourier coefficients
        ↓
give low frequencies more influence
        ↓
inverse FFT creates spatial image values
        ↓
convert latent values to RGB
        ↓
sigmoid restricts RGB to 0–1
        ↓
normalize RGB for ResNet18
        ↓
slightly rotate and resize
        ↓
pass image into ResNet18
        ↓
capture layer2[1].conv2
        ↓
select channel 17
        ↓
calculate channel strength
        ↓
loss = negative strength
        ↓
backpropagate
        ↓
AdamW updates Fourier parameters
        ↓
repeat
```

---

# 4. Imports

```python
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
```

Purpose of each library:

```text
random
    Produces random resize factors.

Path
    Handles the output filename and directory.

NumPy
    Creates initial random values and frequency labels.

PyTorch
    Handles tensors, gradients, inverse FFT, and optimization.

torch.nn.functional
    Provides differentiable image resizing.

Pillow
    Saves the final image.

torchvision
    Provides ResNet18 and image transforms.
```

There is no DreamLens import.

---

# 5. Configuration

```python
SEED = 133
SIZE = 160
STEPS = 100
LEARNING_RATE = 0.009
GRADIENT_CLIP = 1.0
TARGET_CHANNEL = 17
DEVICE = torch.device("cpu")
```

Meanings:

```text
SEED
    Controls random-number generation.

SIZE
    Generated image is 160×160 pixels.

STEPS
    Number of image-improvement iterations.

LEARNING_RATE
    Controls how large each AdamW update is.

GRADIENT_CLIP
    Prevents an unusually large update.

TARGET_CHANNEL
    Internal channel we want to strengthen.

DEVICE
    CPU performs the calculations.
```

---

# 6. Random seeds

```python
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
```

Three systems generate random numbers:

```text
Python random
NumPy random
PyTorch random
```

All three receive the same seed so repeated runs are reproducible.

A seed does not mean every random value equals 133. It means the pseudo-random
sequence starts from the state identified by 133.

---

# 7. Load ResNet18

```python
model = resnet18(
    weights=ResNet18_Weights.DEFAULT
)
```

This creates ResNet18 and loads pretrained ImageNet weights.

```python
model = model.to(DEVICE)
model.eval()
```

`to(DEVICE)` places model tensors on CPU. `eval()` selects evaluation behavior
for layers such as batch normalization.

---

# 8. Freeze the model

```python
for model_parameter in model.parameters():
    model_parameter.requires_grad_(False)
```

ResNet18 contains many learned tensors. Setting `requires_grad=False` means:

```text
Do not calculate gradients for model weights.
Do not update model weights.
```

It does not stop a gradient from traveling through model operations to the
input image.

The model is a frozen road: gradients can travel through it, but the road does
not change.

---

# 9. Select the target layer

```python
target_layer = model.layer2[1].conv2
```

ResNet18 contains:

```text
layer1
layer2
layer3
layer4
```

Inside `layer2`, `[1]` selects its second residual block. `.conv2` selects the
second convolution in that block.

---

# 10. Attach a forward hook

Normally:

```python
prediction = model(image)
```

returns the final prediction. We need an internal layer.

Create storage:

```python
captured = {}
```

Define the hook:

```python
def capture_activation(module, inputs, output):
    captured["activation"] = output
```

PyTorch supplies:

```text
module
    The layer that executed.

inputs
    Values given to the layer.

output
    Values produced by the layer.
```

Attach it:

```python
hook_handle = target_layer.register_forward_hook(
    capture_activation
)
```

Now the model flow includes:

```text
input image
    ↓
early layers
    ↓
layer2[1].conv2
    ├── continue through model
    └── copy output into captured["activation"]
```

The hook observes but does not modify the activation.

---

# 11. Create initial random values

```python
initial_values = np.random.normal(
    size=(1, 3, SIZE, SIZE),
    scale=0.01,
).astype("float32")
```

For `SIZE=160`, shape is:

```text
[1, 3, 160, 160]
```

Meanings:

```text
1
    One generated image.

3
    Three latent color channels.

160
    Vertical-frequency storage.

160
    Interleaved real and imaginary storage.
```

The values might look like:

```text
 0.003
-0.009
 0.012
-0.002
```

They are near zero because the standard deviation is `0.01`.

They are not visible pixels yet.

---

# 12. Convert to a trainable PyTorch parameter

```python
initial_tensor = torch.tensor(
    initial_values,
    device=DEVICE,
)
```

This converts a NumPy array into a PyTorch tensor.

```python
fourier_parameter = torch.nn.Parameter(
    initial_tensor
)
```

This makes:

```python
fourier_parameter.requires_grad == True
```

PyTorch will track how the loss depends on this tensor. Later:

```python
loss.backward()
```

calculates:

```text
d(loss) / d(fourier_parameter)
```

and stores the result in:

```python
fourier_parameter.grad
```

Wrapping a tensor in `Parameter` does not immediately change the values. It
marks them as trainable.

---

# 13. What image frequency means

Frequency describes how quickly a wave changes across space.

Zero frequency:

```text
constant everywhere
```

Low frequency:

```text
slow change across a large region
```

High frequency:

```text
rapid repeated changes over a small distance
```

Examples:

```text
Low:
dark → medium → bright

High:
dark bright dark bright dark bright
```

Every two-dimensional wave has:

```text
horizontal frequency fx
vertical frequency fy
```

---

# 14. Sample spacing

```python
frequency_distance = 0.5**0.5
```

This equals approximately `0.7071`. A clearer name is `sample_spacing`.

It is a fixed coordinate convention inherited from the reference recipe. It is
not learned, updated, calculated from the generated image, or passed to
ResNet18.

---

# 15. Vertical frequency labels

```python
frequency_y = np.fft.fftfreq(
    SIZE,
    d=frequency_distance,
)[:, None]
```

`fftfreq` does not transform an image. It creates frequency labels.

With `SIZE=160`, adjacent frequency labels differ by approximately:

```text
1 / (160 × 0.7071) ≈ 0.00884
```

The labels are ordered approximately as:

```text
0
small positive
larger positive
...
large positive
large negative
...
small negative
```

Negative frequencies represent the opposite complex phase direction. Their
sign disappears in the scale calculation because they are squared.

`[:, None]` changes shape:

```text
[160] → [160, 1]
```

This makes a column that can broadcast against horizontal labels.

---

# 16. Horizontal frequency labels

```python
frequency_x = np.fft.rfftfreq(
    SIZE,
    d=frequency_distance,
)[: SIZE // 2]
```

`rfftfreq` also only creates labels.

For `SIZE=160`:

```text
SIZE // 2 = 80
```

so its shape is `[80]`.

Only approximately half of the horizontal spectrum is stored because the
Fourier spectrum of a real-valued image has conjugate symmetry. The omitted
side can be inferred during inverse FFT.

Now:

```text
frequency_y shape = [160, 1]
frequency_x shape = [80]
```

---

# 17. Combine horizontal and vertical frequencies

```python
frequency_power = (
    frequency_x**2
    + frequency_y**2
) ** 0.75
```

For each combination:

```text
overall radial frequency depends on fx² + fy²
```

The standard radius is:

```text
r = sqrt(fx² + fy²)
```

The code uses:

```text
(fx² + fy²)^0.75 = r^1.5
```

The exponent controls how strongly high-frequency influence is reduced later.

NumPy broadcasting combines `[160,1]` and `[80]` into `[160,80]`. Every table
location represents one two-dimensional wave frequency.

---

# 18. Protect frequency zero

At `fx=0` and `fy=0`, `frequency_power=0`.

Later we calculate a reciprocal. Division by zero is invalid, so the code uses:

```python
minimum_frequency = 1.0 / (
    SIZE * frequency_distance
)
```

For size 160 this is approximately `0.00884`.

---

# 19. Calculate frequency scaling

```python
frequency_scale_numpy = 1.0 / np.maximum(
    frequency_power,
    minimum_frequency,
)
```

`np.maximum` applies the safe floor element by element. The reciprocal reverses
the relationship:

```text
frequency value 0.01 → scale 100
frequency value 0.50 → scale 2
```

Therefore:

```text
low frequency  → large scale
high frequency → small scale
```

This encourages broad structures over tiny noise.

Convert the fixed scale to PyTorch:

```python
frequency_scale = torch.tensor(
    frequency_scale_numpy,
    dtype=torch.float32,
    device=DEVICE,
)
```

Only `fourier_parameter` is trainable. `frequency_scale` is fixed.

---

# 20. Color matrix

```python
color_matrix = torch.tensor(
    [
        [0.26, 0.09, 0.02],
        [0.27, 0.00, -0.05],
        [0.27, -0.09, 0.03],
    ],
    dtype=torch.float32,
    device=DEVICE,
)
```

Inverse FFT produces three latent channels. This matrix converts them into red,
green, and blue.

Approximately:

```text
red   = 0.26×latent1 + 0.09×latent2 + 0.02×latent3
green = 0.27×latent1 + 0.00×latent2 - 0.05×latent3
blue  = 0.27×latent1 - 0.09×latent2 + 0.03×latent3
```

The first latent channel changes all RGB channels similarly, so it mostly
controls brightness. Other channels introduce color differences.

Normalize the matrix:

```python
column_lengths = torch.linalg.vector_norm(
    color_matrix,
    dim=0,
)

largest_column_length = column_lengths.max()

color_matrix = color_matrix / largest_column_length
```

This keeps its numerical scale controlled.

---

# 21. ImageNet normalization constants

```python
imagenet_mean = torch.tensor(
    [0.485, 0.456, 0.406],
    device=DEVICE,
).view(1, 3, 1, 1)

imagenet_std = torch.tensor(
    [0.229, 0.224, 0.225],
    device=DEVICE,
).view(1, 3, 1, 1)
```

ResNet18 expects:

```text
normalized = (pixel - mean) / standard deviation
```

Shape `[1,3,1,1]` lets one red, green, and blue constant broadcast across every
pixel.

---

# 22. Start `make_image()`

```python
def make_image():
```

This function converts the current Fourier parameter into RGB pixels. It is
called during every optimization iteration.

---

# 23. Reshape into real and imaginary pairs

```python
real_imaginary_pairs = fourier_parameter.reshape(
    1,
    3,
    SIZE,
    SIZE // 2,
    2,
)
```

Shape changes:

```text
[1, 3, 160, 160]
        ↓
[1, 3, 160, 80, 2]
```

The final dimension means:

```text
index 0 = real
index 1 = imaginary
```

Reshape does not change the stored numbers. It changes their organization and
interpretation.

---

# 24. Select real and imaginary parts

```python
real_values = real_imaginary_pairs[..., 0]
imaginary_values = real_imaginary_pairs[..., 1]
```

Both have shape `[1,3,160,80]`.

---

# 25. Create a complex spectrum

```python
spectrum = torch.complex(
    real_values,
    imaginary_values,
)
```

Every coefficient becomes:

```text
real + imaginary × i
```

This is a frequency-domain representation, not a pixel image.

---

# 26. Apply frequency preference

```python
scaled_spectrum = spectrum * frequency_scale
```

Shapes:

```text
spectrum         [1, 3, 160, 80]
frequency_scale        [160, 80]
```

PyTorch broadcasts the scale across the batch and latent-channel dimensions.

---

# 27. Inverse FFT creates spatial values

```python
latent_image = torch.fft.irfft2(
    scaled_spectrum,
    s=(SIZE, SIZE),
    norm="ortho",
)
```

This is the exact moment the spatial image grid is created.

Before this line:

```text
frequency-domain wave coefficients
```

After this line:

```text
values at spatial image locations
```

Output shape is `[1,3,160,160]`. It is called latent because its channels have
not yet been converted to ordinary RGB.

---

# 28. Convert latent channels to RGB

```python
rgb_logits = torch.einsum(
    "oc,bchw->bohw",
    color_matrix,
    latent_image,
)
```

This applies the 3×3 color matrix at every pixel. Output shape remains
`[1,3,160,160]`.

---

# 29. Constrain RGB values

```python
visible_image = torch.sigmoid(rgb_logits)
```

Before sigmoid, values can be any real number. Sigmoid maps them between zero
and one. Now the tensor is a valid visible RGB image.

---

# 30. Create AdamW

```python
optimizer = torch.optim.AdamW(
    [fourier_parameter],
    lr=LEARNING_RATE,
    weight_decay=0.0,
)
```

The square brackets contain the only object AdamW may update. Model weights are
not included.

---

# 31. Start optimization loop

```python
for step in range(1, STEPS + 1):
```

Everything below repeats for each step.

---

# 32. Clear old gradients

```python
optimizer.zero_grad(set_to_none=True)
```

Gradients from the preceding iteration must not mix with the new iteration.

---

# 33. Construct current image

```python
image = make_image()
```

This performs:

```text
fourier_parameter
    ↓
real/imaginary pairs
    ↓
complex spectrum
    ↓
frequency scaling
    ↓
inverse FFT
    ↓
latent image
    ↓
RGB matrix
    ↓
sigmoid
    ↓
image
```

Image shape is `[1,3,160,160]`.

---

# 34. Normalize for ResNet18

```python
model_input = (
    image - imagenet_mean
) / imagenet_std
```

ResNet receives normalized values rather than raw 0–1 pixels.

---

# 35. Random transforms

```python
model_input = transforms.RandomAffine(
    degrees=10.0,
    translate=(0.02, 0.02),
)(model_input)
```

This slightly rotates and translates the temporary model view.

```python
height_factor = random.uniform(0.7, 1.15)
width_factor = random.uniform(0.7, 1.15)

model_input = F.interpolate(
    model_input,
    scale_factor=(height_factor, width_factor),
    mode="bilinear",
)
```

This independently resizes height and width. Transforms encourage a feature
that works under small geometric changes rather than one exact view.

---

# 36. Run ResNet18

```python
captured.clear()
model(model_input)
```

ResNet receives transformed, normalized RGB pixels. It does not receive the
Fourier parameter, spectrum, or frequency scale.

When the target layer executes, the hook stores its output.

---

# 37. Read the target activation

```python
activations = captured["activation"]
```

A possible shape is `[1,128,19,19]`:

```text
1       one image
128     layer channels
19×19   spatial response positions
```

---

# 38. Select channel 17

```python
selected_channel = activations[
    :,
    TARGET_CHANNEL,
    :,
    :,
]
```

Shape changes:

```text
[1,128,19,19] → [1,19,19]
```

---

# 39. Calculate score

```python
score = selected_channel.norm()
```

The L2 norm is:

```text
sqrt(sum of all selected activations squared)
```

For values `[3,4]`, the norm is `sqrt(9+16)=5`. A larger score means channel 17
responded more strongly.

---

# 40. Create negative loss

```python
loss = -score
```

AdamW minimizes. Therefore:

```text
score 2  → loss -2
score 10 → loss -10
```

Negative ten is smaller than negative two, so minimizing loss increases score.

---

# 41. Backpropagate

```python
loss.backward()
```

PyTorch follows the computation backward:

```text
loss
  ↓
channel norm
  ↓
channel 17
  ↓
target layer
  ↓
earlier ResNet layers
  ↓
model input
  ↓
random transformations
  ↓
normalization
  ↓
sigmoid
  ↓
color matrix
  ↓
inverse FFT
  ↓
frequency-scaled spectrum
  ↓
fourier_parameter
```

The result is stored in `fourier_parameter.grad`, with the same shape as the
parameter.

---

# 42. Clip gradient

```python
torch.nn.utils.clip_grad_norm_(
    [fourier_parameter],
    max_norm=GRADIENT_CLIP,
)
```

If the complete gradient is too large, it is scaled down to prevent an unstable
jump.

---

# 43. Update Fourier parameters

```python
optimizer.step()
```

AdamW reads `fourier_parameter.grad` and changes `fourier_parameter`.

The model and frequency scale do not change.

On the next iteration:

```text
updated Fourier parameter
        ↓
inverse FFT
        ↓
different image
        ↓
new channel score
```

---

# 44. Repeat

Conceptually:

```text
Step 1:
random-looking image, small score

Step 10:
some preferred structures, larger score

Step 50:
strong repeated patterns, larger score

Final step:
optimized feature image
```

The score can fluctuate because every iteration uses a different random
transform.

---

# 45. Evaluate final clean image

After training:

```python
final_image = make_image()
```

The final image can be evaluated without random transforms to obtain a stable
clean score.

---

# 46. Remove hook

```python
hook_handle.remove()
```

The internal-layer listener is no longer needed.

---

# 47. Save image

Remove the batch dimension:

```python
image_to_save = final_image[0]
```

Shape becomes `[3,160,160]`.

Move channels to the end:

```python
image_to_save = image_to_save.permute(1, 2, 0)
```

Shape becomes `[160,160,3]`.

Convert 0–1 floating-point values to 0–255 bytes:

```python
image_array = (
    image_to_save.cpu().numpy() * 255
).astype(np.uint8)
```

Save:

```python
Image.fromarray(image_array).save(OUTPUT_PATH)
```

---

# 48. Exact model input

The model receives `model_input`, which is created by:

```text
Fourier parameter
    ↓ inverse FFT
visible RGB image
    ↓ ImageNet normalization
normalized RGB image
    ↓ random transform
transformed normalized RGB image
    ↓
ResNet18
```

It never receives Fourier coefficients directly.

---

# 49. Exact trainable object

Only this changes:

```python
fourier_parameter
```

These remain fixed:

```text
model weights
frequency labels
frequency scale
color matrix
ImageNet mean
ImageNet standard deviation
target channel
```

---

# 50. Final mental model

Imagine a synthesizer.

The Fourier parameters are thousands of knobs controlling waves.

Inverse FFT combines those waves into an image.

ResNet18 looks at the image and gives channel 17 a strength score.

Backpropagation determines which wave knobs influenced that score.

AdamW adjusts those wave knobs.

Then inverse FFT builds a new image from the adjusted waves.

```text
wave knobs
  → image
  → ResNet
  → channel score
  → gradient
  → adjust wave knobs
  → new image
```

That is the complete native PyTorch maximize flow.

