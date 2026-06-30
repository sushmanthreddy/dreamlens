import numpy as np
import torch

from .layers import model_device, resolve_module


def collect_activations(
    model,
    layer,
    inputs,
    preprocess=None,
    device=None,
    activation_format="NCHW",
    spatial="center",
    random_seed=None,
):
    """Collect layer activation vectors from a tensor or iterable of tensors.

    Args:
      model: PyTorch module to run.
      layer: Module name or module object to capture with a forward hook.
      inputs: A batch tensor or iterable yielding batch tensors in NCHW format.
      preprocess: Optional callable applied before the model.
      activation_format: ``"NCHW"`` for normal conv outputs, ``"NHWC"`` for
        channels-last outputs.
      spatial: ``"center"`` for one vector per image, ``"random"`` for one
        random spatial vector per image, or ``"all"`` for one vector per
        spatial location.
    """

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")

    device = torch.device(device) if device is not None else model_device(model)
    model.to(device)
    model.eval()
    preprocess = (lambda x: x) if preprocess is None else preprocess
    feature_module = resolve_module(model, layer)
    rng = None
    if random_seed is not None:
        rng = torch.Generator(device="cpu")
        rng.manual_seed(random_seed)

    batches = _as_batches(inputs)
    collected = []

    with _LayerCapture(feature_module) as capture:
        with torch.no_grad():
            for batch in batches:
                batch = batch.to(device)
                capture.clear()
                model(preprocess(batch))
                acts = capture.output
                if acts is None:
                    raise RuntimeError("The requested layer was not called by model.")
                vectors = _activation_vectors(
                    acts,
                    activation_format=activation_format,
                    spatial=spatial,
                    rng=rng,
                )
                collected.append(vectors.detach().cpu().numpy())

    if not collected:
        raise ValueError("No input batches were provided.")
    return np.concatenate(collected, axis=0).astype("float32")


def _activation_vectors(acts, activation_format="NCHW", spatial="center", rng=None):
    if acts.dim() == 2:
        return acts
    if acts.dim() == 3:
        if spatial == "center":
            return acts[:, acts.shape[1] // 2, :]
        if spatial == "random":
            positions = torch.randint(0, acts.shape[1], (acts.shape[0],), generator=rng)
            positions = positions.to(device=acts.device)
            return acts[torch.arange(acts.shape[0], device=acts.device), positions]
        if spatial == "all":
            return acts.reshape(-1, acts.shape[-1])
    if acts.dim() != 4:
        raise ValueError("Expected 2D, 3D, or 4D layer activations.")

    activation_format = activation_format.upper()
    if activation_format == "NCHW":
        if spatial == "center":
            return acts[:, :, acts.shape[2] // 2, acts.shape[3] // 2]
        if spatial == "random":
            ys = torch.randint(0, acts.shape[2], (acts.shape[0],), generator=rng)
            xs = torch.randint(0, acts.shape[3], (acts.shape[0],), generator=rng)
            ys = ys.to(device=acts.device)
            xs = xs.to(device=acts.device)
            batch = torch.arange(acts.shape[0], device=acts.device)
            return acts[batch, :, ys, xs]
        if spatial == "all":
            return acts.permute(0, 2, 3, 1).reshape(-1, acts.shape[1])
    elif activation_format == "NHWC":
        if spatial == "center":
            return acts[:, acts.shape[1] // 2, acts.shape[2] // 2, :]
        if spatial == "random":
            ys = torch.randint(0, acts.shape[1], (acts.shape[0],), generator=rng)
            xs = torch.randint(0, acts.shape[2], (acts.shape[0],), generator=rng)
            ys = ys.to(device=acts.device)
            xs = xs.to(device=acts.device)
            batch = torch.arange(acts.shape[0], device=acts.device)
            return acts[batch, ys, xs, :]
        if spatial == "all":
            return acts.reshape(-1, acts.shape[-1])
    else:
        raise ValueError("activation_format must be 'NCHW' or 'NHWC'.")

    raise ValueError("spatial must be 'center', 'random', or 'all'.")


def _as_batches(inputs):
    if isinstance(inputs, torch.Tensor):
        return [inputs]
    return inputs


class _LayerCapture:
    def __init__(self, module):
        self.module = module
        self.output = None
        self._handle = None

    def __enter__(self):
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._handle.remove()

    def clear(self):
        self.output = None

    def _hook(self, module, inputs, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        self.output = output
