from collections.abc import Mapping, Sequence

import torch

try:
    from torch.func import functional_call
except ImportError:  # pragma: no cover - PyTorch 2.x provides torch.func.
    from torch.nn.utils.stateless import functional_call


class ModelEnsemble(torch.nn.Module):
    """Run several named models from one module.

    The ensemble is useful when a single optimized image should satisfy
    objectives from layers that live in different networks. Forward outputs can
    be returned as an ordered dictionary, tuple, or list without affecting layer
    hooks registered on the child modules.
    """

    def __init__(self, models, return_format="dict"):
        super().__init__()
        self.models = torch.nn.ModuleDict(_normalize_models(models))
        if return_format not in {"dict", "tuple", "list"}:
            raise ValueError("return_format must be 'dict', 'tuple', or 'list'")
        self.return_format = return_format

    def forward(self, image, only=None):
        selected = self._select_names(only)
        outputs = {name: self.models[name](image) for name in selected}
        if self.return_format == "dict":
            return outputs
        values = tuple(outputs.values())
        if self.return_format == "tuple":
            return values
        return list(values)

    def names(self):
        return tuple(self.models.keys())

    def get(self, name):
        return self.models[name]

    def items(self):
        return self.models.items()

    def _select_names(self, only):
        if only is None:
            return self.names()
        if isinstance(only, str):
            only = [only]
        names = tuple(only)
        missing = [name for name in names if name not in self.models]
        if missing:
            raise KeyError("Unknown model name(s): {}".format(", ".join(missing)))
        return names


class ParameterNoise(torch.nn.Module):
    """Evaluate a module with multiplicative parameter noise.

    The wrapped module is never mutated. Each forward pass builds a transient
    parameter state and dispatches with ``torch.func.functional_call`` so hooks,
    buffers, and input gradients still behave like a normal module call.
    """

    def __init__(self, module, mean=1.0, std=0.2, enabled=True):
        super().__init__()
        if not isinstance(module, torch.nn.Module):
            raise TypeError("module must be a torch.nn.Module")
        if std < 0:
            raise ValueError("std must be >= 0")
        self.module = module
        self.mean = float(mean)
        self.std = float(std)
        self.enabled = bool(enabled)

    def forward(self, *args, **kwargs):
        if not self.enabled or self.std == 0:
            return self.module(*args, **kwargs)

        parameters = {
            name: self._perturb(parameter)
            for name, parameter in self.module.named_parameters()
        }
        buffers = dict(self.module.named_buffers())
        return functional_call(self.module, (parameters, buffers), args, kwargs)

    def _perturb(self, parameter):
        noise = torch.empty_like(parameter).normal_(mean=self.mean, std=self.std)
        return parameter * noise

    def set_enabled(self, enabled=True):
        self.enabled = bool(enabled)
        return self


def _normalize_models(models):
    if isinstance(models, Mapping):
        items = list(models.items())
    elif isinstance(models, Sequence):
        items = []
        for index, model in enumerate(models):
            if isinstance(model, tuple) and len(model) == 2:
                items.append(model)
            else:
                items.append(("model_{}".format(index), model))
    else:
        raise TypeError("models must be a mapping or sequence of modules")

    if not items:
        raise ValueError("models must contain at least one module")

    normalized = {}
    for name, model in items:
        if not isinstance(name, str) or not name:
            raise ValueError("model names must be non-empty strings")
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model '{}' must be a torch.nn.Module".format(name))
        if name in normalized:
            raise ValueError("duplicate model name: {}".format(name))
        normalized[name] = model
    return normalized
