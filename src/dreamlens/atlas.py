from enum import Enum, auto

import numpy as np
import torch

from .layout import aligned_umap
from .render import render_icons


NUMBER_OF_AVAILABLE_SAMPLES = 100000


def activation_atlas(
    model,
    layer,
    activations=None,
    layer_name=None,
    grid_size=10,
    icon_size=96,
    number_activations=NUMBER_OF_AVAILABLE_SAMPLES,
    icon_batch_size=32,
    verbose=False,
    umap_options=None,
    threshold=5,
    render_kwargs=None,
):
    """Render an Activation Atlas for a PyTorch model layer.

    Args:
      model: PyTorch module to visualize.
      layer: Target module name, module object, or object with ``name`` and
        ``activations`` attributes.
      activations: Activation samples shaped ``[samples, channels]``. Optional
        only when ``layer.activations`` exists.
      layer_name: Optional module name override.
      render_kwargs: Keyword arguments forwarded to ``render_icons``.
    """

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")

    activations = _as_numpy_activations(layer, activations)
    activations = activations[:number_activations, ...]
    render_layer = _resolve_layer_name(layer, layer_name=layer_name)
    render_kwargs = {} if render_kwargs is None else dict(render_kwargs)

    layout = aligned_umap(activations, umap_options=umap_options, verbose=verbose)
    directions, coordinates, _ = bin_laid_out_activations(
        layout, activations, grid_size, threshold=threshold
    )

    render_options = {"num_attempts": 1}
    render_options.update(render_kwargs)

    icons = []
    for directions_batch in chunked(directions, icon_batch_size):
        icon_batch, _ = render_icons(
            directions_batch,
            model,
            layer=render_layer,
            size=icon_size,
            **render_options
        )
        icons += icon_batch

    return make_canvas(icons, coordinates, grid_size)


def aligned_activation_atlas(
    model1,
    layer1,
    model2,
    layer2,
    activations1=None,
    activations2=None,
    layer1_name=None,
    layer2_name=None,
    grid_size=10,
    icon_size=80,
    num_steps=1024,
    whiten_layers=True,
    number_activations=NUMBER_OF_AVAILABLE_SAMPLES,
    icon_batch_size=32,
    verbose=False,
    umap_options=None,
    render_kwargs=None,
):
    """Render aligned Activation Atlases for two PyTorch model layers.

    Returns one progressive canvas iterator per model, matching the original
    Lucid recipe's calling pattern.
    """

    render_kwargs = {} if render_kwargs is None else dict(render_kwargs)
    layer_activations = (
        _as_numpy_activations(layer1, activations1)[:number_activations, ...],
        _as_numpy_activations(layer2, activations2)[:number_activations, ...],
    )
    combined_activations = _combine_activations(
        layer_activations[0],
        layer_activations[1],
        number_activations=number_activations,
    )
    layouts = aligned_umap(
        combined_activations,
        umap_options=umap_options,
        verbose=verbose,
    )
    render_layers = (
        _resolve_layer_name(layer1, layer_name=layer1_name),
        _resolve_layer_name(layer2, layer_name=layer2_name),
    )

    for model, activations, render_layer, layout in zip(
        (model1, model2), layer_activations, render_layers, layouts
    ):
        directions, coordinates, _ = bin_laid_out_activations(
            layout, activations, grid_size, threshold=10
        )

        def _progressive_canvas_iterator(
            model=model,
            activations=activations,
            render_layer=render_layer,
            directions=directions,
            coordinates=coordinates,
        ):
            icons = []
            S = _inverse_covariance(activations) if whiten_layers else None
            render_options = {"alpha": False, "n_steps": num_steps, "S": S}
            render_options.update(render_kwargs)
            for directions_batch in chunked(directions, icon_batch_size):
                icon_batch, _ = render_icons(
                    directions_batch,
                    model,
                    layer=render_layer,
                    size=icon_size,
                    **render_options
                )
                icons += icon_batch
                yield make_canvas(icons, coordinates, grid_size)

        yield _progressive_canvas_iterator()


class ActivationTranslation(Enum):
    ONE_TO_TWO = auto()
    BIDIRECTIONAL = auto()


def _combine_activations(
    activations1,
    activations2,
    mode=ActivationTranslation.BIDIRECTIONAL,
    number_activations=NUMBER_OF_AVAILABLE_SAMPLES,
):
    activations1 = _as_numpy_activations(None, activations1)[:number_activations, ...]
    activations2 = _as_numpy_activations(None, activations2)[:number_activations, ...]

    if mode is ActivationTranslation.ONE_TO_TWO:
        acts_1_to_2 = _push_activations(activations1, activations1, activations2)
        return acts_1_to_2, activations2

    if mode is ActivationTranslation.BIDIRECTIONAL:
        acts_1_to_2 = _push_activations(activations1, activations1, activations2)
        acts_2_to_1 = _push_activations(activations2, activations2, activations1)

        activations_model1 = np.concatenate((activations1, acts_1_to_2), axis=1)
        activations_model2 = np.concatenate((acts_2_to_1, activations2), axis=1)

        return activations_model1, activations_model2

    raise ValueError("Unsupported activation translation mode: {}".format(mode))


def bin_laid_out_activations(layout, activations, grid_size, threshold=5):
    """Overlay a grid on the layout and average activations per occupied cell."""

    if layout.shape[0] != activations.shape[0]:
        raise ValueError("layout and activations must have the same sample count")

    bins = np.linspace(0, 1, num=grid_size + 1)
    bins[-1] = np.inf
    indices = np.digitize(layout, bins) - 1

    means, coordinates, counts = [], [], []
    grid_coordinates = np.indices((grid_size, grid_size)).transpose().reshape(-1, 2)
    for xy_coordinates in grid_coordinates:
        mask = np.equal(xy_coordinates, indices).all(axis=1)
        count = np.count_nonzero(mask)
        if count > threshold:
            counts.append(count)
            coordinates.append(xy_coordinates)
            means.append(np.average(activations[mask], axis=0))

    if len(coordinates) == 0:
        raise RuntimeError("Binning activations led to 0 cells containing activations.")

    return means, coordinates, counts


def make_canvas(icon_batch, coordinates, grid_size):
    """Place rendered NHWC icons on a white canvas."""

    if not icon_batch:
        raise ValueError("icon_batch must contain at least one icon")

    grid_shape = (grid_size, grid_size)
    icon_shape = icon_batch[0].shape
    canvas = np.ones((*grid_shape, *icon_shape), dtype=icon_batch[0].dtype)

    for icon, (x, y) in zip(icon_batch, coordinates):
        canvas[x, y] = icon

    return np.hstack(np.hstack(canvas))


def chunked(iterable, size):
    if size <= 0:
        raise ValueError("chunk size must be positive")
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _as_numpy_activations(layer, activations=None):
    if activations is None:
        if not hasattr(layer, "activations"):
            raise ValueError(
                "Provide activations or pass a layer object with an activations attribute."
            )
        activations = layer.activations

    if isinstance(activations, torch.Tensor):
        activations = activations.detach().cpu().numpy()
    activations = np.asarray(activations, dtype="float32")
    if activations.ndim != 2:
        raise ValueError(
            "Activation atlas expects activation samples shaped [samples, channels]."
        )
    return activations


def _covariance(act1, act2=None):
    act2 = act1 if act2 is None else act2
    if act1.shape[0] != act2.shape[0]:
        raise ValueError("Activation arrays must have the same sample count.")
    return np.matmul(act1.T, act2) / float(act1.shape[0])


def _inverse_covariance(activations):
    return np.linalg.pinv(_covariance(activations))


def _push_activations(activations, from_activations, to_activations):
    activations_decorrelated = np.dot(
        _inverse_covariance(from_activations), activations.T
    ).T
    covariance_matrix = _covariance(from_activations, to_activations)
    return np.dot(activations_decorrelated, covariance_matrix)


def _resolve_layer_name(layer, layer_name=None):
    if layer_name is not None:
        return layer_name
    if isinstance(layer, (str, torch.nn.Module)):
        return layer
    if hasattr(layer, "name"):
        return layer.name
    raise ValueError(
        "Pass layer as a PyTorch module name/module, or provide layer_name explicitly."
    )
