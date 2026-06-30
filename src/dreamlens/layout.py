import logging

import numpy as np


log = logging.getLogger(__name__)


def normalize_layout(layout, min_percentile=1, max_percentile=99, relative_margin=0.1):
    """Remove outliers and scale a 2D layout into [0, 1]."""

    mins = np.percentile(layout, min_percentile, axis=0)
    maxs = np.percentile(layout, max_percentile, axis=0)

    mins -= relative_margin * (maxs - mins)
    maxs += relative_margin * (maxs - mins)

    clipped = np.clip(layout, mins, maxs)
    clipped -= clipped.min(axis=0)
    clipped /= np.maximum(clipped.max(axis=0), 1e-12)

    return clipped


def aligned_umap(activations, umap_options=None, normalize=True, verbose=False):
    """Fit UMAP to one activation array or a list of aligned activation arrays."""

    try:
        from umap import UMAP
    except ImportError as exc:
        raise ImportError(
            "activation_atlas requires umap-learn. Install with: "
            'pip install -e ".[atlas]"'
        ) from exc

    umap_options = {} if umap_options is None else dict(umap_options)
    umap_defaults = dict(
        n_components=2,
        n_neighbors=50,
        min_dist=0.05,
        verbose=verbose,
        metric="cosine",
    )
    umap_defaults.update(umap_options)

    if isinstance(activations, (list, tuple)):
        num_activation_groups = len(activations)
        combined_activations = np.concatenate(activations)
    else:
        num_activation_groups = 1
        combined_activations = activations

    try:
        layout = UMAP(**umap_defaults).fit_transform(combined_activations)
    except (RecursionError, SystemError) as exception:
        log.error("UMAP failed to fit these activations.")
        raise ValueError("UMAP failed to fit activations: %s" % exception)

    if normalize:
        layout = normalize_layout(layout)

    if num_activation_groups > 1:
        return np.split(layout, num_activation_groups, axis=0)
    return layout
