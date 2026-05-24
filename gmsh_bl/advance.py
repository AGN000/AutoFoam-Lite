"""Advancing-layer march with geometric growth, smoothing, and collapse.

Given surface nodes + smoothed normals, produce N layers of marched node
positions. Each layer height = first_height * ratio**i. Front-front collisions
trigger early termination (clamp) for affected nodes — see collision.py for
the per-step collision check that produces `max_height_per_node`.
"""
from __future__ import annotations

import numpy as np

from .normals import laplacian_smooth


def layer_heights(first: float, ratio: float, n_layers: int,
                   outer_ratio: float | None = None,
                   outer_layers: int = 0) -> np.ndarray:
    """Cumulative layer top heights with optional outer-layer stretching.

    For layers 0..(n_layers - outer_layers - 1), the per-step thickness grows
    by `ratio`. For the last `outer_layers` layers, the per-step thickness
    grows by `outer_ratio` instead. This is used to close the size mismatch
    between the cap and the surrounding cavity tets without dramatically
    changing the near-wall thickness budget.

    `outer_ratio=None` (default) reproduces the legacy single-ratio behaviour.
    """
    if outer_ratio is None or outer_layers <= 0:
        if abs(ratio - 1.0) < 1e-12:
            steps = first * np.ones(n_layers)
        else:
            steps = first * ratio ** np.arange(n_layers)
        return np.cumsum(steps)

    n_inner = max(n_layers - outer_layers, 0)
    inner = first * ratio ** np.arange(n_inner)
    last_inner = inner[-1] if n_inner else first
    outer = last_inner * outer_ratio ** np.arange(1, outer_layers + 1)
    steps = np.concatenate([inner, outer])[:n_layers]
    return np.cumsum(steps)


def march_layers(
    nodes: np.ndarray,
    tris: np.ndarray,
    normals: np.ndarray,
    first_height: float,
    ratio: float,
    n_layers: int,
    max_height_per_node: np.ndarray | None = None,
    smooth_sweeps: int = 2,
    smoothing_fixed_mask: np.ndarray | None = None,
    outer_ratio: float | None = None,
    outer_layers: int = 0,
) -> np.ndarray:
    """March `n_layers` of prism nodes outward.

    Returns array of shape (n_layers+1, N, 3) where layer 0 is the wall surface
    and layers 1..N are the marched fronts. Per-node clamping via
    `max_height_per_node` (length-N) ensures collapsed nodes stop early.
    Nodes flagged in `smoothing_fixed_mask` keep their unsmoothed displacement
    — used to protect ridge nodes whose marches must not be averaged across a
    sharp fold into nodes on the opposite side.
    """
    cum_h = layer_heights(first_height, ratio, n_layers,
                           outer_ratio=outer_ratio, outer_layers=outer_layers)
    if max_height_per_node is None:
        max_height_per_node = np.full(nodes.shape[0], np.inf)

    out = np.empty((n_layers + 1, nodes.shape[0], 3))
    out[0] = nodes
    for i, h in enumerate(cum_h, start=1):
        h_clamped = np.minimum(h, max_height_per_node)
        disp = normals * h_clamped[:, None]
        disp = laplacian_smooth(disp, tris, n_sweeps=smooth_sweeps,
                                fixed_mask=smoothing_fixed_mask)
        out[i] = nodes + disp
    return out
