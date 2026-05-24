"""Front-front collision detection via KD-tree.

For each surface node, query the KD-tree of all *other* surface nodes for
neighbours within a search radius proportional to the projected total layer
thickness. If a neighbour's marching ray approaches within `safety` of ours,
clamp our maximum march height to half the inter-node gap.

This is the simplest variant of Aubry/Lohner front-collision and Garimella's
graceful-exit logic. It is intentionally conservative — a feasibility test, not
a production heuristic.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _surface_neighbors(tris: np.ndarray, N: int, rings: int = 2) -> list[set[int]]:
    """Per-node set of k-ring neighbours along the surface mesh graph.

    Used to exclude topological neighbours from spatial collision queries: a
    node sitting close in 3D *because it shares an edge or a 2-ring path* with
    us is not an opposing front, just an adjacent point on our own surface.
    """
    one_ring = [set() for _ in range(N)]
    for tri in tris:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        one_ring[a].update((b, c))
        one_ring[b].update((a, c))
        one_ring[c].update((a, b))
    if rings <= 1:
        return one_ring
    out = [set(s) for s in one_ring]
    for _ in range(rings - 1):
        new = [set(s) for s in out]
        for i, s in enumerate(out):
            for j in s:
                new[i].update(one_ring[j])
            new[i].discard(i)
        out = new
    return out


def estimate_max_heights(
    nodes: np.ndarray,
    normals: np.ndarray,
    desired_total: float,
    tris: np.ndarray | None = None,
    safety: float = 0.5,
    exclude_rings: int = 2,
) -> np.ndarray:
    """Per-node clamp on march height based on opposing-front collisions.

    Excludes surface-topological neighbours (within `exclude_rings` rings on
    the wall mesh graph) — close geometric proximity caused by being adjacent
    on the same smooth surface is not a collision, it's the resolution of the
    surface mesh. Real collisions come from *non-neighbour* nodes that happen
    to be close in 3D (opposite walls of a narrow gap, fold across a concave
    ridge, etc.).
    """
    N = nodes.shape[0]
    tree = cKDTree(nodes)
    k = min(16, N)
    dists, idxs = tree.query(nodes, k=k)

    excl: list[set[int]] | None = None
    if tris is not None and exclude_rings > 0:
        excl = _surface_neighbors(tris, N, rings=exclude_rings)

    out = np.full(N, desired_total)
    for i in range(N):
        excluded = excl[i] if excl is not None else set()
        for j in range(1, k):
            other = int(idxs[i, j])
            if other in excluded:
                continue
            sep = nodes[other] - nodes[i]
            d = float(dists[i, j])
            if d <= 0:
                continue
            forward = float(np.dot(sep / d, normals[i]))
            if forward <= 0.1:
                continue
            cap = safety * d / max(forward, 0.2)
            if cap < out[i]:
                out[i] = cap
    return out
