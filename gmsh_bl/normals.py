"""Area-weighted smoothed wall normals + ridge classification.

Inputs are surface triangle meshes from gmsh: nodes (N,3) and tris (M,3) of
node indices. Outputs a unit normal per surface node, plus per-node ridge tags
(SMOOTH, CONVEX_SHARP, CONCAVE_SHARP) used downstream by the march loop.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

SMOOTH = 0
CONVEX_SHARP = 1
CONCAVE_SHARP = 2


def triangle_normals_and_areas(nodes: np.ndarray, tris: np.ndarray):
    v0 = nodes[tris[:, 0]]
    v1 = nodes[tris[:, 1]]
    v2 = nodes[tris[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    area2 = np.linalg.norm(cross, axis=1)
    n = cross / np.where(area2[:, None] > 0, area2[:, None], 1.0)
    return n, 0.5 * area2


def orient_tris_outward(nodes: np.ndarray, tris: np.ndarray,
                        reference: np.ndarray | None = None) -> np.ndarray:
    """Flip tri winding so each tri's RH-rule normal points away from `reference`.

    For closed star-shaped geometries (airfoils, sphere-in-box, etc.), pass the
    domain centroid as `reference` (default: mean of all nodes). Each tri whose
    centroid->reference vector is *aligned* with its current normal is flipped
    so the normal points outward.

    For non-star surfaces (flat plate, BFS) this still produces a consistent
    orientation, just one chosen by the centroid heuristic — both choices yield
    valid prism columns since the march direction tracks whichever side we pick.
    """
    if reference is None:
        reference = nodes.mean(axis=0)
    tri_n, _ = triangle_normals_and_areas(nodes, tris)
    centroids = nodes[tris].mean(axis=1)
    outward = centroids - reference
    align = np.einsum("ij,ij->i", tri_n, outward)
    flipped = tris.copy()
    mask = align < 0  # normal points inward -> flip winding
    flipped[mask] = flipped[mask][:, [0, 2, 1]]
    return flipped


def smoothed_normals(nodes: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Per-node unit normal, area-weighted average of incident triangle normals."""
    tri_n, tri_area = triangle_normals_and_areas(nodes, tris)
    N = nodes.shape[0]
    M = tris.shape[0]
    # build sparse incidence: row=node, col=tri, val=area (one row per node)
    rows = tris.reshape(-1)
    cols = np.repeat(np.arange(M), 3)
    data = np.repeat(tri_area, 3)
    incidence = csr_matrix((data, (rows, cols)), shape=(N, M))
    weighted = incidence @ tri_n  # (N, 3)
    norms = np.linalg.norm(weighted, axis=1)
    out = weighted / np.where(norms[:, None] > 0, norms[:, None], 1.0)
    return out


def laplacian_smooth(values: np.ndarray, tris: np.ndarray, n_sweeps: int = 2,
                     fixed_mask: np.ndarray | None = None) -> np.ndarray:
    """1-ring Laplacian smoothing of a per-node vector field over the surface graph."""
    N = values.shape[0]
    edges = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    edges = np.vstack([edges, edges[:, ::-1]])
    rows = edges[:, 0]
    cols = edges[:, 1]
    deg = np.bincount(rows, minlength=N).astype(float)
    deg[deg == 0] = 1.0
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
    out = values.copy()
    for _ in range(n_sweeps):
        avg = (A @ out) / deg[:, None]
        if fixed_mask is not None:
            avg[fixed_mask] = out[fixed_mask]
        out = 0.5 * out + 0.5 * avg
    return out


def dihedral_classify(nodes: np.ndarray, tris: np.ndarray,
                      sharp_deg: float = 30.0) -> np.ndarray:
    """Tag each node SMOOTH / CONVEX_SHARP / CONCAVE_SHARP from local dihedrals.

    For each interior edge (shared by two tris), compute dihedral angle and the
    sign of (n_a + n_b) . (centroid_b - centroid_a) to distinguish concave/convex.
    Aggregate per node: take the most-extreme classification across incident edges.
    """
    N = nodes.shape[0]
    tri_n, _ = triangle_normals_and_areas(nodes, tris)
    M = tris.shape[0]
    centroids = nodes[tris].mean(axis=1)

    # build edge -> [tri ids] map; canonicalize each edge as sorted pair
    edge_to_tris: dict[tuple[int, int], list[int]] = {}
    for ti in range(M):
        a, b, c = tris[ti]
        for u, v in ((a, b), (b, c), (c, a)):
            key = (int(min(u, v)), int(max(u, v)))
            edge_to_tris.setdefault(key, []).append(ti)

    tags = np.zeros(N, dtype=np.int8)
    # sharp = angle between adjacent triangle normals exceeds `sharp_deg`.
    # For a coplanar surface, normals are parallel (angle 0). A 90° fold gives
    # a 90° normal-angle. So smooth-test = cosang > cos(sharp_deg).
    cos_thresh = np.cos(np.deg2rad(sharp_deg))

    for (u, v), ts in edge_to_tris.items():
        if len(ts) != 2:
            continue
        a, b = ts
        cosang = float(np.clip(np.dot(tri_n[a], tri_n[b]), -1.0, 1.0))
        if cosang >= cos_thresh:
            continue  # nearly coplanar -> smooth, no ridge tag
        # determine convex vs concave: convex if average normal points away from
        # the line connecting centroids (outward-bulging fold), concave if inward.
        avg_n = tri_n[a] + tri_n[b]
        sep = centroids[b] - centroids[a]
        # convex: (n_a + n_b) . (c_b - c_a) > 0 means normals fan outward
        sign = float(np.dot(avg_n, sep))
        cls = CONVEX_SHARP if sign > 0 else CONCAVE_SHARP
        # promote nodes' tags: CONCAVE > CONVEX > SMOOTH (concave is the dangerous case)
        for node in (u, v):
            if cls == CONCAVE_SHARP:
                tags[node] = CONCAVE_SHARP
            elif cls == CONVEX_SHARP and tags[node] != CONCAVE_SHARP:
                tags[node] = CONVEX_SHARP
    return tags
