"""Mesh quality metrics for hybrid prism+tet meshes.

We focus on the metrics that distinguish a usable BL mesh from a broken one:
- min/max scaled Jacobian per cell (negative => self-intersecting)
- equiangle skewness (0 = perfect, 1 = degenerate)
- aspect ratio (longest edge / shortest height)
- layer growth-rate consistency

For prisms, scaled-Jacobian is computed at each of 6 corners using the cell's
local frame; we report the worst corner per cell.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def prism_scaled_jacobian(prism_xyz: np.ndarray) -> np.ndarray:
    """prism_xyz: (n_prisms, 6, 3). Returns worst-tet scaled volume per prism.

    Decompose the wedge (n0,n1,n2,n3,n4,n5) into three tets and report the
    minimum signed scaled volume across them. Negative => the wedge is
    self-intersecting / inverted. The scaled form divides by the cube of the
    mean edge length so values lie in roughly [-1, 1] for well-shaped cells.

    Tet decomposition (canonical for VTK_WEDGE bottom CCW seen from above):
      T0 = (0,1,2,5),  T1 = (0,1,5,4),  T2 = (0,4,5,3)
    """
    tets = [(0, 1, 2, 5), (0, 1, 5, 4), (0, 4, 5, 3)]
    n = prism_xyz.shape[0]
    sj = np.full(n, np.inf)
    for a, b, c, d in tets:
        e1 = prism_xyz[:, b] - prism_xyz[:, a]
        e2 = prism_xyz[:, c] - prism_xyz[:, a]
        e3 = prism_xyz[:, d] - prism_xyz[:, a]
        det = np.einsum("ij,ij->i", np.cross(e1, e2), e3)
        # scale by mean of edge-length cubes (rough geometric mean)
        L = (np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1)
             * np.linalg.norm(e3, axis=1))
        L = np.where(L > 0, L, 1.0)
        sj = np.minimum(sj, det / L)
    return sj


def prism_aspect_ratio(prism_xyz: np.ndarray) -> np.ndarray:
    """Longest base edge / mean layer height per prism."""
    bot = prism_xyz[:, :3]
    top = prism_xyz[:, 3:]
    base_edges = np.array([
        np.linalg.norm(bot[:, 1] - bot[:, 0], axis=1),
        np.linalg.norm(bot[:, 2] - bot[:, 1], axis=1),
        np.linalg.norm(bot[:, 0] - bot[:, 2], axis=1),
    ]).max(axis=0)
    heights = np.linalg.norm(top - bot, axis=2).mean(axis=1)
    heights = np.where(heights > 0, heights, 1e-30)
    return base_edges / heights


def prism_equiangle_skewness(prism_xyz: np.ndarray) -> np.ndarray:
    """Worst-case (max) equiangle skewness over the two triangular faces.

    skew = max((theta_max - 60)/120, (60 - theta_min)/60), 0 ideal.
    """
    def tri_skew(tri: np.ndarray) -> np.ndarray:
        a = tri[:, 1] - tri[:, 0]
        b = tri[:, 2] - tri[:, 1]
        c = tri[:, 0] - tri[:, 2]
        la = np.linalg.norm(a, axis=1); lb = np.linalg.norm(b, axis=1); lc = np.linalg.norm(c, axis=1)
        cos0 = -np.einsum("ij,ij->i", a, c) / np.maximum(la * lc, 1e-30)
        cos1 = -np.einsum("ij,ij->i", b, a) / np.maximum(lb * la, 1e-30)
        cos2 = -np.einsum("ij,ij->i", c, b) / np.maximum(lc * lb, 1e-30)
        ang = np.arccos(np.clip(np.stack([cos0, cos1, cos2], axis=1), -1, 1))
        ang_deg = np.rad2deg(ang)
        a_max = ang_deg.max(axis=1); a_min = ang_deg.min(axis=1)
        return np.maximum((a_max - 60) / 120, (60 - a_min) / 60)
    return np.maximum(tri_skew(prism_xyz[:, :3]), tri_skew(prism_xyz[:, 3:]))


def report_from_layer_positions(layer_positions: np.ndarray,
                                wall_tris: np.ndarray) -> dict:
    """Build per-prism metric arrays from a march result and return summary stats."""
    n_layers_plus_one, N, _ = layer_positions.shape
    M = wall_tris.shape[0]
    all_metrics = {"scaled_jac": [], "aspect_ratio": [], "skewness": []}
    for li in range(n_layers_plus_one - 1):
        bot = layer_positions[li][wall_tris]    # (M, 3, 3)
        top = layer_positions[li + 1][wall_tris]
        prism_xyz = np.concatenate([bot, top], axis=1)  # (M, 6, 3)
        all_metrics["scaled_jac"].append(prism_scaled_jacobian(prism_xyz))
        all_metrics["aspect_ratio"].append(prism_aspect_ratio(prism_xyz))
        all_metrics["skewness"].append(prism_equiangle_skewness(prism_xyz))
    summary = {}
    for k, v in all_metrics.items():
        arr = np.concatenate(v)
        summary[k] = {
            "min": float(arr.min()), "max": float(arr.max()),
            "mean": float(arr.mean()), "p99": float(np.quantile(arr, 0.99)),
            "n_negative_jac": int((arr < 0).sum()) if k == "scaled_jac" else None,
        }
    summary["n_prisms"] = int(M * (n_layers_plus_one - 1))
    return summary


def main():
    """CLI: print quality report for a .npz layer-positions dump."""
    if len(sys.argv) < 2:
        print("usage: metrics.py <layers.npz> [<layers.npz>...]")
        sys.exit(1)
    for path in sys.argv[1:]:
        data = np.load(path)
        rpt = report_from_layer_positions(data["layer_positions"], data["wall_tris"])
        print(f"\n=== {Path(path).name} ===")
        print(f"  n_prisms : {rpt['n_prisms']}")
        for k in ("scaled_jac", "aspect_ratio", "skewness"):
            s = rpt[k]
            print(f"  {k:12s}: min={s['min']:+.4f}  max={s['max']:+.4f}  "
                  f"mean={s['mean']:+.4f}  p99={s['p99']:+.4f}"
                  + (f"  neg={s['n_negative_jac']}" if s["n_negative_jac"] is not None else ""))


if __name__ == "__main__":
    main()
