"""End-to-end driver: surface (gmsh) -> prism column (us) -> hybrid mesh.

For the feasibility prototype we keep the tet-cavity step minimal: the prism
cap is exported as a discrete surface that gmsh can tet-fill in a separate
pass, and we also emit the prism mesh standalone via meshio for inspection.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import gmsh

from .normals import smoothed_normals, dihedral_classify, laplacian_smooth, orient_tris_outward
from .collision import estimate_max_heights
from .advance import march_layers, layer_heights


def _orient_for_outward_march(tris: np.ndarray, nodes: np.ndarray,
                              normals: np.ndarray) -> np.ndarray:
    """Flip each triangle so its RH-rule normal points opposite the march dir."""
    v0 = nodes[tris[:, 0]]
    v1 = nodes[tris[:, 1]]
    v2 = nodes[tris[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    avg_normal = (normals[tris[:, 0]] + normals[tris[:, 1]] + normals[tris[:, 2]]) / 3.0
    align = np.einsum("ij,ij->i", cross, avg_normal)
    flipped = tris.copy()
    # Need bottom RH normal aligned WITH march for positive-volume wedge.
    mask = align < 0
    flipped[mask] = flipped[mask][:, [0, 2, 1]]
    return flipped


@dataclass
class BLConfig:
    first_height: float
    ratio: float = 1.2
    n_layers: int = 8
    smooth_sweeps: int = 2
    collision_safety: float = 0.4
    concave_clamp_frac: float = 0.3   # graceful-exit clamp for concave folds
    convex_clamp_frac: float = 0.5    # clamp at convex sharp ridges (e.g. TE)
                                       # to keep cap from over-stretching
    use_ridge_clamps: bool = True     # Phase 3 ablation: set False to disable
                                       # the per-class concave/convex clamps
                                       # while keeping smoothing and collision
    outer_ratio: float | None = None  # Outer-layer stretching for cap-tet
                                       # size matching. None = single ratio
                                       # everywhere (legacy).
    outer_layers: int = 0             # Count of layers using outer_ratio


def extract_wall_surface(wall_physical_tag: int):
    """Return (wall_node_tags, wall_xyz, wall_tris_local) from current gmsh model.

    wall_tris_local indexes into wall_xyz (0..N-1), not gmsh tags. wall_node_tags
    gives the gmsh tag at each local index.
    """
    surfaces = gmsh.model.getEntitiesForPhysicalGroup(2, wall_physical_tag)

    # Pull all nodes used by the wall surfaces in bulk
    raw_tris: list[np.ndarray] = []
    used_tags_set: set[int] = set()
    for surf in surfaces:
        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(2, surf)
        for et, ents in zip(elem_types, elem_node_tags):
            if et != 2:
                continue
            arr = np.asarray(ents, dtype=np.int64).reshape(-1, 3)
            raw_tris.append(arr)
            used_tags_set.update(arr.flatten().tolist())
    if not raw_tris:
        raise RuntimeError(f"no triangles found on wall physical tag {wall_physical_tag}")
    raw_tris_arr = np.concatenate(raw_tris, axis=0)

    used_tags = np.array(sorted(used_tags_set), dtype=np.int64)
    # Bulk fetch coordinates for these tags
    xyz = np.empty((len(used_tags), 3), dtype=float)
    for i, t in enumerate(used_tags):
        coord, _, _, _ = gmsh.model.mesh.getNode(int(t))
        xyz[i] = coord[:3]
    # Build tag -> local index map
    tag_to_local = {int(t): i for i, t in enumerate(used_tags)}
    tris_local = np.vectorize(tag_to_local.__getitem__)(raw_tris_arr)
    return used_tags, xyz, tris_local.astype(np.int64)


def build_hybrid_mesh(wall_physical_tag: int, cfg: BLConfig):
    """Run full BL pipeline against the current gmsh model. Assumes surface mesh
    has been generated. Returns (layer_positions, wall_tris_local, wall_node_tags,
    normals, ridge_tags) — all the data the benchmark scripts need to inspect or
    export the result.
    """
    wall_node_tags, wall_xyz, wall_tris = extract_wall_surface(wall_physical_tag)

    # Enforce a consistent outward winding on the surface tris before computing
    # normals — gmsh's plane-surface meshes use whatever winding the curve loop
    # implies, which is inconsistent for closed bodies built from many panels
    # (e.g. an airfoil skin). Centroid heuristic works for star-shaped surfaces.
    wall_tris = orient_tris_outward(wall_xyz, wall_tris)

    # Area-weighted normals already encode local smoothing. Avoid an additional
    # Laplacian pass over the surface graph — at sharp ridges (e.g. an airfoil
    # trailing edge) such smoothing drags normals across the ridge toward the
    # bisector and can rotate nearby normals INTO the wall. Robustness for
    # ridge regions is delegated to (a) ridge classification + per-node
    # height clamps and (b) Laplacian smoothing of displacements during march.
    normals = smoothed_normals(wall_xyz, wall_tris)
    ridge_tags = dihedral_classify(wall_xyz, wall_tris)

    desired_total = float(layer_heights(
        cfg.first_height, cfg.ratio, cfg.n_layers,
        outer_ratio=cfg.outer_ratio, outer_layers=cfg.outer_layers,
    )[-1])
    max_h = estimate_max_heights(
        wall_xyz, normals, desired_total,
        tris=wall_tris, safety=cfg.collision_safety,
    )
    # Graceful-exit clamp at concave ridges: nodes where adjacent triangles fold
    # inward toward the fluid volume have neighbouring nodes whose marches
    # converge. Reducing max-height here prevents head-on collision.
    from .normals import CONCAVE_SHARP, CONVEX_SHARP
    if cfg.use_ridge_clamps:
        concave_mask = (ridge_tags == CONCAVE_SHARP)
        convex_mask = (ridge_tags == CONVEX_SHARP)
        if concave_mask.any():
            max_h = np.where(
                concave_mask, np.minimum(max_h, cfg.concave_clamp_frac * desired_total), max_h
            )
        if convex_mask.any():
            max_h = np.where(
                convex_mask, np.minimum(max_h, cfg.convex_clamp_frac * desired_total), max_h
            )

    layer_positions = march_layers(
        wall_xyz, wall_tris, normals,
        first_height=cfg.first_height,
        ratio=cfg.ratio,
        n_layers=cfg.n_layers,
        max_height_per_node=max_h,
        smooth_sweeps=cfg.smooth_sweeps,
        outer_ratio=cfg.outer_ratio,
        outer_layers=cfg.outer_layers,
    )
    # Orient wall tris so the prism base (bottom) winding has RH normal pointing
    # *into* the wall (opposite march direction). This yields positive-Jacobian
    # wedges with both meshio and the gmsh prism element type.
    wall_tris = _orient_for_outward_march(wall_tris, wall_xyz, normals)
    return {
        "layer_positions": layer_positions,
        "wall_tris": wall_tris,
        "wall_node_tags": wall_node_tags,
        "normals": normals,
        "ridge_tags": ridge_tags,
        "max_height": max_h,
    }


def to_meshio_prism_only(result: dict):
    """Build a meshio Mesh containing just the prism column (for inspection)."""
    import meshio
    lp = result["layer_positions"]
    tris = result["wall_tris"]
    n_layers_plus_one, N, _ = lp.shape
    n_layers = n_layers_plus_one - 1
    points = lp.reshape(-1, 3)
    # node index in flat array: layer * N + local
    def gidx(layer: int, local: np.ndarray) -> np.ndarray:
        return layer * N + local
    cells = []
    for li in range(n_layers):
        bot = gidx(li, tris)
        top = gidx(li + 1, tris)
        # meshio "wedge" expects (n0,n1,n2,n3,n4,n5) with bottom then top
        wedge = np.hstack([bot, top])
        cells.append(("wedge", wedge))
    # also include the wall surface tris for visualization
    cells.append(("triangle", tris))
    return meshio.Mesh(points=points, cells=cells)
