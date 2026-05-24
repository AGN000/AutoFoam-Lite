"""Tet-cavity generation: fill the volume between the prism cap and a farfield
box with tetrahedra, then assemble a hybrid prism+tet mesh for export.

The prism column produced by build_hybrid_mesh() owns its nodes in Python.
For a CFD-runnable mesh we need an enclosing flow domain. We use a SECOND
gmsh session that:
  1. ingests the cap (top of the prism column) as a discrete surface entity
  2. builds CAD geometry for the farfield box around it
  3. defines a volume bounded by [farfield, cap] (cavity = box minus cap interior)
  4. meshes the cavity volume with tets

This requires the wall — and therefore the cap — to be a closed manifold (so
the cavity is a well-defined volume). For our benchmarks: NACA airfoil is
closed once side caps at z=0/z=span are added; flat plate / BFS need a more
involved construction (flat-cap perimeter must match farfield-side at the cap
height). For Day 8 we cover the closed-body case; the open-cap case is added
in Day 9 along with BC tagging.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import gmsh
import meshio


@dataclass
class FarfieldBox:
    bbox: tuple             # (xmin, ymin, zmin, xmax, ymax, zmax)
    lc: float               # mesh size on the farfield surfaces
    cap_match_lc: float | None = None
    """Target mesh size at the cap. When set, cavity tets near the cap are
    refined to roughly this scale (Distance + Threshold field on the cap
    surface), which closes the cap-to-tet face-size mismatch and reduces
    non-orthogonality at the prism-tet interface. Leave None to use the
    farfield lc everywhere (legacy behaviour)."""
    cap_match_dist: float | None = None
    """Distance over which the cap-side tet size grows from cap_match_lc up
    to lc. Defaults to 5x cap_match_lc when cap_match_lc is set."""


def _trace_polygon(edges: list[tuple[int, int]]) -> list[int]:
    """Order a list of undirected edges into a closed cyclic polygon."""
    if not edges:
        return []
    remaining = list(edges)
    a, b = remaining.pop(0)
    poly = [a, b]
    while remaining:
        last = poly[-1]
        for i, (u, v) in enumerate(remaining):
            if u == last:
                poly.append(v); remaining.pop(i); break
            if v == last:
                poly.append(u); remaining.pop(i); break
        else:
            raise RuntimeError("open polygon — boundary edges don't form a closed loop")
    if poly[0] == poly[-1]:
        poly = poly[:-1]
    return poly


def _point_in_polygon_2d(pt: np.ndarray, poly_xy: np.ndarray) -> bool:
    """Standard ray-cast point-in-polygon test in 2D."""
    n = len(poly_xy)
    inside = False
    j = n - 1
    px, py = float(pt[0]), float(pt[1])
    for i in range(n):
        xi, yi = poly_xy[i]
        xj, yj = poly_xy[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-30) + xi):
            inside = not inside
        j = i
    return inside


def close_cap_at_z_planes(cap_xyz: np.ndarray, cap_tris: np.ndarray,
                          z_tolerance: float = 1e-6) -> np.ndarray:
    """Close an open cap with triangulated z=zmin and z=zmax flat patches.

    Returns extended cap_tris with the closure tris appended. Closure tris
    have their normals pointing OUT of the prism column body (-z at zmin,
    +z at zmax) so the resulting closed manifold has consistent outward
    orientation.

    For non-convex profiles (e.g. NACA airfoil) we use scipy Delaunay then
    filter out tris whose centroids fall outside the boundary polygon —
    avoids spurious tris that bridge concave parts of the profile.
    """
    from scipy.spatial import Delaunay

    z_vals = cap_xyz[:, 2]
    z_min, z_max = float(z_vals.min()), float(z_vals.max())

    # Find boundary edges of the open cap (each shared by exactly 1 tri)
    edge_count: dict[tuple[int, int], int] = {}
    for tri in cap_tris:
        for u, v in ((int(tri[0]), int(tri[1])),
                      (int(tri[1]), int(tri[2])),
                      (int(tri[2]), int(tri[0]))):
            key = (min(u, v), max(u, v))
            edge_count[key] = edge_count.get(key, 0) + 1
    boundary_edges = [k for k, c in edge_count.items() if c == 1]

    edges_at_zmin: list[tuple[int, int]] = []
    edges_at_zmax: list[tuple[int, int]] = []
    for u, v in boundary_edges:
        zu, zv = z_vals[u], z_vals[v]
        if abs(zu - z_min) < z_tolerance and abs(zv - z_min) < z_tolerance:
            edges_at_zmin.append((u, v))
        elif abs(zu - z_max) < z_tolerance and abs(zv - z_max) < z_tolerance:
            edges_at_zmax.append((u, v))

    new_tris: list[list[int]] = []
    for edges, z_target, sign in [(edges_at_zmin, z_min, -1.0),
                                    (edges_at_zmax, z_max, +1.0)]:
        if not edges:
            continue
        poly = _trace_polygon(edges)
        xy = cap_xyz[poly, :2]
        tri = Delaunay(xy)
        for simplex in tri.simplices:
            cx = xy[simplex].mean(axis=0)
            if not _point_in_polygon_2d(cx, xy):
                continue
            global_tri = [poly[int(s)] for s in simplex]
            # Orient so RH normal aligns with `sign` * z_hat (outward from
            # prism column body): cross((p1-p0), (p2-p0)).z must have sign.
            v0 = cap_xyz[global_tri[0]]
            v1 = cap_xyz[global_tri[1]]
            v2 = cap_xyz[global_tri[2]]
            nz = (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v1[1] - v0[1]) * (v2[0] - v0[0])
            if nz * sign < 0:
                global_tri = [global_tri[0], global_tri[2], global_tri[1]]
            new_tris.append(global_tri)

    if not new_tris:
        return cap_tris
    return np.concatenate([cap_tris, np.array(new_tris, dtype=np.int64)], axis=0)


_CAP_DISCRETE_TAG = 9001     # explicit, well outside OCC default ranges
_CAP_NODE_TAG_OFFSET = 100000
_CAP_ELEM_TAG_OFFSET = 100000


def _add_cap_as_discrete_surface(cap_xyz: np.ndarray, cap_tris: np.ndarray) -> int:
    """Add the prism cap to the current gmsh model as a 2D discrete entity.

    Uses explicit non-overlapping tags so we coexist cleanly with OCC entities
    that gmsh assigns starting at 1.
    """
    surf_tag = gmsh.model.addDiscreteEntity(2, _CAP_DISCRETE_TAG)
    N = cap_xyz.shape[0]
    node_tags = list(range(_CAP_NODE_TAG_OFFSET + 1, _CAP_NODE_TAG_OFFSET + 1 + N))
    coords = cap_xyz.reshape(-1).tolist()
    gmsh.model.mesh.addNodes(2, surf_tag, node_tags, coords)
    M = cap_tris.shape[0]
    elem_tags = list(range(_CAP_ELEM_TAG_OFFSET + 1, _CAP_ELEM_TAG_OFFSET + 1 + M))
    # local indices -> our offset node tags
    elem_node_tags = (cap_tris.astype(np.int64) + _CAP_NODE_TAG_OFFSET + 1).reshape(-1).tolist()
    gmsh.model.mesh.addElementsByType(surf_tag, 2, elem_tags, elem_node_tags)
    return surf_tag


def _add_farfield_box(ff: FarfieldBox) -> tuple[list[int], dict[str, int]]:
    """Add the farfield box as 6 plane surfaces using the OCC kernel.

    Returns (list of farfield surface tags, name->surface_tag for BC tagging).
    """
    xmin, ymin, zmin, xmax, ymax, zmax = ff.bbox
    box_tag = gmsh.model.occ.addBox(
        xmin, ymin, zmin, xmax - xmin, ymax - ymin, zmax - zmin,
    )
    gmsh.model.occ.synchronize()
    # the box surfaces are tagged automatically; identify them by face center
    surfs = gmsh.model.getEntities(2)
    # box was just added — the only surfaces present are its 6 faces
    occ_surfs = [s for (d, s) in surfs if d == 2]
    name_to_tag: dict[str, int] = {}
    for s in occ_surfs:
        cx, cy, cz = gmsh.model.occ.getCenterOfMass(2, s)
        if abs(cx - xmin) < 1e-9:    name_to_tag["xmin"] = s
        elif abs(cx - xmax) < 1e-9:  name_to_tag["xmax"] = s
        elif abs(cy - ymin) < 1e-9:  name_to_tag["ymin"] = s
        elif abs(cy - ymax) < 1e-9:  name_to_tag["ymax"] = s
        elif abs(cz - zmin) < 1e-9:  name_to_tag["zmin"] = s
        elif abs(cz - zmax) < 1e-9:  name_to_tag["zmax"] = s
    return occ_surfs, name_to_tag


# Module-level state used so _add_farfield_box can filter out the discrete cap;
# set by tet_mesh_cavity_around_closed_body before adding the box.
_CAP_SURF_TAG: int | None = None


def tet_mesh_cavity_around_closed_body(
    cap_xyz: np.ndarray,
    cap_tris: np.ndarray,
    farfield: FarfieldBox,
    cap_winding_outward: bool = True,
):
    """Fresh-gmsh-session cavity mesher for a closed-body prism cap.

    Returns:
        cavity_nodes: (N_total, 3) all nodes in the cavity mesh — first len(cap_xyz)
                      are exactly the input cap nodes, the rest are added by gmsh
        cavity_tets: (n_tets, 4) tet connectivity into cavity_nodes
        farfield_surface_tris: (n_ff_tris, 3) farfield boundary tri connectivity
        farfield_named_tris: dict name -> (n_tris_on_face, 3) for BC tagging
    """
    global _CAP_SURF_TAG
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("cavity")
    try:
        # 1. farfield box first (OCC), so it gets the low surface tags
        ff_surfs, ff_named = _add_farfield_box(farfield)

        # 2. discrete cap surface with explicit out-of-range tags
        cap_surf = _add_cap_as_discrete_surface(cap_xyz, cap_tris)
        _CAP_SURF_TAG = cap_surf

        # 3. set mesh size on farfield surface boundaries (cap is fixed)
        for s in ff_surfs:
            for c in gmsh.model.getBoundary([(2, s)], oriented=False):
                gmsh.model.mesh.setSize([c], farfield.lc)

        # 4. drop the auto-created OCC volume (we'll define our cavity volume)
        all_vols = gmsh.model.getEntities(3)
        if all_vols:
            gmsh.model.removeEntities(all_vols, recursive=False)

        # 5. rebuild as: outer (box surfaces) + inner (cap) surface loops, then volume
        outer_loop = gmsh.model.geo.addSurfaceLoop(ff_surfs)
        inner_loop = gmsh.model.geo.addSurfaceLoop([cap_surf])
        cavity_vol = gmsh.model.geo.addVolume([outer_loop, inner_loop])
        gmsh.model.geo.synchronize()

        # 5b. Optional: distance-based refinement around the cap so cavity tets
        # near the cap match cap-face size — collapses the worst non-orth at
        # the prism-tet interface from ~88 deg to <70 deg in our tests.
        if farfield.cap_match_lc is not None:
            d_field = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(d_field, "SurfacesList", [cap_surf])
            gmsh.model.mesh.field.setNumber(d_field, "Sampling", 200)
            t_field = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(t_field, "InField", d_field)
            gmsh.model.mesh.field.setNumber(t_field, "SizeMin", farfield.cap_match_lc)
            gmsh.model.mesh.field.setNumber(t_field, "SizeMax", farfield.lc)
            gmsh.model.mesh.field.setNumber(t_field, "DistMin", 0.0)
            dist_max = farfield.cap_match_dist or 5.0 * farfield.cap_match_lc
            gmsh.model.mesh.field.setNumber(t_field, "DistMax", dist_max)
            gmsh.model.mesh.field.setAsBackgroundMesh(t_field)

        # 6. mesh: 2D first (so farfield faces get tris), then 3D for the cavity
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.generate(3)

        # 7. extract output
        all_node_tags, all_node_coords, _ = gmsh.model.mesh.getNodes()
        all_node_coords = np.asarray(all_node_coords).reshape(-1, 3)
        all_node_tags = np.asarray(all_node_tags, dtype=np.int64)
        tag_to_idx = {int(t): i for i, t in enumerate(all_node_tags)}

        # build cap_index_in_cavity: for each prism-cap-local index i (0..N-1),
        # query the cap entity for its current node tags (gmsh renumbered them
        # during meshing). The returned order matches our original insertion.
        cap_node_tags_now, _, _ = gmsh.model.mesh.getNodes(2, cap_surf)
        cap_node_tags_now = np.asarray(cap_node_tags_now, dtype=np.int64)
        N_cap = cap_xyz.shape[0]
        if cap_node_tags_now.size != N_cap:
            raise RuntimeError(
                f"cap entity has {cap_node_tags_now.size} nodes after mesh, "
                f"expected {N_cap}"
            )
        cap_indices = np.array(
            [tag_to_idx[int(t)] for t in cap_node_tags_now], dtype=np.int64
        )

        # tet elements (type 4 = 4-node tet)
        tet_tags, tet_node_tags = gmsh.model.mesh.getElementsByType(4)
        tet_node_tags = np.asarray(tet_node_tags, dtype=np.int64).reshape(-1, 4)
        cavity_tets = np.vectorize(tag_to_idx.__getitem__)(tet_node_tags)

        # farfield surface tris per named face (3-node tri = type 2)
        ff_named_tris: dict[str, np.ndarray] = {}
        for name, surf in ff_named.items():
            t_types, t_tags, t_node_tags = gmsh.model.mesh.getElements(2, surf)
            for et, ents in zip(t_types, t_node_tags):
                if et == 2:
                    arr = np.asarray(ents, dtype=np.int64).reshape(-1, 3)
                    arr_local = np.vectorize(tag_to_idx.__getitem__)(arr)
                    ff_named_tris[name] = arr_local

        return all_node_coords, cavity_tets, ff_named_tris, cap_indices
    finally:
        gmsh.finalize()
        _CAP_SURF_TAG = None


def assemble_hybrid_mesh(
    layer_positions: np.ndarray,
    wall_tris: np.ndarray,            # oriented outward
    cavity_nodes: np.ndarray,
    cavity_tets: np.ndarray,
    farfield_named_tris: dict[str, np.ndarray],
    cap_indices_in_cavity: np.ndarray,  # cavity row idx for each prism-cap-local i
    out_path: str,
):
    """Combine prism column + tet cavity into a single meshio.Mesh and write it.

    The cap nodes are duplicated between the prism column (last layer) and the
    cavity. We de-duplicate by remapping cavity rows that ARE cap rows back to
    the prism cap-row global indices, and assigning fresh global indices only
    to non-cap cavity rows.
    """
    n_layers_plus_one, N_wall, _ = layer_positions.shape
    n_layers = n_layers_plus_one - 1

    # 1. prism nodes: laid out as layer * N_wall + local
    prism_points = layer_positions.reshape(-1, 3)
    cap_row_global_offset = n_layers * N_wall  # index of layer N (cap) in prism_points
    n_prism_pts = prism_points.shape[0]

    # 2. sanity check: each cap_indices_in_cavity[i] should give cavity_nodes
    # row matching layer_positions[-1, i]
    cap_in_prism = layer_positions[-1]
    diff = np.linalg.norm(cavity_nodes[cap_indices_in_cavity] - cap_in_prism, axis=1).max()
    if diff > 1e-7:
        raise RuntimeError(f"cavity cap mismatch (max diff {diff:.3e})")

    # 3. build cavity_row -> global index map
    n_cav = cavity_nodes.shape[0]
    cav_to_global = np.full(n_cav, -1, dtype=np.int64)
    cap_set = set(int(c) for c in cap_indices_in_cavity)
    # cap rows -> prism cap row global index
    for local_i, cav_row in enumerate(cap_indices_in_cavity):
        cav_to_global[int(cav_row)] = cap_row_global_offset + local_i
    # non-cap rows get fresh global indices appended after prism points
    extra_pts = []
    cursor = n_prism_pts
    for cav_row in range(n_cav):
        if cav_row not in cap_set:
            cav_to_global[cav_row] = cursor
            extra_pts.append(cavity_nodes[cav_row])
            cursor += 1
    extra_pts_arr = np.array(extra_pts) if extra_pts else np.zeros((0, 3))
    points = np.concatenate([prism_points, extra_pts_arr], axis=0)

    def remap_cav(idx: np.ndarray) -> np.ndarray:
        return cav_to_global[idx]

    tets_global = remap_cav(cavity_tets.astype(np.int64))

    # 4. prism (wedge) elements
    def prism_idx(layer: int, local: np.ndarray) -> np.ndarray:
        return layer * N_wall + local
    wedges = np.empty((n_layers * wall_tris.shape[0], 6), dtype=np.int64)
    M = wall_tris.shape[0]
    for li in range(n_layers):
        bot = prism_idx(li, wall_tris)
        top = prism_idx(li + 1, wall_tris)
        wedges[li * M:(li + 1) * M] = np.hstack([bot, top])

    # 5. wall surface tris (for BC tagging "wall")
    wall_tris_global = wall_tris.copy()  # already indexes into prism_points layer 0

    # 6. farfield named tris — also need the same cavity-node remapping
    ff_cells = []
    for name, t in farfield_named_tris.items():
        if t.size == 0:
            continue
        ff_cells.append((name, remap_cav(t.astype(np.int64))))

    cells = [
        ("triangle", wall_tris_global),  # for tagging
        ("wedge", wedges),
        ("tetra", tets_global),
    ]
    cell_data = {"gmsh:physical": [
        np.full(wall_tris_global.shape[0], 1, dtype=np.int32),  # wall = 1
        np.full(wedges.shape[0], 100, dtype=np.int32),           # prism volume
        np.full(tets_global.shape[0], 100, dtype=np.int32),      # tet volume = same physical
    ]}
    # add farfield faces with their own physical IDs
    for k, (name, t) in enumerate(ff_cells):
        cells.append(("triangle", t))
        cell_data["gmsh:physical"].append(
            np.full(t.shape[0], 10 + k, dtype=np.int32)
        )

    mesh = meshio.Mesh(points=points, cells=cells, cell_data=cell_data)
    # also stash the BC name->physical-id mapping in field_data so gmshToFoam
    # picks it up
    field_data = {"wall": np.array([1, 2]), "fluid": np.array([100, 3])}
    for k, (name, _) in enumerate(ff_cells):
        field_data[name] = np.array([10 + k, 2])
    mesh.field_data = field_data
    mesh.write(out_path, file_format="gmsh22", binary=False)
    return mesh
