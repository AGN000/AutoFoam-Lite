"""Emit prism (wedge) elements from marched layer positions into a gmsh model.

gmsh element type 6 = 6-node prism (wedge). Node ordering: bottom triangle
(n0,n1,n2) followed by top triangle (n3,n4,n5) where ni and ni+3 are
connected. The orientation of the bottom must match the wall surface tris so
the prism has a positive Jacobian when marching outward.
"""
from __future__ import annotations

import numpy as np
import gmsh

PRISM_TYPE = 6  # 6-node first-order wedge


def _flip_for_outward_march(tris: np.ndarray, nodes: np.ndarray,
                            normals: np.ndarray) -> np.ndarray:
    """Ensure each tri's winding has its right-hand normal aligned with march dir.

    For a positive-Jacobian prism, the bottom triangle's outward normal (by
    right-hand rule on its winding) must point opposite the march direction
    (i.e. into the wall). If aligned with the march, flip the winding.
    """
    v0 = nodes[tris[:, 0]]
    v1 = nodes[tris[:, 1]]
    v2 = nodes[tris[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    avg_normal = (normals[tris[:, 0]] + normals[tris[:, 1]] + normals[tris[:, 2]]) / 3.0
    align = np.einsum("ij,ij->i", cross, avg_normal)
    flipped = tris.copy()
    mask = align < 0  # need RH normal aligned with march for +Jac wedge
    flipped[mask] = flipped[mask][:, [0, 2, 1]]
    return flipped


def emit_prism_elements(
    layer_positions: np.ndarray,   # (n_layers+1, N, 3)
    tris: np.ndarray,              # (M, 3) wall surface tris
    normals: np.ndarray,           # (N, 3) for orientation check
    volume_tag: int,
):
    """Add nodes for layers 1..N and prism elements to gmsh model.

    layer_positions[0] are wall nodes already in the gmsh model — we do NOT
    re-add them. We add nodes for layers 1..N with fresh tags and build prism
    elements between consecutive layers.

    Returns:
        cap_nodes_xyz: (N, 3) coordinates of the top layer (for tet-cavity step)
        cap_tris: (M, 3) triangle node tags on the cap surface (for gmsh discrete)
    """
    n_layers_plus_one, N, _ = layer_positions.shape
    n_layers = n_layers_plus_one - 1
    M = tris.shape[0]

    # Get current max node tag so we don't collide
    all_nodes = gmsh.model.mesh.getNodes()
    existing_tags = all_nodes[0]
    max_tag = int(existing_tags.max()) if len(existing_tags) > 0 else 0

    # Map: surface-local index (0..N-1) -> wall node tag in gmsh.
    # Caller passes wall nodes such that layer_positions[0,i] corresponds to
    # the gmsh node at wall_node_tags[i]. We require this mapping below.
    # For now, we expect the caller to have populated `_wall_tags` on the array.
    if not hasattr(layer_positions, "_wall_tags"):
        raise ValueError("layer_positions must carry attr `_wall_tags` (N,)")
    wall_tags: np.ndarray = layer_positions._wall_tags  # type: ignore[attr-defined]

    # Allocate new node tags for layers 1..n_layers
    new_tags = np.arange(max_tag + 1, max_tag + 1 + n_layers * N).reshape(n_layers, N)

    # Add new nodes to the volume entity (so gmsh sees them as belonging there)
    for li in range(n_layers):
        coords = layer_positions[li + 1].reshape(-1)
        gmsh.model.mesh.addNodes(3, volume_tag, new_tags[li].tolist(), coords.tolist())

    # Build prism elements for each layer slab
    fixed_tris = _flip_for_outward_march(tris, layer_positions[0], normals)
    # Convert tri indices (0..N-1) into per-layer node tag arrays
    layer_node_tags = np.vstack([wall_tags[None, :], new_tags])  # (n_layers+1, N)

    prism_node_tags_per_layer: list[np.ndarray] = []
    elem_tag_cursor = _next_element_tag()
    for li in range(n_layers):
        bot_tags = layer_node_tags[li][fixed_tris]      # (M, 3)
        top_tags = layer_node_tags[li + 1][fixed_tris]  # (M, 3)
        prism_nodes = np.hstack([bot_tags, top_tags]).astype(np.uint64).reshape(-1)
        prism_node_tags_per_layer.append(prism_nodes)
        elem_tags = np.arange(elem_tag_cursor, elem_tag_cursor + M, dtype=np.uint64)
        elem_tag_cursor += M
        gmsh.model.mesh.addElementsByType(
            volume_tag, PRISM_TYPE, elem_tags.tolist(), prism_nodes.tolist()
        )

    cap_xyz = layer_positions[-1]
    cap_tris_with_tags = layer_node_tags[-1][fixed_tris]
    return cap_xyz, cap_tris_with_tags, layer_node_tags[-1]


def _next_element_tag() -> int:
    """Find a safe starting element tag past anything currently in the model."""
    types, tags_per_type, _ = gmsh.model.mesh.getElements()
    max_tag = 0
    for tags in tags_per_type:
        if len(tags) > 0:
            m = int(np.max(tags))
            if m > max_tag:
                max_tag = m
    return max_tag + 1
