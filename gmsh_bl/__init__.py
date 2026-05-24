"""gmsh_bl — robust advancing-layer prism mesher as a gmsh Python plugin.

Pipeline: surface (gmsh) -> prism column (us) -> tet cavity (gmsh) -> hybrid mesh.
"""
from .normals import smoothed_normals, dihedral_classify
from .advance import march_layers
from .prism_mesh import emit_prism_elements
from .pipeline import build_hybrid_mesh

__all__ = [
    "smoothed_normals",
    "dihedral_classify",
    "march_layers",
    "emit_prism_elements",
    "build_hybrid_mesh",
]
