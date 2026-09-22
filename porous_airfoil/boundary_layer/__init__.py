"""Boundary-layer solver package for viscous-inviscid interaction."""

from .state import BLStation, BoundaryLayerState
from .surface import BLSurfacePath, build_bl_surface_paths, find_stagnation_point
from .closures import laminar_closures, turbulent_closures, Hk_from_Hstar
from .transition import critical_reynolds_theta, dN_ds
from .wake import compute_squire_young_drag
from .solver import solve_boundary_layer
from .coupling import compute_displacement_transpiration

__all__ = [
    "BLStation",
    "BoundaryLayerState",
    "BLSurfacePath",
    "build_bl_surface_paths",
    "find_stagnation_point",
    "laminar_closures",
    "turbulent_closures",
    "Hk_from_Hstar",
    "critical_reynolds_theta",
    "dN_ds",
    "compute_squire_young_drag",
    "solve_boundary_layer",
    "compute_displacement_transpiration",
]