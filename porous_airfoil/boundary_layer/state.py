"""Boundary layer state data containers."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class BLStation:
    """Boundary-layer station geometry and external velocity data."""

    panel_id: int
    x: float
    y: float
    s: float
    Ue: float
    Cp: float
    dUe_ds: float


@dataclass
class BoundaryLayerState:
    """Complete boundary-layer solution vector along a marching path."""

    side: str
    s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    Ue: np.ndarray
    dUe_ds: np.ndarray
    Cp: np.ndarray

    theta: np.ndarray
    delta_star: np.ndarray
    H: np.ndarray
    H_star: np.ndarray
    Cf: np.ndarray
    Re_theta: np.ndarray
    n_factor: np.ndarray
    ctau: np.ndarray
    turbulent: np.ndarray

    xtr: float = float("nan")
    str_val: float = float("nan")
    transition_index: int = -1

    def __len__(self) -> int:
        return len(self.s)