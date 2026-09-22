"""Viscous-inviscid interaction equivalent displacement transpiration velocity."""

from __future__ import annotations

import numpy as np

from .state import BoundaryLayerState


def compute_displacement_transpiration(
    upper_state: BoundaryLayerState,
    lower_state: BoundaryLayerState,
    geom: object,
) -> np.ndarray:
    """Compute equivalent panel normal transpiration velocity $v_{\\delta^*}(s)$.

    $$v_{\\delta^*} = \\frac{d(U_e \\delta^*)}{ds}$$

    Maps boundary-layer displacement thickness directly to the panel array shape $(N_{\\rm pan},)$.
    """
    vn_displacement = np.zeros(geom.num_pan, dtype=float)

    def _apply_transpiration(state: BoundaryLayerState) -> None:
        ue_delta = state.Ue * state.delta_star
        s_arr = state.s

        due_delta_ds = np.zeros_like(ue_delta)
        if len(s_arr) > 1:
            due_delta_ds[1:-1] = (ue_delta[2:] - ue_delta[:-2]) / (s_arr[2:] - s_arr[:-2])
            due_delta_ds[0] = (ue_delta[1] - ue_delta[0]) / (s_arr[1] - s_arr[0])
            due_delta_ds[-1] = (ue_delta[-1] - ue_delta[-2]) / (s_arr[-1] - s_arr[-2])

        # Assign transpiration values to target panels
        for i, station in enumerate(state.s):
            if i < len(state.s):
                pid = getattr(state, "panel_ids", None)
                if pid is not None and i < len(pid):
                    panel_idx = pid[i]
                    vn_displacement[panel_idx] = due_delta_ds[i]

    _apply_transpiration(upper_state)
    _apply_transpiration(lower_state)

    return vn_displacement