"""Wake boundary-layer extension and Squire-Young profile drag evaluation."""

from __future__ import annotations

import numpy as np
from .state import BoundaryLayerState


def compute_squire_young_drag(
    upper_state: BoundaryLayerState,
    lower_state: BoundaryLayerState,
    v_inf: float,
    chord: float,
) -> float:
    """Compute profile drag coefficient $C_D$ using the Squire-Young relation."""
    v_inf = max(1.0e-6, float(v_inf))
    chord = max(1.0e-6, float(chord))

    def _get_extrapolated_te(state: BoundaryLayerState) -> tuple[float, float, float]:
        """Extract BL parameters avoiding the unphysical inviscid TE stagnation."""
        n = len(state.x)
        idx = n - 1
        
        # Step back to ~98% chord where Ue is physical and attached
        for i in range(n - 1, 0, -1):
            if state.x[i] <= 0.98 * chord:
                idx = i
                break
        
        dx = chord - state.x[idx]
        dtheta_dx = 0.0
        if idx > 0:
            ds_step = state.x[idx] - state.x[idx-1]
            if ds_step > 1e-8:
                dtheta_dx = (state.theta[idx] - state.theta[idx-1]) / ds_step
                
        # Extrapolate theta to the geometric trailing edge
        theta_te = state.theta[idx] + dtheta_dx * dx
        delta_star_te = state.delta_star[idx]
        Ue_te = state.Ue[idx]  # Use the healthy Ue, not the crashed TE Ue
        
        return max(1.0e-8, theta_te), max(1.0e-8, delta_star_te), max(1.0e-4, Ue_te)

    th_u, ds_u, ue_u = _get_extrapolated_te(upper_state)
    th_l, ds_l, ue_l = _get_extrapolated_te(lower_state)

    theta_te = th_u + th_l
    delta_star_te = ds_u + ds_l
    Ue_te = 0.5 * (ue_u + ue_l)

    H_te = delta_star_te / max(1.0e-12, theta_te)

    ratio = Ue_te / v_inf
    exponent = 0.5 * (H_te + 5.0)

    Cd_profile = 2.0 * (theta_te / chord) * (ratio**exponent)
    return float(Cd_profile)