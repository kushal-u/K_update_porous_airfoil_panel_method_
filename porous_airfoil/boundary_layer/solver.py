"""Boundary-layer integration solver along surface paths."""

from __future__ import annotations

import numpy as np

from .state import BoundaryLayerState
from .surface import BLSurfacePath, build_bl_surface_paths
from .closures import laminar_closures, turbulent_closures, Hk_from_Hstar
from .transition import dN_ds
from .wake import compute_squire_young_drag


def get_derivatives(
    theta: float, 
    H_star: float, 
    ctau: float, 
    Ue: float, 
    due_ds: float, 
    nu: float, 
    is_turbulent: bool,
    v_inf: float
) -> tuple[float, float, float, float]:
    """Calculate spatial derivatives for the two-equation integral boundary layer."""
    theta = max(1.0e-8, theta)
    
    # 1) Forward Stagnation Bypass: 
    # Avoid the 1/Ue singularity. Hold initial analytical stagnation values constant 
    # until the flow safely accelerates away from the exact stagnation point.
    if Ue < 0.02 * v_inf and due_ds > 0.0:
        return 0.0, 0.0, 0.0, 0.0
        
    Ue = max(1.0e-4, Ue)
    
    if not is_turbulent:
        Hk = Hk_from_Hstar(H_star, turbulent=False)
        Hk = float(np.clip(Hk, 1.05, 8.0))
        Re_th = max(1.0e-3, Ue * theta / nu)
        Hs_eq, Cf, cD2 = laminar_closures(Hk, Re_th)
        
        # Dampen Cf in severely separated regions to prevent wild feedback
        if Hk > 4.0:
            Cf = 0.0
            
        dtheta_ds = 0.5 * Cf - (2.0 + Hk) * (theta / Ue) * due_ds
        dHs_ds = (H_star / theta) * (cD2 - 0.5 * Cf + (Hk - 1.0) * (theta / Ue) * due_ds)
        dctau_ds = 0.0
        
        dRe_ds = (Ue * dtheta_ds + theta * due_ds) / nu
        dn_ds = dN_ds(Hk, Re_th, dRe_ds)
        
    else:
        Hk = Hk_from_Hstar(H_star, turbulent=True)
        Hk = float(np.clip(Hk, 1.01, 8.0))
        Re_th = max(10.0, Ue * theta / nu)
        ctau_safe = max(1.0e-8, ctau)
        Hs_eq, Cf, ctau_eq = turbulent_closures(Hk, Re_th, ctau_safe)
        
        if Hk > 3.0:
            Cf = max(Cf, 0.0)
            
        us_ue = 0.5 * H_star * (1.0 - 4.0 / 3.0 * (Hk - 1.0) / (3.0 * Hk))
        cD = 0.5 * Cf * us_ue + ctau_safe * (1.0 - us_ue)
        cD2 = 2.0 * cD / H_star
        
        dtheta_ds = 0.5 * Cf - (2.0 + Hk) * (theta / Ue) * due_ds
        dHs_ds = (H_star / theta) * (cD2 - 0.5 * Cf + (Hk - 1.0) * (theta / Ue) * due_ds)
        dctau_ds = (5.6 / max(1.0e-8, 3.0 * theta)) * (np.sqrt(ctau_eq) - np.sqrt(ctau_safe))
        dn_ds = 0.0
        
    # 2) Rear Stagnation Clamps:
    # A fixed inviscid trailing-edge stagnation point imposes an infinite 
    # adverse pressure gradient, which breaks RK2 without bounds. We clamp
    # the gradients generously to allow physical thickening, but prevent blow-up.
    dtheta_ds = float(np.clip(dtheta_ds, -0.2, 0.2))
    dHs_ds = float(np.clip(dHs_ds, -5.0, 5.0))
    dctau_ds = float(np.clip(dctau_ds, -10.0, 10.0))
    
    return dtheta_ds, dHs_ds, dctau_ds, dn_ds


def solve_boundary_layer(
    geom: object,
    aero_result: object,
    flow: object,
    N_crit: float = 9.0,
) -> tuple[BoundaryLayerState, BoundaryLayerState, float]:
    """Solve two-equation integral boundary layer on upper and lower surfaces."""
    nu = float(flow.mu_inf / flow.rho_inf)
    v_inf = float(flow.v_inf)
    chord = float(geom.chord)

    upper_path, lower_path = build_bl_surface_paths(geom, aero_result)

    def _march_path(path: BLSurfacePath) -> BoundaryLayerState:
        n = len(path.stations)

        s_arr = path.s
        x_arr = path.x
        y_arr = path.y
        ue_arr = path.Ue
        due_ds_arr = path.dUe_ds
        cp_arr = path.Cp

        theta = np.zeros(n, dtype=float)
        delta_star = np.zeros(n, dtype=float)
        H = np.zeros(n, dtype=float)
        H_star = np.zeros(n, dtype=float)
        Cf = np.zeros(n, dtype=float)
        Re_theta = np.zeros(n, dtype=float)
        n_factor = np.zeros(n, dtype=float)
        ctau = np.zeros(n, dtype=float)
        turbulent = np.zeros(n, dtype=bool)

        # Stagnation Initial Condition (Exact for Hiemenz flow)
        due_ds_0 = max(1.0e-3, due_ds_arr[0])
        theta[0] = np.sqrt(0.075 * nu / due_ds_0)
        H[0] = 2.61
        delta_star[0] = H[0] * theta[0]
        Re_theta[0] = max(1.0e-3, theta[0] * ue_arr[0] / nu)
        H_star[0], Cf[0], _ = laminar_closures(H[0], Re_theta[0])

        xtr, str_val = float("nan"), float("nan")
        trans_idx = -1
        is_turbulent = False

        for i in range(n - 1):
            ds = s_arr[i + 1] - s_arr[i]
            if ds <= 1.0e-12:
                theta[i + 1] = theta[i]
                delta_star[i + 1] = delta_star[i]
                H[i + 1] = H[i]
                H_star[i + 1] = H_star[i]
                Cf[i + 1] = Cf[i]
                Re_theta[i + 1] = Re_theta[i]
                n_factor[i + 1] = n_factor[i]
                ctau[i + 1] = ctau[i]
                turbulent[i + 1] = is_turbulent
                continue

            # Runge-Kutta 2nd Order (Heun's Method)
            dth1, dHs1, dctau1, dn1 = get_derivatives(
                theta[i], H_star[i], ctau[i], ue_arr[i], due_ds_arr[i], nu, is_turbulent, v_inf
            )
            
            # Predictor Step
            th_p = theta[i] + dth1 * ds
            Hs_p = H_star[i] + dHs1 * ds
            ctau_p = ctau[i] + dctau1 * ds
            
            # Domain safety bounds for corrector evaluation
            th_p = max(1.0e-8, min(th_p, 0.2 * chord))
            Hs_p = max(1.5, min(Hs_p, 4.0))
            ctau_p = max(1.0e-8, min(ctau_p, 0.2))

            dth2, dHs2, dctau2, dn2 = get_derivatives(
                th_p, Hs_p, ctau_p, ue_arr[i + 1], due_ds_arr[i + 1], nu, is_turbulent, v_inf
            )
            
            # Corrector Step
            theta_next = theta[i] + 0.5 * ds * (dth1 + dth2)
            Hs_next = H_star[i] + 0.5 * ds * (dHs1 + dHs2)
            ctau_next = ctau[i] + 0.5 * ds * (dctau1 + dctau2)
            n_next = n_factor[i] + 0.5 * ds * (dn1 + dn2)

            # Final step bounds to prevent domain errors
            theta_next = max(1.0e-8, min(theta_next, 0.2 * chord))
            Hs_next = max(1.5, min(Hs_next, 4.0))
            ctau_next = max(1.0e-8, min(ctau_next, 0.2))
            
            # Transition Trigger
            if not is_turbulent and n_next >= N_crit:
                is_turbulent = True
                xtr = float(x_arr[i + 1])
                str_val = float(s_arr[i + 1])
                trans_idx = i + 1
                ctau_next = 0.03

            # Apply new state
            theta[i + 1] = theta_next
            H_star[i + 1] = Hs_next
            ctau[i + 1] = ctau_next
            n_factor[i + 1] = n_next
            
            Hk_next = Hk_from_Hstar(Hs_next, turbulent=is_turbulent)
            H[i + 1] = Hk_next
            delta_star[i + 1] = Hk_next * theta_next
            Re_theta[i + 1] = max(1.0e-3, ue_arr[i + 1] * theta_next / nu)
            turbulent[i + 1] = is_turbulent

            if is_turbulent:
                _, Cf[i + 1], _ = turbulent_closures(Hk_next, Re_theta[i + 1], ctau_next)
            else:
                _, Cf[i + 1], _ = laminar_closures(Hk_next, Re_theta[i + 1])

        return BoundaryLayerState(
            side=path.side,
            s=s_arr,
            x=x_arr,
            y=y_arr,
            Ue=ue_arr,
            dUe_ds=due_ds_arr,
            Cp=cp_arr,
            theta=theta,
            delta_star=delta_star,
            H=H,
            H_star=H_star,
            Cf=Cf,
            Re_theta=Re_theta,
            n_factor=n_factor,
            ctau=ctau,
            turbulent=turbulent,
            xtr=xtr,
            str_val=str_val,
            transition_index=trans_idx,
        )

    upper_state = _march_path(upper_path)
    lower_state = _march_path(lower_path)

    Cd_profile = compute_squire_young_drag(upper_state, lower_state, v_inf, chord)

    return upper_state, lower_state, Cd_profile