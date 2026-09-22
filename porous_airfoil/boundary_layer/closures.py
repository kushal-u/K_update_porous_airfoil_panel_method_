"""Drela/XFOIL boundary-layer closure relations for laminar and turbulent regimes."""

from __future__ import annotations

import numpy as np


def laminar_closures(Hk: float, Re_theta: float) -> tuple[float, float, float]:
    """Drela/XFOIL incompressible laminar closure relationships."""
    Hk = float(np.clip(Hk, 1.05, 10.0))
    Re_theta = float(np.clip(Re_theta, 1.0e-3, 1.0e8))

    # Energy shape parameter H*
    if Hk <= 4.0:
        H_star = 1.515 + 0.076 * (4.0 - Hk) ** 2 / Hk
    else:
        H_star = 1.515 + 0.015 * (Hk - 4.0) ** 2 / Hk

    # Skin friction coefficient Cf
    if Hk < 4.0:
        cf_factor = 0.0707 * (4.0 - Hk) ** 2.26 - 0.07
    else:
        cf_factor = 0.015 * (1.0 - 4.0 / Hk) ** 2 - 0.07

    Cf = (2.0 / Re_theta) * cf_factor

    # Dissipation parameter 2*CD / H*
    if Hk < 4.0:
        cD_factor = 0.00205 * (4.0 - Hk) ** 5.5 + 0.207
    else:
        cD_factor = 0.0007 * (Hk - 4.0) ** 2 / Hk + 0.207

    dissipation_rate = (2.0 / Re_theta) * cD_factor

    return float(H_star), float(Cf), float(dissipation_rate)


def turbulent_closures(
    Hk: float, Re_theta: float, ctau: float
) -> tuple[float, float, float]:
    """Drela/XFOIL incompressible turbulent closure relationships."""
    Hk = float(np.clip(Hk, 1.01, 10.0))
    Re_theta = float(np.clip(Re_theta, 10.0, 1.0e8))
    ctau = float(np.clip(ctau, 1.0e-8, 0.5))

    Hk0 = 1.505 + 4.0 / Re_theta

    # Energy shape parameter H*
    if Hk < Hk0:
        H_star = 1.505 + 4.0 / Re_theta + (0.165 - 1.6 / np.sqrt(Re_theta)) * (Hk0 - Hk) ** 1.6 / Hk
    else:
        denom = max(1.0e-4, (Hk - Hk0 + 4.0 / np.log(Re_theta)))
        H_star = (
            1.505
            + 4.0 / Re_theta
            + (Hk - Hk0) ** 2 * (0.04 / Hk + 0.007 * np.log(Re_theta) / (denom**2))
        )

    # Skin friction Cf (Ludwieg-Tillmann type fit)
    log_re = max(1.001, np.log10(Re_theta))
    power_exp = float(np.clip(1.74 + 0.31 * Hk, 1.0, 5.0))
    
    Cf = (
        0.3 * np.exp(-1.33 * Hk) / (log_re**power_exp)
        + 0.00011 * (np.tanh(4.0 - Hk / 0.875) - 1.0)
    )

    # Equilibrium shear stress parameter C_tau,eq
    us_ue = 0.5 * H_star * (1.0 - 4.0 / 3.0 * (Hk - 1.0) / (3.0 * Hk))
    ctau_eq = 0.015 / max(1.0e-4, Hk - 1.0) * us_ue**2

    return float(H_star), float(Cf), float(max(1.0e-8, ctau_eq))


def Hk_from_Hstar(H_star: float, turbulent: bool = False) -> float:
    """Invert H*(Hk) relation to extract kinematic shape parameter Hk."""
    H_star = float(H_star)
    if H_star <= 1.515:
        return 4.0 if not turbulent else 2.5

    if not turbulent:
        if H_star < 1.8:
            a = 0.076
            b = 1.515 - H_star
            roots = np.roots([a, b - 8.0 * a, 16.0 * a])
            valid = [r for r in roots if 1.0 <= r <= 4.0]
            return float(valid[0]) if valid else 2.6
        else:
            return float(min(10.0, 4.0 + np.sqrt((H_star - 1.515) / 0.015)))
    else:
        return float(min(10.0, 1.5 + np.sqrt(max(0.0, (H_star - 1.505) / 0.04))))