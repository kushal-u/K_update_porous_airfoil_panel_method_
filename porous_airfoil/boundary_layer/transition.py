"""Drela $e^N$ spatial envelope transition criterion."""

from __future__ import annotations

import numpy as np


def critical_reynolds_theta(Hk: float) -> float:
    """Critical momentum Reynolds number $Re_{\\theta,crit}$ as a function of $H_k$."""
    Hk = float(np.clip(Hk, 1.05, 10.0))
    if Hk <= 1.05:
        return 1.0e6

    term1 = (1.415 / (Hk - 1.0) - 0.489) * np.tanh(20.0 / (Hk - 1.0) - 12.9)
    term2 = 3.29 / (Hk - 1.0) + 0.215
    log_re_crit = float(np.clip(term1 + term2, 0.0, 8.0))

    return float(10.0**log_re_crit)


def dN_ds(Hk: float, Re_theta: float, dRe_theta_ds: float) -> float:
    """Spatial growth rate of the amplification factor $N$."""
    Hk = float(Hk)
    Re_theta = float(Re_theta)

    Re_crit = critical_reynolds_theta(Hk)
    if Re_theta < Re_crit:
        return 0.0

    term = 2.4 * Hk - 3.7 + 2.5 * np.tanh(1.5 * Hk - 4.65)
    dN_dRe = 0.01 * np.sqrt(term**2 + 0.25)

    dRe_ds_clamped = float(np.clip(dRe_theta_ds, -1.0e6, 1.0e6))
    return float(max(0.0, dN_dRe * dRe_ds_clamped))