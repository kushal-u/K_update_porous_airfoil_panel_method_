"""
solver.py

Core aerodynamic solver and geometry utilities.

This file provides:
- flow, airfoil, reference, and coupling configuration dataclasses
- NACA 4-digit airfoil generation
- panel-geometry construction
- a source-vortex panel method solver
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la
from matplotlib.path import Path as MplPath

# ---------------------------------------------------------------------------
# Optional Numba acceleration for the velocity-field kernel.
# When numba is available the inner double loop (field-points x panels) is
# JIT-compiled and parallelised with prange.  When numba is absent the pure-
# NumPy fallback is used automatically â€” identical maths, no change required
# by the caller.
# ---------------------------------------------------------------------------
try:
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False


def _make_velocity_kernel():
    """
    Return the appropriate velocity-field kernel.

    Kernel signature:
        fn(xp, yp, XB0, YB0, S, cos_phi, sin_phi, lam, gamma,
           u_inf, v_inf)  ->  (u, v)   [1-D float64 arrays, len == len(xp)]

    Methodology: source + vortex.
    The outer loop is over panels k; the inner loop accumulates onto every
    exterior field point.  In the numba path, prange parallelises over field
    points so each thread owns an independent slice and touches a disjoint
    portion of u/v â€” no race condition.
    """
    if _NUMBA_AVAILABLE:
        @njit(parallel=True, cache=True, fastmath=True)
        def _kernel(xp, yp, XB0, YB0, S, cos_phi, sin_phi, lam, gamma,
                    u_inf, v_inf):
            """Numba JIT + parallel kernel â€” Clement SPM/VPM methodology."""
            npts   = len(xp)
            npan   = len(S)
            inv_2pi = 1.0 / (2.0 * np.pi)

            u = np.empty(npts)
            v = np.empty(npts)

            # prange: each thread handles a contiguous slice of field points.
            # The panel loop inside is read-only on shared arrays â†’ no races.
            for ip in prange(npts):          # pylint: disable=not-an-iterable
                xpi = xp[ip]
                ypi = yp[ip]
                ui  = u_inf
                vi  = v_inf

                for k in range(npan):
                    dx = xpi - XB0[k]
                    dy = ypi - YB0[k]
                    cp = cos_phi[k]
                    sp = sin_phi[k]
                    sk = S[k]

                    A  = -(dx * cp + dy * sp)
                    B  = dx * dx + dy * dy
                    E2 = B - A * A
                    if E2 < 1e-14:
                        E2 = 1e-14
                    E = E2 ** 0.5

                    log_arg = (sk * sk + 2.0 * A * sk + B) / B
                    if log_arg < 1e-14:
                        log_arg = 1e-14
                    log_term  = np.log(log_arg)
                    atan_term = np.arctan2(sk + A, E) - np.arctan2(A, E)

                    # SOURCE kernel â€” Clement's STREAMLINE_SPM
                    # Cx = -cos(phi),  Dx = dx   (XP - XB[j])
                    # Cy = -sin(phi),  Dy = dy   (YP - YB[j])
                    Mx = 0.5 * (-cp) * log_term + ((dx - A * (-cp)) / E) * atan_term
                    My = 0.5 * (-sp) * log_term + ((dy - A * (-sp)) / E) * atan_term

                    # VORTEX kernel â€” Clement's STREAMLINE_VPM
                    # Cx =  sin(phi),  Dx = -dy  (-(YP - YB[j]))
                    # Cy = -cos(phi),  Dy =  dx  (  XP - XB[j] )
                    Nx = 0.5 *   sp  * log_term + ((-dy - A *   sp ) / E) * atan_term
                    Ny = 0.5 * (-cp) * log_term + (( dx - A * (-cp)) / E) * atan_term

                    ui += (lam[k] * Mx - gamma * Nx) * inv_2pi
                    vi += (lam[k] * My - gamma * Ny) * inv_2pi

                u[ip] = ui
                v[ip] = vi

            return u, v

    else:
        # ------------------------------------------------------------------
        # Pure-NumPy fallback â€” vectorised over field points for each panel.
        # Identical maths to the numba path; used when numba is not installed.
        # ------------------------------------------------------------------
        def _kernel(xp, yp, XB0, YB0, S, cos_phi, sin_phi, lam, gamma,
                    u_inf, v_inf):
            """NumPy fallback kernel â€” Clement SPM/VPM methodology."""
            inv_2pi = 1.0 / (2.0 * np.pi)
            u = np.full(len(xp), u_inf)
            v = np.full(len(yp), v_inf)

            for k in range(len(S)):
                dx = xp - XB0[k]
                dy = yp - YB0[k]
                cp = cos_phi[k]
                sp = sin_phi[k]
                sk = S[k]

                A  = -(dx * cp + dy * sp)
                B  = dx * dx + dy * dy
                E  = np.sqrt(np.maximum(B - A * A, 1e-14))

                log_term  = np.log((sk * sk + 2.0 * A * sk + B) / B)
                atan_term = np.arctan2(sk + A, E) - np.arctan2(A, E)

                # SOURCE kernel â€” Clement's STREAMLINE_SPM
                Mx = 0.5 * (-cp) * log_term + ((dx - A * (-cp)) / E) * atan_term
                My = 0.5 * (-sp) * log_term + ((dy - A * (-sp)) / E) * atan_term

                # VORTEX kernel â€” Clement's STREAMLINE_VPM
                Nx = 0.5 *   sp  * log_term + ((-dy - A *   sp ) / E) * atan_term
                Ny = 0.5 * (-cp) * log_term + (( dx - A * (-cp)) / E) * atan_term

                u += (lam[k] * Mx - gamma * Nx) * inv_2pi
                v += (lam[k] * My - gamma * Ny) * inv_2pi

            return u, v

    return _kernel


# Module-level singleton â€” compiled once on first import (numba path) or
# defined once as a plain function (numpy fallback).
_velocity_kernel = _make_velocity_kernel()


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class FlowConfig:
    """Freestream and fluid properties."""

    aoa_deg: float = 4.0
    v_inf: float = 30.0
    rho_inf: float = 1.225
    mu_inf: float = 1.8e-5
    p_inf: float = 101325.0


@dataclass
class AirfoilConfig:
    """Airfoil geometry settings."""

    airfoil_name: str = "0018"
    chord: float = 1.0
    n_panels: int = 220


@dataclass
class ReferenceConfig:
    """Aerodynamic reference point in the same length units as the geometry."""

    x_ref: float = 0.25
    y_ref: float = 0.0


@dataclass
class CouplingConfig:
    """Nonlinear aero-hydraulic coupling settings.

    ``max_iter`` limits nonlinear residual evaluations. The coupled solver
    does not use under-relaxation or user-defined convergence tolerances.
    """

    max_iter: int = 10000

def generate_naca4(airfoil_str, n_points=100): # or whatever the parameter name is
    code = str(code).lower().replace("naca", "").replace(" ", "").replace("-", "").strip()
# =============================================================================
# GEOMETRY HELPERS
# =============================================================================
def generate_naca4(
    airfoil_name: str,
    n_panels: int,
    chord: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a closed NACA 4-digit airfoil with cosine spacing.

    Parameters
    ----------
    airfoil_name
        NACA 4-digit identifier.
    n_panels
        Number of panels on the closed boundary.
    chord
        Physical chord length.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Closed airfoil boundary coordinates `(XB, YB)`.
    """
    # Sanitize 'airfoil_name' into 4-digit 'code'
    code = str(airfoil_name).lower().replace("naca", "").replace(" ", "").replace("-", "").strip()
    
    # Validate the sanitized 'code', not 'airfoil_name'
    if len(code) != 4 or not code.isdigit():
        raise ValueError("Only NACA 4-digit airfoils are supported.")
    if n_panels < 10:
        raise ValueError("n_panels must be at least 10.")
    if chord <= 0.0:
        raise ValueError("chord must be positive.")

    m = int(airfoil_name[0]) / 100.0
    p = int(airfoil_name[1]) / 10.0
    t = int(airfoil_name[2:]) / 100.0

    beta = np.linspace(0.0, np.pi, n_panels // 2 + 1)
    x = 0.5 * (1.0 - np.cos(beta))

    yt = 5.0 * t * (
        0.2969 * np.sqrt(np.maximum(x, 1e-14))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1036 * x**4
    )

    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)

    if m > 0.0 and p > 0.0:
        i1 = x <= p
        i2 = x > p

        yc[i1] = m / p**2 * (2.0 * p * x[i1] - x[i1] ** 2)
        yc[i2] = m / (1.0 - p) ** 2 * ((1.0 - 2.0 * p) + 2.0 * p * x[i2] - x[i2] ** 2)

        dyc_dx[i1] = 2.0 * m / p**2 * (p - x[i1])
        dyc_dx[i2] = 2.0 * m / (1.0 - p) ** 2 * (p - x[i2])

    theta = np.arctan(dyc_dx)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # Force an exactly closed trailing edge.
    xu[-1], yu[-1] = 1.0, 0.0
    xl[-1], yl[-1] = 1.0, 0.0

    XB = np.concatenate([xu[::-1], xl[1:]]) * chord
    YB = np.concatenate([yu[::-1], yl[1:]]) * chord

    if abs(XB[0] - XB[-1]) > 1e-14 or abs(YB[0] - YB[-1]) > 1e-14:
        XB = np.append(XB, XB[0])
        YB = np.append(YB, YB[0])

    return XB, YB


def ensure_clockwise_boundary(XB: np.ndarray, YB: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ensure the airfoil boundary is ordered clockwise."""
    num_pan = len(XB) - 1
    edge = np.zeros(num_pan)
    for i in range(num_pan):
        edge[i] = (XB[i + 1] - XB[i]) * (YB[i + 1] + YB[i])

    if np.sum(edge) < 0.0:
        XB = np.flipud(XB)
        YB = np.flipud(YB)

    return XB, YB


def _collapse_duplicate_x(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse repeated x-values by averaging their y-values."""
    idx = np.argsort(x)
    x_sorted = x[idx]
    y_sorted = y[idx]

    xu, inv = np.unique(x_sorted, return_inverse=True)
    yu = np.zeros_like(xu, dtype=float)
    cnt = np.zeros_like(xu, dtype=float)

    for i, g in enumerate(inv):
        yu[g] += y_sorted[i]
        cnt[g] += 1.0

    yu /= np.maximum(cnt, 1.0)
    return xu, yu


@dataclass
class SurfacePoint:
    """Surface point associated with a mapped pore location."""

    x: float
    y: float
    nx: float
    ny: float
    tx: float
    ty: float
    panel_id: int


class PanelGeometry:
    """Store airfoil panel geometry and derived quantities."""

    def __init__(self, XB: np.ndarray, YB: np.ndarray, aoa_deg: float):
        XB, YB = ensure_clockwise_boundary(np.asarray(XB, dtype=float), np.asarray(YB, dtype=float))
        self.XB = XB
        self.YB = YB

        self.aoa_deg = float(aoa_deg)
        self.alpha = np.deg2rad(self.aoa_deg)

        self.num_pts = len(XB)
        self.num_pan = self.num_pts - 1

        self.x_le = float(np.min(XB))
        self.x_te = float(np.max(XB))
        self.chord = float(self.x_te - self.x_le)
        if self.chord <= 0.0:
            raise ValueError("Invalid geometry: chord must be positive.")

        self.XC = np.zeros(self.num_pan)
        self.YC = np.zeros(self.num_pan)
        self.S = np.zeros(self.num_pan)
        self.phi = np.zeros(self.num_pan)
        self.delta = np.zeros(self.num_pan)
        self.beta = np.zeros(self.num_pan)

        self.tx = np.zeros(self.num_pan)
        self.ty = np.zeros(self.num_pan)
        self.nx = np.zeros(self.num_pan)
        self.ny = np.zeros(self.num_pan)

        for i in range(self.num_pan):
            x1, y1 = XB[i], YB[i]
            x2, y2 = XB[i + 1], YB[i + 1]

            self.XC[i] = 0.5 * (x1 + x2)
            self.YC[i] = 0.5 * (y1 + y2)

            dx = x2 - x1
            dy = y2 - y1
            self.S[i] = np.hypot(dx, dy)

            phi_i = np.arctan2(dy, dx)
            if phi_i < 0.0:
                phi_i += 2.0 * np.pi
            self.phi[i] = phi_i

            self.delta[i] = phi_i + np.pi / 2.0

            beta_i = self.delta[i] - self.alpha
            while beta_i < 0.0:
                beta_i += 2.0 * np.pi
            while beta_i >= 2.0 * np.pi:
                beta_i -= 2.0 * np.pi
            self.beta[i] = beta_i

            self.tx[i] = np.cos(phi_i)
            self.ty[i] = np.sin(phi_i)
            self.nx[i] = np.cos(self.delta[i])
            self.ny[i] = np.sin(self.delta[i])

        self.cos_phi = np.cos(self.phi)
        self.sin_phi = np.sin(self.phi)
        self.cos_beta = np.cos(self.beta)
        self.sin_beta = np.sin(self.beta)

        self._prepare_surface_interpolants()

    def x_from_fraction(self, x_frac: float) -> float:
        """Convert chord fraction x/c into absolute x-coordinate."""
        return self.x_le + float(x_frac) * self.chord

    def local_thickness_at_x(self, x_query: float) -> float:
        """Compute local airfoil thickness at an absolute x-coordinate."""
        x_min = max(self.upper_x.min(), self.lower_x.min())
        x_max = min(self.upper_x.max(), self.lower_x.max())
        xq = float(np.clip(x_query, x_min, x_max))

        yu = np.interp(xq, self.upper_x, self.upper_y)
        yl = np.interp(xq, self.lower_x, self.lower_y)
        return abs(yu - yl)

    def local_thickness_at_fraction(self, x_frac: float) -> float:
        """Compute local airfoil thickness at chord fraction x/c."""
        return self.local_thickness_at_x(self.x_from_fraction(x_frac))

    def _prepare_surface_interpolants(self) -> None:
        """Build upper/lower surface interpolants and robust panel-side maps."""
        le_idx = int(np.argmin(self.XB))

        surf1_x = self.XB[: le_idx + 1][::-1]
        surf1_y = self.YB[: le_idx + 1][::-1]

        surf2_x = self.XB[le_idx:-1]
        surf2_y = self.YB[le_idx:-1]

        s1x, s1y = _collapse_duplicate_x(surf1_x, surf1_y)
        s2x, s2y = _collapse_duplicate_x(surf2_x, surf2_y)

        if np.mean(s1y) >= np.mean(s2y):
            self.upper_x, self.upper_y = s1x, s1y
            self.lower_x, self.lower_y = s2x, s2y
        else:
            self.upper_x, self.upper_y = s2x, s2y
            self.lower_x, self.lower_y = s1x, s1y

        upper_ids: list[int] = []
        lower_ids: list[int] = []

        for i, (xc, yc) in enumerate(zip(self.XC, self.YC)):
            x_min = max(self.upper_x.min(), self.lower_x.min())
            x_max = min(self.upper_x.max(), self.lower_x.max())
            x_clip = float(np.clip(xc, x_min, x_max))

            yu = float(np.interp(x_clip, self.upper_x, self.upper_y))
            yl = float(np.interp(x_clip, self.lower_x, self.lower_y))

            if abs(yc - yu) <= abs(yc - yl):
                upper_ids.append(i)
            else:
                lower_ids.append(i)

        self.upper_panel_ids = np.asarray(upper_ids, dtype=int)
        self.lower_panel_ids = np.asarray(lower_ids, dtype=int)

    def surface_point(self, x_query: float, side: str) -> SurfacePoint:
        """Map an absolute x-location and side to a surface point.

        The returned ``x`` and ``y`` coordinates are the interpolated surface
        coordinates, not the nearest panel-control-point coordinates.  The
        nearest panel id is still returned and is used to sample pressure,
        normals, and tangents.  Keeping the interpolated coordinates is
        important for geometry exports because a requested vertical or
        horizontal porous channel should remain visually vertical or horizontal
        instead of being slightly skewed by panel-centre snapping.
        """
        side = side.lower()
        if side not in ("upper", "lower"):
            raise ValueError("side must be 'upper' or 'lower'.")

        x_min = max(self.upper_x.min(), self.lower_x.min())
        x_max = min(self.upper_x.max(), self.lower_x.max())
        x_surface = float(np.clip(x_query, x_min, x_max))

        if side == "upper":
            y_surface = float(np.interp(x_surface, self.upper_x, self.upper_y))
            candidate_ids = self.upper_panel_ids
        else:
            y_surface = float(np.interp(x_surface, self.lower_x, self.lower_y))
            candidate_ids = self.lower_panel_ids

        if candidate_ids.size == 0:
            raise RuntimeError(f"No candidate panels found for side '{side}'.")

        dx = self.XC[candidate_ids] - x_surface
        dy = self.YC[candidate_ids] - y_surface
        panel_id = int(candidate_ids[np.argmin(dx * dx + dy * dy)])

        return SurfacePoint(
            x=x_surface,
            y=y_surface,
            nx=float(self.nx[panel_id]),
            ny=float(self.ny[panel_id]),
            tx=float(self.tx[panel_id]),
            ty=float(self.ty[panel_id]),
            panel_id=panel_id,
        )

    def surface_point_from_fraction(self, x_frac: float, side: str) -> SurfacePoint:
        """Map chord fraction x/c and side to the nearest panel control point."""
        return self.surface_point(self.x_from_fraction(x_frac), side)


# =============================================================================
# PANEL METHOD
# =============================================================================
def compute_IJKL_vectorized(geom: PanelGeometry) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute source and vortex influence matrices."""
    N = geom.num_pan
    eps = 1e-14

    XC_i = geom.XC[:, None]
    YC_i = geom.YC[:, None]
    phi_i = geom.phi[:, None]

    XB_j = geom.XB[:-1][None, :]
    YB_j = geom.YB[:-1][None, :]
    phi_j = geom.phi[None, :]
    S_j = geom.S[None, :]

    cos_phi_i = np.cos(phi_i)
    sin_phi_i = np.sin(phi_i)
    cos_phi_j = np.cos(phi_j)
    sin_phi_j = np.sin(phi_j)

    dX = XC_i - XB_j
    dY = YC_i - YB_j

    A = -(dX * cos_phi_j + dY * sin_phi_j)
    B = dX**2 + dY**2

    E2 = np.maximum(B - A**2, eps)
    E = np.sqrt(E2)

    log_term = np.log((S_j**2 + 2.0 * A * S_j + B) / B)
    atan_term = np.arctan2(S_j + A, E) - np.arctan2(A, E)

    diag = np.eye(N, dtype=bool)
    log_term[diag] = 0.0
    atan_term[diag] = 0.0
    E[diag] = 1.0

    dphi = phi_i - phi_j

    Cn_s = np.sin(dphi)
    Dn_s = -(dX * sin_phi_i) + (dY * cos_phi_i)
    Ct_s = -np.cos(dphi)
    Dt_s = (dX * cos_phi_i) + (dY * sin_phi_i)

    I = 0.5 * Cn_s * log_term + ((Dn_s - A * Cn_s) / E) * atan_term
    J = 0.5 * Ct_s * log_term + ((Dt_s - A * Ct_s) / E) * atan_term

    Cn_v = -np.cos(dphi)
    Dn_v = (dX * cos_phi_i) + (dY * sin_phi_i)
    Ct_v = np.sin(phi_j - phi_i)
    Dt_v = (dX * sin_phi_i) - (dY * cos_phi_i)

    K = 0.5 * Cn_v * log_term + ((Dn_v - A * Cn_v) / E) * atan_term
    L = 0.5 * Ct_v * log_term + ((Dt_v - A * Ct_v) / E) * atan_term

    np.fill_diagonal(I, 0.0)
    np.fill_diagonal(J, 0.0)
    np.fill_diagonal(K, 0.0)
    np.fill_diagonal(L, 0.0)

    return I, J, K, L


@dataclass
class SPVPResult:
    """Aerodynamic result from the source-vortex panel method."""

    lambda_src: np.ndarray
    gamma: float
    Vt: np.ndarray
    Cp: np.ndarray
    Cx: float
    Cy: float
    CL: float
    CD: float
    CM: float
    CL_kj: float
    source_sum: float
    system_residual_norm: float
    kutta_residual: float
    normal_bc_residual_max: float


class SourceVortexPanelMethod:
    """Constant-source + constant-vortex panel solver."""

    def __init__(self, geom: PanelGeometry, flow: FlowConfig, ref: ReferenceConfig):
        self.geom = geom
        self.flow = flow
        self.ref = ref

        self.I, self.J, self.K, self.L = compute_IJKL_vectorized(geom)
        self.A = self._build_system_matrix()
        self.lu_A, self.piv_A = la.lu_factor(self.A)

    def _build_system_matrix(self) -> np.ndarray:
        """Assemble the aerodynamic system matrix."""
        N = self.geom.num_pan
        A = np.zeros((N + 1, N + 1))

        A[:N, :N] = self.I
        np.fill_diagonal(A[:N, :N], np.pi)
        A[:N, N] = -np.sum(self.K, axis=1)

        A[N, :N] = self.J[0, :] + self.J[N - 1, :]
        A[N, N] = -np.sum(self.L[0, :] + self.L[N - 1, :]) + 2.0 * np.pi
        return A

    def _build_rhs(self, normal_transpiration: np.ndarray | None = None) -> np.ndarray:
        """Assemble the right-hand side including prescribed normal transpiration."""
        N = self.geom.num_pan
        Vinf = self.flow.v_inf

        vn = np.zeros(N) if normal_transpiration is None else np.asarray(normal_transpiration, dtype=float)
        if vn.shape != (N,):
            raise ValueError(f"normal_transpiration must have shape ({N},).")

        b = np.zeros(N + 1)
        b[:N] = -2.0 * np.pi * Vinf * self.geom.cos_beta + 2.0 * np.pi * vn
        b[N] = -2.0 * np.pi * Vinf * (self.geom.sin_beta[0] + self.geom.sin_beta[-1])
        return b
        
    def solve(
        self,
        normal_transpiration: np.ndarray | None = None,
    ) -> SPVPResult:
        """Solve the aerodynamic problem."""
        N = self.geom.num_pan
        Vinf = self.flow.v_inf
        c = self.geom.chord

        if Vinf <= 0.0:
            raise ValueError("Freestream velocity must be positive.")
        if c <= 0.0:
            raise ValueError("Reference chord must be positive.")

        # Prescribed normal velocity at every panel.
        # Solid airfoil: vn = 0
        # Porous airfoil: vn is supplied by the porous coupling.
        if normal_transpiration is None:
            vn = np.zeros(N, dtype=float)
        else:
            vn = np.asarray(normal_transpiration, dtype=float)

            if vn.shape != (N,):
                raise ValueError(
                    f"normal_transpiration must have shape ({N},), "
                    f"but received {vn.shape}."
                )

            if not np.all(np.isfinite(vn)):
                raise ValueError(
                    "normal_transpiration contains NaN or infinite values."
                )

        b = self._build_rhs(normal_transpiration=vn)
        x = la.lu_solve((self.lu_A, self.piv_A), b)

        lam = x[:-1]
        gamma = x[-1]

        Vt = np.zeros(N, dtype=float)
        Cp = np.zeros(N, dtype=float)

        for i in range(N):
            Vt[i] = (
                Vinf * self.geom.sin_beta[i]
                + (1.0 / (2.0 * np.pi))
                * np.sum(lam * self.J[i, :])
                + gamma / 2.0
                - (gamma / (2.0 * np.pi))
                * np.sum(self.L[i, :])
            )

            # Local surface velocity magnitude:
            # |V|^2 = Vt^2 + vn^2
            Cp[i] = 1.0 - (
                Vt[i] ** 2 + vn[i] ** 2
            ) / Vinf**2

        dCx = -Cp * self.geom.nx * self.geom.S / c
        dCy = -Cp * self.geom.ny * self.geom.S / c

        Cx = float(np.sum(dCx))
        Cy = float(np.sum(dCy))

        alpha = np.deg2rad(self.flow.aoa_deg)

        CL = Cy * np.cos(alpha) - Cx * np.sin(alpha)
        CD = Cx * np.cos(alpha) + Cy * np.sin(alpha)

        xbar = self.geom.XC - self.ref.x_ref
        ybar = self.geom.YC - self.ref.y_ref

        dCM = -(xbar * dCy - ybar * dCx) / c
        CM = float(np.sum(dCM))

        circulation = gamma * np.sum(self.geom.S)
        CL_kj = float(2.0 * circulation / (Vinf * c))

        system_residual = self.A @ x - b

        return SPVPResult(
            lambda_src=lam,
            gamma=float(gamma),
            Vt=Vt,
            Cp=Cp,
            Cx=Cx,
            Cy=Cy,
            CL=float(CL),
            CD=float(CD),
            CM=float(CM),
            CL_kj=CL_kj,
            source_sum=float(np.sum(lam * self.geom.S)),
            system_residual_norm=float(
                np.linalg.norm(system_residual)
            ),
            kutta_residual=float(system_residual[-1]),
            normal_bc_residual_max=float(
                np.max(np.abs(system_residual[:-1]))
            ),
        )

    def panel_pressures(self, result: SPVPResult) -> np.ndarray:
        """Convert pressure coefficient to absolute panel pressure."""
        q_inf = 0.5 * self.flow.rho_inf * self.flow.v_inf**2
        return self.flow.p_inf + q_inf * result.Cp

    def build_inside_mask(self, XX: np.ndarray, YY: np.ndarray) -> np.ndarray:
        """Build an inside-body mask for plotting grids."""
        poly = MplPath(np.column_stack([self.geom.XB, self.geom.YB]))
        pts = np.column_stack([XX.ravel(), YY.ravel()])
        return poly.contains_points(pts).reshape(XX.shape)

    def velocity_field(
        self,
        XX: np.ndarray,
        YY: np.ndarray,
        result: SPVPResult,
        inside: np.ndarray | None = None,
    ):
        """
        Compute velocity and pressure-coefficient fields for streamline plotting.

        Kernels follow Clement's STREAMLINE_SPM (source) and STREAMLINE_VPM
        (vortex) exactly.  The heavy double loop (field-points x panels) is
        delegated to the module-level ``_velocity_kernel``, which is:

        * **numba** ``@njit(parallel=True)`` when numba is installed â€” the
          field-point loop is parallelised with ``prange`` and the inner
          panel loop is JIT-compiled with ``fastmath``.
        * a **pure-NumPy** fallback (vectorised over field-points for each
          panel) when numba is absent.

        Source kernel (Clement STREAMLINE_SPM):
            Cx = -cos(phi_j),  Dx = XP - XB_j
            Cy = -sin(phi_j),  Dy = YP - YB_j
            Mx = 0.5*Cx*log + ((Dx - A*Cx)/E)*atan
            My = 0.5*Cy*log + ((Dy - A*Cy)/E)*atan

        Vortex kernel (Clement STREAMLINE_VPM):
            Cx =  sin(phi_j),  Dx = -(YP - YB_j)
            Cy = -cos(phi_j),  Dy =  (XP - XB_j)
            Nx = 0.5*Cx*log + ((Dx - A*Cx)/E)*atan
            Ny = 0.5*Cy*log + ((Dy - A*Cy)/E)*atan

        Total:  u += (1/2pi) * sum_j [ lam_j*Mx_j - gamma*Nx_j ]
                v += (1/2pi) * sum_j [ lam_j*My_j - gamma*Ny_j ]
        """
        if inside is None:
            inside = self.build_inside_mask(XX, YY)

        u_inf = self.flow.v_inf * np.cos(np.deg2rad(self.flow.aoa_deg))
        v_inf = self.flow.v_inf * np.sin(np.deg2rad(self.flow.aoa_deg))

        inside_flat = inside.ravel()
        valid_mask  = ~inside_flat

        xp_valid = np.ascontiguousarray(XX.ravel()[valid_mask], dtype=np.float64)
        yp_valid = np.ascontiguousarray(YY.ravel()[valid_mask], dtype=np.float64)

        # All panel geometry arrays are already float64 from PanelGeometry.
        u_valid, v_valid = _velocity_kernel(
            xp_valid, yp_valid,
            np.ascontiguousarray(self.geom.XB[:-1], dtype=np.float64),
            np.ascontiguousarray(self.geom.YB[:-1], dtype=np.float64),
            np.ascontiguousarray(self.geom.S,       dtype=np.float64),
            np.ascontiguousarray(self.geom.cos_phi, dtype=np.float64),
            np.ascontiguousarray(self.geom.sin_phi, dtype=np.float64),
            np.ascontiguousarray(result.lambda_src, dtype=np.float64),
            float(result.gamma),
            float(u_inf),
            float(v_inf),
        )

        U = np.full(XX.size, np.nan)
        V = np.full(YY.size, np.nan)
        U[valid_mask] = u_valid
        V[valid_mask] = v_valid
        U = U.reshape(XX.shape)
        V = V.reshape(YY.shape)

        Vmag = np.sqrt(U**2 + V**2)
        Cp_field = 1.0 - (Vmag / self.flow.v_inf) ** 2
        Cp_field[inside] = np.nan

        return U, V, Cp_field, inside