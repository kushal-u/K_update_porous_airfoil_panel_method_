from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, root
from .solver import (
    CouplingConfig,
    FlowConfig,
    PanelGeometry,
    ReferenceConfig,
    SourceVortexPanelMethod,
    SurfacePoint,
    generate_naca4,
)
from .plotting.aerodynamic import (
    plot_airfoil_with_porous_network,
    plot_aoa_sweep_comparison,
    plot_cp_distribution_with_xfoil,
    plot_pressure_contours_comparison,
    plot_velocity_contours_comparison,
)
from .integrations.xfoil import (
    load_xfoil_cp,
    load_xfoil_polar_dataframe,
    run_xfoil_cp,
    run_xfoil_polar_sweep,
)
from .config import (
    AIRFOIL_NAME,
    AOA_DEG,
    AOA_SWEEP_END_DEG,
    AOA_SWEEP_START_DEG,
    AOA_SWEEP_STEP_DEG,
    AOA_SWEEP_CHANNEL_HEIGHTS,
    CHORD,
    COUPLING,
    EXPORT_AOA_SWEEP_PARAVIEW,
    MAKE_ZOOMED_CONTOUR_PLOTS,
    ZOOMED_CONTOUR_XLIM,
    ZOOMED_CONTOUR_YLIM,
    GeometryLimits,
    Chamber,
    Pore,
    PassageState,
    PassageSpec,
    PorousModelSpec,
    MultiPassageResult,
    P_INF,
    REYNOLDS_EXTERNAL,
    RHO_INF,
    MU_INF,
    SPAN_WIDTH,
    USE_XFOIL,
    XFOIL_EXE_PATH,
    XFOIL_MACH,
    XFOIL_OUTPUT_ROOT,
    XFOIL_TIMEOUT,
    rectangular_laminar_resistance,
    reynolds_to_velocity,
)

from .integrations.paraview import export_aoa_sweep_collections, export_paraview_files
# =============================================================================
# INTERNAL CHANNEL HYDRAULIC RESISTANCE
# =============================================================================
INTERNAL_RE_TRANSITION = 2300.0

# Reduced interface model, not a resolved jet/mixing calculation.
# Inlet direction follows the static-pressure difference. Inlet tangential
# kinetic energy is dissipated (zero recovery). The separate bulk outlet jet
# matches exterior static pressure and carries alpha_j*rho*(q'/H)^2/2.
# Set "legacy_static" only for comparison with old results.
CHANNEL_ENERGY_CLOSURE = "zero_tangential_recovery_jet"
JET_KINETIC_ENERGY_FACTOR = 1.0  # uniform bulk jet; not a fitted loss coefficient

# -----------------------------------------------------------------------------
# NONLINEAR CONTINUATION SETTINGS
# -----------------------------------------------------------------------------
# These settings change only the numerical path used to reach the requested
# physical state.  Every reported solution is still checked at the requested
# AoA, Reynolds number and full coupling strength lambda=1.
ADAPTIVE_AOA_MIN_STEP_DEG = 0.05
RE_CONTINUATION_REFERENCE = 5.0e5
RE_CONTINUATION_INITIAL_DIVISIONS = 5
RE_CONTINUATION_MIN_STEP = 1.0e4
HOMOTOPY_INITIAL_STEP = 0.20
HOMOTOPY_MIN_STEP = 0.025

# Branch-boundary refinement.  These tolerances control only how accurately a
# loss-of-root boundary is bracketed; they do not relax the nonlinear root
# acceptance criterion used for physical results.
AOA_BOUNDARY_TOL_DEG = 0.01
RE_BOUNDARY_TOL = 2.0e3
HOMOTOPY_BOUNDARY_TOL = 2.0e-3

# Pseudo-arclength continuation settings.  The arclength state contains the
# dimensionless passage-flow vector plus alpha/PAL_ALPHA_SCALE_DEG.
PAL_ALPHA_SCALE_DEG = 1.0
PAL_INITIAL_DS = 0.05
PAL_MIN_DS = 0.005
PAL_MAX_DS = 0.20
PAL_MAX_STEPS = 50
PAL_CORRECTOR_TOL = 1.0e-8
PAL_TARGET_TOL_DEG = 0.02
PAL_ALPHA_LIMIT_DEG = 30.0



def _smooth_transition_weight(
    reynolds: float,
    re_low: float = 2000.0,
    re_high: float = 3000.0,
) -> float:
    """Return a C1 smooth transition weight from laminar to turbulent flow."""
    if reynolds <= re_low:
        return 0.0
    if reynolds >= re_high:
        return 1.0
    xi = (reynolds - re_low) / (re_high - re_low)
    return float(xi * xi * (3.0 - 2.0 * xi))


def _rectangular_2d_pressure_drop_abs(
    mean_velocity: float,
    rho: float,
    mu: float,
    length: float,
    height: float,
) -> tuple[float, float, float, str]:
    """Pressure drop for a 2D parallel-wall channel at a given mean velocity.

    The hydraulic diameter is ``D_h = 2H``. The laminar Darcy factor
    ``f = 96/Re_Dh`` reproduces exactly ``Delta p = 12 mu L U/H^2`` and hence
    ``R' = 12 mu L/H^3``. A smooth blend to Blasius is used from Re=2000 to
    Re=3000 so the pressure-flow law remains continuous during coupling.
    """
    u = abs(float(mean_velocity))
    if u <= 0.0:
        return 0.0, 0.0, float("inf"), "zero_flow"

    hydraulic_diameter = 2.0 * float(height)
    reynolds = rho * u * hydraulic_diameter / max(mu, 1.0e-30)

    f_laminar = 96.0 / max(reynolds, 1.0e-30)
    f_turbulent = 0.3164 / max(reynolds, 1.0e-30) ** 0.25
    blend = _smooth_transition_weight(reynolds)
    friction_factor = (1.0 - blend) * f_laminar + blend * f_turbulent

    pressure_drop = (
        friction_factor
        * (length / hydraulic_diameter)
        * 0.5
        * rho
        * u**2
    )

    if reynolds <= 2000.0:
        model_name = "laminar_2d_rectangular_12muL_over_H3"
    elif reynolds < 3000.0:
        model_name = "transition_2d_rectangular_smooth_blend"
    else:
        model_name = "turbulent_2d_rectangular_blasius"

    return (
        float(pressure_drop),
        float(reynolds),
        float(friction_factor),
        model_name,
    )


def solve_rectangular_channel_flow(
    dp: float,
    rho: float,
    mu: float,
    length: float,
    height: float,
    outlet_kinetic_factor: float = 0.0,
) -> tuple[float, float, float, float, str]:
    """Solve the 2D rectangular-channel flow rate per unit span.

    Returns
    -------
    tuple
        ``q_prime, R_prime_equivalent, Re_Dh, f_D, model_name``.

    Notes
    -----
    ``q_prime`` has units m^2/s. The flow-through area per unit span is H,
    therefore ``U_bar = q_prime/H``. The visible in-plane area ``L*H`` is not
    used as the flow-through area.
    When outlet_kinetic_factor > 0, dp is the available inlet-to-outlet
    head in pressure units, not just the distributed friction drop. The
    scalar equation includes outlet_kinetic_factor*rho*U_bar**2/2. This
    kinetic energy leaves with the jet; it is not a friction loss.
    """
    dp = float(dp)
    outlet_kinetic_factor = float(outlet_kinetic_factor)
    if not np.isfinite(outlet_kinetic_factor) or outlet_kinetic_factor < 0.0:
        raise ValueError("outlet_kinetic_factor must be finite and nonnegative.")
    rho = float(rho)
    mu = float(mu)
    length = float(length)
    height = float(height)

    if not all(np.isfinite(value) for value in (dp, rho, mu, length, height)):
        raise FloatingPointError("Non-finite rectangular-channel input.")
    if rho <= 0.0 or mu <= 0.0 or length <= 0.0 or height <= 0.0:
        raise ValueError("Density, viscosity, length and channel height must be positive.")

    dp_abs = abs(dp)
    if dp_abs <= 1.0e-30:
        return 0.0, float("inf"), 0.0, float("inf"), "zero_flow"

    sign = 1.0 if dp >= 0.0 else -1.0

    # Bracket and solve the monotonic pressure-drop law in mean velocity.
    u_low = 0.0
    u_high = 1.0
    for _ in range(100):
        dp_high, _, _, _ = _rectangular_2d_pressure_drop_abs(
            mean_velocity=u_high,
            rho=rho,
            mu=mu,
            length=length,
            height=height,
        )
        dp_high += outlet_kinetic_factor * 0.5 * rho * u_high**2
        if dp_high >= dp_abs:
            break
        u_high *= 2.0
    else:
        raise RuntimeError("Could not bracket the rectangular-channel flow solution.")

    for _ in range(80):
        u_mid = 0.5 * (u_low + u_high)
        dp_mid, _, _, _ = _rectangular_2d_pressure_drop_abs(
            mean_velocity=u_mid,
            rho=rho,
            mu=mu,
            length=length,
            height=height,
        )
        dp_mid += outlet_kinetic_factor * 0.5 * rho * u_mid**2
        if dp_mid < dp_abs:
            u_low = u_mid
        else:
            u_high = u_mid

    mean_velocity = 0.5 * (u_low + u_high)
    _, reynolds, friction_factor, model_name = _rectangular_2d_pressure_drop_abs(
        mean_velocity=mean_velocity,
        rho=rho,
        mu=mu,
        length=length,
        height=height,
    )

    q_prime_abs = mean_velocity * height
    if outlet_kinetic_factor > 0.0:
        model_name += "+outlet_jet_kinetic_head"
    equivalent_resistance = dp_abs / max(q_prime_abs, 1.0e-30)

    if not all(
        np.isfinite(value)
        for value in (q_prime_abs, equivalent_resistance, reynolds, friction_factor)
    ):
        raise FloatingPointError("Non-finite rectangular-channel result.")

    return (
        float(sign * q_prime_abs),
        float(equivalent_resistance),
        float(reynolds),
        float(friction_factor),
        str(model_name),
    )


# =============================================================================
# ONE INDEPENDENT CHANNEL
# =============================================================================
class IndependentPassage:
    def __init__(
        self,
        name: str,
        pore1: Pore,
        pore2: Pore,
        chamber: Chamber,
        span_width: float,
        layout_kind: str = "surface_pair",
    ) -> None:
        self.name = name
        self.pore1 = pore1
        self.pore2 = pore2
        self.chamber = chamber
        self.span_width = float(span_width)
        self.layout_kind = str(layout_kind)

    def pore_surface_point(self, geom: PanelGeometry, pore: Pore) -> SurfacePoint:
        return geom.surface_point_from_fraction(pore.x_frac, pore.side)

    def passage_length(self, geom: PanelGeometry) -> float:
        sp1 = self.pore_surface_point(geom, self.pore1)
        sp2 = self.pore_surface_point(geom, self.pore2)

        def safe_thickness(x_frac: float) -> float:
            if x_frac <= 1e-8 or x_frac >= 1.0 - 1e-8:
                return 0.0
            return geom.local_thickness_at_fraction(x_frac)

        t1 = safe_thickness(self.pore1.x_frac)
        t2 = safe_thickness(self.pore2.x_frac)

        straight = np.hypot(sp2.x - sp1.x, sp2.y - sp1.y)

        # Straight in-plane channels use the direct surface-to-surface centreline
        # distance. For a lower-to-upper perpendicular passage on a symmetric
        # airfoil, this is exactly the local thickness; adding t1/2 + t2/2 would
        # double-count the same geometric distance.
        if self.layout_kind in {
            "internal_horizontal",
            "internal_chordwise_horizontal",
            "perpendicular",
        }:
            return max(float(straight), 1e-6)

        return max(float(straight + 0.5 * t1 + 0.5 * t2), 1e-6)


    def validate_geometry(self, geom: PanelGeometry, limits: GeometryLimits) -> None:
        for label, pore in (("pore1", self.pore1), ("pore2", self.pore2)):
            if not (limits.x_min_frac <= pore.x_frac <= limits.x_max_frac):
                raise ValueError(
                    f"{self.name}:{label} x/c={pore.x_frac:.4f} is outside "
                    f"[{limits.x_min_frac}, {limits.x_max_frac}]"
                )

            is_endpoint_pore = (
                abs(pore.x_frac - 0.0) <= limits.endpoint_tol
                or abs(pore.x_frac - 1.0) <= limits.endpoint_tol
            )

            # Exact LE/TE surface points have zero local thickness in 2D.
            # They are allowed only as endpoint pores.
            if not is_endpoint_pore:
                thickness = geom.local_thickness_at_fraction(pore.x_frac)

                if pore.height > limits.thickness_fraction_limit * thickness:
                    raise ValueError(
                        f"{self.name}:{label} height={pore.height:.4e} is too large "
                        f"for local thickness={thickness:.4e}"
                    )

        sp1 = self.pore_surface_point(geom, self.pore1)
        sp2 = self.pore_surface_point(geom, self.pore2)

        dist = float(np.hypot(sp2.x - sp1.x, sp2.y - sp1.y))
        required = 0.5 * (self.pore1.height + self.pore2.height) + limits.min_gap

        if dist < required:
            raise ValueError(
                f"{self.name} pore centers are too close: distance={dist:.4e}, "
                f"required={required:.4e}"
            )

        x_min = min(self.pore1.x_frac, self.pore2.x_frac)
        x_max = max(self.pore1.x_frac, self.pore2.x_frac)

        # Skip exact LE/TE zero-thickness locations for chamber fit checking.
        check_x_min = max(x_min, limits.endpoint_fit_margin)
        check_x_max = min(x_max, 1.0 - limits.endpoint_fit_margin)

        if check_x_min <= check_x_max:
            if abs(check_x_max - check_x_min) <= 1e-12:
                xs = np.array([check_x_min])
            else:
                xs = np.linspace(check_x_min, check_x_max, 40)

            t_min = min(geom.local_thickness_at_fraction(float(x)) for x in xs)

            if self.chamber.height > limits.fit_fraction * t_min:
                raise ValueError(
                    f"{self.name} rectangular channel is too tall for the minimum local thickness. "
                    f"height={self.chamber.height:.4e}, t_min={t_min:.4e}"
                )

    def equivalent_reynolds(self, q_abs: float, flow: FlowConfig) -> float:
        """Return Re based on q' per unit span and D_h=2H."""
        u_internal = q_abs / max(self.chamber.height, 1e-16)
        return (
            flow.rho_inf
            * abs(u_internal)
            * (2.0 * self.chamber.height)
            / max(flow.mu_inf, 1e-16)
        )


    def _opening_segments(
        self, geom: PanelGeometry, pore: Pore,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Intersect the channel strip with the polygonal airfoil surface.

        H is measured perpendicular to the centreline. Each returned segment
        is [t0,t1] within its original panel, with 0 <= t0 < t1 <= 1.
        Select the connected opening containing this end of the centreline;
        do not confuse it with the opening at the other end of the channel.
        Geometry is exact for the supplied straight-panel boundary.
        """
        cache = getattr(self, "_opening_cache", None)
        if cache is None or cache[0] is not geom:
            cache = (geom, {})
            self._opening_cache = cache
        key = (pore, self.pore1, self.pore2, self.chamber.height)
        if key in cache[1]:
            return cache[1][key]
        if not np.isclose(pore.height, self.chamber.height):
            raise ValueError("Opening and channel heights must agree.")
        a = self.pore_surface_point(geom, self.pore1)
        b = self.pore_surface_point(geom, self.pore2)
        origin = np.array([a.x, a.y])
        direction = np.array([b.x - a.x, b.y - a.y])
        length = np.linalg.norm(direction)
        if length <= 0.0:
            raise ValueError("Channel endpoints must be distinct.")
        direction /= length
        transverse = np.array([-direction[1], direction[0]])
        points = np.column_stack((geom.XB, geom.YB))
        z = (points - origin) @ transverse
        half_height = 0.5 * self.chamber.height
        dz = np.diff(z)
        lo = np.zeros(geom.num_pan)
        hi = np.ones(geom.num_pan)
        nonparallel = np.abs(dz) > 1e-14 * max(geom.chord, self.chamber.height)
        t_a = np.zeros_like(lo)
        t_b = np.zeros_like(lo)
        t_a[nonparallel] = (-half_height-z[:-1][nonparallel])/dz[nonparallel]
        t_b[nonparallel] = ( half_height-z[:-1][nonparallel])/dz[nonparallel]
        lo[nonparallel] = np.maximum(0.0, np.minimum(t_a, t_b)[nonparallel])
        hi[nonparallel] = np.minimum(1.0, np.maximum(t_a, t_b)[nonparallel])
        hi[~nonparallel & (np.abs(z[:-1]) > half_height)] = -1.0
        active = (hi-lo)*geom.S > 1e-14*geom.chord
        ids = np.flatnonzero(active)
        if not ids.size:
            raise ValueError(f"{self.name}: channel does not open on the boundary.")
        # Components in cyclic panel order, joining only across covered vertices.
        groups = [[int(ids[0])]]
        for i in ids[1:]:
            prev = groups[-1][-1]
            if i == prev+1 and hi[prev] >= 1.0-1e-12 and lo[i] <= 1e-12:
                groups[-1].append(int(i))
            else:
                groups.append([int(i)])
        if (len(groups) > 1 and groups[0][0] == 0
                and groups[-1][-1] == geom.num_pan-1
                and lo[0] <= 1e-12 and hi[-1] >= 1.0-1e-12):
            groups[0] = groups[-1]+groups[0]
            groups.pop()
        target = self.pore_surface_point(geom, pore)
        target_xy = np.array([target.x, target.y])
        def distance(group):
            ii = np.asarray(group)
            edges = points[ii+1]-points[ii]
            t = np.sum((target_xy-points[ii])*edges, axis=1)/geom.S[ii]**2
            t = np.clip(t, lo[ii], hi[ii])
            return np.min(np.linalg.norm(points[ii]+t[:, None]*edges-target_xy, axis=1))
        selected = min(groups, key=distance)
        if distance(selected) > 1e-5*geom.chord:
            raise ValueError(f"{self.name}: no opening contains the specified endpoint.")
        ii = np.asarray(selected, dtype=int)
        # A physical end must lie on its declared surface. Openings wrapping
        # around LE/TE need explicit geometry, rather than silently shifting them.
        side_ids = geom.upper_panel_ids if pore.side == "upper" else geom.lower_panel_ids
        if not np.all(np.isin(ii, side_ids)):
            raise ValueError(f"{self.name}: opening wraps around LE/TE; revise channel geometry.")
        result = (ii, lo[ii].copy(), hi[ii].copy())
        cache[1][key] = result
        return result

    def _surface_patch_weights(
        self, geom: PanelGeometry, pore: Pore,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compatibility API: actual panel intersection lengths, not fitted weights.

        Used only for conservative transfer of channel flow to panel velocities.
        Pressure is integrated independently by _area_average_pressure_at_pore.
        """
        ids, t0, t1 = self._opening_segments(geom, pore)
        return ids, (t1-t0)*geom.S[ids]

    def _area_average_pressure_at_pore(
        self, geom: PanelGeometry, pore: Pore,
        panel_pressures: np.ndarray, min_panels: int = 0,
    ) -> float:
        """Integrate reconstructed p(s) over the geometrically resolved opening.

        p(s) is piecewise linear through surface panel-centre pressures, with
        constant continuation over the half-panels at the surface endpoints.
        Integration splits at every interpolation knot, so it is exact for
        this reconstruction. Accuracy of physical pressure still depends on
        panel resolution and the aerodynamic model. No overlap-weighted mean
        of piecewise-constant panel pressures is used.
        """
        del min_panels
        pressures = np.asarray(panel_pressures, dtype=float)
        if pressures.shape != (geom.num_pan,) or not np.all(np.isfinite(pressures)):
            raise ValueError("Panel pressures must be finite and match the panel count.")
        ids, t0, t1 = self._opening_segments(geom, pore)
        edges = np.concatenate(([0.0], np.cumsum(geom.S)))
        centres = 0.5*(edges[:-1]+edges[1:])
        side = geom.upper_panel_ids if pore.side == "upper" else geom.lower_panel_ids
        side = np.sort(side)
        knots = centres[side]
        values = pressures[side]
        integral = 0.0
        opening_length = 0.0
        for i, left, right in zip(ids, t0, t1):
            start = edges[i]+left*geom.S[i]
            end = edges[i]+right*geom.S[i]
            x = np.concatenate(([start], knots[(knots > start) & (knots < end)], [end]))
            y = np.interp(x, knots, values)
            integral += float(np.sum(0.5*(y[:-1]+y[1:])*np.diff(x)))
            opening_length += end-start
        return integral/opening_length


    def solve(
        self,
        geom: PanelGeometry,
        panel_pressures: np.ndarray,
        flow: FlowConfig,
        aero_result: object | None = None,
    ) -> PassageState:
        """Static-pressure-selected inlet, zero tangential recovery, separate jet.

        p_in + <rho*Vn_in^2/2> - p_out
            = dp_friction(q') + alpha_j*rho*(q'/H)^2/2.

        Inlet tangential kinetic energy is removed before entering the channel.
        Outlet static pressure matches the exterior, but its velocity/total
        pressure are those of the bulk jet, not the surrounding potential flow.
        This is a conservative reduced closure hypothesis; jet mixing and
        tangential recovery are not resolved or calibrated.
        """
        p1 = self._area_average_pressure_at_pore(geom, self.pore1, panel_pressures)
        p2 = self._area_average_pressure_at_pore(geom, self.pore2, panel_pressures)
        dp = p1 - p2
        length = self.passage_length(geom)
        closure = CHANNEL_ENERGY_CLOSURE
        if closure not in {"legacy_static", "zero_tangential_recovery_jet"}:
            raise ValueError(f"Unknown CHANNEL_ENERGY_CLOSURE: {closure}")
        normal_head = tangential_head = 0.0
        kinetic_factor = 0.0
        if closure == "zero_tangential_recovery_jet":
            if aero_result is None or getattr(aero_result, "Vn", None) is None:
                raise ValueError("Install the matching solver.py and pass aero_result "
                                 "to network.solve for the inlet/jet energy closure.")
            vn = np.asarray(aero_result.Vn, dtype=float)
            vt = np.asarray(aero_result.Vt, dtype=float)
            if vn.shape != (geom.num_pan,) or vt.shape != (geom.num_pan,):
                raise ValueError("Invalid aerodynamic velocity arrays for channel energy closure.")
            # Same linear averaging operator for pressure and kinetic heads:
            # <p> + <rho*(Vt^2+Vn^2)/2> = p0_inf, up to roundoff.
            inlet = self.pore1 if dp >= 0.0 else self.pore2
            normal_head = self._area_average_pressure_at_pore(
                geom, inlet, 0.5*flow.rho_inf*vn**2)
            tangential_head = self._area_average_pressure_at_pore(
                geom, inlet, 0.5*flow.rho_inf*vt**2)
            kinetic_factor = float(JET_KINETIC_ENERGY_FACTOR)
            if not np.isfinite(kinetic_factor) or kinetic_factor < 1.0:
                raise ValueError("JET_KINETIC_ENERGY_FACTOR must be finite and >= 1.")

        # This reduced pressure-driven model excludes inertia-driven flow against
        # the static-pressure gradient. At an exact tie it selects zero flow.
        direction = float(np.sign(dp))
        available_head = abs(dp) + normal_head if direction != 0.0 else 0.0
        q, _, reynolds, friction_factor, resistance_model = solve_rectangular_channel_flow(
            dp=direction*available_head,
            rho=flow.rho_inf,
            mu=flow.mu_inf,
            length=length,
            height=self.chamber.height,
            outlet_kinetic_factor=kinetic_factor,
        )
        speed = abs(q)/self.chamber.height
        friction_head = _rectangular_2d_pressure_drop_abs(
            speed, flow.rho_inf, flow.mu_inf, length, self.chamber.height)[0]
        jet_head = kinetic_factor*0.5*flow.rho_inf*speed**2
        # Rs retains the original static-pressure-difference / flow meaning.
        rs = abs(dp)/abs(q) if q != 0.0 else float("inf")
        state = PassageState(
            Q=float(q), p1=p1, p2=p2, dp_total=float(abs(dp)), Rs=float(rs),
            reynolds_equivalent=float(reynolds), friction_factor=float(friction_factor),
            resistance_model=str(resistance_model),
        )
        p_in, p_out = (p1, p2) if dp >= 0.0 else (p2, p1)
        active = direction != 0.0
        energy_residual = available_head-friction_head-jet_head
        state.energy_diagnostics = {
            "energy_closure": closure,
            "energy_balance_active": bool(active),
            "inlet_pore": 1 if dp > 0.0 else (2 if dp < 0.0 else 0),
            "inlet_normal_kinetic_head_Pa": float(normal_head),
            "inlet_tangential_head_removed_Pa": float(tangential_head),
            "available_channel_head_Pa": float(available_head),
            "channel_friction_loss_Pa": float(friction_head),
            "outlet_jet_kinetic_head_Pa": float(jet_head),
            "outlet_jet_speed_m_s": float(speed),
            "channel_inlet_total_pressure_Pa": float(p_in+normal_head) if active else float("nan"),
            "outlet_jet_total_pressure_Pa": float(p_out+jet_head) if active else float("nan"),
            "energy_balance_residual_Pa": float(energy_residual),
            "friction_power_W_per_m_span": float(abs(q)*friction_head),
            "inlet_turning_power_W_per_m_span": float(abs(q)*tangential_head),
            "outlet_kinetic_energy_flux_W_per_m_span": float(abs(q)*jet_head),
        }
        if abs(energy_residual) > 1e-10*max(1.0, available_head):
            raise FloatingPointError("Channel energy equation failed its residual check.")
        return state


    def transpiration_from_flow(
        self,
        geom: PanelGeometry,
        state: PassageState,
    ) -> np.ndarray:
        """Map q' [m^2/s] over the actual channel/surface intersection.

        The panel-average values conserve flow exactly:

            sum(v_n,i S_i) = +/- q'

        No out-of-plane span width appears because both the internal and
        external coupling are formulated per unit span.
        """
        vn = np.zeros(geom.num_pan, dtype=float)

        q_prime = float(state.Q)
        if abs(q_prime) < 1.0e-30:
            return vn

        ids1, overlap1 = self._surface_patch_weights(geom, self.pore1)
        ids2, overlap2 = self._surface_patch_weights(geom, self.pore2)

        opening1 = max(float(np.sum(overlap1)), 1.0e-16)
        opening2 = max(float(np.sum(overlap2)), 1.0e-16)

        # Uniform physical velocity over the opening, represented as a
        # panel-average when an opening cuts through a partial panel.
        vn[ids1] -= q_prime * overlap1 / (opening1 * geom.S[ids1])
        vn[ids2] += q_prime * overlap2 / (opening2 * geom.S[ids2])

        return vn

# =============================================================================
# NETWORK OF INDEPENDENT CHANNELS
# =============================================================================
class IndependentPassageNetwork:
    def __init__(self, passages: list[IndependentPassage]) -> None:
        if not passages:
            raise ValueError("At least one passage is required.")
        self.passages = passages

    def validate_geometry(self, geom: PanelGeometry, limits: GeometryLimits) -> None:
        for passage in self.passages:
            passage.validate_geometry(geom, limits)

        # Check real opening intervals, including partial panels. Centre
        # distances alone cannot detect oblique rectangular-opening overlap.
        openings = []
        for passage in self.passages:
            for label, pore in (("pore1", passage.pore1), ("pore2", passage.pore2)):
                ids, t0, t1 = passage._opening_segments(geom, pore)
                intervals = {int(i): (float(a), float(b)) for i, a, b in zip(ids, t0, t1)}
                for previous_name, previous in openings:
                    for i in intervals.keys() & previous.keys():
                        overlap = min(intervals[i][1], previous[i][1])-max(intervals[i][0], previous[i][0])
                        if overlap*geom.S[i] > 1e-12*geom.chord:
                            raise ValueError(
                                f"Actual rectangular openings overlap: {previous_name} "
                                f"and {passage.name}:{label}. Revise the channel layout."
                            )
                openings.append((f"{passage.name}:{label}", intervals))

        pore_refs: list[tuple[str, Pore, SurfacePoint]] = []
        for passage in self.passages:
            pore_refs.append(
                (
                    f"{passage.name}:pore1",
                    passage.pore1,
                    passage.pore_surface_point(geom, passage.pore1),
                )
            )
            pore_refs.append(
                (
                    f"{passage.name}:pore2",
                    passage.pore2,
                    passage.pore_surface_point(geom, passage.pore2),
                )
            )

        for i in range(len(pore_refs)):
            name_i, pore_i, sp_i = pore_refs[i]

            for j in range(i + 1, len(pore_refs)):
                name_j, pore_j, sp_j = pore_refs[j]

                dist = float(np.hypot(sp_i.x - sp_j.x, sp_i.y - sp_j.y))
                required = 0.5 * (pore_i.height + pore_j.height) + limits.min_gap

                if dist < required:
                    raise ValueError(
                        f"Pore overlap between {name_i} and {name_j}: "
                        f"distance={dist:.4e}, required={required:.4e}"
                    )

    def solve(
        self,
        geom: PanelGeometry,
        panel_pressures: np.ndarray,
        flow: FlowConfig,
        aero_result: object | None = None,
    ) -> list[PassageState]:
        return [
            passage.solve(
                geom=geom,
                panel_pressures=panel_pressures,
                aero_result=aero_result,
                flow=flow,
            )
            for passage in self.passages
        ]

    def transpiration_from_flow(
        self,
        geom: PanelGeometry,
        states: list[PassageState],
    ) -> np.ndarray:
        vn = np.zeros(geom.num_pan, dtype=float)

        for passage, state in zip(self.passages, states):
            vn += passage.transpiration_from_flow(
                geom=geom,
                state=state,
            )

        return vn



# =============================================================================
# COUPLED PANEL + POROUS SOLVER
# =============================================================================
class MultiPassageCoupledSolver:
    def __init__(
        self,
        aero_solver: SourceVortexPanelMethod,
        network: IndependentPassageNetwork,
        coupling: CouplingConfig,
        limits: GeometryLimits,
    ) -> None:
        self.aero_solver = aero_solver
        self.network = network
        self.coupling = coupling
        self.limits = limits

    def solve(
        self,
        verbose: bool = False,
        initial_vn: np.ndarray | None = None,
        initial_q: np.ndarray | None = None,
        coupling_strength: float = 1.0,
    ) -> MultiPassageResult:
        """Solve the coupled pressure-flow equations as a nonlinear root problem.

        Parameters
        ----------
        verbose
            Print nonlinear residual evaluations.
        initial_vn
            Backward-compatible initial transpiration field.  It is used only
            to construct an initial flow-rate guess.
        initial_q
            Preferred continuation seed: physical passage flow rates per unit
            span [m^2/s].  Passing Q directly is robust when AoA or Reynolds
            number changes because the solver rescales it using the new H*Vinf.
        coupling_strength
            Homotopy parameter lambda in [0,1].  The trial transpiration is
            lambda times the physical Q-to-vn mapping.  lambda=1 is exactly the
            original fully coupled model; lambda<1 is used only for numerical
            continuation and must never be reported as a physical final state.

        Notes
        -----
        The nonlinear unknown is the dimensionless passage-flow vector.  For a
        trial Q the code maps Q to surface transpiration, solves the external
        panel method, calculates opening pressures, obtains the pressure-driven
        channel flows, and forms

            R = (Q_trial - Q_pressure_driven) / (H V_inf).

        HYBR is tried with two step factors, followed by TRF least-squares.
        Solver status alone is never accepted: convergence requires the original
        scaled residual to satisfy the machine-resolution-based criterion.
        """
        geom = self.aero_solver.geom
        flow = self.aero_solver.flow
        n_passages = len(self.network.passages)

        coupling_strength = float(coupling_strength)
        if not np.isfinite(coupling_strength) or not (0.0 <= coupling_strength <= 1.0):
            raise ValueError("coupling_strength must be finite and lie in [0, 1].")
        if initial_q is not None and initial_vn is not None:
            raise ValueError("Provide initial_q or initial_vn, not both.")

        try:
            self.network.validate_geometry(geom, self.limits)
        except ValueError as exc:
            zero_vn = np.zeros(geom.num_pan, dtype=float)
            aero = self.aero_solver.solve(normal_transpiration=zero_vn)
            return MultiPassageResult(
                aero_result=aero,
                passage_states=[],
                normal_transpiration=zero_vn,
                converged=False,
                iterations=0,
                max_vn=float("nan"),
                max_vn_over_vinf=float("nan"),
                coupling_status="invalid_geometry",
                solver_message=str(exc),
            )

        q_scale = np.asarray(
            [
                max(
                    float(passage.chamber.height) * float(flow.v_inf),
                    np.finfo(float).tiny,
                )
                for passage in self.network.passages
            ],
            dtype=float,
        )

        def states_for_trial_flow(q_trial: np.ndarray) -> list[PassageState]:
            """Create lightweight states for the conservative Q-to-vn mapping."""
            return [
                PassageState(
                    Q=float(q_value),
                    p1=0.0,
                    p2=0.0,
                    dp_total=0.0,
                    Rs=0.0,
                    reynolds_equivalent=0.0,
                )
                for q_value in q_trial
            ]

        # Construct a physically consistent starting state.
        if initial_q is not None:
            seed_q = np.asarray(initial_q, dtype=float).copy()
            if seed_q.shape != (n_passages,):
                raise ValueError(
                    f"initial_q must have shape ({n_passages},), got {seed_q.shape}."
                )
            if not np.all(np.isfinite(seed_q)):
                raise ValueError("initial_q must contain only finite values.")
            full_seed_vn = self.network.transpiration_from_flow(
                geom=geom,
                states=states_for_trial_flow(seed_q),
            )
            starting_vn = coupling_strength * full_seed_vn
            initial_q_dimensionless = seed_q / q_scale
        else:
            if initial_vn is None:
                starting_vn = np.zeros(geom.num_pan, dtype=float)
            else:
                starting_vn = np.asarray(initial_vn, dtype=float).copy()
                if starting_vn.shape != (geom.num_pan,):
                    raise ValueError(
                        f"initial_vn must have shape ({geom.num_pan},), "
                        f"got {starting_vn.shape}."
                    )
                if not np.all(np.isfinite(starting_vn)):
                    raise ValueError("initial_vn must contain only finite values.")

            seed_aero = self.aero_solver.solve(normal_transpiration=starting_vn)
            seed_pressures = self.aero_solver.panel_pressures(seed_aero)
            seed_states = self.network.solve(
                geom=geom,
                panel_pressures=seed_pressures,
                aero_result=seed_aero,
                flow=flow,
            )
            seed_q = np.asarray([float(state.Q) for state in seed_states], dtype=float)
            initial_q_dimensionless = seed_q / q_scale

        last_finite_vn = starting_vn.copy()
        last_finite_aero = self.aero_solver.solve(normal_transpiration=starting_vn)
        last_finite_pressures = self.aero_solver.panel_pressures(last_finite_aero)
        last_finite_states = self.network.solve(
            geom=geom,
            panel_pressures=last_finite_pressures,
            aero_result=last_finite_aero,
            flow=flow,
        )

        residual_norm = float("nan")
        floating_point_resolution = float("nan")
        solver_success = False
        solver_message = ""
        coupling_status = "not_converged"
        evaluation_count = 0
        numerical_failure = False

        def coupling_residual(q_trial_dimensionless: np.ndarray) -> np.ndarray:
            nonlocal evaluation_count
            nonlocal last_finite_vn
            nonlocal last_finite_aero
            nonlocal last_finite_states

            evaluation_count += 1
            q_trial_dimensionless = np.asarray(q_trial_dimensionless, dtype=float)
            if (
                q_trial_dimensionless.shape != (n_passages,)
                or not np.all(np.isfinite(q_trial_dimensionless))
            ):
                raise FloatingPointError("Non-finite passage-flow trial state.")

            q_trial = q_trial_dimensionless * q_scale

            with np.errstate(over="raise", invalid="raise", divide="raise"):
                vn_full = self.network.transpiration_from_flow(
                    geom=geom,
                    states=states_for_trial_flow(q_trial),
                )
                vn_trial = coupling_strength * vn_full
                aero_result = self.aero_solver.solve(normal_transpiration=vn_trial)
                panel_pressures = self.aero_solver.panel_pressures(aero_result)
                hydraulic_states = self.network.solve(
                    geom=geom,
                    panel_pressures=panel_pressures,
                    aero_result=aero_result,
                    flow=flow,
                )

            q_pressure_driven = np.asarray(
                [float(state.Q) for state in hydraulic_states],
                dtype=float,
            )
            residual = (q_trial - q_pressure_driven) / q_scale

            if (
                not np.all(np.isfinite(vn_trial))
                or not np.all(np.isfinite(panel_pressures))
                or not np.all(np.isfinite(q_pressure_driven))
                or not np.all(np.isfinite(residual))
                or not np.isfinite(aero_result.CL)
            ):
                raise FloatingPointError("Non-finite coupled state.")

            last_finite_vn = vn_trial.copy()
            last_finite_aero = aero_result
            last_finite_states = hydraulic_states

            if verbose:
                print(
                    f"[Coupling] lambda={coupling_strength:.3f} | "
                    f"evaluation={evaluation_count:03d} | "
                    f"CL={aero_result.CL: .8f} | "
                    f"max|Q|={np.max(np.abs(q_trial)): .6e} | "
                    f"max|scaled residual|={np.max(np.abs(residual)): .3e}"
                )

            return residual

        # ------------------------------------------------------------------
        # Robust nonlinear solution hierarchy
        # ------------------------------------------------------------------
        best_q_dimensionless = initial_q_dimensionless.copy()
        best_residual_norm = float("inf")
        best_residual_tolerance = float("nan")
        best_method = "initial_guess"
        attempt_messages: list[str] = []
        converged = False
        final_iteration = 0
        selected_solver_message = ""

        def assess_candidate(
            q_dimensionless: np.ndarray,
            method_name: str,
        ) -> tuple[float, float, bool]:
            nonlocal best_q_dimensionless
            nonlocal best_residual_norm
            nonlocal best_residual_tolerance
            nonlocal best_method

            q_dimensionless = np.asarray(q_dimensionless, dtype=float)
            if (
                q_dimensionless.shape != (n_passages,)
                or not np.all(np.isfinite(q_dimensionless))
            ):
                return float("inf"), float("nan"), False

            residual = coupling_residual(q_dimensionless)
            norm = float(np.max(np.abs(residual)))
            state_scale = max(1.0, float(np.max(np.abs(q_dimensionless))))
            tolerance = float(np.sqrt(np.finfo(float).eps) * state_scale)

            if np.isfinite(norm) and norm < best_residual_norm:
                best_residual_norm = norm
                best_residual_tolerance = tolerance
                best_q_dimensionless = q_dimensionless.copy()
                best_method = method_name

            return norm, tolerance, bool(norm <= tolerance)

        try:
            current_guess = initial_q_dimensionless.copy()

            for factor in (0.5, 0.2):
                try:
                    hybr_result = root(
                        coupling_residual,
                        current_guess,
                        method="hybr",
                        options={
                            "maxfev": int(self.coupling.max_iter),
                            "factor": float(factor),
                            "xtol": 1.0e-10,
                        },
                    )
                    q_candidate = np.asarray(hybr_result.x, dtype=float)
                    norm, tolerance, resolved = assess_candidate(
                        q_candidate,
                        method_name=f"hybr(factor={factor:g})",
                    )
                    message = str(hybr_result.message).replace("\n", " ")
                    attempt_messages.append(
                        f"HYBR factor={factor:g}: success={bool(hybr_result.success)}, "
                        f"residual={norm:.3e}, message={message}"
                    )
                    if np.all(np.isfinite(q_candidate)):
                        current_guess = q_candidate.copy()
                    if resolved:
                        converged = True
                        residual_norm = norm
                        floating_point_resolution = tolerance
                        solver_success = bool(hybr_result.success)
                        selected_solver_message = message
                        best_method = f"hybr(factor={factor:g})"
                        break
                except (FloatingPointError, OverflowError, ValueError, RuntimeError) as exc:
                    attempt_messages.append(
                        f"HYBR factor={factor:g}: {type(exc).__name__}: {exc}"
                    )

            if not converged:
                try:
                    lsq_result = least_squares(
                        coupling_residual,
                        x0=current_guess,
                        method="trf",
                        jac="3-point",
                        x_scale="jac",
                        ftol=1.0e-12,
                        xtol=1.0e-12,
                        gtol=1.0e-12,
                        max_nfev=max(1000, 4 * int(self.coupling.max_iter)),
                    )
                    q_candidate = np.asarray(lsq_result.x, dtype=float)
                    norm, tolerance, resolved = assess_candidate(
                        q_candidate,
                        method_name="least_squares(TRF)",
                    )
                    message = str(lsq_result.message).replace("\n", " ")
                    attempt_messages.append(
                        f"TRF: success={bool(lsq_result.success)}, "
                        f"residual={norm:.3e}, message={message}"
                    )
                    if resolved:
                        converged = True
                        residual_norm = norm
                        floating_point_resolution = tolerance
                        solver_success = bool(lsq_result.success)
                        selected_solver_message = message
                        best_method = "least_squares(TRF)"
                except (FloatingPointError, OverflowError, ValueError, RuntimeError) as exc:
                    attempt_messages.append(f"TRF: {type(exc).__name__}: {exc}")

            # Synchronize the returned aerodynamic/hydraulic state with the best
            # candidate, whether or not it is an accepted root.
            if np.all(np.isfinite(best_q_dimensionless)):
                try:
                    norm, tolerance, _ = assess_candidate(
                        best_q_dimensionless,
                        method_name=best_method,
                    )
                    if not converged:
                        residual_norm = norm
                        floating_point_resolution = tolerance
                except (FloatingPointError, OverflowError, ValueError, RuntimeError):
                    pass

            if not converged:
                solver_success = False
                selected_solver_message = (
                    f"No retry reached the residual criterion; best method={best_method}."
                )

            final_iteration = evaluation_count
            coupling_status = "converged" if converged else "not_converged"
            solver_message = (
                f"lambda={coupling_strength:.3f}; selected={best_method}; "
                f"{selected_solver_message} "
                + " | ".join(attempt_messages)
            ).strip()

            if verbose:
                print(
                    f"[Coupling] robust nonlinear solve: "
                    f"lambda={coupling_strength:.3f}, converged={converged}, "
                    f"selected={best_method}, evaluations={final_iteration}, "
                    f"max|scaled residual|={residual_norm:.3e}"
                )
                for attempt in attempt_messages:
                    print(f"[Coupling]   {attempt}")

        except (FloatingPointError, OverflowError, ValueError, RuntimeError) as exc:
            converged = False
            numerical_failure = True
            coupling_status = "numerical_failure"
            solver_success = False
            solver_message = (
                f"lambda={coupling_strength:.3f}; {type(exc).__name__}: {exc}"
            )
            residual_norm = (
                best_residual_norm if np.isfinite(best_residual_norm) else float("nan")
            )
            floating_point_resolution = best_residual_tolerance
            final_iteration = evaluation_count
            if verbose:
                print(
                    f"[Coupling] robust nonlinear solve stopped after "
                    f"{evaluation_count} evaluations: {type(exc).__name__}: {exc}"
                )

        max_vn = float(np.max(np.abs(last_finite_vn)))
        return MultiPassageResult(
            aero_result=last_finite_aero,
            passage_states=last_finite_states,
            normal_transpiration=last_finite_vn,
            converged=bool(converged and not numerical_failure),
            iterations=final_iteration,
            max_vn=max_vn,
            max_vn_over_vinf=max_vn / max(flow.v_inf, 1e-16),
            coupling_status=coupling_status,
            residual_norm=residual_norm,
            residual_tolerance=floating_point_resolution,
            solver_success=solver_success,
            solver_message=solver_message,
            function_evaluations=evaluation_count,
        )


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================
def _arc_length_spaced_points_on_side(
    geom: PanelGeometry,
    side: str,
    x_start_frac: float,
    x_end_frac: float,
    n_points: int,
    n_sample: int = 800,
) -> list[tuple[float, str]]:
    """
    Return n_points equally spaced by surface arc length on one airfoil side.

    Output:
        [(x_frac, side), ...]
    """
    side = side.lower()

    if side == "upper":
        sx, sy = geom.upper_x, geom.upper_y
    elif side == "lower":
        sx, sy = geom.lower_x, geom.lower_y
    else:
        raise ValueError("side must be 'upper' or 'lower'")

    x_frac = np.linspace(x_start_frac, x_end_frac, n_sample)
    x_abs = geom.x_le + x_frac * geom.chord
    y_abs = np.interp(x_abs, sx, sy)

    ds = np.hypot(np.diff(x_abs), np.diff(y_abs))
    s = np.concatenate([[0.0], np.cumsum(ds)])

    targets = np.linspace(0.0, s[-1], n_points)
    x_out = np.interp(targets, s, x_frac)

    return [(float(x), side) for x in x_out]


def _equally_spaced_surface_band(
    geom: PanelGeometry,
    center_x_frac: float,
    outer_x_frac: float,
    n_points: int = 4,
    n_sample: int = 800,
) -> list[tuple[float, str]]:
    """
    Build equally spaced pore points around one rounded airfoil end.

    The path starts on the lower surface at outer_x_frac, travels to the
    centre endpoint, and continues on the upper surface back to outer_x_frac.
    For Model 1, n_points=4 gives four surface pores around the LE and four
    matching surface pores around the TE.

    This avoids the old 9-point pattern:
        4 lower + 1 centre + 4 upper

    New Model 1 pattern:
        4 total points, uniformly spaced by local surface arc length.
    """
    if n_points < 2:
        raise ValueError("n_points must be at least 2.")

    # Lower-side arc: outer point -> centre endpoint.
    lower_x_frac = np.linspace(outer_x_frac, center_x_frac, n_sample)
    lower_x_abs = geom.x_le + lower_x_frac * geom.chord
    lower_y_abs = np.interp(lower_x_abs, geom.lower_x, geom.lower_y)

    # Upper-side arc: centre endpoint -> outer point.
    upper_x_frac = np.linspace(center_x_frac, outer_x_frac, n_sample)
    upper_x_abs = geom.x_le + upper_x_frac * geom.chord
    upper_y_abs = np.interp(upper_x_abs, geom.upper_x, geom.upper_y)

    # Skip the duplicated centre point in the upper-side array.
    x_path_frac = np.concatenate([lower_x_frac, upper_x_frac[1:]])
    x_path_abs = np.concatenate([lower_x_abs, upper_x_abs[1:]])
    y_path_abs = np.concatenate([lower_y_abs, upper_y_abs[1:]])
    side_path = np.array(["lower"] * len(lower_x_frac) + ["upper"] * (len(upper_x_frac) - 1))

    ds = np.hypot(np.diff(x_path_abs), np.diff(y_path_abs))
    s_arc = np.concatenate([[0.0], np.cumsum(ds)])

    targets = np.linspace(0.0, s_arc[-1], n_points)
    x_out = np.interp(targets, s_arc, x_path_frac)

    points: list[tuple[float, str]] = []
    for target, x_frac in zip(targets, x_out):
        idx = int(np.searchsorted(s_arc, target, side="right") - 1)
        idx = max(0, min(idx, len(side_path) - 1))
        points.append((float(x_frac), str(side_path[idx])))

    return points


def _unique_sorted(values: list[float], tol: float = 1e-8) -> list[float]:
    """Return sorted unique values with a tolerance."""
    if not values:
        return []

    out: list[float] = []
    for value in sorted(float(v) for v in values):
        if not out or abs(value - out[-1]) > tol:
            out.append(value)
    return out


def _surface_x_intersections_for_y(
    geom: PanelGeometry,
    side: str,
    y_target: float,
    n_sample: int = 5000,
) -> tuple[float, float]:
    """
    Find the left and right x/c locations where one surface crosses y_target.

    For a symmetric NACA 0018, a positive y_target intersects the upper
    surface twice and a negative y_target intersects the lower surface twice.
    The returned pair defines one constant-y horizontal channel inside the
    airfoil, with both pores located on the airfoil surface.
    """
    side = side.lower()
    if side == "upper":
        sx, sy = geom.upper_x, geom.upper_y
    elif side == "lower":
        sx, sy = geom.lower_x, geom.lower_y
    else:
        raise ValueError("side must be 'upper' or 'lower'.")

    x_grid = np.linspace(float(sx.min()), float(sx.max()), n_sample)
    y_grid = np.interp(x_grid, sx, sy)
    f = y_grid - float(y_target)

    crossings: list[float] = []

    for k in range(len(x_grid) - 1):
        f0 = float(f[k])
        f1 = float(f[k + 1])

        if abs(f0) <= 1e-12:
            crossings.append(float(x_grid[k]))
            continue

        if f0 * f1 < 0.0:
            # Linear interpolation between the two sampled surface points.
            x0 = float(x_grid[k])
            x1 = float(x_grid[k + 1])
            x_cross = x0 - f0 * (x1 - x0) / (f1 - f0)
            crossings.append(float(x_cross))

    if abs(float(f[-1])) <= 1e-12:
        crossings.append(float(x_grid[-1]))

    crossings = _unique_sorted(crossings, tol=1e-7 * geom.chord)

    if len(crossings) < 2:
        raise ValueError(
            f"Could not find two {side} surface intersections for "
            f"Model 1 horizontal channel at y={y_target:.6e} m."
        )

    x1_frac = (crossings[0] - geom.x_le) / geom.chord
    x2_frac = (crossings[-1] - geom.x_le) / geom.chord

    return float(x1_frac), float(x2_frac)


def model1_horizontal_surface_pores(
    geom: PanelGeometry,
    n_channels: int = 4,
    max_y_fraction: float = 0.60,
) -> list[tuple[float, str, float, str, float]]:
    """
    Build Model 1 as horizontal, straight internal channels.

    The channels are placed at equally spaced constant-y levels inside the
    airfoil.  A positive y/c level uses the upper surface for both pores; a
    negative y/c level uses the lower surface for both pores.  Therefore each
    channel is a straight horizontal line between two surface intersections.

    Returns
    -------
    list[tuple]
        Each tuple is ``(x1_frac, side1, x2_frac, side2, y_level_m)``.
    """
    if n_channels < 1:
        raise ValueError("n_channels must be at least one.")
    if not (0.0 < max_y_fraction < 1.0):
        raise ValueError("max_y_fraction must be between 0 and 1.")

    upper_clearance = float(np.max(geom.upper_y))
    lower_clearance = float(abs(np.min(geom.lower_y)))
    usable_half_height = max_y_fraction * min(upper_clearance, lower_clearance)

    if usable_half_height <= 0.0:
        raise ValueError("Airfoil thickness is zero; horizontal channels cannot be created.")

    # Avoid placing channels exactly on y=0 when n_channels is even.  For the
    # default 4-channel model this gives two lower and two upper horizontal
    # channels, symmetrically spaced about the chord line.
    y_levels = np.linspace(
        -usable_half_height,
        usable_half_height,
        n_channels + 2,
    )[1:-1]

    points: list[tuple[float, str, float, str, float]] = []

    for y_level in y_levels:
        side = "upper" if y_level >= 0.0 else "lower"
        x1_frac, x2_frac = _surface_x_intersections_for_y(
            geom=geom,
            side=side,
            y_target=float(y_level),
        )
        points.append((x1_frac, side, x2_frac, side, float(y_level)))

    return points


def model1_le_te_surface_pores(
    geom: PanelGeometry,
) -> tuple[list[tuple[float, str]], list[tuple[float, str]]]:
    """
    Backward-compatible Model 1 pore helper.

    The old implementation returned LE-band and TE-band pore lists.  Model 1
    is now horizontal, so this function returns the left and right pore lists
    from the constant-y channel definition.
    """
    horizontal_channels = model1_horizontal_surface_pores(geom)
    left_points = [(x1, side1) for x1, side1, _, _, _ in horizontal_channels]
    right_points = [(x2, side2) for _, _, x2, side2, _ in horizontal_channels]
    return left_points, right_points


def build_model1_chordwise(
    geom: PanelGeometry,
    channel_height_m: float,
) -> list[PassageSpec]:
    """Build four horizontal in-plane rectangular channels."""
    horizontal_channels = model1_horizontal_surface_pores(
        geom=geom,
        n_channels=4,
        max_y_fraction=0.85,
    )
    passages: list[PassageSpec] = []
    for i, (x1, side1, x2, side2, y_level) in enumerate(horizontal_channels, start=1):
        y_label = "upper" if y_level >= 0.0 else "lower"
        passages.append(
            PassageSpec(
                name=f"m1_{i:02d}_{y_label}_horizontal_channel",
                x1_frac=x1,
                side1=side1,
                x2_frac=x2,
                side2=side2,
                channel_height_m=float(channel_height_m),
                layout_kind="internal_horizontal",
            )
        )
    return passages


def build_vertical_passages(
    x_values: np.ndarray,
    channel_height_m: float,
    prefix: str,
) -> list[PassageSpec]:
    """Build lower-to-upper in-plane rectangular channels."""
    passages: list[PassageSpec] = []
    for i, x in enumerate(x_values, start=1):
        passages.append(
            PassageSpec(
                name=f"{prefix}_{i:02d}_perpendicular",
                x1_frac=float(x),
                side1="lower",
                x2_frac=float(x),
                side2="upper",
                channel_height_m=float(channel_height_m),
                layout_kind="perpendicular",
            )
        )
    return passages


def _copy_passages_with_prefix(
    passages: list[PassageSpec],
    prefix: str,
    label: str,
) -> list[PassageSpec]:
    """Copy rectangular passage definitions and rename them."""
    return [
        PassageSpec(
            name=f"{prefix}_{i:02d}_{label}",
            x1_frac=passage.x1_frac,
            side1=passage.side1,
            x2_frac=passage.x2_frac,
            side2=passage.side2,
            channel_height_m=passage.channel_height_m,
            layout_kind=passage.layout_kind,
        )
        for i, passage in enumerate(passages, start=1)
    ]


def build_model4_saved_case1(
    channel_height_m: float | None = None,
) -> list[PassageSpec]:
    """Build saved case 1 using one-parameter rectangular passages."""
    heights = (1.2e-02, 0.9e-02) if channel_height_m is None else (
        float(channel_height_m), float(channel_height_m)
    )
    locations = (
        ("m4_case1_passage_01_lower_to_upper", 0.400094, 0.933001),
        ("m4_case1_passage_02_lower_to_upper", 0.925069, 0.955118),
    )
    return [
        PassageSpec(
            name=name,
            x1_frac=x1,
            side1="lower",
            x2_frac=x2,
            side2="upper",
            channel_height_m=height,
            layout_kind="saved_case_1",
        )
        for (name, x1, x2), height in zip(locations, heights)
    ]


def build_model_specs(
    geom: PanelGeometry,
    channel_height_m: float | None = None,
    *,
    diameter_m: float | None = None,
    channel_width_m: float | None = None,
) -> list[PorousModelSpec]:
    """Build all 2D rectangular models from the in-plane opening H.

    ``diameter_m`` remains a compatibility alias for H. ``channel_width_m`` is
    ignored because the active equations are per unit span.
    """
    if channel_height_m is None:
        if diameter_m is None:
            raise ValueError("channel_height_m must be provided.")
        channel_height_m = float(diameter_m)
    elif diameter_m is not None and not np.isclose(channel_height_m, diameter_m):
        raise ValueError("Provide channel_height_m or diameter_m, not conflicting values.")

    height = float(channel_height_m)
    # Accepted only so older callers do not fail. The value is not used by the
    # active per-unit-span hydraulic equations.
    _ = channel_width_m

    model_1 = build_model1_chordwise(geom, height)

    # Keep Model 2 and the vertical part of Model 3 identical.  The old rear
    # location x/c=0.900 causes the 9 and 10 mm vertical surface opening to
    # intersect the rear opening of the inner horizontal channel in Model 3.
    # Ending the equally spaced set at x/c=0.890 provides finite clearance at
    # H=10 mm while preserving the four-channel lower-to-upper layout.
    vertical_x_values = np.linspace(0.100, 0.890, 4)

    model_2 = build_vertical_passages(vertical_x_values, height, "m2")
    model_3h = _copy_passages_with_prefix(model_1, "m3c", "LE_to_TE_internal_channel")
    model_3v = build_vertical_passages(vertical_x_values, height, "m3v")
    # model_4 = build_model4_saved_case1(height)

    size_text = (
        f"in-plane opening H={1000.0*height:.1f} mm; "
        "2D flow per unit span; D_h=2H"
    )
    return [
        PorousModelSpec(
            name="model_1_4_chordwise",
            description="4 straight horizontal rectangular channels; " + size_text + ".",
            passages=tuple(model_1),
        ),
        PorousModelSpec(
            name="model_2_9_perpendicular",
            description="4 independent lower-to-upper rectangular channels; " + size_text + ".",
            passages=tuple(model_2),
        ),
        PorousModelSpec(
            name="model_3_combined_independent",
            description=(
                "Combined horizontal and vertical rectangular channels. "
                "All 8 passages are independent; internal intersections are not connected; "
                + size_text + "."
            ),
            passages=tuple(model_3h + model_3v),
        ),
        # PorousModelSpec(
        #     name="model_4_saved_case_1",
        #     description="Saved case 1 with two independent rectangular passages; " + size_text + ".",
        #     passages=tuple(model_4),
        # ),
    ]


# =============================================================================
# BUILDERS
# =============================================================================
def build_base_problem(
    airfoil_name: str,
    chord: float,
    n_panels: int,
    aoa_deg: float,
    reynolds_external: float,
    rho_inf: float,
    mu_inf: float,
    p_inf: float,
) -> tuple[FlowConfig, PanelGeometry, ReferenceConfig, SourceVortexPanelMethod]:
    v_inf = reynolds_to_velocity(
        reynolds=reynolds_external,
        rho=rho_inf,
        mu=mu_inf,
        chord=chord,
    )

    flow = FlowConfig(
        aoa_deg=aoa_deg,
        v_inf=v_inf,
        rho_inf=rho_inf,
        mu_inf=mu_inf,
        p_inf=p_inf,
    )

    xb, yb = generate_naca4(
        airfoil_name,
        n_panels,
        chord=chord,
    )

    geom = PanelGeometry(xb, yb, aoa_deg)
    ref = ReferenceConfig(x_ref=0.25 * chord, y_ref=0.0)
    aero_solver = SourceVortexPanelMethod(geom, flow, ref)

    return flow, geom, ref, aero_solver


def build_network_from_model(
    model: PorousModelSpec,
    span_width: float,
) -> IndependentPassageNetwork:
    """Build the 2D rectangular network.

    ``span_width`` is accepted for backward compatibility but is not used by
    the per-unit-span hydraulic or transpiration equations.
    """
    if span_width <= 0.0:
        raise ValueError("span_width must be positive.")

    passages: list[IndependentPassage] = []
    for spec in model.passages:
        height = float(spec.channel_height_m)
        passages.append(
            IndependentPassage(
                name=spec.name,
                pore1=Pore(spec.x1_frac, spec.side1, height),
                pore2=Pore(spec.x2_frac, spec.side2, height),
                chamber=Chamber(height=height),
                span_width=float(span_width),
                layout_kind=spec.layout_kind,
            )
        )
    return IndependentPassageNetwork(passages)



# =============================================================================
# XFOIL HELPERS
# =============================================================================
def run_xfoil_cp_for_case(
    airfoil_name: str,
    aoa_deg: float,
    output_dir: Path,
) -> dict | None:
    """
    Run XFOIL Cp export and return loaded Cp data.
    """
    if not USE_XFOIL or XFOIL_EXE_PATH is None:
        print("[XFOIL] Skipping Cp comparison because xfoil.exe was not found.")
        return None

    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        cp_path = run_xfoil_cp(
            xfoil_exe_path=XFOIL_EXE_PATH,
            airfoil_name=airfoil_name,
            aoa_deg=aoa_deg,
            output_dir=output_dir,
            mach=XFOIL_MACH,
            timeout=XFOIL_TIMEOUT,
            reynolds=None,
        )

        print(f"[XFOIL] Cp file created: {cp_path}")
        return load_xfoil_cp(cp_path)

    except Exception as exc:
        print(f"[XFOIL] Cp comparison failed: {exc}")
        return None


def run_xfoil_aoa_sweep_for_case(
    airfoil_name: str,
    output_dir: Path,
    alpha_start: float = -5.0,
    alpha_end: float = 15.0,
    alpha_step: float = 1.0,
) -> pd.DataFrame | None:
    """
    Run XFOIL AoA sweep.
    """
    if not USE_XFOIL or XFOIL_EXE_PATH is None:
        print("[XFOIL] Skipping AoA sweep because xfoil.exe was not found.")
        return None

    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        sweep_path = run_xfoil_polar_sweep(
            xfoil_exe_path=XFOIL_EXE_PATH,
            airfoil_name=airfoil_name,
            alpha_start=alpha_start,
            alpha_end=alpha_end,
            alpha_step=alpha_step,
            output_dir=output_dir,
            mach=XFOIL_MACH,
            timeout=XFOIL_TIMEOUT,
            reynolds=None,  # inviscid XFOIL polar for stable comparison overlay
        )

        print(f"[XFOIL] AoA sweep file created: {sweep_path}")
        return load_xfoil_polar_dataframe(sweep_path)

    except Exception as exc:
        print(f"[XFOIL] AoA sweep failed: {exc}")
        return None


def add_xfoil_to_sweep_comparison(
    sweep_df: pd.DataFrame,
    xfoil_sweep_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge XFOIL polar data into panel-method sweep data.
    """
    xfoil_df = xfoil_sweep_df.copy()
    xfoil_df.columns = [str(c).strip() for c in xfoil_df.columns]

    rename_map: dict[str, str] = {}

    for col in xfoil_df.columns:
        c = col.strip().lower()

        if c in {"alpha", "alpha_deg", "aoa", "aoa_deg"}:
            rename_map[col] = "alpha_deg"
        elif c == "cl":
            rename_map[col] = "xfoil_CL"
        elif c == "cd":
            rename_map[col] = "xfoil_CD"
        elif c == "cm":
            rename_map[col] = "xfoil_CM"

    xfoil_df = xfoil_df.rename(columns=rename_map)

    required = {"alpha_deg", "xfoil_CL", "xfoil_CD", "xfoil_CM"}
    missing = required - set(xfoil_df.columns)

    if missing:
        raise ValueError(
            f"XFOIL sweep dataframe missing columns after normalization: {sorted(missing)}. "
            f"Available columns: {list(xfoil_sweep_df.columns)}"
        )

    return sweep_df.merge(
        xfoil_df[["alpha_deg", "xfoil_CL", "xfoil_CD", "xfoil_CM"]],
        on="alpha_deg",
        how="left",
    )



# =============================================================================
# CSV EXPORTS
# =============================================================================
def build_passage_dataframe(
    model: PorousModelSpec,
    network: IndependentPassageNetwork,
    geom: PanelGeometry,
    result: MultiPassageResult,
) -> pd.DataFrame:
    rows = []

    for passage_id, (passage, state) in enumerate(
        zip(network.passages, result.passage_states),
        start=1,
    ):
        sp1 = passage.pore_surface_point(geom, passage.pore1)
        sp2 = passage.pore_surface_point(geom, passage.pore2)

        rows.append(
            {
                "model_name": model.name,
                "passage_id": passage_id,
                "passage_name": passage.name,
                "pore1_side": passage.pore1.side,
                "pore2_side": passage.pore2.side,
                "x1_frac": passage.pore1.x_frac,
                "x2_frac": passage.pore2.x_frac,
                "x1_surface_m": sp1.x,
                "y1_surface_m": sp1.y,
                "x2_surface_m": sp2.x,
                "y2_surface_m": sp2.y,
                "x1_over_c_surface": sp1.x / geom.chord,
                "y1_over_c_surface": sp1.y / geom.chord,
                "x2_over_c_surface": sp2.x / geom.chord,
                "y2_over_c_surface": sp2.y / geom.chord,
                "channel_height_m": passage.chamber.height,
                "flow_area_per_unit_span_m": passage.chamber.area,
                "channel_planform_area_m2": passage.chamber.planform_area(
                    passage.passage_length(geom)
                ),
                "hydraulic_diameter_m": passage.chamber.hydraulic_diameter,
                # Compatibility column used by existing plotting scripts.
                "pore_diameter_m": passage.pore1.height,
                "passage_length_m": passage.passage_length(geom),
                "q_per_unit_span_m2_s": state.Q,
                "abs_q_per_unit_span_m2_s": abs(state.Q),
                "Rprime_Pa_s_per_m2": state.Rs,
                "dp_total_Pa": state.dp_total,
                "reynolds_internal_equivalent": state.reynolds_equivalent,
                "p1_Pa": state.p1,
                "p2_Pa": state.p2,
                "friction_factor": state.friction_factor,
                "resistance_model": state.resistance_model,
                **getattr(state, "energy_diagnostics", {}),
            }
        )

    return pd.DataFrame(rows)


def representative_channel_height(model: PorousModelSpec) -> float:
    """Return the mean rectangular channel height for summaries."""
    values = [float(p.channel_height_m) for p in model.passages]
    return float(np.mean(values)) if values else float("nan")


def representative_channel_width(model: PorousModelSpec) -> float:
    """Compatibility value: the active model is per unit span."""
    del model
    return 1.0


def representative_pore_diameter(model: PorousModelSpec) -> float:
    """Compatibility alias returning rectangular channel height."""
    return representative_channel_height(model)


def coupling_diagnostics(result: MultiPassageResult) -> dict:
    """Diagnostics are retained even when physical results are NaN."""
    return {
        "energy_closure": CHANNEL_ENERGY_CLOSURE,
        "energy_model_validated": False,
        "aerodynamic_force_definition": "closed_contour_pressure_integral",
        "max_channel_energy_balance_residual_Pa": max(
            (abs(getattr(s, "energy_diagnostics", {}).get("energy_balance_residual_Pa", float("nan")))
             for s in result.passage_states), default=float("nan")),
        "coupling_status": result.coupling_status,
        "coupling_residual_max": result.residual_norm,
        "coupling_residual_tolerance": result.residual_tolerance,
        "coupling_solver_success": result.solver_success,
        "coupling_solver_message": result.solver_message,
        "coupling_function_evaluations": result.function_evaluations,
    }


def nan_if_not_converged(value: float, converged: bool) -> float:
    """Return value for converged cases and NaN for non-converged cases."""
    return float(value) if converged else float("nan")


def make_nonconverged_summary_row(
    model: PorousModelSpec,
    flow: FlowConfig,
    solid_result: object,
    porous_result: MultiPassageResult,
    elapsed_seconds: float,
) -> dict:
    """Summary row used when coupling did not converge.

    The requested behaviour is: if a porous case does not converge, report NaN
    for the porous aerodynamic/internal-flow quantities and do not treat the
    unconverged values as valid results.
    """
    nan = float("nan")
    return {
        "model_name": model.name,
        "description": model.description,
        "reference_unit_span_m": 1.0,
        "channel_height_m": representative_channel_height(model),
        "pore_diameter_m": representative_channel_height(model),
        "num_independent_passages": len(model.passages),
        "reynolds_external": REYNOLDS_EXTERNAL,
        "v_inf_m_s": flow.v_inf,
        "aoa_deg": flow.aoa_deg,
        "solid_CL": solid_result.CL,
        "solid_CD": solid_result.CD,
        "solid_CM": solid_result.CM,
        "porous_CL": nan,
        "porous_CD": nan,
        "porous_CM": nan,
        "delta_CL": nan,
        "delta_CD": nan,
        "delta_CM": nan,
        "gain_percent_vs_solid_panel": nan,
        "max_abs_q_per_unit_span_m2_s": nan,
        "sum_abs_q_per_unit_span_m2_s": nan,
        "max_dp_total_Pa": nan,
        "max_internal_reynolds_equivalent": nan,
        "max_vn_m_s": nan,
        "max_vn_over_vinf": nan,
        "coupling_converged": False,
        "coupling_iterations": porous_result.iterations,
        **coupling_diagnostics(porous_result),
        "elapsed_seconds": elapsed_seconds,
    }


def export_surface_data(
    output_dir: Path,
    geom: PanelGeometry,
    solid_result: object,
    porous_result: MultiPassageResult,
) -> None:
    aero = porous_result.aero_result

    df = pd.DataFrame(
        {
            "panel_id": np.arange(geom.num_pan),
            "XB_start_m": geom.XB[:-1],
            "YB_start_m": geom.YB[:-1],
            "XB_end_m": geom.XB[1:],
            "YB_end_m": geom.YB[1:],
            "XC_m": geom.XC,
            "YC_m": geom.YC,
            "XC_over_c": geom.XC / geom.chord,
            "YC_over_c": geom.YC / geom.chord,
            "panel_length_m": geom.S,
            "porous_Vt_m_s": aero.Vt,
            "porous_Cp": aero.Cp,
            "solid_Vt_m_s": solid_result.Vt,
            "solid_Cp": solid_result.Cp,
            "normal_transpiration_m_s": porous_result.normal_transpiration,
        }
    )

    path = output_dir / "surface_data.csv"
    df.to_csv(path, index=False)
    print(f"[CSV Saved] {path}")


def build_model_summary_row(
    model: PorousModelSpec,
    flow: FlowConfig,
    solid_result: object,
    porous_result: MultiPassageResult,
    elapsed_seconds: float,
) -> dict:
    if not porous_result.converged:
        return make_nonconverged_summary_row(
            model=model,
            flow=flow,
            solid_result=solid_result,
            porous_result=porous_result,
            elapsed_seconds=elapsed_seconds,
        )

    aero = porous_result.aero_result

    q_abs = [abs(float(s.Q)) for s in porous_result.passage_states]
    dp_values = [float(s.dp_total) for s in porous_result.passage_states]
    re_values = [float(s.reynolds_equivalent) for s in porous_result.passage_states]

    return {
        "model_name": model.name,
        "description": model.description,
        "reference_unit_span_m": 1.0,
        "channel_height_m": representative_channel_height(model),
        "pore_diameter_m": representative_channel_height(model),
        "num_independent_passages": len(model.passages),
        "reynolds_external": REYNOLDS_EXTERNAL,
        "v_inf_m_s": flow.v_inf,
        "aoa_deg": flow.aoa_deg,
        "solid_CL": solid_result.CL,
        "solid_CD": solid_result.CD,
        "solid_CM": solid_result.CM,
        "porous_CL": aero.CL,
        "porous_CD": aero.CD,
        "porous_CM": aero.CM,
        "delta_CL": aero.CL - solid_result.CL,
        "delta_CD": aero.CD - solid_result.CD,
        "delta_CM": aero.CM - solid_result.CM,
        "gain_percent_vs_solid_panel": 100.0 * (aero.CL / solid_result.CL - 1.0),
        "max_abs_q_per_unit_span_m2_s": max(q_abs) if q_abs else 0.0,
        "sum_abs_q_per_unit_span_m2_s": sum(q_abs),
        "max_dp_total_Pa": max(dp_values) if dp_values else 0.0,
        "max_internal_reynolds_equivalent": max(re_values) if re_values else 0.0,
        "max_vn_m_s": porous_result.max_vn,
        "max_vn_over_vinf": porous_result.max_vn_over_vinf,
        "coupling_converged": porous_result.converged,
        "coupling_iterations": porous_result.iterations,
        **coupling_diagnostics(porous_result),
        "elapsed_seconds": elapsed_seconds,
    }



def _alpha_directory_name(alpha_deg: float) -> str:
    """Return a stable folder name for one AoA sweep case."""
    sign = "p" if alpha_deg >= 0.0 else "m"
    magnitude = f"{abs(alpha_deg):06.2f}".replace(".", "p")
    return f"alpha_{sign}{magnitude}_deg"


# =============================================================================
# CONTINUATION HELPERS
# =============================================================================
def _result_flow_vector(result: MultiPassageResult) -> np.ndarray | None:
    """Return the converged physical Q vector [m^2/s], or None."""
    if not result.converged or not result.passage_states:
        return None
    q = np.asarray([float(state.Q) for state in result.passage_states], dtype=float)
    return q if np.all(np.isfinite(q)) else None


def _prefer_result(
    first: MultiPassageResult,
    second: MultiPassageResult,
) -> MultiPassageResult:
    """Prefer a converged result; otherwise keep the smaller finite residual."""
    if second.converged and not first.converged:
        return second
    if first.converged and not second.converged:
        return first
    r1 = float(first.residual_norm)
    r2 = float(second.residual_norm)
    if np.isfinite(r2) and (not np.isfinite(r1) or r2 < r1):
        return second
    return first


def _build_coupled_case_context(
    model: PorousModelSpec,
    airfoil_name: str,
    chord: float,
    n_panels: int,
    alpha_deg: float,
    reynolds_external: float,
    rho_inf: float,
    mu_inf: float,
    p_inf: float,
) -> tuple[
    FlowConfig,
    PanelGeometry,
    SourceVortexPanelMethod,
    object,
    IndependentPassageNetwork,
]:
    """Build one requested physical case without solving the porous coupling."""
    flow_i, geom_i, _, aero_solver_i = build_base_problem(
        airfoil_name=airfoil_name,
        chord=chord,
        n_panels=n_panels,
        aoa_deg=float(alpha_deg),
        reynolds_external=float(reynolds_external),
        rho_inf=rho_inf,
        mu_inf=mu_inf,
        p_inf=p_inf,
    )
    solid_i = aero_solver_i.solve()
    network_i = build_network_from_model(model=model, span_width=SPAN_WIDTH)
    return flow_i, geom_i, aero_solver_i, solid_i, network_i


def _solve_context(
    aero_solver: SourceVortexPanelMethod,
    network: IndependentPassageNetwork,
    coupling: CouplingConfig,
    limits: GeometryLimits,
    initial_q: np.ndarray | None = None,
    coupling_strength: float = 1.0,
    verbose: bool = False,
) -> MultiPassageResult:
    return MultiPassageCoupledSolver(
        aero_solver=aero_solver,
        network=network,
        coupling=coupling,
        limits=limits,
    ).solve(
        verbose=verbose,
        initial_q=initial_q,
        coupling_strength=coupling_strength,
    )


def _solve_full_coupling_with_cold_retry(
    aero_solver: SourceVortexPanelMethod,
    network: IndependentPassageNetwork,
    coupling: CouplingConfig,
    limits: GeometryLimits,
    initial_q: np.ndarray | None,
) -> MultiPassageResult:
    """Solve lambda=1, retrying from a cold start if a warm start fails."""
    warm = _solve_context(
        aero_solver=aero_solver,
        network=network,
        coupling=coupling,
        limits=limits,
        initial_q=initial_q,
        coupling_strength=1.0,
        verbose=False,
    )
    if warm.converged or initial_q is None:
        return warm

    cold = _solve_context(
        aero_solver=aero_solver,
        network=network,
        coupling=coupling,
        limits=limits,
        initial_q=None,
        coupling_strength=1.0,
        verbose=False,
    )
    return _prefer_result(warm, cold)


def _solve_with_coupling_homotopy(
    aero_solver: SourceVortexPanelMethod,
    network: IndependentPassageNetwork,
    coupling: CouplingConfig,
    limits: GeometryLimits,
    initial_q: np.ndarray | None,
    original_full_result: MultiPassageResult,
) -> MultiPassageResult:
    """Continue lambda from 0 to 1 and bracket a coupling branch limit.

    A converged state is always kept on the low-lambda side.  Once a failed
    state is encountered, the interval is bisected rather than repeatedly
    retrying exactly lambda=1.  Only lambda=1 may be returned as a converged
    physical result.
    """
    q_seed = None if initial_q is None else np.asarray(initial_q, dtype=float).copy()
    stages: list[str] = []

    stage0 = _solve_context(
        aero_solver=aero_solver,
        network=network,
        coupling=coupling,
        limits=limits,
        initial_q=q_seed,
        coupling_strength=0.0,
        verbose=False,
    )
    stages.append(f"lambda=0:{'ok' if stage0.converged else 'fail'}")
    if not stage0.converged:
        original_full_result.solver_message += " | homotopy failed at lambda=0"
        return original_full_result

    low_lam = 0.0
    low_q = _result_flow_vector(stage0)
    high_lam: float | None = None
    step = float(HOMOTOPY_INITIAL_STEP)

    while low_lam < 1.0 - 1.0e-12:
        if high_lam is None:
            trial_lam = min(1.0, low_lam + step)
        else:
            trial_lam = 0.5 * (low_lam + high_lam)

        stage = _solve_context(
            aero_solver=aero_solver,
            network=network,
            coupling=coupling,
            limits=limits,
            initial_q=low_q,
            coupling_strength=trial_lam,
            verbose=False,
        )

        if stage.converged:
            stages.append(f"lambda={trial_lam:.5f}:ok")
            low_lam = float(trial_lam)
            low_q = _result_flow_vector(stage)
            if low_lam >= 1.0 - 1.0e-12:
                stage.solver_message = (
                    "continuation=coupling_homotopy; "
                    + ", ".join(stages)
                    + " | "
                    + stage.solver_message
                )
                return stage
            if high_lam is None:
                step = min(0.25, step * 1.4)
        else:
            stages.append(
                f"lambda={trial_lam:.5f}:fail(R={stage.residual_norm:.3e})"
            )
            high_lam = float(trial_lam)

        if high_lam is not None and (high_lam - low_lam) <= HOMOTOPY_BOUNDARY_TOL:
            original_full_result.solver_message += (
                " | homotopy branch boundary bracketed: "
                f"lambda_converged={low_lam:.6f}, lambda_failed={high_lam:.6f}; "
                + ", ".join(stages)
            )
            return original_full_result

        if high_lam is None and step < HOMOTOPY_MIN_STEP:
            original_full_result.solver_message += (
                " | homotopy did not reach lambda=1; " + ", ".join(stages)
            )
            return original_full_result

    return original_full_result


def _solve_with_reynolds_continuation(
    model: PorousModelSpec,
    airfoil_name: str,
    chord: float,
    n_panels: int,
    alpha_deg: float,
    target_reynolds: float,
    rho_inf: float,
    mu_inf: float,
    p_inf: float,
    coupling: CouplingConfig,
    limits: GeometryLimits,
    initial_q: np.ndarray | None,
    original_target_result: MultiPassageResult,
) -> tuple[
    FlowConfig,
    PanelGeometry,
    SourceVortexPanelMethod,
    object,
    IndependentPassageNetwork,
    MultiPassageResult,
]:
    """Continue Reynolds number and bracket a loss-of-root boundary."""
    target_reynolds = float(target_reynolds)
    start_re = min(float(RE_CONTINUATION_REFERENCE), target_reynolds)
    target_context = _build_coupled_case_context(
        model, airfoil_name, chord, n_panels, alpha_deg, target_reynolds,
        rho_inf, mu_inf, p_inf,
    )

    if target_reynolds <= start_re + 1.0:
        return (*target_context, original_target_result)

    start_context = _build_coupled_case_context(
        model, airfoil_name, chord, n_panels, alpha_deg, start_re,
        rho_inf, mu_inf, p_inf,
    )
    start_result = _solve_full_coupling_with_cold_retry(
        aero_solver=start_context[2],
        network=start_context[4],
        coupling=coupling,
        limits=limits,
        initial_q=initial_q,
    )
    if not start_result.converged:
        original_target_result.solver_message += (
            f" | Re continuation could not establish start Re={start_re:.6g}"
        )
        return (*target_context, original_target_result)

    low_re = float(start_re)
    low_q = _result_flow_vector(start_result)
    high_re: float | None = None
    step = max(
        (target_reynolds - start_re) / max(1, RE_CONTINUATION_INITIAL_DIVISIONS),
        RE_CONTINUATION_MIN_STEP,
    )
    stages = [f"Re={low_re:.6g}:ok"]

    while low_re < target_reynolds - 1.0:
        if high_re is None:
            trial_re = min(target_reynolds, low_re + step)
        else:
            trial_re = 0.5 * (low_re + high_re)

        trial_context = _build_coupled_case_context(
            model, airfoil_name, chord, n_panels, alpha_deg, trial_re,
            rho_inf, mu_inf, p_inf,
        )
        trial_result = _solve_full_coupling_with_cold_retry(
            aero_solver=trial_context[2],
            network=trial_context[4],
            coupling=coupling,
            limits=limits,
            initial_q=low_q,
        )

        if trial_result.converged:
            stages.append(f"Re={trial_re:.6g}:ok")
            low_re = float(trial_re)
            low_q = _result_flow_vector(trial_result)
            if low_re >= target_reynolds - 1.0:
                trial_result.solver_message = (
                    "continuation=reynolds; " + ", ".join(stages)
                    + " | " + trial_result.solver_message
                )
                return (*trial_context, trial_result)
            if high_re is None:
                step = min(step * 1.4, target_reynolds - low_re)
        else:
            stages.append(
                f"Re={trial_re:.6g}:fail(R={trial_result.residual_norm:.3e})"
            )
            high_re = float(trial_re)

        if high_re is not None and (high_re - low_re) <= RE_BOUNDARY_TOL:
            original_target_result.solver_message += (
                " | Reynolds branch boundary bracketed: "
                f"Re_converged={low_re:.6g}, Re_failed={high_re:.6g}; "
                + ", ".join(stages)
            )
            return (*target_context, original_target_result)

    return (*target_context, original_target_result)


def _adaptive_aoa_continuation_to_target(
    model: PorousModelSpec,
    airfoil_name: str,
    chord: float,
    n_panels: int,
    previous_alpha: float,
    target_alpha: float,
    reynolds_external: float,
    rho_inf: float,
    mu_inf: float,
    p_inf: float,
    coupling: CouplingConfig,
    limits: GeometryLimits,
    previous_q: np.ndarray,
    original_target_result: MultiPassageResult,
) -> tuple[
    FlowConfig,
    PanelGeometry,
    SourceVortexPanelMethod,
    object,
    IndependentPassageNetwork,
    MultiPassageResult,
    dict,
]:
    """Continue in AoA and refine the converged/failed branch boundary.

    The requested target is retried from every newly converged midpoint.  If it
    remains unresolved, the routine returns a bracket [alpha_converged,
    alpha_failed] plus the converged Q vector on the low side.  This metadata is
    used by pseudo-arclength continuation.
    """
    previous_alpha = float(previous_alpha)
    target_alpha = float(target_alpha)
    target_context = _build_coupled_case_context(
        model, airfoil_name, chord, n_panels, target_alpha, reynolds_external,
        rho_inf, mu_inf, p_inf,
    )
    metadata = {
        "boundary_found": False,
        "alpha_converged": float("nan"),
        "alpha_failed": float("nan"),
        "q_converged": None,
        "stages": [],
    }

    if abs(target_alpha - previous_alpha) <= AOA_BOUNDARY_TOL_DEG:
        return (*target_context, original_target_result, metadata)

    direction = 1.0 if target_alpha > previous_alpha else -1.0
    low_alpha = previous_alpha
    low_q = np.asarray(previous_q, dtype=float).copy()
    high_alpha = target_alpha
    stages: list[str] = [f"alpha={low_alpha:.6g}:known_ok"]

    while abs(high_alpha - low_alpha) > AOA_BOUNDARY_TOL_DEG:
        mid_alpha = 0.5 * (low_alpha + high_alpha)
        mid_context = _build_coupled_case_context(
            model, airfoil_name, chord, n_panels, mid_alpha, reynolds_external,
            rho_inf, mu_inf, p_inf,
        )
        mid_result = _solve_full_coupling_with_cold_retry(
            aero_solver=mid_context[2],
            network=mid_context[4],
            coupling=coupling,
            limits=limits,
            initial_q=low_q,
        )

        if mid_result.converged:
            stages.append(f"alpha={mid_alpha:.6g}:ok")
            low_alpha = float(mid_alpha)
            low_q = _result_flow_vector(mid_result)

            # A target that failed from the old seed may become reachable from a
            # state much closer to it, so retry the exact requested equations.
            retry = _solve_full_coupling_with_cold_retry(
                aero_solver=target_context[2],
                network=target_context[4],
                coupling=coupling,
                limits=limits,
                initial_q=low_q,
            )
            if retry.converged:
                stages.append(f"alpha={target_alpha:.6g}:ok_from_refined_seed")
                retry.solver_message = (
                    "continuation=adaptive_aoa_boundary; " + ", ".join(stages)
                    + " | " + retry.solver_message
                )
                metadata.update({
                    "boundary_found": False,
                    "alpha_converged": target_alpha,
                    "alpha_failed": float("nan"),
                    "q_converged": _result_flow_vector(retry),
                    "stages": stages,
                })
                return (*target_context, retry, metadata)
        else:
            stages.append(f"alpha={mid_alpha:.6g}:fail(R={mid_result.residual_norm:.3e})")
            high_alpha = float(mid_alpha)

        # Keep the bracket ordered in the physical marching direction.
        if direction < 0.0 and low_alpha < high_alpha:
            low_alpha, high_alpha = high_alpha, low_alpha

    metadata.update({
        "boundary_found": True,
        "alpha_converged": float(low_alpha),
        "alpha_failed": float(high_alpha),
        "q_converged": None if low_q is None else np.asarray(low_q, dtype=float).copy(),
        "stages": stages,
    })
    original_target_result.solver_message += (
        " | AoA branch boundary bracketed: "
        f"alpha_converged={low_alpha:.6f}, alpha_failed={high_alpha:.6f}; "
        + ", ".join(stages)
    )
    return (*target_context, original_target_result, metadata)




def _q_scale_for_context(
    network: IndependentPassageNetwork,
    flow: FlowConfig,
) -> np.ndarray:
    """Return the H*Vinf scaling used by the nonlinear passage-flow solve."""
    return np.asarray(
        [
            max(
                float(passage.chamber.height) * float(flow.v_inf),
                np.finfo(float).tiny,
            )
            for passage in network.passages
        ],
        dtype=float,
    )


def _states_for_fixed_q(q_values: np.ndarray) -> list[PassageState]:
    """Create lightweight passage states for a prescribed physical Q vector."""
    return [
        PassageState(
            Q=float(q_value),
            p1=0.0,
            p2=0.0,
            dp_total=0.0,
            Rs=0.0,
            reynolds_equivalent=0.0,
        )
        for q_value in np.asarray(q_values, dtype=float)
    ]


def _evaluate_coupled_residual_at_q_alpha(
    model: PorousModelSpec,
    airfoil_name: str,
    chord: float,
    n_panels: int,
    alpha_deg: float,
    reynolds_external: float,
    rho_inf: float,
    mu_inf: float,
    p_inf: float,
    q_dimensionless: np.ndarray,
) -> np.ndarray:
    """Evaluate the original full-coupling residual at arbitrary (Q, alpha).

    This is deliberately the same residual used by MultiPassageCoupledSolver at
    lambda=1.  It is exposed here only so pseudo-arclength continuation can
    treat alpha as an additional unknown.
    """
    flow_i, geom_i, aero_solver_i, _, network_i = _build_coupled_case_context(
        model=model,
        airfoil_name=airfoil_name,
        chord=chord,
        n_panels=n_panels,
        alpha_deg=float(alpha_deg),
        reynolds_external=float(reynolds_external),
        rho_inf=rho_inf,
        mu_inf=mu_inf,
        p_inf=p_inf,
    )
    q_scale = _q_scale_for_context(network_i, flow_i)
    q_dimensionless = np.asarray(q_dimensionless, dtype=float)
    if q_dimensionless.shape != q_scale.shape or not np.all(np.isfinite(q_dimensionless)):
        raise FloatingPointError("Invalid pseudo-arclength Q state.")

    q_physical = q_dimensionless * q_scale
    vn = network_i.transpiration_from_flow(
        geom=geom_i,
        states=_states_for_fixed_q(q_physical),
    )
    aero = aero_solver_i.solve(normal_transpiration=vn)
    pressures = aero_solver_i.panel_pressures(aero)
    hydraulic_states = network_i.solve(
        geom=geom_i,
        panel_pressures=pressures,
        aero_result=aero,
        flow=flow_i,
    )
    q_pressure = np.asarray([float(state.Q) for state in hydraulic_states], dtype=float)
    residual = (q_physical - q_pressure) / q_scale

    if not np.all(np.isfinite(residual)):
        raise FloatingPointError("Non-finite pseudo-arclength residual.")
    return residual


def _pseudo_arclength_to_target(
    model: PorousModelSpec,
    airfoil_name: str,
    chord: float,
    n_panels: int,
    reynolds_external: float,
    rho_inf: float,
    mu_inf: float,
    p_inf: float,
    coupling: CouplingConfig,
    limits: GeometryLimits,
    alpha0: float,
    q0: np.ndarray,
    alpha1: float,
    q1: np.ndarray,
    target_alpha: float,
    original_target_result: MultiPassageResult,
) -> tuple[MultiPassageResult, dict]:
    """Trace the full-coupling branch through a possible fold.

    Two already-converged points seed a secant tangent.  The augmented system is

        R(Q, alpha) = 0
        t . (z - z_predictor) = 0,

    with z=[Q/(H Vinf), alpha/PAL_ALPHA_SCALE_DEG].  Pseudo-arclength points are
    diagnostic continuation states only.  A physical sweep result is accepted
    only after the exact requested alpha is solved again at lambda=1 and the
    normal residual criterion is satisfied.
    """
    target_alpha = float(target_alpha)
    alpha0 = float(alpha0)
    alpha1 = float(alpha1)
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)

    base_context = _build_coupled_case_context(
        model, airfoil_name, chord, n_panels, alpha1, reynolds_external,
        rho_inf, mu_inf, p_inf,
    )
    q_scale = _q_scale_for_context(base_context[4], base_context[0])
    if q0.shape != q_scale.shape or q1.shape != q_scale.shape:
        metadata = {"attempted": False, "reason": "seed_shape_mismatch"}
        return original_target_result, metadata

    z_prev = np.concatenate((q0 / q_scale, [alpha0 / PAL_ALPHA_SCALE_DEG]))
    z_curr = np.concatenate((q1 / q_scale, [alpha1 / PAL_ALPHA_SCALE_DEG]))
    secant = z_curr - z_prev
    secant_norm = float(np.linalg.norm(secant))
    if not np.isfinite(secant_norm) or secant_norm <= 1.0e-14:
        metadata = {"attempted": False, "reason": "degenerate_seed_secant"}
        return original_target_result, metadata

    tangent = secant / secant_norm
    desired_alpha_direction = np.sign(target_alpha - alpha1)
    if desired_alpha_direction != 0.0 and tangent[-1] * desired_alpha_direction < 0.0:
        tangent = -tangent

    ds = float(PAL_INITIAL_DS)
    branch_points: list[tuple[float, np.ndarray]] = [
        (alpha0, q0.copy()),
        (alpha1, q1.copy()),
    ]
    stages: list[str] = []
    fold_detected = False
    best_distance = abs(alpha1 - target_alpha)
    best_q = q1.copy()
    previous_alpha = alpha1

    target_context = _build_coupled_case_context(
        model, airfoil_name, chord, n_panels, target_alpha, reynolds_external,
        rho_inf, mu_inf, p_inf,
    )

    for step_id in range(1, PAL_MAX_STEPS + 1):
        z_predictor = z_curr + ds * tangent

        def augmented_residual(z: np.ndarray) -> np.ndarray:
            z = np.asarray(z, dtype=float)
            q_dim = z[:-1]
            alpha_trial = float(z[-1] * PAL_ALPHA_SCALE_DEG)
            if not np.isfinite(alpha_trial) or abs(alpha_trial) > PAL_ALPHA_LIMIT_DEG:
                return np.full(z.shape, 1.0e3, dtype=float)
            physical_residual = _evaluate_coupled_residual_at_q_alpha(
                model=model,
                airfoil_name=airfoil_name,
                chord=chord,
                n_panels=n_panels,
                alpha_deg=alpha_trial,
                reynolds_external=reynolds_external,
                rho_inf=rho_inf,
                mu_inf=mu_inf,
                p_inf=p_inf,
                q_dimensionless=q_dim,
            )
            arc_residual = float(np.dot(z - z_predictor, tangent))
            return np.concatenate((physical_residual, [arc_residual]))

        best_z = None
        best_norm = float("inf")
        method = "none"

        try:
            rr = root(
                augmented_residual,
                z_predictor,
                method="hybr",
                options={
                    "maxfev": max(500, int(coupling.max_iter)),
                    "factor": 0.5,
                    "xtol": 1.0e-10,
                },
            )
            zr = np.asarray(rr.x, dtype=float)
            if np.all(np.isfinite(zr)):
                nr = float(np.max(np.abs(augmented_residual(zr))))
                if nr < best_norm:
                    best_z, best_norm, method = zr.copy(), nr, "PAL-HYBR"
        except (FloatingPointError, OverflowError, ValueError, RuntimeError):
            pass

        if best_norm > PAL_CORRECTOR_TOL:
            try:
                lr = least_squares(
                    augmented_residual,
                    x0=z_predictor if best_z is None else best_z,
                    method="trf",
                    jac="3-point",
                    x_scale="jac",
                    ftol=1.0e-12,
                    xtol=1.0e-12,
                    gtol=1.0e-12,
                    max_nfev=max(1500, 5 * int(coupling.max_iter)),
                )
                zl = np.asarray(lr.x, dtype=float)
                if np.all(np.isfinite(zl)):
                    nl = float(np.max(np.abs(augmented_residual(zl))))
                    if nl < best_norm:
                        best_z, best_norm, method = zl.copy(), nl, "PAL-TRF"
            except (FloatingPointError, OverflowError, ValueError, RuntimeError):
                pass

        if best_z is None or best_norm > PAL_CORRECTOR_TOL:
            stages.append(f"s={step_id}:corrector_fail(R={best_norm:.3e},ds={ds:.4g})")
            ds *= 0.5
            if ds < PAL_MIN_DS:
                break
            continue

        z_new = best_z
        alpha_new = float(z_new[-1] * PAL_ALPHA_SCALE_DEG)
        q_new = z_new[:-1] * q_scale
        stages.append(
            f"s={step_id}:{method}:alpha={alpha_new:.6f},R={best_norm:.2e},ds={ds:.4g}"
        )
        branch_points.append((alpha_new, q_new.copy()))

        # Detect a turn in alpha along the continuously oriented branch.
        if desired_alpha_direction != 0.0:
            if (alpha_new - previous_alpha) * desired_alpha_direction < -1.0e-8:
                fold_detected = True

        distance = abs(alpha_new - target_alpha)
        if distance < best_distance:
            best_distance = distance
            best_q = q_new.copy()

        crossed_target = (
            (previous_alpha - target_alpha) * (alpha_new - target_alpha) <= 0.0
            and abs(alpha_new - previous_alpha) > 1.0e-12
        )
        if crossed_target or distance <= PAL_TARGET_TOL_DEG:
            target_try = _solve_full_coupling_with_cold_retry(
                aero_solver=target_context[2],
                network=target_context[4],
                coupling=coupling,
                limits=limits,
                initial_q=q_new,
            )
            if target_try.converged:
                target_try.solver_message = (
                    "continuation=pseudo_arclength; "
                    f"fold_detected={fold_detected}; "
                    + ", ".join(stages)
                    + " | " + target_try.solver_message
                )
                metadata = {
                    "attempted": True,
                    "fold_detected": fold_detected,
                    "alpha_min": float(min(a for a, _ in branch_points)),
                    "alpha_max": float(max(a for a, _ in branch_points)),
                    "steps": len(branch_points) - 2,
                    "reached_target": True,
                }
                return target_try, metadata

        new_tangent = z_new - z_curr
        new_norm = float(np.linalg.norm(new_tangent))
        if not np.isfinite(new_norm) or new_norm <= 1.0e-14:
            break
        new_tangent /= new_norm
        if float(np.dot(new_tangent, tangent)) < 0.0:
            new_tangent = -new_tangent

        z_prev = z_curr
        z_curr = z_new
        tangent = new_tangent
        previous_alpha = alpha_new
        ds = min(PAL_MAX_DS, ds * 1.2)

    # One final exact-target attempt from the closest arclength point.
    final_try = _solve_full_coupling_with_cold_retry(
        aero_solver=target_context[2],
        network=target_context[4],
        coupling=coupling,
        limits=limits,
        initial_q=best_q,
    )
    if final_try.converged:
        final_try.solver_message = (
            "continuation=pseudo_arclength_nearest_seed; "
            f"fold_detected={fold_detected}; "
            + ", ".join(stages)
            + " | " + final_try.solver_message
        )
        result = final_try
        reached = True
    else:
        original_target_result.solver_message += (
            " | pseudo-arclength diagnostic: "
            f"fold_detected={fold_detected}, "
            f"alpha_min={min(a for a, _ in branch_points):.6f}, "
            f"alpha_max={max(a for a, _ in branch_points):.6f}, "
            f"closest_to_target={best_distance:.6f} deg; "
            + ", ".join(stages)
        )
        result = _prefer_result(original_target_result, final_try)
        reached = bool(result.converged)

    metadata = {
        "attempted": True,
        "fold_detected": fold_detected,
        "alpha_min": float(min(a for a, _ in branch_points)),
        "alpha_max": float(max(a for a, _ in branch_points)),
        "steps": len(branch_points) - 2,
        "reached_target": reached,
    }
    return result, metadata

# =============================================================================
# AOA SWEEP
# =============================================================================
def run_porous_aoa_sweep_for_model(
    model: PorousModelSpec,
    airfoil_name: str,
    chord: float,
    n_panels: int,
    rho_inf: float,
    mu_inf: float,
    p_inf: float,
    reynolds_external: float,
    coupling: CouplingConfig,
    limits: GeometryLimits,
    alpha_start: float = -5.0,
    alpha_end: float = 15.0,
    alpha_step: float = 1.0,
    export_paraview: bool = False,
    paraview_output_dir: Path | None = None,
    field_nx: int = 300,
    field_ny: int = 300,
) -> pd.DataFrame:
    """Run one porous-model AoA sweep using robust continuation.

    Requested AoAs are solved from the point nearest zero outwards in two
    independent branches.  A failed requested point is retried using, in order:
    adaptive AoA continuation with boundary bracketing, Reynolds-number
    continuation with boundary bracketing, coupling homotopy with boundary
    bracketing, and finally pseudo-arclength continuation through a possible
    AoA fold.  Only requested AoAs at the requested Reynolds number and
    lambda=1 are written as converged results.
    """
    if alpha_step == 0.0:
        raise ValueError("alpha_step must be non-zero.")

    raw_alphas = np.arange(
        alpha_start,
        alpha_end + 0.1 * alpha_step,
        alpha_step,
        dtype=float,
    )
    requested_alphas = np.asarray(sorted(set(float(a) for a in raw_alphas)), dtype=float)
    if requested_alphas.size == 0:
        return pd.DataFrame()

    if export_paraview:
        if paraview_output_dir is None:
            raise ValueError("paraview_output_dir is required when export_paraview=True.")
        paraview_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[ParaView] AoA sweep export folder: {paraview_output_dir}")

    # Solve the requested point nearest alpha=0 first, then march independently
    # toward positive and negative AoA.  This avoids carrying a high-load branch
    # through zero from the negative end of the original sweep ordering.
    anchor = float(requested_alphas[np.argmin(np.abs(requested_alphas))])
    positive_branch = [float(a) for a in requested_alphas if a > anchor]
    negative_branch = [float(a) for a in requested_alphas if a < anchor][::-1]

    solve_order: list[tuple[str, float]] = [("anchor", anchor)]
    solve_order += [("positive", a) for a in positive_branch]
    solve_order += [("negative", a) for a in negative_branch]

    rows_by_alpha: dict[float, dict] = {}
    paraview_records: dict[float, dict[str, Path]] = {}

    branch_alpha: dict[str, float | None] = {
        "positive": anchor,
        "negative": anchor,
    }
    branch_q: dict[str, np.ndarray | None] = {
        "positive": None,
        "negative": None,
    }

    for branch, alpha in solve_order:
        if branch == "anchor":
            previous_alpha = None
            initial_q = None
        else:
            previous_alpha = branch_alpha[branch]
            initial_q = branch_q[branch]

        flow_i, geom_i, aero_solver_i, solid_i, network_i = _build_coupled_case_context(
            model, airfoil_name, chord, n_panels, alpha, reynolds_external,
            rho_inf, mu_inf, p_inf,
        )

        coupled_i = _solve_full_coupling_with_cold_retry(
            aero_solver=aero_solver_i,
            network=network_i,
            coupling=coupling,
            limits=limits,
            initial_q=initial_q,
        )
        continuation_used = "direct"
        aoa_boundary_meta = {
            "boundary_found": False,
            "alpha_converged": float("nan"),
            "alpha_failed": float("nan"),
            "q_converged": None,
            "stages": [],
        }
        pal_meta = {
            "attempted": False,
            "fold_detected": False,
            "alpha_min": float("nan"),
            "alpha_max": float("nan"),
            "steps": 0,
            "reached_target": False,
        }

        # 1) Adaptive AoA continuation + explicit converged/failed boundary bracket.
        if (
            not coupled_i.converged
            and previous_alpha is not None
            and initial_q is not None
        ):
            (
                flow_try, geom_try, aero_try, solid_try, network_try, result_try,
                aoa_boundary_meta,
            ) = _adaptive_aoa_continuation_to_target(
                model=model,
                airfoil_name=airfoil_name,
                chord=chord,
                n_panels=n_panels,
                previous_alpha=float(previous_alpha),
                target_alpha=alpha,
                reynolds_external=reynolds_external,
                rho_inf=rho_inf,
                mu_inf=mu_inf,
                p_inf=p_inf,
                coupling=coupling,
                limits=limits,
                previous_q=initial_q,
                original_target_result=coupled_i,
            )
            if result_try.converged or result_try.residual_norm < coupled_i.residual_norm:
                flow_i, geom_i, aero_solver_i, solid_i, network_i, coupled_i = (
                    flow_try, geom_try, aero_try, solid_try, network_try, result_try
                )
            if coupled_i.converged:
                continuation_used = "adaptive_aoa"

        # 2) Reynolds-number continuation from Re=5e5 (or the target if lower).
        if not coupled_i.converged and reynolds_external > RE_CONTINUATION_REFERENCE:
            (
                flow_try, geom_try, aero_try, solid_try, network_try, result_try
            ) = _solve_with_reynolds_continuation(
                model=model,
                airfoil_name=airfoil_name,
                chord=chord,
                n_panels=n_panels,
                alpha_deg=alpha,
                target_reynolds=reynolds_external,
                rho_inf=rho_inf,
                mu_inf=mu_inf,
                p_inf=p_inf,
                coupling=coupling,
                limits=limits,
                initial_q=initial_q,
                original_target_result=coupled_i,
            )
            if result_try.converged or result_try.residual_norm < coupled_i.residual_norm:
                flow_i, geom_i, aero_solver_i, solid_i, network_i, coupled_i = (
                    flow_try, geom_try, aero_try, solid_try, network_try, result_try
                )
            if coupled_i.converged:
                continuation_used = "reynolds"

        # 3) Coupling homotopy lambda: 0 -> 1 at the true target alpha and Re.
        if not coupled_i.converged:
            homotopy_result = _solve_with_coupling_homotopy(
                aero_solver=aero_solver_i,
                network=network_i,
                coupling=coupling,
                limits=limits,
                initial_q=initial_q,
                original_full_result=coupled_i,
            )
            coupled_i = _prefer_result(coupled_i, homotopy_result)
            if coupled_i.converged:
                continuation_used = "coupling_homotopy"

        # 4) Pseudo-arclength continuation through a possible AoA fold.
        # This is attempted only when ordinary AoA continuation has established
        # a converged/failed boundary bracket.  The branch trace itself is
        # diagnostic; the requested case is accepted only after an exact
        # lambda=1 target-alpha solve satisfies the original residual criterion.
        if (
            not coupled_i.converged
            and previous_alpha is not None
            and initial_q is not None
            and bool(aoa_boundary_meta.get("boundary_found", False))
            and aoa_boundary_meta.get("q_converged") is not None
        ):
            pal_result, pal_meta = _pseudo_arclength_to_target(
                model=model,
                airfoil_name=airfoil_name,
                chord=chord,
                n_panels=n_panels,
                reynolds_external=reynolds_external,
                rho_inf=rho_inf,
                mu_inf=mu_inf,
                p_inf=p_inf,
                coupling=coupling,
                limits=limits,
                alpha0=float(previous_alpha),
                q0=np.asarray(initial_q, dtype=float),
                alpha1=float(aoa_boundary_meta["alpha_converged"]),
                q1=np.asarray(aoa_boundary_meta["q_converged"], dtype=float),
                target_alpha=alpha,
                original_target_result=coupled_i,
            )
            coupled_i = _prefer_result(coupled_i, pal_result)
            if coupled_i.converged:
                continuation_used = "pseudo_arclength"

        # Update only the branch that this point belongs to, and only after a
        # genuine target-state root has been found.
        if coupled_i.converged:
            q_converged = _result_flow_vector(coupled_i)
            if branch == "anchor":
                branch_alpha["positive"] = alpha
                branch_alpha["negative"] = alpha
                branch_q["positive"] = q_converged
                branch_q["negative"] = None if q_converged is None else q_converged.copy()
            else:
                branch_alpha[branch] = alpha
                branch_q[branch] = q_converged

        if coupled_i.converged and export_paraview and paraview_output_dir is not None:
            alpha_dir = paraview_output_dir / _alpha_directory_name(alpha)
            paraview_records[alpha] = export_paraview_files(
                output_dir=alpha_dir,
                network=network_i,
                geom=geom_i,
                aero_solver=aero_solver_i,
                solid_result=solid_i,
                porous_result=coupled_i,
                field_nx=field_nx,
                field_ny=field_ny,
                include_solid_exports=True,
                collection_name="paraview_results.pvd",
            )

        if coupled_i.converged:
            row = {
                "alpha_deg": alpha,
                "reference_unit_span_m": 1.0,
                "channel_height_m": representative_channel_height(model),
                "pore_diameter_m": representative_channel_height(model),
                "porous_CL": float(coupled_i.aero_result.CL),
                "porous_CD": float(coupled_i.aero_result.CD),
                "porous_CM": float(coupled_i.aero_result.CM),
                "solid_panel_CL": float(solid_i.CL),
                "solid_panel_CD": float(solid_i.CD),
                "solid_panel_CM": float(solid_i.CM),
                "delta_CL": float(coupled_i.aero_result.CL - solid_i.CL),
                "delta_CD": float(coupled_i.aero_result.CD - solid_i.CD),
                "delta_CM": float(coupled_i.aero_result.CM - solid_i.CM),
                "max_abs_q_per_unit_span_m2_s": max(abs(float(s.Q)) for s in coupled_i.passage_states),
                "sum_abs_q_per_unit_span_m2_s": sum(abs(float(s.Q)) for s in coupled_i.passage_states),
                "max_dp_total_Pa": max(float(s.dp_total) for s in coupled_i.passage_states),
                "max_internal_reynolds_equivalent": max(float(s.reynolds_equivalent) for s in coupled_i.passage_states),
                "max_vn_m_s": float(coupled_i.max_vn),
                "coupling_converged": True,
                "coupling_iterations": int(coupled_i.iterations),
                "max_vn_over_vinf": float(coupled_i.max_vn_over_vinf),
                "continuation_method": continuation_used,
            }
        else:
            nan = float("nan")
            row = {
                "alpha_deg": alpha,
                "reference_unit_span_m": 1.0,
                "channel_height_m": representative_channel_height(model),
                "pore_diameter_m": representative_channel_height(model),
                "porous_CL": nan,
                "porous_CD": nan,
                "porous_CM": nan,
                "solid_panel_CL": float(solid_i.CL),
                "solid_panel_CD": float(solid_i.CD),
                "solid_panel_CM": float(solid_i.CM),
                "delta_CL": nan,
                "delta_CD": nan,
                "delta_CM": nan,
                "max_abs_q_per_unit_span_m2_s": nan,
                "sum_abs_q_per_unit_span_m2_s": nan,
                "max_dp_total_Pa": nan,
                "max_internal_reynolds_equivalent": nan,
                "max_vn_m_s": nan,
                "coupling_converged": False,
                "coupling_iterations": int(coupled_i.iterations),
                "max_vn_over_vinf": nan,
                "continuation_method": "failed_after_all_continuation",
            }

        row.update({
            "aoa_boundary_found": bool(aoa_boundary_meta.get("boundary_found", False)),
            "aoa_boundary_converged_deg": float(aoa_boundary_meta.get("alpha_converged", float("nan"))),
            "aoa_boundary_failed_deg": float(aoa_boundary_meta.get("alpha_failed", float("nan"))),
            "pseudo_arclength_attempted": bool(pal_meta.get("attempted", False)),
            "pseudo_arclength_fold_detected": bool(pal_meta.get("fold_detected", False)),
            "pseudo_arclength_alpha_min_deg": float(pal_meta.get("alpha_min", float("nan"))),
            "pseudo_arclength_alpha_max_deg": float(pal_meta.get("alpha_max", float("nan"))),
            "pseudo_arclength_steps": int(pal_meta.get("steps", 0)),
            "pseudo_arclength_reached_target": bool(pal_meta.get("reached_target", False)),
        })
        row.update(coupling_diagnostics(coupled_i))
        rows_by_alpha[alpha] = row

        height_mm = 1000.0 * representative_channel_height(model)
        if coupled_i.converged:
            print(
                f"[Case] alpha={alpha:g} deg, H={height_mm:g} mm: "
                f"converged; method={continuation_used}; "
                f"CL={coupled_i.aero_result.CL:.8f}; "
                f"residual={coupled_i.residual_norm:.3e}; "
                f"evaluations={coupled_i.function_evaluations}"
            )
        else:
            print(
                f"[Case] alpha={alpha:g} deg, H={height_mm:g} mm: "
                f"{coupled_i.coupling_status}; residual={coupled_i.residual_norm:.3e}; "
                f"all continuation methods exhausted; {coupled_i.solver_message}"
            )

    if export_paraview and paraview_output_dir is not None and paraview_records:
        exported_alphas = sorted(paraview_records)
        export_aoa_sweep_collections(
            output_dir=paraview_output_dir,
            cases=[paraview_records[a] for a in exported_alphas],
            alpha_values=exported_alphas,
        )

    rows = [rows_by_alpha[float(a)] for a in requested_alphas if float(a) in rows_by_alpha]
    return pd.DataFrame(rows)


# =============================================================================
# RUN ONE MODEL
# =============================================================================
def run_one_model(
    model: PorousModelSpec,
    geom: PanelGeometry,
    flow: FlowConfig,
    aero_solver: SourceVortexPanelMethod,
    solid_result: object,
    output_root: Path,
    make_contours: bool,
    contour_nx: int,
    contour_ny: int,
    export_paraview: bool = False,
    make_zoomed_contours: bool = MAKE_ZOOMED_CONTOUR_PLOTS,
    zoomed_xlim: tuple[float, float] = ZOOMED_CONTOUR_XLIM,
    zoomed_ylim: tuple[float, float] = ZOOMED_CONTOUR_YLIM,
    run_aoa_sweep: bool = True,
    output_name: str | None = None,
    aoa_sweep_diameters: tuple[float, ...] | None = None,
) -> dict:
    model_dir = output_root / (output_name or model.name)
    model_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 80)
    print(f"RUNNING {model.name}")
    print("=" * 80)
    print(model.description)
    print(f"Independent channels: {len(model.passages)}")

    network = build_network_from_model(model, SPAN_WIDTH)

    t0 = time.perf_counter()

    coupled = MultiPassageCoupledSolver(
        aero_solver=aero_solver,
        network=network,
        coupling=COUPLING,
        limits=GeometryLimits(),
    ).solve(verbose=True)

    elapsed = time.perf_counter() - t0

    print()
    print(f"[Result] converged              : {coupled.converged}")
    print(f"[Result] coupling iterations    : {coupled.iterations}")
    print(f"[Result] solid CL               : {solid_result.CL:.8f}")

    if coupled.converged:
        print(f"[Result] porous CL              : {coupled.aero_result.CL:.8f}")
        print(
            f"[Result] lift gain vs solid [%] : "
            f"{100.0 * (coupled.aero_result.CL / solid_result.CL - 1.0):.6f}"
        )
        print(f"[Result] max |vn| / Vinf        : {coupled.max_vn_over_vinf:.8f}")
    else:
        print("[Result] porous CL              : NaN")
        print("[Result] lift gain vs solid [%] : NaN")
        print("[Result] max |vn| / Vinf        : NaN")

    print(f"[Result] elapsed [s]            : {elapsed:.3f}")

    if not coupled.converged:
        summary_row = build_model_summary_row(
            model=model,
            flow=flow,
            solid_result=solid_result,
            porous_result=coupled,
            elapsed_seconds=elapsed,
        )
        summary_path = model_dir / "model_summary.csv"
        pd.DataFrame([summary_row]).to_csv(summary_path, index=False)
        print(f"[CSV Saved] {summary_path}")
        print("[Result] Non-converged case: only NaN summary values were reported.")
        return summary_row

    passage_df = build_passage_dataframe(
        model=model,
        network=network,
        geom=geom,
        result=coupled,
    )
    passage_path = model_dir / "passage_summary.csv"
    passage_df.to_csv(passage_path, index=False)
    print(f"[CSV Saved] {passage_path}")

    export_surface_data(
        output_dir=model_dir,
        geom=geom,
        solid_result=solid_result,
        porous_result=coupled,
    )
    if export_paraview:
        export_paraview_files(
            output_dir=model_dir,
            network=network,
            geom=geom,
            aero_solver=aero_solver,
            solid_result=solid_result,
            porous_result=coupled,
            field_nx=contour_nx,
            field_ny=contour_ny,
        )
    else:
        print("[ParaView] Fixed-AoA ParaView export disabled for this model.")

    summary_row = build_model_summary_row(
        model=model,
        flow=flow,
        solid_result=solid_result,
        porous_result=coupled,
        elapsed_seconds=elapsed,
    )
    summary_path = model_dir / "model_summary.csv"
    pd.DataFrame([summary_row]).to_csv(summary_path, index=False)
    print(f"[CSV Saved] {summary_path}")

    xfoil_model_dir = XFOIL_OUTPUT_ROOT / (output_name or model.name)

    xfoil_cp_data = run_xfoil_cp_for_case(
        airfoil_name=AIRFOIL_NAME,
        aoa_deg=AOA_DEG,
        output_dir=xfoil_model_dir,
    )

    plot_airfoil_with_porous_network(
        geom=geom,
        network=network,
        coupled_result=coupled,
        output_dir=model_dir,
        title=f"{model.name}: Airfoil with Porous Network",
    )

    plot_cp_distribution_with_xfoil(
        geom=geom,
        porous_result=coupled.aero_result,
        solid_result=solid_result,
        output_dir=model_dir,
        xfoil_cp_data=xfoil_cp_data,
    )

    if make_contours:
        plot_velocity_contours_comparison(
            porous_solver=aero_solver,
            porous_result=coupled.aero_result,
            solid_solver=aero_solver,
            solid_result=solid_result,
            output_dir=model_dir,
            nx=contour_nx,
            ny=contour_ny,
        )

        plot_pressure_contours_comparison(
            porous_solver=aero_solver,
            porous_result=coupled.aero_result,
            solid_solver=aero_solver,
            solid_result=solid_result,
            output_dir=model_dir,
            nx=contour_nx,
            ny=contour_ny,
        )

        if make_zoomed_contours:
            plot_velocity_contours_comparison(
                porous_solver=aero_solver,
                porous_result=coupled.aero_result,
                solid_solver=aero_solver,
                solid_result=solid_result,
                output_dir=model_dir,
                xlim=zoomed_xlim,
                ylim=zoomed_ylim,
                nx=contour_nx,
                ny=contour_ny,
                save_filename="velocity_contours_zoomed.png",
                view_label="zoomed",
            )

            plot_pressure_contours_comparison(
                porous_solver=aero_solver,
                porous_result=coupled.aero_result,
                solid_solver=aero_solver,
                solid_result=solid_result,
                output_dir=model_dir,
                xlim=zoomed_xlim,
                ylim=zoomed_ylim,
                nx=contour_nx,
                ny=contour_ny,
                save_filename="pressure_contours_zoomed.png",
                view_label="zoomed",
            )
    else:
        print("[Plot] Skipping contour plots because --no-contours was used.")

    if not run_aoa_sweep:
        return summary_row

    print(
        "[Sweep] Running porous + solid panel AoA sweep "
        f"from {AOA_SWEEP_START_DEG:g} to {AOA_SWEEP_END_DEG:g} deg "
        f"in {AOA_SWEEP_STEP_DEG:g} deg steps..."
    )

    if EXPORT_AOA_SWEEP_PARAVIEW:
        print("[ParaView] Exporting all AoA sweep cases for ParaView...")

    if aoa_sweep_diameters is None:
        aoa_sweep_diameters = tuple(float(d) for d in AOA_SWEEP_CHANNEL_HEIGHTS)

    # All models use the configured rectangular-channel-height sweep.
    # Each model is rebuilt for every height in AOA_SWEEP_CHANNEL_HEIGHTS.
    # Build all sweep models first.
    sweep_models = []

    for diameter_m in aoa_sweep_diameters:
        diameter_m = float(diameter_m)

        matching = [
            candidate
            for candidate in build_model_specs(geom=geom, channel_height_m=diameter_m)
            if candidate.name == model.name
        ]

        if not matching:
            raise RuntimeError(
                f"Could not build {model.name} for channel height {diameter_m:g} m"
            )

        sweep_models.append(matching[0])


    # ParaView AoA sweep export should be written for only one diameter.
    # Use the first AoA sweep diameter as the ParaView reference diameter.
    paraview_diameter_m = (
        float(aoa_sweep_diameters[0])
        if aoa_sweep_diameters
        else representative_pore_diameter(model)
    )

    print("[DEBUG] sweep models to run:")
    for i, sm in enumerate(sweep_models, start=1):
        print(
            f"  {i}: {sm.name}, "
            f"H={representative_channel_height(sm):.6e} m"
        )


    # Now run the sweep models only once.
    sweep_frames: list[pd.DataFrame] = []

    for sweep_model in sweep_models:
        diameter_m = representative_pore_diameter(sweep_model)
        print(f"[Sweep] {sweep_model.name}, channel height = {diameter_m:.6e} m")

        export_paraview_for_this_diameter = (
            EXPORT_AOA_SWEEP_PARAVIEW
            and np.isclose(diameter_m, paraview_diameter_m)
        )

        if export_paraview_for_this_diameter:
            print(
                f"[ParaView] Exporting AoA sweep for channel height "
                f"{diameter_m:.6e} m only."
            )

        sweep_frames.append(
            run_porous_aoa_sweep_for_model(
                model=sweep_model,
                airfoil_name=AIRFOIL_NAME,
                chord=CHORD,
                n_panels=geom.num_pan,
                rho_inf=RHO_INF,
                mu_inf=MU_INF,
                p_inf=P_INF,
                reynolds_external=REYNOLDS_EXTERNAL,
                coupling=COUPLING,
                limits=GeometryLimits(),
                alpha_start=AOA_SWEEP_START_DEG,
                alpha_end=AOA_SWEEP_END_DEG,
                alpha_step=AOA_SWEEP_STEP_DEG,
                export_paraview=export_paraview_for_this_diameter,
                paraview_output_dir=model_dir / "aoa_sweep_paraview",
                field_nx=contour_nx,
                field_ny=contour_ny,
            )
        )

    sweep_df = (
        pd.concat(sweep_frames, ignore_index=True)
        if sweep_frames
        else pd.DataFrame()
    )
    xfoil_sweep_df = run_xfoil_aoa_sweep_for_case(
        airfoil_name=AIRFOIL_NAME,
        output_dir=xfoil_model_dir,
        alpha_start=AOA_SWEEP_START_DEG,
        alpha_end=AOA_SWEEP_END_DEG,
        alpha_step=AOA_SWEEP_STEP_DEG,
    )

    if xfoil_sweep_df is not None:
        sweep_df = add_xfoil_to_sweep_comparison(
            sweep_df=sweep_df,
            xfoil_sweep_df=xfoil_sweep_df,
        )

    sweep_csv_path = model_dir / "aoa_sweep_comparison.csv"
    sweep_df.to_csv(sweep_csv_path, index=False)
    print(f"[CSV Saved] {sweep_csv_path}")

    plot_aoa_sweep_comparison(
        sweep_df=sweep_df,
        output_dir=model_dir,
    )

    return summary_row
