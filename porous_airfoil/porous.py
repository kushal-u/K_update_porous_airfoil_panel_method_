from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
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
    AOA_SWEEP_PORE_DIAMETERS,
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
    circular_laminar_resistance,
    reynolds_to_velocity,
)

from .integrations.paraview import export_aoa_sweep_collections, export_paraview_files
# =============================================================================
# INTERNAL CHANNEL HYDRAULIC RESISTANCE
# =============================================================================
INTERNAL_RE_TRANSITION = 2300.0


def _darcy_friction_factor(reynolds: float) -> tuple[float, str]:
    """
    Return Darcy friction factor for the internal channel.

    Model switch:
        Re <= 2300 : laminar circular-pipe relation, f = 64/Re
        Re >  2300 : Blasius turbulent smooth-pipe relation, f = 0.3164/Re^0.25

    Notes
    -----
    This uses the Darcy friction factor, not the Fanning friction factor.
    """
    re = float(reynolds)

    if re <= 1e-12:
        return float("inf"), "zero_flow"

    if re <= INTERNAL_RE_TRANSITION:
        return 64.0 / re, "laminar_poiseuille"

    return 0.3164 / (re ** 0.25), "turbulent_blasius"


def _internal_channel_pressure_drop_abs(
    q_abs: float,
    rho: float,
    mu: float,
    length: float,
    diameter: float,
    area: float,
) -> tuple[float, float, float, str]:
    """
    Pressure drop magnitude for a given internal-channel flow magnitude.

    Uses Darcy-Weisbach:

        dp = f * (L/D) * 0.5 * rho * U^2

    with:
        U = Q/A

    The friction factor is selected from Reynolds number:
        Re <= 2300 : f = 64/Re
        Re >  2300 : f = 0.3164/Re^0.25
    """
    q_abs = abs(float(q_abs))

    if q_abs <= 0.0:
        return 0.0, 0.0, float("inf"), "zero_flow"

    velocity = q_abs / max(area, 1e-16)
    reynolds = rho * abs(velocity) * diameter / max(mu, 1e-16)

    friction_factor, model_name = _darcy_friction_factor(reynolds)

    dp_abs = (
        friction_factor
        * (length / max(diameter, 1e-16))
        * 0.5
        * rho
        * velocity**2
    )

    return float(dp_abs), float(reynolds), float(friction_factor), model_name


def solve_internal_channel_flow(
    dp: float,
    rho: float,
    mu: float,
    length: float,
    diameter: float,
    area: float,
) -> tuple[float, float, float, float, str]:
    """
    Solve internal channel flow rate from pressure difference.

    Unlike the old code, this does not assume Q = dp/Rs for all Reynolds
    numbers. For turbulent flow, Darcy-Weisbach + Blasius is nonlinear, so the
    flow rate is found by bisection.

    Returns
    -------
    tuple
        Q, equivalent_Rs, reynolds, friction_factor, resistance_model
    """
    dp = float(dp)
    dp_abs = abs(dp)

    if dp_abs <= 0.0:
        return 0.0, float("inf"), 0.0, float("inf"), "zero_flow"

    sign = 1.0 if dp >= 0.0 else -1.0

    # Bracket the solution.
    q_low = 0.0
    q_high = max(area * 1.0, 1e-12)

    for _ in range(100):
        dp_high, _, _, _ = _internal_channel_pressure_drop_abs(
            q_abs=q_high,
            rho=rho,
            mu=mu,
            length=length,
            diameter=diameter,
            area=area,
        )

        if dp_high >= dp_abs:
            break

        q_high *= 2.0
    else:
        raise RuntimeError(
            "Could not bracket internal-channel flow solution. "
            "Pressure drop may be too large or geometry may be invalid."
        )

    # Bisection solve for |Q|.
    for _ in range(80):
        q_mid = 0.5 * (q_low + q_high)

        dp_mid, _, _, _ = _internal_channel_pressure_drop_abs(
            q_abs=q_mid,
            rho=rho,
            mu=mu,
            length=length,
            diameter=diameter,
            area=area,
        )

        if dp_mid < dp_abs:
            q_low = q_mid
        else:
            q_high = q_mid

    q_abs = 0.5 * (q_low + q_high)

    dp_check, reynolds, friction_factor, model_name = _internal_channel_pressure_drop_abs(
        q_abs=q_abs,
        rho=rho,
        mu=mu,
        length=length,
        diameter=diameter,
        area=area,
    )

    q_signed = sign * q_abs
    equivalent_rs = dp_abs / max(q_abs, 1e-30)

    return (
        float(q_signed),
        float(equivalent_rs),
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

        # Model 1 uses constant-y horizontal, straight internal channels.
        # Do not add extra endpoint-thickness length for this geometry.
        if self.layout_kind in {"internal_horizontal", "internal_chordwise_horizontal"}:
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

                if pore.diameter > limits.thickness_fraction_limit * thickness:
                    raise ValueError(
                        f"{self.name}:{label} diameter={pore.diameter:.4e} is too large "
                        f"for local thickness={thickness:.4e}"
                    )

        sp1 = self.pore_surface_point(geom, self.pore1)
        sp2 = self.pore_surface_point(geom, self.pore2)

        dist = float(np.hypot(sp2.x - sp1.x, sp2.y - sp1.y))
        required = 0.5 * (self.pore1.diameter + self.pore2.diameter) + limits.min_gap

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

            if self.chamber.hydraulic_diameter > limits.fit_fraction * t_min:
                raise ValueError(
                    f"{self.name} chamber diameter is too large for the minimum local thickness. "
                    f"diameter={self.chamber.hydraulic_diameter:.4e}, t_min={t_min:.4e}"
                )

    def equivalent_reynolds(self, q_abs: float, flow: FlowConfig) -> float:
        u_internal = q_abs / max(self.chamber.area, 1e-16)
        return (
            flow.rho_inf
            * abs(u_internal)
            * self.chamber.hydraulic_diameter
            / max(flow.mu_inf, 1e-16)
        )


    def _nearest_surface_patch_ids(
        self,
        geom: PanelGeometry,
        pore: Pore,
        min_panels: int = 9,
    ) -> np.ndarray:
        """
        Return nearby panel IDs on the same surface as the pore.

        The patch size scales with total panel count, so the same physical pore
        patch is used for N=1000, N=3000, etc.

        For example:
            N=1000 -> about 9 panels
            N=3000 -> about 27 panels
        """
        sp = self.pore_surface_point(geom, pore)

        if pore.side == "upper":
            candidate_ids = np.asarray(geom.upper_panel_ids, dtype=int)
        else:
            candidate_ids = np.asarray(geom.lower_panel_ids, dtype=int)

        dx = geom.XC[candidate_ids] - sp.x
        dy = geom.YC[candidate_ids] - sp.y
        distance2 = dx * dx + dy * dy

        order = np.argsort(distance2)

        # Scale patch size with mesh density.
        scaled_panels = int(np.ceil(0.009 * geom.num_pan))

        # Also make larger pores use a larger patch.
        diameter_based_panels = int(np.ceil(3.0 * pore.diameter / max(np.mean(geom.S), 1e-16)))

        n_take = max(
            int(min_panels),
            scaled_panels,
            diameter_based_panels,
        )

        n_take = min(n_take, len(candidate_ids))

        return candidate_ids[order[:n_take]]


    def _area_average_pressure_at_pore(
        self,
        geom: PanelGeometry,
        pore: Pore,
        panel_pressures: np.ndarray,
        min_panels: int = 9,
    ) -> float:
        """
        Area/length-average the pressure around the pore instead of sampling one panel.
        """
        ids = self._nearest_surface_patch_ids(
            geom=geom,
            pore=pore,
            min_panels=min_panels,
        )

        weights = geom.S[ids]
        return float(np.sum(panel_pressures[ids] * weights) / max(np.sum(weights), 1e-16))


    def solve(
        self,
        geom: PanelGeometry,
        panel_pressures: np.ndarray,
        flow: FlowConfig,
    ) -> PassageState:
        """
        Solve one internal channel using surface-patch averaged pore pressures.
        """
        p1 = self._area_average_pressure_at_pore(
            geom=geom,
            pore=self.pore1,
            panel_pressures=panel_pressures,
            min_panels=9,
        )
        p2 = self._area_average_pressure_at_pore(
            geom=geom,
            pore=self.pore2,
            panel_pressures=panel_pressures,
            min_panels=9,
        )

        dp = p1 - p2

        length = self.passage_length(geom)
        rs, _ = circular_laminar_resistance(
            mu=flow.mu_inf,
            length=length,
            diameter=self.chamber.hydraulic_diameter,
        )
        q = dp / rs

        return PassageState(
            Q=float(q),
            p1=p1,
            p2=p2,
            dp_total=float(abs(dp)),
            Rs=float(rs),
            reynolds_equivalent=self.equivalent_reynolds(abs(q), flow),
        )


    def transpiration_from_flow(
        self,
        geom: PanelGeometry,
        state: PassageState,
    ) -> np.ndarray:
        """
        Convert internal channel flow rate Q into normal transpiration velocity.

        The total pore flow is conserved, but it is distributed over a small
        surface patch instead of being applied to one panel.
        """
        vn = np.zeros(geom.num_pan, dtype=float)

        Q = float(state.Q)
        if abs(Q) < 1e-30:
            return vn

        ids1 = self._nearest_surface_patch_ids(
            geom=geom,
            pore=self.pore1,
            min_panels=9,
        )
        ids2 = self._nearest_surface_patch_ids(
            geom=geom,
            pore=self.pore2,
            min_panels=9,
        )

        q1 = -Q
        q2 = +Q

        total_area_1 = self.span_width * max(np.sum(geom.S[ids1]), 1e-16)
        total_area_2 = self.span_width * max(np.sum(geom.S[ids2]), 1e-16)

        vn[ids1] += q1 / total_area_1
        vn[ids2] += q2 / total_area_2

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
                required = 0.5 * (pore_i.diameter + pore_j.diameter) + limits.min_gap

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
    ) -> list[PassageState]:
        return [
            passage.solve(
                geom=geom,
                panel_pressures=panel_pressures,
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

    def solve(self, verbose: bool = False) -> MultiPassageResult:
        geom = self.aero_solver.geom
        flow = self.aero_solver.flow

        self.network.validate_geometry(geom, self.limits)

        vn = np.zeros(geom.num_pan, dtype=float)
        previous_q: list[float] | None = None

        converged = False
        final_iteration = 0

        for iteration in range(1, self.coupling.max_iter + 1):
            aero_result = self.aero_solver.solve(normal_transpiration=vn)
            panel_pressures = self.aero_solver.panel_pressures(aero_result)

            states = self.network.solve(
                geom=geom,
                panel_pressures=panel_pressures,
                flow=flow,
            )

            vn_target = self.network.transpiration_from_flow(
                geom=geom,
                states=states,
            )

            vn_new = (
                (1.0 - self.coupling.relaxation) * vn
                + self.coupling.relaxation * vn_target
            )

            current_q = [float(s.Q) for s in states]
            if previous_q is None:
                dq = np.inf
            else:
                dq = max(abs(q_new - q_old) for q_new, q_old in zip(current_q, previous_q))

            dvn = float(np.max(np.abs(vn_new - vn)))

            if verbose:
                max_q = max(abs(q) for q in current_q)
                print(
                    f"[Coupling] iter={iteration:03d} | "
                    f"CL={aero_result.CL: .8f} | "
                    f"max|Q|={max_q: .6e} | "
                    f"dvn={dvn: .3e} | dq={dq: .3e}"
                )

            vn = vn_new
            previous_q = current_q
            final_iteration = iteration

            if dvn < self.coupling.tol_vn and dq < self.coupling.tol_q:
                converged = True
                break

        aero_result = self.aero_solver.solve(normal_transpiration=vn)
        panel_pressures = self.aero_solver.panel_pressures(aero_result)

        states = self.network.solve(
            geom=geom,
            panel_pressures=panel_pressures,
            flow=flow,
        )

        max_vn = float(np.max(np.abs(vn)))

        return MultiPassageResult(
            aero_result=aero_result,
            passage_states=states,
            normal_transpiration=vn,
            converged=converged,
            iterations=final_iteration,
            max_vn=max_vn,
            max_vn_over_vinf=max_vn / max(flow.v_inf, 1e-16),
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
    diameter_m: float,
) -> list[PassageSpec]:
    """
    Model 1:
        4 horizontal straight internal channels.

    Each channel is generated from a constant y/c level.  The two pore
    positions are the left and right intersections of that horizontal line
    with the airfoil surface, so the plotted/exported internal passage is a
    straight horizontal line.
    """
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
                diameter_m=diameter_m,
                layout_kind="internal_horizontal",
            )
        )

    return passages

def build_vertical_passages(
    x_values: np.ndarray,
    diameter_m: float,
    prefix: str,
) -> list[PassageSpec]:
    """
    Build lower-to-upper perpendicular channels.
    """
    passages: list[PassageSpec] = []

    for i, x in enumerate(x_values, start=1):
        passages.append(
            PassageSpec(
                name=f"{prefix}_{i:02d}_perpendicular",
                x1_frac=float(x),
                side1="lower",
                x2_frac=float(x),
                side2="upper",
                diameter_m=diameter_m,
                layout_kind="perpendicular",
            )
        )

    return passages


def _copy_passages_with_prefix(
    passages: list[PassageSpec],
    prefix: str,
    label: str,
) -> list[PassageSpec]:
    """
    Copy passage definitions and rename them.

    Used by Model 3 so it can reuse the exact Model 1 channel geometry.
    """
    copied: list[PassageSpec] = []

    for i, passage in enumerate(passages, start=1):
        copied.append(
            PassageSpec(
                name=f"{prefix}_{i:02d}_{label}",
                x1_frac=passage.x1_frac,
                side1=passage.side1,
                x2_frac=passage.x2_frac,
                side2=passage.side2,
                diameter_m=passage.diameter_m,
                layout_kind=passage.layout_kind,
            )
        )

    return copied


# =============================================================================
# MODEL 4: SAVED CASE 1 FROM OLD run_test.py
# =============================================================================
def build_model4_saved_case1(diameter_m: float | None = None) -> list[PassageSpec]:
    """
    Model 4:
        Saved case_1 from the old run_test.py validation script.

    If diameter_m is provided, both passages use that common diameter.
    This allows Model 4 to participate in the same AoA diameter sweep as
    Models 1, 2 and 3.
    """
    if diameter_m is None:
        d1 = 1.2e-02
        d2 = 0.9e-02
    else:
        d1 = float(diameter_m)
        d2 = float(diameter_m)

    return [
        PassageSpec(
            name="m4_case1_passage_01_lower_to_upper",
            x1_frac=0.400094,
            side1="lower",
            x2_frac=0.933001,
            side2="upper",
            diameter_m=d1,
            layout_kind="saved_case_1",
        ),
        PassageSpec(
            name="m4_case1_passage_02_lower_to_upper",
            x1_frac=0.925069,
            side1="lower",
            x2_frac=0.955118,
            side2="upper",
            diameter_m=d2,
            layout_kind="saved_case_1",
        ),
    ]


def build_model_specs(
    geom: PanelGeometry,
    diameter_m: float,
) -> list[PorousModelSpec]:
    """
    Build the fixed porous models.

    Models 1-3 use the common diameter_m.
    Model 4 is the saved case_1 design from the old run_test.py and uses
    its own two fixed diameters.
    """
    model_1_chordwise = build_model1_chordwise(
        geom=geom,
        diameter_m=diameter_m,
    )

    model_2_perpendicular = build_vertical_passages(
        x_values=np.linspace(0.100, 0.900, 4),
        diameter_m=diameter_m,
        prefix="m2",
    )

    model_3_chordwise = _copy_passages_with_prefix(
        passages=model_1_chordwise,
        prefix="m3c",
        label="LE_to_TE_internal_channel",
    )

    model_3_perpendicular = build_vertical_passages(
        x_values=np.linspace(0.100, 0.900, 4),
        diameter_m=diameter_m,
        prefix="m3v",
    )

    model_3_combined = model_3_chordwise + model_3_perpendicular

    
    model_4_saved_case1 = build_model4_saved_case1(diameter_m=diameter_m)

    return [
        PorousModelSpec(
            name="model_1_4_chordwise",
            description=(
                "4 straight horizontal internal channels at constant y/c levels. "
                "Each channel connects the left and right surface intersections."
            ),
            passages=tuple(model_1_chordwise),
        ),
        PorousModelSpec(
            name="model_2_9_perpendicular",
            description="4 independent lower-to-upper perpendicular channels.",
            passages=tuple(model_2_perpendicular),
        ),
        PorousModelSpec(
            name="model_3_combined_independent",
            description=(
                "Combined Model 1 + Model 2: 4 straight horizontal channels "
                "plus 4 lower-to-upper perpendicular channels. All 8 passages "
                "are independent; internal intersections are not connected."
            ),
            passages=tuple(model_3_combined),
        ),
        PorousModelSpec(
            name="model_4_saved_case_1",
            description=(
                "Saved case_1 from old run_test.py: two independent lower-to-upper "
                "dual porous passages with fixed pore locations and fixed diameters. "
                "This model does not use the common DEFAULT_PORE_DIAMETER."
            ),
            passages=tuple(model_4_saved_case1),
        ),
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
    passages: list[IndependentPassage] = []

    for spec in model.passages:
        d = float(spec.diameter_m)

        passages.append(
            IndependentPassage(
                name=spec.name,
                pore1=Pore(spec.x1_frac, spec.side1, d),
                pore2=Pore(spec.x2_frac, spec.side2, d),
                chamber=Chamber(
                    hydraulic_diameter=d,
                    area=0.25 * np.pi * d**2,
                ),
                span_width=span_width,
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
                "pore_diameter_m": passage.pore1.diameter,
                "passage_length_m": passage.passage_length(geom),
                "Q_m3_s": state.Q,
                "abs_Q_m3_s": abs(state.Q),
                "Rs_Pa_s_per_m3": state.Rs,
                "dp_total_Pa": state.dp_total,
                "reynolds_internal_equivalent": state.reynolds_equivalent,
                "p1_Pa": state.p1,
                "p2_Pa": state.p2,
                "friction_factor": state.friction_factor,
                "resistance_model": state.resistance_model,
            }
        )

    return pd.DataFrame(rows)


def representative_pore_diameter(model: PorousModelSpec) -> float:
    """Return a representative pore diameter for CSV summaries.

    Models 1-3 normally use one common diameter. Model 4 uses two fixed
    diameters, so its representative value is the mean of its passage diameters.
    """
    diameters = [float(p.diameter_m) for p in model.passages]
    if not diameters:
        return float("nan")
    return float(np.mean(diameters))


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
        "pore_diameter_m": representative_pore_diameter(model),
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
        "max_abs_Q_m3_s": nan,
        "sum_abs_Q_m3_s": nan,
        "max_dp_total_Pa": nan,
        "max_internal_reynolds_equivalent": nan,
        "max_vn_m_s": nan,
        "max_vn_over_vinf": nan,
        "coupling_converged": False,
        "coupling_iterations": porous_result.iterations,
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
        "pore_diameter_m": representative_pore_diameter(model),
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
        "max_abs_Q_m3_s": max(q_abs) if q_abs else 0.0,
        "sum_abs_Q_m3_s": sum(q_abs),
        "max_dp_total_Pa": max(dp_values) if dp_values else 0.0,
        "max_internal_reynolds_equivalent": max(re_values) if re_values else 0.0,
        "max_vn_m_s": porous_result.max_vn,
        "max_vn_over_vinf": porous_result.max_vn_over_vinf,
        "coupling_converged": porous_result.converged,
        "coupling_iterations": porous_result.iterations,
        "elapsed_seconds": elapsed_seconds,
    }



def _alpha_directory_name(alpha_deg: float) -> str:
    """Return a stable folder name for one AoA sweep case."""
    sign = "p" if alpha_deg >= 0.0 else "m"
    magnitude = f"{abs(alpha_deg):06.2f}".replace(".", "p")
    return f"alpha_{sign}{magnitude}_deg"


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
    rows: list[dict[str, float]] = []
    paraview_cases: list[dict[str, Path]] = []
    paraview_alphas: list[float] = []

    if export_paraview:
        if paraview_output_dir is None:
            raise ValueError("paraview_output_dir is required when export_paraview=True.")

        paraview_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[ParaView] AoA sweep export folder: {paraview_output_dir}")

    for alpha in np.arange(alpha_start, alpha_end + 0.1 * alpha_step, alpha_step):
        alpha = float(alpha)

        flow_i, geom_i, _, aero_solver_i = build_base_problem(
            airfoil_name=airfoil_name,
            chord=chord,
            n_panels=n_panels,
            aoa_deg=alpha,
            reynolds_external=reynolds_external,
            rho_inf=rho_inf,
            mu_inf=mu_inf,
            p_inf=p_inf,
        )

        solid_i = aero_solver_i.solve()

        network_i = build_network_from_model(
            model=model,
            span_width=SPAN_WIDTH,
        )

        coupled_i = MultiPassageCoupledSolver(
            aero_solver=aero_solver_i,
            network=network_i,
            coupling=coupling,
            limits=limits,
        ).solve(verbose=False)

        if coupled_i.converged and export_paraview and paraview_output_dir is not None:
            alpha_dir = paraview_output_dir / _alpha_directory_name(alpha)

            exported_files = export_paraview_files(
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

            paraview_cases.append(exported_files)
            paraview_alphas.append(alpha)

        if coupled_i.converged:
            row = {
                "alpha_deg": alpha,
                "pore_diameter_m": representative_pore_diameter(model),
                "porous_CL": float(coupled_i.aero_result.CL),
                "porous_CD": float(coupled_i.aero_result.CD),
                "porous_CM": float(coupled_i.aero_result.CM),
                "solid_panel_CL": float(solid_i.CL),
                "solid_panel_CD": float(solid_i.CD),
                "solid_panel_CM": float(solid_i.CM),
                "delta_CL": float(coupled_i.aero_result.CL - solid_i.CL),
                "delta_CD": float(coupled_i.aero_result.CD - solid_i.CD),
                "delta_CM": float(coupled_i.aero_result.CM - solid_i.CM),
                "max_abs_Q_m3_s": max(abs(float(s.Q)) for s in coupled_i.passage_states),
                "sum_abs_Q_m3_s": sum(abs(float(s.Q)) for s in coupled_i.passage_states),
                "max_dp_total_Pa": max(float(s.dp_total) for s in coupled_i.passage_states),
                "max_internal_reynolds_equivalent": max(
                    float(s.reynolds_equivalent) for s in coupled_i.passage_states
                ),
                "max_vn_m_s": float(coupled_i.max_vn),
                "coupling_converged": True,
                "coupling_iterations": int(coupled_i.iterations),
                "max_vn_over_vinf": float(coupled_i.max_vn_over_vinf),
            }
        else:
            nan = float("nan")
            row = {
                "alpha_deg": alpha,
                "pore_diameter_m": representative_pore_diameter(model),
                "porous_CL": nan,
                "porous_CD": nan,
                "porous_CM": nan,
                "solid_panel_CL": float(solid_i.CL),
                "solid_panel_CD": float(solid_i.CD),
                "solid_panel_CM": float(solid_i.CM),
                "delta_CL": nan,
                "delta_CD": nan,
                "delta_CM": nan,
                "max_abs_Q_m3_s": nan,
                "sum_abs_Q_m3_s": nan,
                "max_dp_total_Pa": nan,
                "max_internal_reynolds_equivalent": nan,
                "max_vn_m_s": nan,
                "coupling_converged": False,
                "coupling_iterations": int(coupled_i.iterations),
                "max_vn_over_vinf": nan,
            }

        rows.append(row)

    if export_paraview and paraview_output_dir is not None and paraview_cases:
        export_aoa_sweep_collections(
            output_dir=paraview_output_dir,
            cases=paraview_cases,
            alpha_values=paraview_alphas,
        )

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
        aoa_sweep_diameters = tuple(float(d) for d in AOA_SWEEP_PORE_DIAMETERS)

    # All models use the three-diameter AoA sweep.
    # Each model is rebuilt for every diameter in AOA_SWEEP_PORE_DIAMETERS.
    # Build all sweep models first.
    sweep_models = []

    for diameter_m in aoa_sweep_diameters:
        diameter_m = float(diameter_m)

        matching = [
            candidate
            for candidate in build_model_specs(geom=geom, diameter_m=diameter_m)
            if candidate.name == model.name
        ]

        if not matching:
            raise RuntimeError(
                f"Could not build {model.name} for diameter {diameter_m:g} m"
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
            f"D={representative_pore_diameter(sm):.6e} m"
        )


    # Now run the sweep models only once.
    sweep_frames: list[pd.DataFrame] = []

    for sweep_model in sweep_models:
        diameter_m = representative_pore_diameter(sweep_model)
        print(f"[Sweep] {sweep_model.name}, pore diameter = {diameter_m:.6e} m")

        export_paraview_for_this_diameter = (
            EXPORT_AOA_SWEEP_PARAVIEW
            and np.isclose(diameter_m, paraview_diameter_m)
        )

        if export_paraview_for_this_diameter:
            print(
                f"[ParaView] Exporting AoA sweep for diameter "
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
