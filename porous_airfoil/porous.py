from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import root
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
    """
    dp = float(dp)
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


    def _surface_patch_weights(
        self,
        geom: PanelGeometry,
        pore: Pore,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return panel IDs and exact opening-overlap lengths for one pore.

        The physical pore opening is exactly ``H`` along the surface. Partial
        end panels receive fractional overlap weights. This removes the old
        mesh-dependent rules based on a minimum panel count or ``3H`` patch.
        """
        if pore.side == "upper":
            candidate_ids = np.asarray(geom.upper_panel_ids, dtype=int)
        else:
            candidate_ids = np.asarray(geom.lower_panel_ids, dtype=int)

        if candidate_ids.size == 0:
            raise RuntimeError(f"No panels available on the {pore.side} surface.")

        # On each individual airfoil side, x is monotonic from LE to TE.
        order = np.argsort(geom.XC[candidate_ids])
        ids = candidate_ids[order]
        panel_lengths = np.asarray(geom.S[ids], dtype=float)

        edges = np.concatenate([[0.0], np.cumsum(panel_lengths)])
        mids = 0.5 * (edges[:-1] + edges[1:])

        surface_point = self.pore_surface_point(geom, pore)
        s_center = float(
            np.interp(
                surface_point.x,
                geom.XC[ids],
                mids,
                left=mids[0],
                right=mids[-1],
            )
        )

        opening = float(pore.height)
        total_surface_length = float(edges[-1])
        if opening > total_surface_length:
            raise ValueError(
                f"{self.name}: pore opening H={opening:.6e} m exceeds the "
                f"available {pore.side} surface length {total_surface_length:.6e} m."
            )

        s_start = s_center - 0.5 * opening
        s_end = s_center + 0.5 * opening

        # Shift, rather than truncate, an opening that touches a side endpoint.
        if s_start < 0.0:
            s_end -= s_start
            s_start = 0.0
        if s_end > total_surface_length:
            s_start -= s_end - total_surface_length
            s_end = total_surface_length
        s_start = max(0.0, s_start)

        overlap = np.maximum(
            0.0,
            np.minimum(edges[1:], s_end) - np.maximum(edges[:-1], s_start),
        )
        active = overlap > 1.0e-15
        selected_ids = ids[active]
        selected_overlap = overlap[active]

        actual_opening = float(np.sum(selected_overlap))
        if not np.isclose(actual_opening, opening, rtol=0.0, atol=1.0e-10):
            raise RuntimeError(
                f"{self.name}: surface patch length {actual_opening:.6e} m does "
                f"not match H={opening:.6e} m."
            )

        return selected_ids, selected_overlap


    def _area_average_pressure_at_pore(
        self,
        geom: PanelGeometry,
        pore: Pore,
        panel_pressures: np.ndarray,
        min_panels: int = 0,
    ) -> float:
        """Length-average pressure over the exact rectangular pore opening."""
        del min_panels
        ids, overlap = self._surface_patch_weights(geom=geom, pore=pore)
        return float(
            np.sum(panel_pressures[ids] * overlap)
            / max(np.sum(overlap), 1.0e-16)
        )


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
        q, rs, reynolds, friction_factor, resistance_model = (
            solve_rectangular_channel_flow(
                dp=dp,
                rho=flow.rho_inf,
                mu=flow.mu_inf,
                length=length,
                height=self.chamber.height,
            )
        )

        return PassageState(
            Q=float(q),
            p1=p1,
            p2=p2,
            dp_total=float(abs(dp)),
            Rs=float(rs),
            reynolds_equivalent=float(reynolds),
            friction_factor=float(friction_factor),
            resistance_model=str(resistance_model),
        )


    def transpiration_from_flow(
        self,
        geom: PanelGeometry,
        state: PassageState,
    ) -> np.ndarray:
        """Map q' [m^2/s] to panel-normal velocity over an opening of length H.

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

    def solve(
        self,
        verbose: bool = False,
        initial_vn: np.ndarray | None = None,
    ) -> MultiPassageResult:
        """Solve the coupled pressure-flow equations as a nonlinear root problem.

        The unknowns are dimensionless flow rates per unit span in the
        independent passages. Each physical flow rate is scaled by ``H V_inf``
        before it is passed to the nonlinear solver. For a trial flow vector,
        the method maps the flow to panel transpiration, solves the external
        panel method, calculates the pore pressures, and obtains a new
        hydraulic flow vector. The dimensionless residual is

            residual = (trial flow - pressure-driven flow) / (H V_inf).

        No under-relaxation or user-defined coupling tolerance is applied.
        SciPy's nonlinear solver supplies its machine-precision-aware stopping
        logic, while ``max_iter`` remains a safety limit on residual
        evaluations. Convergence is classified from the scaled residual using
        the finite-difference resolution of double-precision arithmetic, not
        from SciPy's status flag alone.

        If ``initial_vn`` is provided, it is used only to form a passage-flow
        starting guess. It does not alter the coupled equations.
        """
        geom = self.aero_solver.geom
        flow = self.aero_solver.flow

        self.network.validate_geometry(geom, self.limits)

        n_passages = len(self.network.passages)
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

        last_finite_vn = starting_vn.copy()
        last_finite_aero = self.aero_solver.solve(
            normal_transpiration=starting_vn
        )
        last_finite_pressures = self.aero_solver.panel_pressures(last_finite_aero)
        last_finite_states = self.network.solve(
            geom=geom,
            panel_pressures=last_finite_pressures,
            flow=flow,
        )
        initial_q = np.asarray(
            [float(state.Q) for state in last_finite_states],
            dtype=float,
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
        initial_q_dimensionless = initial_q / q_scale
        evaluation_count = 0
        numerical_failure = False

        def states_for_trial_flow(q_trial: np.ndarray) -> list[PassageState]:
            """Create lightweight states for the linear Q-to-vn mapping."""
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

        def coupling_residual(
            q_trial_dimensionless: np.ndarray,
        ) -> np.ndarray:
            nonlocal evaluation_count
            nonlocal last_finite_vn
            nonlocal last_finite_aero
            nonlocal last_finite_states

            evaluation_count += 1
            q_trial_dimensionless = np.asarray(
                q_trial_dimensionless,
                dtype=float,
            )
            if (
                q_trial_dimensionless.shape != (n_passages,)
                or not np.all(np.isfinite(q_trial_dimensionless))
            ):
                raise FloatingPointError("Non-finite passage-flow trial state.")
            q_trial = q_trial_dimensionless * q_scale

            with np.errstate(over="raise", invalid="raise", divide="raise"):
                vn_trial = self.network.transpiration_from_flow(
                    geom=geom,
                    states=states_for_trial_flow(q_trial),
                )
                aero_result = self.aero_solver.solve(
                    normal_transpiration=vn_trial
                )
                panel_pressures = self.aero_solver.panel_pressures(aero_result)
                hydraulic_states = self.network.solve(
                    geom=geom,
                    panel_pressures=panel_pressures,
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
                    f"[Coupling] evaluation={evaluation_count:03d} | "
                    f"CL={aero_result.CL: .8f} | "
                    f"max|Q|={np.max(np.abs(q_trial)): .6e} | "
                    f"max|scaled residual|={np.max(np.abs(residual)): .3e}"
                )

            return residual

        try:
            nonlinear_result = root(
                coupling_residual,
                initial_q_dimensionless,
                method="hybr",
                options={"maxfev": int(self.coupling.max_iter)},
            )
            final_q_dimensionless = np.asarray(
                nonlinear_result.x,
                dtype=float,
            )
            if np.all(np.isfinite(final_q_dimensionless)):
                final_residual = coupling_residual(final_q_dimensionless)
            else:
                numerical_failure = True
                final_residual = np.full(n_passages, np.inf, dtype=float)

            residual_norm = float(np.max(np.abs(final_residual)))
            state_scale = max(
                1.0,
                float(np.max(np.abs(final_q_dimensionless))),
            )
            floating_point_resolution = (
                np.sqrt(np.finfo(float).eps) * state_scale
            )
            residual_resolved = residual_norm <= floating_point_resolution
            converged = bool(
                not numerical_failure
                and residual_resolved
            )
            final_iteration = int(
                getattr(nonlinear_result, "nfev", evaluation_count)
            )

            if verbose:
                print(
                    f"[Coupling] nonlinear solve: "
                    f"success={nonlinear_result.success}, "
                    f"evaluations={final_iteration}, "
                    f"max|scaled residual|={residual_norm:.3e}, "
                    f"message={nonlinear_result.message}"
                )
        except (FloatingPointError, OverflowError, ValueError, RuntimeError) as exc:
            converged = False
            numerical_failure = True
            final_iteration = evaluation_count
            if verbose:
                print(
                    f"[Coupling] nonlinear solve stopped after "
                    f"{evaluation_count} evaluations: "
                    f"{type(exc).__name__}: {exc}"
                )

        max_vn = float(np.max(np.abs(last_finite_vn)))
        return MultiPassageResult(
            aero_result=last_finite_aero,
            passage_states=last_finite_states,
            normal_transpiration=last_finite_vn,
            converged=bool(converged),
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
    model_2 = build_vertical_passages(np.linspace(0.100, 0.900, 4), height, "m2")
    model_3h = _copy_passages_with_prefix(model_1, "m3c", "LE_to_TE_internal_channel")
    model_3v = build_vertical_passages(np.linspace(0.100, 0.900, 4), height, "m3v")
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

    # Every AoA is solved as an independent physical case.  Do not carry the
    # transpiration field from one angle into the next: near a hydraulic branch
    # change, the previous-angle field can place the fixed-point iteration on an
    # oscillatory/non-convergent path even though a zero-flow start converges.

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

        # Primary solve: cold start for this AoA.  This reproduces the original
        # sweep behaviour in which each angle is independent.
        coupled_i = MultiPassageCoupledSolver(
            aero_solver=aero_solver_i,
            network=network_i,
            coupling=coupling,
            limits=limits,
        ).solve(verbose=False, initial_vn=None)

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
