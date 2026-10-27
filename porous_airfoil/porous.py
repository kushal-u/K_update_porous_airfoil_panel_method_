"""Continuation, sweeps and fixed cases; compatible public entry point.

Extracted from the supplied porous.py; numerical definitions are unchanged.
"""
from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import pandas as pd
from .solver import CouplingConfig, FlowConfig, PanelGeometry, SourceVortexPanelMethod
from .plotting.aerodynamic import plot_airfoil_with_porous_network, plot_aoa_sweep_comparison, plot_cp_distribution_with_xfoil, plot_pressure_contours_comparison, plot_velocity_contours_comparison
from .config import AIRFOIL_NAME, AOA_DEG, AOA_SWEEP_END_DEG, AOA_SWEEP_START_DEG, AOA_SWEEP_STEP_DEG, AOA_SWEEP_CHANNEL_HEIGHTS, CHORD, COUPLING, EXPORT_AOA_SWEEP_PARAVIEW, MAKE_ZOOMED_CONTOUR_PLOTS, ZOOMED_CONTOUR_XLIM, ZOOMED_CONTOUR_YLIM, GeometryLimits, PassageState, PorousModelSpec, MultiPassageResult, P_INF, REYNOLDS_EXTERNAL, RHO_INF, MU_INF, SPAN_WIDTH, XFOIL_OUTPUT_ROOT
from .integrations.paraview import export_aoa_sweep_collections, export_paraview_files
from . import config as settings
from .reporting import timed_stage, aerodynamic_change_metrics, print_case_metrics
from .config import RE_CONTINUATION_REFERENCE, RE_CONTINUATION_INITIAL_DIVISIONS, RE_CONTINUATION_MIN_STEP, HOMOTOPY_INITIAL_STEP, HOMOTOPY_MIN_STEP, AOA_BOUNDARY_TOL_DEG, RE_BOUNDARY_TOL, HOMOTOPY_BOUNDARY_TOL, PAL_ALPHA_SCALE_DEG, PAL_INITIAL_DS, PAL_MIN_DS, PAL_MAX_DS, PAL_MAX_STEPS, PAL_CORRECTOR_TOL, PAL_TARGET_TOL_DEG, PAL_ALPHA_LIMIT_DEG
from .porous_hydraulics import (
    IndependentPassage,
    IndependentPassageNetwork,
    _channel_head_slope,
    _rectangular_2d_pressure_drop_abs,
    _smooth_transition_weight,
    solve_rectangular_channel_flow,
)
from .porous_coupling import (
    MultiPassageCoupledSolver,
    ReducedFlowProblem,
    _NewtonOutcome,
    _damped_newton,
)
from .porous_models import (
    _arc_length_spaced_points_on_side,
    _copy_passages_with_prefix,
    _equally_spaced_surface_band,
    _surface_x_intersections_for_y,
    _unique_sorted,
    add_xfoil_to_sweep_comparison,
    build_base_problem,
    build_model1_chordwise,
    build_model4_saved_case1,
    build_model_specs,
    build_model_summary_row,
    build_network_from_model,
    build_passage_dataframe,
    build_vertical_passages,
    coupling_diagnostics,
    export_surface_data,
    make_nonconverged_summary_row,
    model1_horizontal_surface_pores,
    model1_le_te_surface_pores,
    nan_if_not_converged,
    representative_channel_height,
    representative_channel_width,
    representative_pore_diameter,
    run_xfoil_aoa_sweep_for_case,
    run_xfoil_cp_for_case,
)

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

    # [subsec:coupled_solution -- coupling homotopy]
    # Continue chi from 0 to 1 in Vn=chi*M*q. chi=0 retains pressure-driven
    # hydraulic predictions but removes their aerodynamic feedback.
    # Only the chi=1 endpoint is a physical full-coupling sweep result.
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
    context = _build_coupled_case_context(
        model, airfoil_name, chord, n_panels, float(alpha_deg),
        reynolds_external, rho_inf, mu_inf, p_inf,
    )
    problem = ReducedFlowProblem(context[2], context[4])
    return problem.evaluate(q_dimensionless, alpha_deg)[0]


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
    branch_problem = ReducedFlowProblem(base_context[2], base_context[4])
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

        # [subsec:coupled_solution -- pseudo-arclength continuation]
        # Augment R(z_flow,alpha)=0 with t.(z_aug-z_predictor)=0.
        # z_aug=[z_flow,alpha_deg/PAL_ALPHA_SCALE_DEG]. Its last Jacobian column
        # converts dR/dalpha_radian using deg2rad(PAL_ALPHA_SCALE_DEG).
        def augmented_evaluation(z):
            z = np.asarray(z, dtype=float)
            alpha_trial = float(z[-1]*PAL_ALPHA_SCALE_DEG)
            if not np.isfinite(alpha_trial) or abs(alpha_trial) > PAL_ALPHA_LIMIT_DEG:
                raise FloatingPointError("Pseudo-arclength angle exceeded its allowed range.")
            residual, jacobian, alpha_derivative = branch_problem.evaluate(z[:-1], alpha_trial)
            augmented = np.r_[residual, np.dot(z-z_predictor, tangent)]
            matrix = np.empty((z.size, z.size))
            matrix[:-1, :-1] = jacobian
            matrix[:-1, -1] = alpha_derivative*np.deg2rad(PAL_ALPHA_SCALE_DEG)
            matrix[-1] = tangent
            return augmented, matrix

        corrector = _damped_newton(
            augmented_evaluation, z_predictor, coupling, tolerance=PAL_CORRECTOR_TOL)
        best_z = corrector.x if np.isfinite(corrector.residual_norm) else None
        best_norm = corrector.residual_norm
        method = "PAL-"+corrector.method

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
@timed_stage("model_height_sweep")
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

    # [subsec:coupled_solution; subsec:verification_outputs]
    # Sweep order starts nearest zero, then follows separate positive/negative
    # branches. Failed target solves may use AoA, Re_c, homotopy, then PAL
    # continuation; only verified requested operating points are reported.
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
        case_started = time.perf_counter()
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
        row.update(aerodynamic_change_metrics(solid_i, coupled_i.aero_result, coupled_i.converged))
        row["elapsed_seconds"] = time.perf_counter()-case_started
        if settings.PRINT_METRICS_PER_CASE:
            print_case_metrics(row, f"{model.name}, alpha={alpha:g} deg")
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
@timed_stage("fixed_case")
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
    case_started = time.perf_counter()
    make_contours = bool(make_contours and settings.ENABLE_PLOTS and settings.RUN_CONTOURS)
    run_aoa_sweep = bool(run_aoa_sweep and settings.RUN_AOA_SWEEP)

    def finish_case(row):
        row["solve_elapsed_seconds"] = row.get("elapsed_seconds", float("nan"))
        row["elapsed_seconds"] = time.perf_counter()-case_started
        pd.DataFrame([row]).to_csv(model_dir / "model_summary.csv", index=False)
        return row

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
    ).solve(verbose=settings.COUPLING_VERBOSE)

    elapsed = time.perf_counter() - t0

    print()
    print(f"[Result] converged              : {coupled.converged}")
    print(f"[Result] coupling iterations    : {coupled.iterations}")
    print(f"[Result] solid CL               : {solid_result.CL:.8f}")

    if coupled.converged:
        print(f"[Result] porous CL              : {coupled.aero_result.CL:.8f}")
        print(
            f"[Result] lift gain vs solid [%] : "
            f"{aerodynamic_change_metrics(solid_result, coupled.aero_result)['gain_percent_vs_solid_panel']:.6f}"
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
        return finish_case(summary_row)

    passage_df = build_passage_dataframe(
        model=model,
        network=network,
        geom=geom,
        result=coupled,
    )
    passage_path = model_dir / "passage_summary.csv"
    if settings.EXPORT_PASSAGE_DATA:
        passage_df.to_csv(passage_path, index=False)
        print(f"[CSV Saved] {passage_path}")

    if settings.EXPORT_SURFACE_DATA:
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
            field_nx=int(settings.PARAVIEW_FIELD_NX),
            field_ny=int(settings.PARAVIEW_FIELD_NY),
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
    if settings.PRINT_METRICS_PER_CASE:
        print_case_metrics(summary_row, model.name)
    summary_path = model_dir / "model_summary.csv"
    pd.DataFrame([summary_row]).to_csv(summary_path, index=False)
    print(f"[CSV Saved] {summary_path}")

    xfoil_model_dir = XFOIL_OUTPUT_ROOT / (output_name or model.name)

    xfoil_cp_data = run_xfoil_cp_for_case(
        airfoil_name=AIRFOIL_NAME,
        aoa_deg=AOA_DEG,
        output_dir=xfoil_model_dir,
    )

    if settings.ENABLE_PLOTS and settings.MAKE_NETWORK_PLOTS and settings.RUN_GEOMETRY:
        plot_airfoil_with_porous_network(
            geom=geom,
            network=network,
            coupled_result=coupled,
            output_dir=model_dir,
            title=f"{model.name}: Airfoil with Porous Network",
        )

    if settings.ENABLE_PLOTS and settings.MAKE_CP_PLOTS and settings.RUN_AERODYNAMIC_PLOTS:
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
        print("[Plot] Contour plots disabled by the function argument or config.py.")

    if not run_aoa_sweep:
        return finish_case(summary_row)

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
                field_nx=int(settings.PARAVIEW_FIELD_NX),
                field_ny=int(settings.PARAVIEW_FIELD_NY),
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

    if settings.ENABLE_PLOTS and settings.RUN_AERODYNAMIC_PLOTS and settings.MAKE_AOA_SWEEP_PLOTS:
        plot_aoa_sweep_comparison(sweep_df=sweep_df, output_dir=model_dir)

    return finish_case(summary_row)
