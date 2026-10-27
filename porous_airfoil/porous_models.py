"""Model geometry, problem builders and result/export helpers.

Extracted from the supplied porous.py; numerical definitions are unchanged.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from .solver import FlowConfig, PanelGeometry, ReferenceConfig, SourceVortexPanelMethod, generate_naca4
from .integrations.xfoil import load_xfoil_cp, load_xfoil_polar_dataframe, run_xfoil_cp, run_xfoil_polar_sweep
from .config import Chamber, Pore, PassageSpec, PorousModelSpec, MultiPassageResult, REYNOLDS_EXTERNAL, USE_XFOIL, XFOIL_EXE_PATH, XFOIL_MACH, XFOIL_TIMEOUT, reynolds_to_velocity
from . import config as settings
from .reporting import aerodynamic_change_metrics
from .config import CHANNEL_ENERGY_CLOSURE
from .porous_hydraulics import (
    IndependentPassage,
    IndependentPassageNetwork,
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
        n_channels=settings.MODEL1_CHANNEL_COUNT,
        max_y_fraction=settings.MODEL1_MAX_Y_FRACTION,
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
    # [subsec:model_overview]
    # Current vertical positions are x/c=(0.100,0.363333...,0.626667...,0.890).
    # The combined model copies the horizontal and vertical channels without
    # adding junctions at their crossings.
    vertical_x_values = tuple(settings.MODEL2_X_LOCATIONS)

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
            description=f"{len(model_1)} straight horizontal rectangular channels; " + size_text + ".",
            passages=tuple(model_1),
        ),
        PorousModelSpec(
            name="model_2_9_perpendicular",
            description=f"{len(model_2)} independent lower-to-upper rectangular channels; " + size_text + ".",
            passages=tuple(model_2),
        ),
        PorousModelSpec(
            name="model_3_combined_independent",
            description=(
                "Combined horizontal and vertical rectangular channels. "
                f"All {len(model_3h)+len(model_3v)} passages are independent; internal intersections are not connected; "
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
    ref = ReferenceConfig(x_ref=settings.MOMENT_REFERENCE_X_C*chord,
                          y_ref=settings.MOMENT_REFERENCE_Y_C*chord)
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
        "coupling_newton_iterations": result.newton_iterations,
        "coupling_jacobian_condition_number": result.jacobian_condition_number,
        "coupling_jacobian_smallest_singular_value": result.jacobian_smallest_singular_value,
        "coupling_reduced_residual_max": result.reduced_residual_norm,
        "coupling_original_energy_residual_max_Pa": result.original_energy_residual_max_pa,
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
        **aerodynamic_change_metrics(solid_result, None, False),
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
        "gain_percent_vs_solid_panel": aerodynamic_change_metrics(solid_result, aero)["gain_percent_vs_solid_panel"],
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
        **aerodynamic_change_metrics(solid_result, aero, True),
    }



