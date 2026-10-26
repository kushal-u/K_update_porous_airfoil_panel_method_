"""Application workflows for the porous-airfoil study.

This is the orchestration layer. It coordinates the numerical core, external
integrations, and plotting modules without placing application entry-point code
inside the solver modules.
"""

from __future__ import annotations

from pathlib import Path
import importlib
import time

from . import config as settings
from .reporting import timed_stage, report_aerodynamic_frame

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .config import (
    AIRFOIL_NAME,
    AOA_DEG,
    AOA_SWEEP_END_DEG,
    AOA_SWEEP_CHANNEL_HEIGHTS,
    AOA_SWEEP_START_DEG,
    AOA_SWEEP_STEP_DEG,
    CHORD,
    CONTOUR_NX,
    CONTOUR_NY,
    COUPLING,
    EXPORT_AOA_SWEEP_PARAVIEW,
    EXPORT_PARAVIEW,
    FIXED_AOA_CHANNEL_HEIGHTS,
    GeometryLimits,
    MAKE_CONTOUR_PLOTS,
    MAKE_FIXED_AOA_CHANNEL_HEIGHT_OUTPUTS,
    MAKE_ZOOMED_CONTOUR_PLOTS,
    MU_INF,
    N_PANELS,
    OUTPUT_ROOT,
    P_INF,
    CHANNEL_HEIGHT,
    REYNOLDS_EXTERNAL,
    RHO_INF,
    SELECTED_MODEL,
    USE_XFOIL,
    XFOIL_OUTPUT_ROOT,
    ZOOMED_CONTOUR_XLIM,
    ZOOMED_CONTOUR_YLIM,
    PorousModelSpec,
)
from .integrations.paraview import export_solid_reference_paraview_files
from .plotting.aerodynamic import plot_aoa_sweep_comparison
from .porous import (
    add_xfoil_to_sweep_comparison,
    build_base_problem,
    build_model_specs,
    representative_channel_height,
    run_one_model,
    run_porous_aoa_sweep_for_model,
    run_xfoil_aoa_sweep_for_case,
)

def plot_sweep_with_gaps(sweep_df: pd.DataFrame, output_dir: Path) -> None:
    """Keep every alpha row: NaNs break lines rather than bridging failures."""
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    for height, frame in sweep_df.groupby("channel_height_m", sort=True):
        frame = frame.sort_values("alpha_deg")
        valid = frame["coupling_converged"].fillna(False).astype(bool).to_numpy()
        for ax, coefficient in zip(axes, ("CL", "CD", "CM")):
            values = frame[f"porous_{coefficient}"].to_numpy(dtype=float).copy()
            values[~valid] = np.nan
            ax.plot(frame["alpha_deg"], values, marker="o", markersize=3,
                    label=f"Porous H={height*1000:g} mm")
    reference = sweep_df.drop_duplicates("alpha_deg").sort_values("alpha_deg")
    for ax, coefficient in zip(axes, ("CL", "CD", "CM")):
        ax.plot(reference["alpha_deg"], reference[f"solid_panel_{coefficient}"],
                "k--", label="Solid panel")
        if f"xfoil_{coefficient}" in reference:
            ax.plot(reference["alpha_deg"], reference[f"xfoil_{coefficient}"],
                    ":", color="0.5", label="Inviscid XFOIL")
        ax.set_ylabel("Pressure CD" if coefficient == "CD" else coefficient)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("Angle of attack [deg]")
    failed = int((~sweep_df["coupling_converged"].fillna(False).astype(bool)).sum())
    fig.suptitle(f"{failed} invalid/unconverged cases omitted; lines break at missing values")
    fig.tight_layout()
    fig.savefig(output_dir / "aoa_sweep_with_gaps.png", dpi=200)
    plt.close(fig)


VALID_MODEL_NAMES = {
    "all",
    "model_1_4_chordwise",
    "model_2_9_perpendicular",
    "model_3_combined_independent",
    "model_4_saved_case_1",
}

def select_models(models: list[PorousModelSpec]) -> list[PorousModelSpec]:
    """Return the model list requested by ``config.SELECTED_MODEL``."""
    if SELECTED_MODEL not in VALID_MODEL_NAMES:
        raise ValueError(
            f"Invalid SELECTED_MODEL={SELECTED_MODEL!r}. "
            f"Choose one of: {sorted(VALID_MODEL_NAMES)}"
        )
    if SELECTED_MODEL == "all":
        return models

    selected = [model for model in models if model.name == SELECTED_MODEL]
    if not selected:
        available = [model.name for model in models]
        raise RuntimeError(
            f"No model matched SELECTED_MODEL={SELECTED_MODEL!r}. "
            f"Available models: {available}"
        )
    return selected


def _unique_channel_heights(values: tuple[float, ...]) -> tuple[float, ...]:
    """Return unique channel heights while retaining their configured order."""
    unique: list[float] = []
    for value in values:
        height = float(value)
        if not any(abs(height - old) <= 1.0e-15 for old in unique):
            unique.append(height)
    return tuple(unique)


def _height_label(channel_height_m: float) -> str:
    return f"H_{1000.0 * float(channel_height_m):.2f}mm".replace(".", "p")


def _build_model_by_name(geom, model_name: str, channel_height_m: float) -> PorousModelSpec:
    matches = [
        model
        for model in build_model_specs(geom=geom, channel_height_m=float(channel_height_m))
        if model.name == model_name
    ]
    if not matches:
        raise RuntimeError(
            f"Could not build {model_name!r} for channel height {channel_height_m:.6e} m."
        )
    return matches[0]


@timed_stage("fixed_cases")
def run_fixed_cases() -> pd.DataFrame:
    """Run configured fixed-angle porous-airfoil cases once per channel height."""
    if not settings.RUN_FIXED_CASES:
        print("[Skip] Fixed cases disabled in config.py.")
        return pd.DataFrame()
    n_panels = int(N_PANELS)
    contour_nx = int(CONTOUR_NX)
    contour_ny = int(CONTOUR_NY)
    output_root = Path(OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RUNNING FIXED POROUS MODELS — PANEL METHOD + XFOIL")
    print("=" * 80)
    print(f"Airfoil                    : NACA {AIRFOIL_NAME}")
    print(f"AoA [deg]                  : {AOA_DEG}")
    print(f"Reynolds number            : {REYNOLDS_EXTERNAL:.6e}")
    print(f"Panels                     : {n_panels}")
    print(f"Selected model             : {SELECTED_MODEL}")
    print(f"Output root                : {output_root}")
    print(f"XFOIL output root          : {XFOIL_OUTPUT_ROOT}")
    print(f"Contour plots              : {bool(MAKE_CONTOUR_PLOTS)}")
    print(f"Contour grid               : {contour_nx} x {contour_ny}")
    print(f"Fixed-AoA ParaView         : {EXPORT_PARAVIEW}")
    print("Hydraulic model            : 2D rectangular channel per unit span")
    print("Flow area per unit span    : H")
    print("Hydraulic diameter         : 2H")
    print(f"XFOIL enabled              : {USE_XFOIL}")

    flow, geom, _, aero_solver = build_base_problem(
        airfoil_name=AIRFOIL_NAME,
        chord=CHORD,
        n_panels=n_panels,
        aoa_deg=AOA_DEG,
        reynolds_external=REYNOLDS_EXTERNAL,
        rho_inf=RHO_INF,
        mu_inf=MU_INF,
        p_inf=P_INF,
    )

    print(f"Freestream speed [m/s]     : {flow.v_inf:.8f}")
    print("[Setup] Solving solid-airfoil baseline...")
    solid_result = aero_solver.solve()
    print(f"[Setup] Solid CL            : {solid_result.CL:.8f}")
    print(f"[Setup] Solid CD            : {solid_result.CD:.8e}")
    print(f"[Setup] Solid CM            : {solid_result.CM:.8f}")

    if EXPORT_PARAVIEW:
        export_solid_reference_paraview_files(
            output_root=output_root,
            geom=geom,
            aero_solver=aero_solver,
            solid_result=solid_result,
            field_nx=int(settings.PARAVIEW_FIELD_NX),
            field_ny=int(settings.PARAVIEW_FIELD_NY),
        )
    else:
        print("[ParaView] Fixed-AoA ParaView export disabled.")

    selected = select_models(
        build_model_specs(geom=geom, channel_height_m=float(CHANNEL_HEIGHT))
    )

    channel_heights = (
        _unique_channel_heights(tuple(float(value) for value in FIXED_AOA_CHANNEL_HEIGHTS))
        if MAKE_FIXED_AOA_CHANNEL_HEIGHT_OUTPUTS
        else (float(CHANNEL_HEIGHT),)
    )
    if not channel_heights:
        channel_heights = (float(CHANNEL_HEIGHT),)

    rows: list[dict] = []
    for base_model in selected:
        for channel_height_m in channel_heights:
            model = _build_model_by_name(
                geom=geom,
                model_name=base_model.name,
                channel_height_m=channel_height_m,
            )
            row = run_one_model(
                model=model,
                geom=geom,
                flow=flow,
                aero_solver=aero_solver,
                solid_result=solid_result,
                output_root=output_root,
                make_contours=bool(MAKE_CONTOUR_PLOTS),
                contour_nx=contour_nx,
                contour_ny=contour_ny,
                export_paraview=bool(EXPORT_PARAVIEW),
                make_zoomed_contours=bool(MAKE_ZOOMED_CONTOUR_PLOTS),
                zoomed_xlim=ZOOMED_CONTOUR_XLIM,
                zoomed_ylim=ZOOMED_CONTOUR_YLIM,
                run_aoa_sweep=False,
                output_name=f"{base_model.name}_{_height_label(channel_height_m)}",
            )
            rows.append(row)

    summary = report_aerodynamic_frame(pd.DataFrame(rows), output_root, "fixed_aerodynamic_change_metrics")
    combined_path = output_root / "all_models_summary.csv"
    summary.to_csv(combined_path, index=False)
    print(f"[CSV Saved] {combined_path}")
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    return summary


def run_sweep_for_model(
    base_model: PorousModelSpec,
    geom,
    output_root: Path,
) -> pd.DataFrame:
    """Run all configured channel heights for one selected model."""
    model_dir = output_root / "aoa_sweep" / base_model.name
    model_dir.mkdir(parents=True, exist_ok=True)

    channel_heights = _unique_channel_heights(
        tuple(float(value) for value in AOA_SWEEP_CHANNEL_HEIGHTS)
    )
    if not channel_heights:
        channel_heights = (float(CHANNEL_HEIGHT),)

    print()
    print("=" * 80)
    print(f"RUNNING AOA SWEEP FOR {base_model.name}")
    print("=" * 80)
    description_prefix = base_model.description.split(
        "; in-plane opening H=",
        maxsplit=1,
    )[0]
    print(
        description_prefix
        + "; 2D flow per unit span; D_h=2H."
    )
    print(
        "AoA range [deg]            : "
        f"{AOA_SWEEP_START_DEG:g} to {AOA_SWEEP_END_DEG:g} "
        f"step {AOA_SWEEP_STEP_DEG:g}"
    )
    print("Sweep channel heights [m]        : " + ", ".join(f"{value:.6e}" for value in channel_heights))
    print(f"Output folder              : {model_dir}")

    frames: list[pd.DataFrame] = []
    paraview_reference_height = channel_heights[0]

    for channel_height_m in channel_heights:
        model = _build_model_by_name(geom, base_model.name, channel_height_m)
        print(
            f"[Sweep] {model.name}, "
            f"channel height = {representative_channel_height(model):.6e} m"
        )
        export_this = (
            bool(EXPORT_AOA_SWEEP_PARAVIEW)
            and abs(channel_height_m - paraview_reference_height) <= 1.0e-15
        )
        frame = run_porous_aoa_sweep_for_model(
            model=model,
            airfoil_name=AIRFOIL_NAME,
            chord=CHORD,
            n_panels=int(N_PANELS),
            rho_inf=RHO_INF,
            mu_inf=MU_INF,
            p_inf=P_INF,
            reynolds_external=REYNOLDS_EXTERNAL,
            coupling=COUPLING,
            limits=GeometryLimits(),
            alpha_start=AOA_SWEEP_START_DEG,
            alpha_end=AOA_SWEEP_END_DEG,
            alpha_step=AOA_SWEEP_STEP_DEG,
            export_paraview=export_this,
            paraview_output_dir=(
                model_dir / "aoa_sweep_paraview" / _height_label(channel_height_m)
            ),
            field_nx=int(settings.PARAVIEW_FIELD_NX),
            field_ny=int(settings.PARAVIEW_FIELD_NY),
        )
        frames.append(frame)

    sweep_df = pd.concat(frames, ignore_index=True)
    xfoil_sweep = run_xfoil_aoa_sweep_for_case(
        airfoil_name=AIRFOIL_NAME,
        output_dir=Path(XFOIL_OUTPUT_ROOT) / "aoa_sweep" / base_model.name,
        alpha_start=AOA_SWEEP_START_DEG,
        alpha_end=AOA_SWEEP_END_DEG,
        alpha_step=AOA_SWEEP_STEP_DEG,
    )
    if xfoil_sweep is not None:
        sweep_df = add_xfoil_to_sweep_comparison(sweep_df, xfoil_sweep)

    # Always save the numerical table before checking convergence so a failed
    # case can be inspected directly from the CSV.
    csv_path = model_dir / "aoa_sweep_comparison.csv"
    sweep_df.to_csv(csv_path, index=False)
    print(f"[CSV Saved] {csv_path}")

    failed = sweep_df.loc[~sweep_df["coupling_converged"].fillna(False).astype(bool)]
    # Rewrite even when empty, so an old failure report is not left behind.
    failed.to_csv(model_dir / "failed_cases.csv", index=False)
    if not failed.empty:
        print(f"[Warning] {len(failed)} cases invalid or unconverged in {base_model.name}. "
              "Results remain NaN; see failed_cases.csv. Continuing.")
    if settings.ENABLE_PLOTS and settings.RUN_AERODYNAMIC_PLOTS and settings.MAKE_AOA_SWEEP_PLOTS:
        plot_sweep_with_gaps(sweep_df=sweep_df, output_dir=model_dir)
    return sweep_df


@timed_stage("aoa_sweep")
def run_aoa_sweep() -> pd.DataFrame:
    """Run the configured angle-of-attack sweep."""
    if not settings.RUN_AOA_SWEEP:
        print("[Skip] AoA sweep disabled in config.py.")
        return pd.DataFrame()
    output_root = Path(OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RUNNING SELECTED-MODEL AOA SWEEP — PANEL METHOD + XFOIL")
    print("=" * 80)
    print(f"Airfoil                    : NACA {AIRFOIL_NAME}")
    print(f"Reynolds number            : {REYNOLDS_EXTERNAL:.6e}")
    print(f"Panels                     : {int(N_PANELS)}")
    print(f"Selected model             : {SELECTED_MODEL}")
    print(f"AoA-sweep ParaView         : {EXPORT_AOA_SWEEP_PARAVIEW}")

    flow, geom, _, _ = build_base_problem(
        airfoil_name=AIRFOIL_NAME,
        chord=CHORD,
        n_panels=int(N_PANELS),
        aoa_deg=0.0,
        reynolds_external=REYNOLDS_EXTERNAL,
        rho_inf=RHO_INF,
        mu_inf=MU_INF,
        p_inf=P_INF,
    )
    print(f"Freestream speed [m/s]     : {flow.v_inf:.8f}")

    selected = select_models(
        build_model_specs(geom=geom, channel_height_m=float(CHANNEL_HEIGHT))
    )
    frames: list[pd.DataFrame] = []
    for base_model in selected:
        frame = run_sweep_for_model(base_model, geom, output_root)
        frame.insert(0, "model_name", base_model.name)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined = report_aerodynamic_frame(combined, output_root / "aoa_sweep", "sweep_aerodynamic_change_metrics")
    combined_path = output_root / "aoa_sweep" / "selected_models_aoa_sweep_summary.csv"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(combined_path, index=False)
    print(f"[CSV Saved] {combined_path}")
    print("=" * 80)
    print("AOA SWEEP DONE")
    print("=" * 80)
    return combined


@timed_stage("pipeline")
def run_all_workflows() -> dict[str, object]:
    """Run only enabled stages; disabled plotters are never imported or called."""
    settings.validate_config()
    output_root = Path(settings.OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)
    results = {}
    stage_status = []
    stages = [
        ("fixed_cases", settings.RUN_FIXED_CASES, run_fixed_cases, None),
        ("aoa_sweep", settings.RUN_AOA_SWEEP, run_aoa_sweep, None),
        ("aero_plots", settings.ENABLE_PLOTS and settings.RUN_AERODYNAMIC_PLOTS, None, "aerodynamic"),
        ("geometry", settings.ENABLE_PLOTS and settings.RUN_GEOMETRY, None, "geometry"),
        ("contours", settings.ENABLE_PLOTS and settings.RUN_CONTOURS, None, "contours"),
        ("streamlines", settings.ENABLE_PLOTS and settings.RUN_STREAMLINES, None, "streamlines"),
    ]
    print("="*80)
    print("POROUS-AIRFOIL PIPELINE -- stages controlled by config.py")
    print("="*80)
    try:
        for number, (name, enabled, operation, module_name) in enumerate(stages, start=1):
            if not enabled:
                results[name] = {"status": "skipped", "message": "disabled in config.py"}
                stage_status.append({"stage": name, "status": "skipped", "elapsed_seconds": 0.0,
                                     "message": "disabled in config.py"})
                print(f"[{number}/{len(stages)}] {name}: disabled")
                continue
            print(f"[{number}/{len(stages)}] Running {name}")
            started = time.perf_counter()
            status, message = "completed", ""
            try:
                if module_name is not None:
                    module = importlib.import_module(f".plotting.{module_name}", __package__)
                    # Prefer this run's authoritative combined sweep table. An
                    # explicit configured input path still takes precedence.
                    if module_name == "aerodynamic" and settings.PLOT_AERO_INPUT_CSV_PATH is None:
                        current_sweep = results.get("aoa_sweep")
                        if isinstance(current_sweep, pd.DataFrame) and not current_sweep.empty:
                            module.INPUT_CSV_PATH = output_root / "aoa_sweep" / "selected_models_aoa_sweep_summary.csv"
                    operation = module.main
                results[name] = operation()
                if isinstance(results[name], dict):
                    status = results[name].get("status", status)
                    message = results[name].get("message", message)
                elif isinstance(results[name], pd.DataFrame) and "coupling_converged" in results[name]:
                    failed = int((~results[name]["coupling_converged"].fillna(False).astype(bool)).sum())
                    if failed:
                        status, message = "completed_with_unconverged_cases", f"{failed} cases retained as NaN"
            except Exception as exc:
                status, message = "failed", f"{type(exc).__name__}: {exc}"
                results[name] = {"status": status, "message": message}
                print(f"[Stage failed] {name}: {message}")
                if not settings.CONTINUE_AFTER_STAGE_ERROR:
                    raise
            finally:
                stage_status.append({"stage": name, "status": status,
                                     "elapsed_seconds": time.perf_counter()-started, "message": message})
    finally:
        pd.DataFrame(stage_status).to_csv(output_root / "pipeline_stage_status.csv", index=False)
    results["status"] = (
        "completed_with_failures" if any(s["status"] == "failed" for s in stage_status) else
        "completed_with_unconverged_cases" if any(s["status"] == "completed_with_unconverged_cases" for s in stage_status) else
        "completed")
    print("[Pipeline] Finished; see pipeline_stage_status.csv and reports/run_timing.csv.")
    return results
