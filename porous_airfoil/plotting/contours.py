"""Journal-ready pressure and velocity difference contours.

This script produces publication figures for the three porous-airfoil layouts:

    1. Horizontal
    2. Vertical
    3. Isotropic

The main journal figures are exported as portrait-style stacked plots:

    - velocity difference contours with one shared colour bar
    - pressure difference contours with one shared colour bar
    - Delta Cp contours with one shared colour bar
    - porous gauge-pressure contours, p_porous - 101325 Pa, with one shared
      colour bar

This layout is better for journal papers than a 1 x 3 landscape figure because
each airfoil remains larger after LaTeX scaling.

Main outputs:
    outputs/final_difference_contours_journal/
        AOA_p10p00_D_9p00mm/
            journal_combined_portrait/
                velocity_difference_percent_portrait_one_colorbar.pdf
                pressure_difference_pa_portrait_one_colorbar.pdf
                pressure_reference_subtracted/
                    porous_pressure_minus_101325_pa_portrait_one_colorbar.pdf
                delta_cp_portrait_one_colorbar.pdf
                velocity_difference_mps_portrait_one_colorbar.pdf
            all_final_contour_cases_summary.csv
            combined_difference_field_ranges.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from ..config import (
    AIRFOIL_NAME,
    CHORD,
    CONTOUR_NX,
    CONTOUR_NY,
    COUPLING,
    GeometryLimits,
    MU_INF,
    N_PANELS,
    OUTPUT_ROOT,
    P_INF,
    REYNOLDS_EXTERNAL,
    RHO_INF,
    SPAN_WIDTH,
)

from ..porous import (
    MultiPassageCoupledSolver,
    build_base_problem,
    build_model_specs,
    build_network_from_model,
    representative_pore_diameter,
)


# =============================================================================
# USER SETTINGS
# =============================================================================
FINAL_CONTOUR_AOA_DEG = 10.0
FINAL_CONTOUR_PORE_DIAMETER_M = 0.0090

FINAL_CONTOUR_MODEL_NAMES = (
    "model_1_4_chordwise",
    "model_2_9_perpendicular",
    "model_3_combined_independent",
)

MODEL_LABELS = {
    "model_1_4_chordwise": "Horizontal",
    "model_2_9_perpendicular": "Vertical",
    "model_3_combined_independent": "Horizontal & Vertical",
}

MODEL_FOLDER_NAMES = {
    "model_1_4_chordwise": "horizontal",
    "model_2_9_perpendicular": "vertical",
    "model_3_combined_independent": "Horizontal & Vertical",
}

OUTPUT_SUBDIR = "final_difference_contours_journal"
FINAL_CONTOUR_OUTPUT_ROOT = Path(OUTPUT_ROOT) / OUTPUT_SUBDIR

SAVE_FORMATS = ("png", "pdf", "svg")
PNG_DPI = 1000

FIELD_NX = int(CONTOUR_NX)
FIELD_NY = int(CONTOUR_NY)

VERBOSE_COUPLING = True


# =============================================================================
# JOURNAL VIEW SETTINGS
# =============================================================================
# This cropped view makes the airfoil much larger in the final journal figure.
# It is better than the wide far-field view for Results and Discussion figures.
JOURNAL_XLIM = (-0.08, 1.08)
JOURNAL_YLIM = (-0.20, 0.20)

# Optional wider view for supplementary material.
EXPORT_WIDE_SUPPORTING_VIEW = False
WIDE_XLIM = (-0.50, 1.50)
WIDE_YLIM = (-0.50, 0.50)

# Main outputs.
EXPORT_VELOCITY_PERCENT_DIFFERENCE = True
EXPORT_PRESSURE_PA_DIFFERENCE = True

# Additional pressure-contour set:
# gauge pressure = porous static pressure - reference pressure.
EXPORT_REFERENCE_SUBTRACTED_PRESSURE = True
REFERENCE_PRESSURE_PA = 101325.0

EXPORT_DELTA_CP_DIFFERENCE = True
EXPORT_VELOCITY_MPS_DIFFERENCE = True

# Recommended journal figure size.
# Portrait 3 x 1 layout gives larger airfoils on the page.
PORTRAIT_FIGSIZE = (6.9, 6.2)
PORTRAIT_FIGSIZE_WIDE = (6.9, 7.1)

# Use no figure suptitle for LaTeX papers.
# Put the explanation in the caption instead.
SHOW_FIGURE_SUPTITLE = False

# Panel labels.
PANEL_LABELS = ("(a)", "(b)", "(c)")
PANEL_LABEL_X = 0.018
PANEL_LABEL_Y = 0.875


# =============================================================================
# CONTOUR STYLE
# =============================================================================
DIFFERENCE_CMAP = "coolwarm"
DIFFERENCE_LEVELS = 101
SYMMETRIC_LIMIT_PERCENTILE = 99.3
MIN_SYMMETRIC_LIMIT = 1.0e-12

# Use one global limit across all three models for each field.
# This is important for fair comparison.
USE_COMMON_COLOUR_LIMITS = True

# Optional manual colour limits. Set to None for automatic robust limits.
MANUAL_VELOCITY_PERCENT_LIMIT = None
MANUAL_PRESSURE_PA_LIMIT = None
MANUAL_REFERENCE_SUBTRACTED_PRESSURE_LIMIT = None
MANUAL_DELTA_CP_LIMIT = None
MANUAL_VELOCITY_MPS_LIMIT = None


# =============================================================================
# AIRFOIL AND NETWORK STYLE
# =============================================================================
AIRFOIL_FACE_COLOR = "white"
AIRFOIL_EDGE_COLOR = "black"
AIRFOIL_EDGE_WIDTH = 0.90

DRAW_INTERNAL_NETWORK = True
NETWORK_CHANNEL_COLOR = "#222222"
NETWORK_CHANNEL_LINEWIDTH = 0.62
NETWORK_CHANNEL_ALPHA = 0.78

DRAW_PORE_MARKERS = False
NETWORK_PORE_FACE_COLOR = "white"
NETWORK_PORE_EDGE_COLOR = "black"
NETWORK_PORE_SIZE = 8.0


# =============================================================================
# PLOT STYLE
# =============================================================================
def apply_publication_style() -> None:
    """Apply clean journal-style Matplotlib settings."""
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": PNG_DPI,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "axes.linewidth": 0.85,
            "axes.labelsize": 10.2,
            "axes.titlesize": 10.2,
            "xtick.labelsize": 9.3,
            "ytick.labelsize": 9.3,
            "legend.fontsize": 8.8,
            "figure.titlesize": 11.2,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 3.8,
            "ytick.major.size": 3.8,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "xtick.major.width": 0.78,
            "ytick.major.width": 0.78,
            "xtick.minor.width": 0.62,
            "ytick.minor.width": 0.62,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass(frozen=True)
class ContourView:
    """One plotting window."""

    folder_name: str
    label: str
    xlim: tuple[float, float]
    ylim: tuple[float, float]


@dataclass
class FieldData:
    """Velocity, pressure and Cp fields on a structured plotting grid."""

    XX: np.ndarray
    YY: np.ndarray
    XXn: np.ndarray
    YYn: np.ndarray

    U: np.ndarray
    V: np.ndarray
    velocity_magnitude: np.ndarray
    pressure: np.ndarray
    pressure_coefficient: np.ndarray

    inside: np.ndarray

    v_inf: float
    q_inf: float


@dataclass
class DifferenceData:
    """Porous-minus-solid difference fields."""

    delta_velocity_m_s: np.ndarray
    delta_velocity_over_vinf_percent: np.ndarray
    delta_pressure_pa: np.ndarray
    porous_pressure_minus_reference_pa: np.ndarray
    delta_cp: np.ndarray


@dataclass
class CombinedContourCase:
    """One solved model result stored for combined plotting."""

    model_name: str
    model_label: str
    view: ContourView
    geom: object
    network: object
    porous_field: FieldData
    solid_field: FieldData
    difference: DifferenceData


# =============================================================================
# PATH HELPERS
# =============================================================================
def windows_long_path(path: Path) -> str:
    """Return a Windows-safe path string for long nested folders."""
    path = Path(path)

    if os.name != "nt":
        return str(path)

    absolute = str(path.resolve())

    if absolute.startswith("\\\\?\\"):
        return absolute

    if absolute.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute.lstrip("\\")

    if len(absolute) >= 240:
        return "\\\\?\\" + absolute

    return absolute


def safe_mkdir(path: Path) -> None:
    """Create directory safely."""
    os.makedirs(windows_long_path(Path(path)), exist_ok=True)


def save_figure_all_formats(fig: mpl.figure.Figure, output_stem: Path) -> None:
    """Save one figure as PNG, PDF and SVG."""
    output_stem = Path(output_stem)
    safe_mkdir(output_stem.parent)

    for ext in SAVE_FORMATS:
        path = output_stem.with_suffix(f".{ext}")

        kwargs = {
            "bbox_inches": "tight",
            "pad_inches": 0.035,
        }

        if ext == "png":
            kwargs["dpi"] = PNG_DPI

        fig.savefig(windows_long_path(path), **kwargs)
        print(f"[Plot Saved] {path}")


def write_csv(path: Path, rows: list[dict] | dict) -> None:
    """Write CSV safely."""
    path = Path(path)
    safe_mkdir(path.parent)

    if isinstance(rows, dict):
        rows = [rows]

    with open(windows_long_path(path), "w", encoding="utf-8", newline="") as f:
        pd.DataFrame(rows).to_csv(f, index=False)


def format_float_label(value: float, decimals: int = 2) -> str:
    """Return folder-safe signed float label."""
    sign = "p" if value >= 0.0 else "m"
    mag = f"{abs(float(value)):.{decimals}f}".replace(".", "p")
    return f"{sign}{mag}"


def diameter_label_mm(diameter_m: float) -> str:
    """Return folder-safe diameter label in mm."""
    return f"{1000.0 * float(diameter_m):.2f}mm".replace(".", "p")


# =============================================================================
# VIEW AND MODEL HELPERS
# =============================================================================
def contour_views() -> tuple[ContourView, ...]:
    """Return plotting views."""
    views = [
        ContourView(
            folder_name="journal_zoom",
            label="Journal zoom",
            xlim=JOURNAL_XLIM,
            ylim=JOURNAL_YLIM,
        )
    ]

    if EXPORT_WIDE_SUPPORTING_VIEW:
        views.append(
            ContourView(
                folder_name="wide_supporting",
                label="Wide view",
                xlim=WIDE_XLIM,
                ylim=WIDE_YLIM,
            )
        )

    return tuple(views)


def model_label(model_name: str) -> str:
    """Return publication model label."""
    return MODEL_LABELS.get(model_name, model_name)


def model_folder_name(model_name: str) -> str:
    """Return folder-safe model name."""
    return MODEL_FOLDER_NAMES.get(model_name, model_name)


def select_contour_models(all_models: list) -> list:
    """Return requested models in fixed order."""
    models_by_name = {model.name: model for model in all_models}

    missing = [
        name for name in FINAL_CONTOUR_MODEL_NAMES
        if name not in models_by_name
    ]

    if missing:
        raise RuntimeError(
            f"Requested contour model(s) not found: {missing}. "
            f"Available models: {list(models_by_name)}"
        )

    return [models_by_name[name] for name in FINAL_CONTOUR_MODEL_NAMES]


# =============================================================================
# FIELD COMPUTATION
# =============================================================================
def build_field_grid(
    geom,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build dimensional and nondimensional contour grids."""
    chord = geom.chord

    x = np.linspace(xlim[0] * chord, xlim[1] * chord, FIELD_NX)
    y = np.linspace(ylim[0] * chord, ylim[1] * chord, FIELD_NY)

    XX, YY = np.meshgrid(x, y)

    return XX, YY, XX / chord, YY / chord


def compute_field_data(
    aero_solver,
    aero_result,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> FieldData:
    """
    Compute velocity magnitude, static pressure and Cp over the grid.

    The velocity field is evaluated without an inside mask first. The display
    fields are then masked inside the airfoil body.
    """
    geom = aero_solver.geom
    flow = aero_solver.flow

    XX, YY, XXn, YYn = build_field_grid(
        geom=geom,
        xlim=xlim,
        ylim=ylim,
    )

    inside = aero_solver.build_inside_mask(XX, YY)
    blank_inside = np.zeros_like(XX, dtype=bool)

    U_raw, V_raw, _, _ = aero_solver.velocity_field(
        XX,
        YY,
        aero_result,
        inside=blank_inside,
    )

    U_raw = np.asarray(U_raw, dtype=float)
    V_raw = np.asarray(V_raw, dtype=float)

    velocity_raw = np.sqrt(U_raw**2 + V_raw**2)

    v_inf = float(flow.v_inf)
    q_inf = 0.5 * float(flow.rho_inf) * v_inf**2

    cp_raw = 1.0 - (velocity_raw / max(v_inf, 1.0e-16)) ** 2
    pressure_raw = float(flow.p_inf) + q_inf * cp_raw

    U = U_raw.copy()
    V = V_raw.copy()
    velocity = velocity_raw.copy()
    pressure = pressure_raw.copy()
    cp = cp_raw.copy()

    U[inside] = np.nan
    V[inside] = np.nan
    velocity[inside] = np.nan
    pressure[inside] = np.nan
    cp[inside] = np.nan

    return FieldData(
        XX=XX,
        YY=YY,
        XXn=XXn,
        YYn=YYn,
        U=U,
        V=V,
        velocity_magnitude=velocity,
        pressure=pressure,
        pressure_coefficient=cp,
        inside=inside,
        v_inf=v_inf,
        q_inf=q_inf,
    )


def compute_difference_data(
    porous_field: FieldData,
    solid_field: FieldData,
) -> DifferenceData:
    """Compute porous-minus-solid difference fields."""
    inside = porous_field.inside | solid_field.inside

    delta_velocity = (
        porous_field.velocity_magnitude
        - solid_field.velocity_magnitude
    )

    delta_velocity_over_vinf_percent = (
        100.0 * delta_velocity / max(solid_field.v_inf, 1.0e-16)
    )

    delta_pressure = porous_field.pressure - solid_field.pressure

    # Gauge pressure for the porous case relative to the requested reference.
    # This is a separate physical field from porous-minus-solid pressure.
    porous_pressure_minus_reference = (
        porous_field.pressure - REFERENCE_PRESSURE_PA
    )

    delta_cp = (
        porous_field.pressure_coefficient
        - solid_field.pressure_coefficient
    )

    for arr in (
        delta_velocity,
        delta_velocity_over_vinf_percent,
        delta_pressure,
        porous_pressure_minus_reference,
        delta_cp,
    ):
        arr[inside] = np.nan

    return DifferenceData(
        delta_velocity_m_s=delta_velocity,
        delta_velocity_over_vinf_percent=delta_velocity_over_vinf_percent,
        delta_pressure_pa=delta_pressure,
        porous_pressure_minus_reference_pa=porous_pressure_minus_reference,
        delta_cp=delta_cp,
    )


# =============================================================================
# NUMERICAL HELPERS
# =============================================================================
def finite_values(arr: np.ndarray) -> np.ndarray:
    """Return finite values from an array."""
    values = np.asarray(arr, dtype=float)
    return values[np.isfinite(values)]


def robust_symmetric_limit_from_arrays(
    arrays: list[np.ndarray],
    percentile: float = SYMMETRIC_LIMIT_PERCENTILE,
) -> float:
    """Return one robust symmetric plotting limit for multiple arrays."""
    finite_sets: list[np.ndarray] = []

    for arr in arrays:
        finite = finite_values(arr)
        if finite.size:
            finite_sets.append(finite)

    if not finite_sets:
        return 1.0

    values = np.concatenate(finite_sets)

    limit = float(
        np.nanpercentile(
            np.abs(values),
            percentile,
        )
    )

    if not np.isfinite(limit) or limit <= MIN_SYMMETRIC_LIMIT:
        limit = float(np.nanmax(np.abs(values)))

    if not np.isfinite(limit) or limit <= MIN_SYMMETRIC_LIMIT:
        return 1.0

    return limit


def manual_or_auto_limit(
    manual_limit: float | None,
    cases: list[CombinedContourCase],
    quantity_name: str,
) -> float:
    """Return manual colour limit if provided; otherwise robust automatic limit."""
    if manual_limit is not None:
        value = abs(float(manual_limit))
        if np.isfinite(value) and value > 0.0:
            return value

    arrays = [
        getattr(case.difference, quantity_name)
        for case in cases
    ]

    return robust_symmetric_limit_from_arrays(arrays)


def contour_levels_and_norm(
    limit: float,
) -> tuple[np.ndarray, TwoSlopeNorm]:
    """Return symmetric contour levels and zero-centred normalisation."""
    limit = max(float(limit), MIN_SYMMETRIC_LIMIT)

    levels = np.linspace(
        -limit,
        limit,
        DIFFERENCE_LEVELS,
    )

    norm = TwoSlopeNorm(
        vmin=-limit,
        vcenter=0.0,
        vmax=limit,
    )

    return levels, norm


# =============================================================================
# AIRFOIL AND NETWORK DRAWING
# =============================================================================
def add_airfoil_mask_and_outline(
    ax: plt.Axes,
    geom,
):
    """Draw white airfoil mask and black outline."""
    chord = geom.chord

    body_patch = ax.fill(
        geom.XB / chord,
        geom.YB / chord,
        facecolor=AIRFOIL_FACE_COLOR,
        edgecolor="none",
        linewidth=0.0,
        zorder=8,
    )[0]

    ax.plot(
        geom.XB / chord,
        geom.YB / chord,
        color=AIRFOIL_EDGE_COLOR,
        linewidth=AIRFOIL_EDGE_WIDTH,
        solid_joinstyle="round",
        solid_capstyle="round",
        zorder=15,
    )

    return body_patch


def add_internal_network_overlay(
    ax: plt.Axes,
    geom,
    network,
    clip_patch=None,
) -> None:
    """Draw internal channel centre-lines and optional pore markers."""
    if not DRAW_INTERNAL_NETWORK:
        return

    chord = geom.chord

    for passage in network.passages:
        sp1 = passage.pore_surface_point(geom, passage.pore1)
        sp2 = passage.pore_surface_point(geom, passage.pore2)

        x1 = float(sp1.x) / chord
        y1 = float(sp1.y) / chord
        x2 = float(sp2.x) / chord
        y2 = float(sp2.y) / chord

        line = ax.plot(
            [x1, x2],
            [y1, y2],
            "-",
            color=NETWORK_CHANNEL_COLOR,
            linewidth=NETWORK_CHANNEL_LINEWIDTH,
            alpha=NETWORK_CHANNEL_ALPHA,
            solid_capstyle="round",
            zorder=13,
            clip_on=True,
        )[0]

        if clip_patch is not None:
            line.set_clip_path(clip_patch)

        if DRAW_PORE_MARKERS:
            points = ax.scatter(
                [x1, x2],
                [y1, y2],
                s=NETWORK_PORE_SIZE,
                marker="o",
                facecolors=NETWORK_PORE_FACE_COLOR,
                edgecolors=NETWORK_PORE_EDGE_COLOR,
                linewidths=0.35,
                zorder=16,
                clip_on=True,
            )

            if clip_patch is not None:
                points.set_clip_path(clip_patch)


# =============================================================================
# AXIS AND COLOUR BAR FORMATTING
# =============================================================================
def format_contour_axis(
    ax: plt.Axes,
    view: ContourView,
    show_xlabel: bool,
) -> None:
    """Format each contour axis."""
    ax.set_xlim(*view.xlim)
    ax.set_ylim(*view.ylim)
    ax.set_aspect("equal", adjustable="box")

    if show_xlabel:
        ax.set_xlabel(r"$x/c$")
    else:
        ax.set_xlabel("")

    ax.set_ylabel(r"$y/c$")

    ax.tick_params(
        axis="both",
        direction="in",
        top=True,
        right=True,
        length=3.4,
        width=0.75,
    )

    # Show x-axis numbers only on the bottom panel, (c).
    ax.tick_params(
        axis="x",
        labelbottom=show_xlabel,
    )
    ax.minorticks_on()


def format_colorbar(cbar, label: str) -> None:
    """Format colour bar."""
    cbar.set_label(label)

    cbar.ax.tick_params(
        direction="in",
        length=3.0,
        width=0.70,
    )

    # Display colour-bar values to one decimal place.
    cbar.ax.yaxis.set_major_formatter(
        mpl.ticker.FormatStrFormatter("%.1f")
    )




# =============================================================================
# JOURNAL COMBINED PORTRAIT FIGURES
# =============================================================================
def plot_combined_portrait_difference_contours(
    cases: list[CombinedContourCase],
    quantity_name: str,
    colourbar_label: str,
    output_stem: Path,
    manual_limit: float | None = None,
    figure_title: str | None = None,
) -> None:
    """
    Plot all models in a 3 x 1 portrait layout with one shared colour bar.

    This is the recommended figure format for the journal paper.
    """
    if not cases:
        return

    view = cases[0].view
    n_cases = len(cases)

    wide = view.folder_name == "wide_supporting"
    figsize = PORTRAIT_FIGSIZE_WIDE if wide else PORTRAIT_FIGSIZE

    limit = manual_or_auto_limit(
        manual_limit=manual_limit,
        cases=cases,
        quantity_name=quantity_name,
    )
    # Round the symmetric colour limit upward to one decimal place.
    # Example: 18.07 becomes 18.1.
    limit = np.ceil(limit * 10.0) / 10.0
    levels, norm = contour_levels_and_norm(limit)

    fig = plt.figure(figsize=figsize)

    grid = fig.add_gridspec(
        nrows=n_cases,
        ncols=2,
        width_ratios=[1.0, 0.040],
        height_ratios=[1.0] * n_cases,
        hspace=0.12,
        wspace=0.045,
    )

    axes = [
        fig.add_subplot(grid[row, 0])
        for row in range(n_cases)
    ]

    cax = fig.add_subplot(grid[:, 1])

    im = None

    for row, (ax, case) in enumerate(zip(axes, cases)):
        data = getattr(case.difference, quantity_name)

        im = ax.contourf(
            case.porous_field.XXn,
            case.porous_field.YYn,
            data,
            levels=levels,
            cmap=DIFFERENCE_CMAP,
            norm=norm,
            extend="both",
        )

        patch = add_airfoil_mask_and_outline(
            ax=ax,
            geom=case.geom,
        )

        add_internal_network_overlay(
            ax=ax,
            geom=case.geom,
            network=case.network,
            clip_patch=patch,
        )

        show_xlabel = row == n_cases - 1

        format_contour_axis(
            ax=ax,
            view=view,
            show_xlabel=show_xlabel,
        )

        panel_label = PANEL_LABELS[row] if row < len(PANEL_LABELS) else f"({row + 1})"

        ax.text(
            PANEL_LABEL_X,
            PANEL_LABEL_Y,
            f"{panel_label} {case.model_label}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.6,
            fontweight="bold",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.72,
                pad=1.6,
            ),
            zorder=30,
        )

    if im is not None:
        # Nine evenly spaced ticks across the symmetric range.
        colorbar_ticks = np.linspace(
            -limit,
            limit,
            9,
        )

        cbar = fig.colorbar(
            im,
            cax=cax,
            ticks=colorbar_ticks,
            format=mpl.ticker.FormatStrFormatter("%.1f"),
        )

        format_colorbar(cbar, colourbar_label)

    if SHOW_FIGURE_SUPTITLE and figure_title:
        fig.suptitle(
            figure_title,
            y=0.995,
        )

    save_figure_all_formats(
        fig=fig,
        output_stem=output_stem,
    )

    plt.close(fig)


def export_combined_portrait_figures(
    combined_cases: list[CombinedContourCase],
    output_root: Path,
) -> None:
    """Export portrait journal figures grouped by view."""
    if not combined_cases:
        return

    combined_dir = output_root / "journal_combined_portrait"
    safe_mkdir(combined_dir)

    for view in contour_views():
        cases_for_view = [
            case
            for case in combined_cases
            if case.view.folder_name == view.folder_name
        ]

        if not cases_for_view:
            continue

        view_dir = combined_dir / view.folder_name
        safe_mkdir(view_dir)

        if EXPORT_VELOCITY_PERCENT_DIFFERENCE:
            plot_combined_portrait_difference_contours(
                cases=cases_for_view,
                quantity_name="delta_velocity_over_vinf_percent",
                colourbar_label=(
                    r"$100(|V|_{\mathrm{porous}}-|V|_{\mathrm{solid}})/V_\infty$ [%]"
                ),
                output_stem=view_dir / "velocity_difference_percent_portrait_one_colorbar",
                manual_limit=MANUAL_VELOCITY_PERCENT_LIMIT,
                figure_title="Velocity-difference contours",
            )

        if EXPORT_PRESSURE_PA_DIFFERENCE:
            plot_combined_portrait_difference_contours(
                cases=cases_for_view,
                quantity_name="delta_pressure_pa",
                colourbar_label=(
                    r"$p_{\mathrm{porous}}-p_{\mathrm{solid}}$ [Pa]"
                ),
                output_stem=view_dir / "pressure_difference_pa_portrait_one_colorbar",
                manual_limit=MANUAL_PRESSURE_PA_LIMIT,
                figure_title="Pressure-difference contours",
            )

        if EXPORT_REFERENCE_SUBTRACTED_PRESSURE:
            reference_pressure_dir = (
                view_dir / "pressure_reference_subtracted"
            )
            safe_mkdir(reference_pressure_dir)

            plot_combined_portrait_difference_contours(
                cases=cases_for_view,
                quantity_name="porous_pressure_minus_reference_pa",
                colourbar_label=(
                    r"$p_{\mathrm{porous}}-p_{\mathrm{ref}}$ [Pa], "
                    rf"$p_{{\mathrm{{ref}}}}={REFERENCE_PRESSURE_PA:.0f}$ Pa"
                ),
                output_stem=(
                    reference_pressure_dir
                    / "porous_pressure_minus_101325_pa_portrait_one_colorbar"
                ),
                manual_limit=MANUAL_REFERENCE_SUBTRACTED_PRESSURE_LIMIT,
                figure_title="Reference-pressure-subtracted contours",
            )

        if EXPORT_DELTA_CP_DIFFERENCE:
            plot_combined_portrait_difference_contours(
                cases=cases_for_view,
                quantity_name="delta_cp",
                colourbar_label=(
                    r"$C_{p,\mathrm{porous}}-C_{p,\mathrm{solid}}$"
                ),
                output_stem=view_dir / "delta_cp_portrait_one_colorbar",
                manual_limit=MANUAL_DELTA_CP_LIMIT,
                figure_title="Pressure-coefficient-difference contours",
            )

        if EXPORT_VELOCITY_MPS_DIFFERENCE:
            plot_combined_portrait_difference_contours(
                cases=cases_for_view,
                quantity_name="delta_velocity_m_s",
                colourbar_label=(
                    r"$|V|_{\mathrm{porous}}-|V|_{\mathrm{solid}}$ [m s$^{-1}$]"
                ),
                output_stem=view_dir / "velocity_difference_mps_portrait_one_colorbar",
                manual_limit=MANUAL_VELOCITY_MPS_LIMIT,
                figure_title="Velocity-difference contours",
            )


# =============================================================================
# SUMMARY HELPERS
# =============================================================================
def difference_range_summary(case: CombinedContourCase) -> dict:
    """Return min/max summary for one case."""
    diff = case.difference

    return {
        "model_name": case.model_name,
        "model_label": case.model_label,
        "view": case.view.folder_name,
        "velocity_delta_min_m_s": float(np.nanmin(diff.delta_velocity_m_s)),
        "velocity_delta_max_m_s": float(np.nanmax(diff.delta_velocity_m_s)),
        "velocity_delta_over_vinf_min_percent": float(
            np.nanmin(diff.delta_velocity_over_vinf_percent)
        ),
        "velocity_delta_over_vinf_max_percent": float(
            np.nanmax(diff.delta_velocity_over_vinf_percent)
        ),
        "pressure_delta_min_pa": float(np.nanmin(diff.delta_pressure_pa)),
        "pressure_delta_max_pa": float(np.nanmax(diff.delta_pressure_pa)),
        "porous_pressure_minus_reference_min_pa": float(
            np.nanmin(diff.porous_pressure_minus_reference_pa)
        ),
        "porous_pressure_minus_reference_max_pa": float(
            np.nanmax(diff.porous_pressure_minus_reference_pa)
        ),
        "reference_pressure_pa": float(REFERENCE_PRESSURE_PA),
        "delta_cp_min": float(np.nanmin(diff.delta_cp)),
        "delta_cp_max": float(np.nanmax(diff.delta_cp)),
    }


# =============================================================================
# MODEL RUNNER
# =============================================================================
def run_one_final_contour_model(
    model,
    geom,
    flow,
    aero_solver,
    solid_result,
    solid_field_cache: dict[str, FieldData],
    output_root: Path,
    combined_cases: list[CombinedContourCase],
) -> dict:
    """Solve one porous model and store its difference fields."""
    label = model_label(model.name)
    folder_name = model_folder_name(model.name)

    print()
    print("=" * 80)
    print(f"RUNNING FINAL JOURNAL CONTOURS: {label}")
    print("=" * 80)
    print(model.description)
    print(f"AoA [deg]          : {flow.aoa_deg:.6g}")
    print(f"Pore diameter [m] : {representative_pore_diameter(model):.6e}")
    print(f"Independent paths : {len(model.passages)}")

    model_dir = output_root / folder_name
    safe_mkdir(model_dir)

    network = build_network_from_model(
        model=model,
        span_width=SPAN_WIDTH,
    )

    t0 = time.perf_counter()

    coupled = MultiPassageCoupledSolver(
        aero_solver=aero_solver,
        network=network,
        coupling=COUPLING,
        limits=GeometryLimits(),
    ).solve(verbose=VERBOSE_COUPLING)

    elapsed = time.perf_counter() - t0

    if not coupled.converged:
        print("[Result] Coupling did not converge. Reporting NaN.")

        summary = {
            "model_name": model.name,
            "model_label": label,
            "aoa_deg": flow.aoa_deg,
            "pore_diameter_m": representative_pore_diameter(model),
            "solid_CL": float(solid_result.CL),
            "solid_CM": float(solid_result.CM),
            "porous_CL": float("nan"),
            "porous_CM": float("nan"),
            "delta_CL": float("nan"),
            "delta_CM": float("nan"),
            "gain_percent_vs_solid_panel": float("nan"),
            "coupling_converged": False,
            "coupling_iterations": int(coupled.iterations),
            "max_vn_m_s": float("nan"),
            "max_vn_over_vinf": float("nan"),
            "elapsed_seconds": float(elapsed),
        }

        write_csv(model_dir / "final_contour_case_summary.csv", summary)
        return summary

    print(f"[Result] Converged          : {coupled.converged}")
    print(f"[Result] Iterations         : {coupled.iterations}")
    print(f"[Result] Solid CL           : {solid_result.CL:.8f}")
    print(f"[Result] Porous CL          : {coupled.aero_result.CL:.8f}")
    print(f"[Result] Solid CM           : {solid_result.CM:.8f}")
    print(f"[Result] Porous CM          : {coupled.aero_result.CM:.8f}")
    print(f"[Result] max |vn| / Vinf    : {coupled.max_vn_over_vinf:.8f}")
    print(f"[Result] elapsed [s]        : {elapsed:.3f}")

    for view in contour_views():
        print(f"[Field] Computing {view.label.lower()} fields for {label}...")

        if view.folder_name not in solid_field_cache:
            solid_field_cache[view.folder_name] = compute_field_data(
                aero_solver=aero_solver,
                aero_result=solid_result,
                xlim=view.xlim,
                ylim=view.ylim,
            )

        porous_field = compute_field_data(
            aero_solver=aero_solver,
            aero_result=coupled.aero_result,
            xlim=view.xlim,
            ylim=view.ylim,
        )

        solid_field = solid_field_cache[view.folder_name]

        diff = compute_difference_data(
            porous_field=porous_field,
            solid_field=solid_field,
        )

        combined_cases.append(
            CombinedContourCase(
                model_name=model.name,
                model_label=label,
                view=view,
                geom=geom,
                network=network,
                porous_field=porous_field,
                solid_field=solid_field,
                difference=diff,
            )
        )

    summary = {
        "model_name": model.name,
        "model_label": label,
        "aoa_deg": flow.aoa_deg,
        "pore_diameter_m": representative_pore_diameter(model),
        "solid_CL": float(solid_result.CL),
        "solid_CM": float(solid_result.CM),
        "porous_CL": float(coupled.aero_result.CL),
        "porous_CM": float(coupled.aero_result.CM),
        "delta_CL": float(coupled.aero_result.CL - solid_result.CL),
        "delta_CM": float(coupled.aero_result.CM - solid_result.CM),
        "gain_percent_vs_solid_panel": float(
            100.0 * (coupled.aero_result.CL / solid_result.CL - 1.0)
        ),
        "coupling_converged": True,
        "coupling_iterations": int(coupled.iterations),
        "max_vn_m_s": float(coupled.max_vn),
        "max_vn_over_vinf": float(coupled.max_vn_over_vinf),
        "elapsed_seconds": float(elapsed),
    }

    write_csv(model_dir / "final_contour_case_summary.csv", summary)

    return summary


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    apply_publication_style()

    aoa_label = format_float_label(FINAL_CONTOUR_AOA_DEG, decimals=2)
    d_label = diameter_label_mm(FINAL_CONTOUR_PORE_DIAMETER_M)

    output_root = FINAL_CONTOUR_OUTPUT_ROOT / f"AOA_{aoa_label}_D_{d_label}"
    safe_mkdir(output_root)

    print("=" * 80)
    print("RUNNING JOURNAL-READY PORTRAIT DIFFERENCE CONTOURS")
    print("=" * 80)
    print(f"Airfoil                    : NACA {AIRFOIL_NAME}")
    print(f"AoA [deg]                  : {FINAL_CONTOUR_AOA_DEG:.6g}")
    print(f"Reynolds number            : {REYNOLDS_EXTERNAL:.6e}")
    print(f"Panels                     : {N_PANELS}")
    print(f"Pore diameter [m]          : {FINAL_CONTOUR_PORE_DIAMETER_M:.6e}")
    print(f"Pore diameter [mm]         : {1000.0 * FINAL_CONTOUR_PORE_DIAMETER_M:.3f}")
    print(f"Reference pressure [Pa]    : {REFERENCE_PRESSURE_PA:.3f}")
    print(
        "Models                     : "
        + ", ".join(model_label(name) for name in FINAL_CONTOUR_MODEL_NAMES)
    )
    print(f"Field grid                 : {FIELD_NX} x {FIELD_NY}")
    print(f"Journal x/c limits         : {JOURNAL_XLIM}")
    print(f"Journal y/c limits         : {JOURNAL_YLIM}")
    print(f"Output root                : {output_root}")

    flow, geom, _, aero_solver = build_base_problem(
        airfoil_name=AIRFOIL_NAME,
        chord=CHORD,
        n_panels=int(N_PANELS),
        aoa_deg=float(FINAL_CONTOUR_AOA_DEG),
        reynolds_external=REYNOLDS_EXTERNAL,
        rho_inf=RHO_INF,
        mu_inf=MU_INF,
        p_inf=P_INF,
    )

    print(f"Freestream speed [m/s]     : {flow.v_inf:.8f}")
    print("[Setup] Solving solid baseline...")

    solid_result = aero_solver.solve()

    print(f"[Setup] Solid CL            : {solid_result.CL:.8f}")
    print(f"[Setup] Solid CD            : {solid_result.CD:.8e}")
    print(f"[Setup] Solid CM            : {solid_result.CM:.8f}")

    all_models = build_model_specs(
        geom=geom,
        diameter_m=float(FINAL_CONTOUR_PORE_DIAMETER_M),
    )

    selected_models = select_contour_models(all_models)

    solid_field_cache: dict[str, FieldData] = {}
    summary_rows: list[dict] = []
    combined_cases: list[CombinedContourCase] = []

    for model in selected_models:
        row = run_one_final_contour_model(
            model=model,
            geom=geom,
            flow=flow,
            aero_solver=aero_solver,
            solid_result=solid_result,
            solid_field_cache=solid_field_cache,
            output_root=output_root,
            combined_cases=combined_cases,
        )
        summary_rows.append(row)

    export_combined_portrait_figures(
        combined_cases=combined_cases,
        output_root=output_root,
    )

    combined_summary_path = output_root / "all_final_contour_cases_summary.csv"
    write_csv(combined_summary_path, summary_rows)
    print(f"[CSV Saved] {combined_summary_path}")

    field_ranges = [
        difference_range_summary(case)
        for case in combined_cases
    ]

    field_ranges_path = output_root / "combined_difference_field_ranges.csv"
    write_csv(field_ranges_path, field_ranges)
    print(f"[CSV Saved] {field_ranges_path}")

    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()