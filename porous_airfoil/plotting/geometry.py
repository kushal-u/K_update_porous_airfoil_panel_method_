"""Export 2D porous-airfoil model figures for Results and Discussion.

This script exports publication-style geometry figures for the first three
porous-airfoil models:

    Model 1: Horizontal / chordwise porous model
    Model 2: Vertical / perpendicular porous model
    Model 3: Isotropic / combined independent porous model

Outputs:
    - Combined three-model comparison figure
    - Individual figure for each porous model
    - CSV file containing exact pore and channel coordinates for each model

The script does not run the aerodynamic solver or coupling loop. It only builds
the airfoil geometry and porous-network model definitions, then exports their
2D layout for documentation and publication figures.
"""

from __future__ import annotations

from .. import config as settings
from ..reporting import timed_stage, aerodynamic_change_metrics, print_case_metrics, report_aerodynamic_frame

from pathlib import Path
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd

from ..config import (
    AIRFOIL_NAME,
    CHORD,
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
    build_base_problem,
    build_model_specs,
    build_network_from_model,
    representative_pore_diameter,
)


# =============================================================================
# USER SETTINGS
# =============================================================================
EXPORT_AOA_DEG = settings.PLOT_GEOMETRY_EXPORT_AOA_DEG
EXPORT_PORE_DIAMETER_M = settings.PLOT_GEOMETRY_EXPORT_PORE_DIAMETER_M

MODELS_TO_EXPORT = settings.PLOT_GEOMETRY_MODELS_TO_EXPORT

MODEL_LABELS = settings.PLOT_GEOMETRY_MODEL_LABELS

MODEL_SHORT_LABELS = settings.PLOT_GEOMETRY_MODEL_SHORT_LABELS

MODEL_FOLDER_NAMES = settings.PLOT_GEOMETRY_MODEL_FOLDER_NAMES

EXPORT_ROOT = settings.PLOT_GEOMETRY_EXPORT_ROOT

SAVE_FORMATS = settings.PLOT_GEOMETRY_SAVE_FORMATS
PNG_DPI = settings.PLOT_GEOMETRY_PNG_DPI

FIGURE_XLIM = settings.PLOT_GEOMETRY_FIGURE_XLIM
FIGURE_YLIM = settings.PLOT_GEOMETRY_FIGURE_YLIM

AIRFOIL_FACE_COLOR = settings.PLOT_GEOMETRY_AIRFOIL_FACE_COLOR
AIRFOIL_EDGE_COLOR = settings.PLOT_GEOMETRY_AIRFOIL_EDGE_COLOR
AIRFOIL_EDGE_WIDTH = settings.PLOT_GEOMETRY_AIRFOIL_EDGE_WIDTH

CHANNEL_COLOR = settings.PLOT_GEOMETRY_CHANNEL_COLOR
CHANNEL_WIDTH = settings.PLOT_GEOMETRY_CHANNEL_WIDTH
CHANNEL_ALPHA = settings.PLOT_GEOMETRY_CHANNEL_ALPHA

PORE_FACE_COLOR = settings.PLOT_GEOMETRY_PORE_FACE_COLOR
PORE_EDGE_COLOR = settings.PLOT_GEOMETRY_PORE_EDGE_COLOR
PORE_EDGE_WIDTH = settings.PLOT_GEOMETRY_PORE_EDGE_WIDTH

SHOW_PORE_CIRCLES = settings.PLOT_GEOMETRY_SHOW_PORE_CIRCLES
SHOW_PORE_CENTRES = settings.PLOT_GEOMETRY_SHOW_PORE_CENTRES
PORE_CENTRE_MARKER_SIZE = settings.PLOT_GEOMETRY_PORE_CENTRE_MARKER_SIZE

SHOW_CHANNEL_LABELS = settings.PLOT_GEOMETRY_SHOW_CHANNEL_LABELS
SHOW_TITLES = settings.PLOT_GEOMETRY_SHOW_TITLES


# =============================================================================
# STYLE
# =============================================================================
def apply_publication_style() -> None:
    """Use the central Matplotlib style configuration."""
    mpl.rcParams.update(settings.PLOT_GEOMETRY_RCPARAMS)


# =============================================================================
# PATH HELPERS
# =============================================================================
def windows_long_path(path: Path) -> str:
    """Return a Windows-safe path string for long nested output folders."""
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
    """Create a folder safely."""
    os.makedirs(windows_long_path(Path(path)), exist_ok=True)


def save_figure_all_formats(fig: mpl.figure.Figure, output_stem: Path) -> None:
    """Save figure as PNG, PDF and SVG."""
    output_stem = Path(output_stem)
    safe_mkdir(output_stem.parent)

    for ext in SAVE_FORMATS:
        path = output_stem.with_suffix(f".{ext}")
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}

        if ext == "png":
            kwargs["dpi"] = PNG_DPI

        fig.savefig(windows_long_path(path), **kwargs)
        print(f"[Figure saved] {path}")


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write list of dictionaries to CSV."""
    path = Path(path)
    safe_mkdir(path.parent)

    with open(windows_long_path(path), "w", encoding="utf-8", newline="") as f:
        pd.DataFrame(rows).to_csv(f, index=False)

    print(f"[CSV saved] {path}")


# =============================================================================
# MODEL HELPERS
# =============================================================================
def model_label(model_name: str) -> str:
    """Return full model label."""
    return MODEL_LABELS.get(model_name, model_name)


def model_short_label(model_name: str) -> str:
    """Return short model label."""
    return MODEL_SHORT_LABELS.get(model_name, model_name)


def model_folder_name(model_name: str) -> str:
    """Return folder-safe model name."""
    return MODEL_FOLDER_NAMES.get(model_name, model_name)


def select_models(all_models: list) -> list:
    """Select models 1 to 3 from all available porous models."""
    models_by_name = {model.name: model for model in all_models}

    missing = [name for name in MODELS_TO_EXPORT if name not in models_by_name]

    if missing:
        raise RuntimeError(
            f"Requested model(s) not found: {missing}. "
            f"Available models: {list(models_by_name)}"
        )

    return [models_by_name[name] for name in MODELS_TO_EXPORT]


def get_passage_diameter(passage) -> float:
    """Return representative pore/channel diameter."""
    chamber = getattr(passage, "chamber", None)

    objects = (
        chamber,
        getattr(passage, "pore1", None),
        getattr(passage, "pore2", None),
        passage,
    )

    for obj in objects:
        if obj is None:
            continue

        for attr in ("hydraulic_diameter", "diameter", "diameter_m", "Dh", "d"):
            value = getattr(obj, attr, None)

            if value is None:
                continue

            try:
                diameter = float(value)
            except (TypeError, ValueError):
                continue

            if np.isfinite(diameter) and diameter > 0.0:
                return diameter

    return float(EXPORT_PORE_DIAMETER_M)


# =============================================================================
# GEOMETRY EXTRACTION
# =============================================================================
def extract_model_geometry_rows(geom, model, network) -> list[dict]:
    """Extract pore and channel coordinates for one porous model."""
    chord = geom.chord
    rows: list[dict] = []

    for passage_id, passage in enumerate(network.passages):
        sp1 = passage.pore_surface_point(geom, passage.pore1)
        sp2 = passage.pore_surface_point(geom, passage.pore2)

        x1 = float(sp1.x)
        y1 = float(sp1.y)
        x2 = float(sp2.x)
        y2 = float(sp2.y)

        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))

        diameter = get_passage_diameter(passage)

        rows.append(
            {
                "model_name": model.name,
                "model_label": model_label(model.name),
                "passage_id": int(passage_id),
                "pore_diameter_m": diameter,
                "pore_diameter_c": diameter / chord,
                "x1_m": x1,
                "y1_m": y1,
                "x2_m": x2,
                "y2_m": y2,
                "x1_c": x1 / chord,
                "y1_c": y1 / chord,
                "x2_c": x2 / chord,
                "y2_c": y2 / chord,
                "channel_length_m": length,
                "channel_length_c": length / chord,
            }
        )

    return rows


# =============================================================================
# DRAWING
# =============================================================================
def draw_airfoil(ax: plt.Axes, geom) -> None:
    """Draw filled airfoil body."""
    chord = geom.chord

    ax.fill(
        geom.XB / chord,
        geom.YB / chord,
        facecolor=AIRFOIL_FACE_COLOR,
        edgecolor="none",
        linewidth=0.0,
        zorder=2,
    )

    ax.plot(
        geom.XB / chord,
        geom.YB / chord,
        color=AIRFOIL_EDGE_COLOR,
        linewidth=AIRFOIL_EDGE_WIDTH,
        solid_joinstyle="round",
        solid_capstyle="round",
        zorder=5,
    )


def draw_porous_network(ax: plt.Axes, geom, model, network) -> None:
    """Draw channels and pore mouths for one porous model."""
    chord = geom.chord

    for passage_id, passage in enumerate(network.passages):
        sp1 = passage.pore_surface_point(geom, passage.pore1)
        sp2 = passage.pore_surface_point(geom, passage.pore2)

        x1 = float(sp1.x) / chord
        y1 = float(sp1.y) / chord
        x2 = float(sp2.x) / chord
        y2 = float(sp2.y) / chord

        diameter = get_passage_diameter(passage)
        radius_c = 0.5 * diameter / chord

        ax.plot(
            [x1, x2],
            [y1, y2],
            "-",
            color=CHANNEL_COLOR,
            linewidth=CHANNEL_WIDTH,
            alpha=CHANNEL_ALPHA,
            solid_capstyle="round",
            zorder=7,
        )

        if SHOW_PORE_CIRCLES:
            for x, y in ((x1, y1), (x2, y2)):
                circle = Circle(
                    (x, y),
                    radius=radius_c,
                    facecolor=PORE_FACE_COLOR,
                    edgecolor=PORE_EDGE_COLOR,
                    linewidth=PORE_EDGE_WIDTH,
                    zorder=9,
                )
                ax.add_patch(circle)

        if SHOW_PORE_CENTRES:
            ax.plot(
                [x1, x2],
                [y1, y2],
                "o",
                color=PORE_EDGE_COLOR,
                markersize=PORE_CENTRE_MARKER_SIZE,
                markeredgewidth=0.0,
                zorder=10,
            )

        if SHOW_CHANNEL_LABELS:
            xm = 0.5 * (x1 + x2)
            ym = 0.5 * (y1 + y2)

            ax.text(
                xm,
                ym,
                str(passage_id + 1),
                ha="center",
                va="center",
                fontsize=6.5,
                color="black",
                zorder=12,
            )


def format_axis(ax: plt.Axes) -> None:
    """Format airfoil figure axis."""
    ax.set_xlim(*FIGURE_XLIM)
    ax.set_ylim(*FIGURE_YLIM)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x/c$")
    ax.set_ylabel(r"$y/c$")
    ax.tick_params(direction="in", top=True, right=True, length=3.5, width=0.75)
    ax.minorticks_on()


def plot_single_model(geom, model, network, output_dir: Path) -> None:
    """Export one figure for a single porous model."""
    fig, ax = plt.subplots(figsize=(7.0, 2.4), constrained_layout=True)

    draw_airfoil(ax, geom)
    draw_porous_network(ax, geom, model, network)

    if SHOW_TITLES:
        ax.set_title(model_label(model.name))

    format_axis(ax)

    save_figure_all_formats(
        fig=fig,
        output_stem=output_dir / model_folder_name(model.name) / "porous_airfoil_layout",
    )

    plt.close(fig)


def plot_combined_models(geom, model_network_pairs: list[tuple]) -> None:
    """Export one combined figure showing all three porous models."""
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.2, 6.5),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    for ax, (model, network) in zip(axes, model_network_pairs):
        draw_airfoil(ax, geom)
        draw_porous_network(ax, geom, model, network)

        ax.set_title(model_label(model.name))
        format_axis(ax)

    fig.suptitle(
        f"NACA {AIRFOIL_NAME}: 2D porous-airfoil model layouts",
        y=1.02,
    )

    save_figure_all_formats(
        fig=fig,
        output_stem=EXPORT_ROOT / "combined_models_1_to_3" / "porous_airfoil_models_1_to_3",
    )

    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
@timed_stage("geometry")
def main() -> None:
    if not (settings.ENABLE_PLOTS and settings.RUN_GEOMETRY):
        print("[Skip] geometry disabled in config.py.")
        return {"status": "skipped", "message": "disabled in config.py"}
    apply_publication_style()
    safe_mkdir(EXPORT_ROOT)

    print("=" * 80)
    print("EXPORTING POROUS AIRFOIL MODELS FOR RESULTS AND DISCUSSION")
    print("=" * 80)
    print(f"Airfoil              : NACA {AIRFOIL_NAME}")
    print(f"Chord [m]            : {CHORD:.6g}")
    print(f"Panels               : {N_PANELS}")
    print(f"Reynolds number      : {REYNOLDS_EXTERNAL:.6e}")
    print(f"AoA used for setup   : {EXPORT_AOA_DEG:.6g} deg")
    print(f"Pore diameter [m]    : {EXPORT_PORE_DIAMETER_M:.6e}")
    print(f"Pore diameter [mm]   : {1000.0 * EXPORT_PORE_DIAMETER_M:.3f}")
    print(f"Output folder        : {EXPORT_ROOT}")

    flow, geom, _, _aero_solver = build_base_problem(
        airfoil_name=AIRFOIL_NAME,
        chord=CHORD,
        n_panels=int(N_PANELS),
        aoa_deg=float(EXPORT_AOA_DEG),
        reynolds_external=REYNOLDS_EXTERNAL,
        rho_inf=RHO_INF,
        mu_inf=MU_INF,
        p_inf=P_INF,
    )

    all_models = build_model_specs(
        geom=geom,
        diameter_m=float(EXPORT_PORE_DIAMETER_M),
    )

    selected_models = select_models(all_models)

    model_network_pairs: list[tuple] = []
    all_rows: list[dict] = []

    for model in selected_models:
        print()
        print("-" * 80)
        print(f"Exporting {model_label(model.name)}")
        print("-" * 80)
        print(model.description)
        print(f"Representative pore diameter [m] : {representative_pore_diameter(model):.6e}")
        print(f"Independent passages             : {len(model.passages)}")

        network = build_network_from_model(
            model=model,
            span_width=SPAN_WIDTH,
        )

        model_network_pairs.append((model, network))

        model_dir = EXPORT_ROOT / model_folder_name(model.name)
        safe_mkdir(model_dir)

        rows = extract_model_geometry_rows(
            geom=geom,
            model=model,
            network=network,
        )

        all_rows.extend(rows)

        write_csv(
            path=model_dir / "pore_and_channel_coordinates.csv",
            rows=rows,
        )

        plot_single_model(
            geom=geom,
            model=model,
            network=network,
            output_dir=EXPORT_ROOT,
        )

    write_csv(
        path=EXPORT_ROOT / "all_models_pore_and_channel_coordinates.csv",
        rows=all_rows,
    )

    plot_combined_models(
        geom=geom,
        model_network_pairs=model_network_pairs,
    )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()