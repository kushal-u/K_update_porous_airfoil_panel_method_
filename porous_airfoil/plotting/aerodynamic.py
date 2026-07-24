"""Plotting API used by the fixed-case and AoA-sweep workflows.

The original project imported these functions from a top-level ``plotter.py``.
They are kept behind the same function signatures so the numerical workflow is
unchanged while the code now lives inside the package.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import safe_mkdir, windows_long_path


def _prepare_output_dir(output_dir: str | Path) -> Path:
    return safe_mkdir(output_dir)


def _save_figure(fig, save_path: str | Path, **kwargs) -> None:
    save_path = Path(save_path)
    safe_mkdir(save_path.parent)
    fig.savefig(windows_long_path(save_path), **kwargs)


def _sorted_side_ids(geom, side: str) -> np.ndarray:
    ids = geom.upper_panel_ids if side == "upper" else geom.lower_panel_ids
    ids = np.asarray(ids, dtype=int)
    return ids[np.argsort(geom.XC[ids])]


def plot_airfoil_with_porous_network(
    geom,
    network,
    coupled_result,
    output_dir: str | Path,
    title: str = "Airfoil with Porous Network",
) -> None:
    out = _prepare_output_dir(output_dir)
    save_path = out / "airfoil_with_porous_network.png"

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill(
        geom.XB / geom.chord,
        geom.YB / geom.chord,
        color="lightgray",
        alpha=0.35,
        label="Airfoil interior",
    )
    ax.plot(
        geom.XB / geom.chord,
        geom.YB / geom.chord,
        "k-",
        lw=1.4,
        label="Airfoil surface",
    )

    for index, passage in enumerate(network.passages):
        sp1 = passage.pore_surface_point(geom, passage.pore1)
        sp2 = passage.pore_surface_point(geom, passage.pore2)
        x1, y1 = sp1.x / geom.chord, sp1.y / geom.chord
        x2, y2 = sp2.x / geom.chord, sp2.y / geom.chord
        state = coupled_result.passage_states[index]
        q_value = float(state.Q)

        line = ax.plot([x1, x2], [y1, y2], "-", lw=1.4, alpha=0.95, label=passage.name)[0]
        color = line.get_color()
        ax.plot(x1, y1, "o", ms=5, color=color)
        ax.plot(x2, y2, "s", ms=5, color=color)

        if abs(q_value) > 1.0e-30:
            if q_value >= 0.0:
                xa, ya, xb, yb = x1, y1, x2, y2
            else:
                xa, ya, xb, yb = x2, y2, x1, y1
            dx, dy = xb - xa, yb - ya
            ax.annotate(
                "",
                xy=(xa + 0.62 * dx, ya + 0.62 * dy),
                xytext=(xa + 0.38 * dx, ya + 0.38 * dy),
                arrowprops={"arrowstyle": "-|>", "lw": 0.8, "color": color},
            )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    if len(network.passages) <= 18:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6)

    fig.tight_layout()
    _save_figure(fig, save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot Saved] {save_path}")


def plot_cp_distribution_with_xfoil(
    geom,
    porous_result,
    solid_result,
    output_dir: str | Path,
    xfoil_cp_data: dict | None = None,
) -> None:
    out = _prepare_output_dir(output_dir)
    save_path = out / "cp_distribution_with_xfoil.png"
    upper = _sorted_side_ids(geom, "upper")
    lower = _sorted_side_ids(geom, "lower")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(geom.XC[upper] / geom.chord, porous_result.Cp[upper], "ro-", ms=3, lw=1.0, label="Porous panel upper")
    ax.plot(geom.XC[lower] / geom.chord, porous_result.Cp[lower], "bo-", ms=3, lw=1.0, label="Porous panel lower")
    ax.plot(geom.XC[upper] / geom.chord, solid_result.Cp[upper], "k--", lw=1.3, label="Solid panel upper")
    ax.plot(geom.XC[lower] / geom.chord, solid_result.Cp[lower], "g--", lw=1.3, label="Solid panel lower")

    if xfoil_cp_data is not None:
        if "x_upper" in xfoil_cp_data and "cp_upper" in xfoil_cp_data:
            ax.plot(xfoil_cp_data["x_upper"], xfoil_cp_data["cp_upper"], "m-.", lw=1.2, label="XFOIL upper")
        if "x_lower" in xfoil_cp_data and "cp_lower" in xfoil_cp_data:
            ax.plot(xfoil_cp_data["x_lower"], xfoil_cp_data["cp_lower"], "c-.", lw=1.2, label="XFOIL lower")

    ax.invert_yaxis()
    ax.set_xlabel("x/c")
    ax.set_ylabel("Cp")
    ax.set_title("Cp Distribution: Porous Panel vs Solid Panel vs XFOIL")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_figure(fig, save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot Saved] {save_path}")


def plot_aoa_sweep_comparison(sweep_df, output_dir: str | Path) -> None:
    out = _prepare_output_dir(output_dir)
    has_diameter = "pore_diameter_m" in sweep_df.columns
    save_path = out / ("aoa_sweep_diameter_CL_CD_CM.png" if has_diameter else "aoa_sweep_CL_CD_CM.png")

    required = {
        "alpha_deg", "porous_CL", "porous_CD", "porous_CM",
        "solid_panel_CL", "solid_panel_CD", "solid_panel_CM",
    }
    missing = required - set(sweep_df.columns)
    if missing:
        raise ValueError(f"sweep_df missing columns: {sorted(missing)}")

    has_xfoil = all(column in sweep_df.columns for column in ("xfoil_CL", "xfoil_CD", "xfoil_CM"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    if has_diameter:
        diameters = sorted(float(value) for value in sweep_df["pore_diameter_m"].dropna().unique())
        solid = sweep_df[["alpha_deg", "solid_panel_CL", "solid_panel_CD", "solid_panel_CM"]].drop_duplicates("alpha_deg").sort_values("alpha_deg")
        xfoil = (
            sweep_df[["alpha_deg", "xfoil_CL", "xfoil_CD", "xfoil_CM"]].drop_duplicates("alpha_deg").sort_values("alpha_deg")
            if has_xfoil else None
        )
        for ax, key in zip(axes, ("CL", "CD", "CM")):
            for diameter_m in diameters:
                curve = sweep_df[np.isclose(sweep_df["pore_diameter_m"], diameter_m)].sort_values("alpha_deg")
                ax.plot(curve["alpha_deg"], curve[f"porous_{key}"], "o-", lw=1.4, ms=3.5, label=f"Porous D={1000.0 * diameter_m:.1f} mm")
            ax.plot(solid["alpha_deg"], solid[f"solid_panel_{key}"], "k--", lw=1.5, label="Solid panel")
            if xfoil is not None:
                ax.plot(xfoil["alpha_deg"], xfoil[f"xfoil_{key}"], "d-.", lw=1.3, ms=3.5, label="XFOIL")
            ax.set_xlabel("AoA [deg]")
            ax.set_ylabel(key)
            ax.set_title(f"{key} vs AoA")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
    else:
        alpha = sweep_df["alpha_deg"].to_numpy()
        for ax, key in zip(axes, ("CL", "CD", "CM")):
            ax.plot(alpha, sweep_df[f"porous_{key}"], "o-", lw=1.5, ms=4, label="Porous panel")
            ax.plot(alpha, sweep_df[f"solid_panel_{key}"], "s--", lw=1.5, ms=4, label="Solid panel")
            if has_xfoil:
                ax.plot(alpha, sweep_df[f"xfoil_{key}"], "d-.", lw=1.5, ms=4, label="XFOIL")
            ax.set_xlabel("AoA [deg]")
            ax.set_ylabel(key)
            ax.set_title(f"{key} vs AoA")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

    fig.tight_layout()
    _save_figure(fig, save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot Saved] {save_path}")


def _build_field_grid(solver, xlim, ylim, nx, ny):
    chord = solver.geom.chord
    x = np.linspace(xlim[0] * chord, xlim[1] * chord, int(nx))
    y = np.linspace(ylim[0] * chord, ylim[1] * chord, int(ny))
    xx, yy = np.meshgrid(x, y)
    return xx, yy, xx / chord, yy / chord


def _velocity_magnitude_for_display(solver, result, xx, yy):
    inside = solver.build_inside_mask(xx, yy)
    blank = np.zeros_like(xx, dtype=bool)
    u, v, _, _ = solver.velocity_field(xx, yy, result, inside=blank)
    magnitude = np.sqrt(u**2 + v**2)
    magnitude[inside] = np.nan
    return magnitude, u, v, inside


def _pressure_for_display(solver, result, xx, yy):
    inside = solver.build_inside_mask(xx, yy)
    blank = np.zeros_like(xx, dtype=bool)
    u, v, _, _ = solver.velocity_field(xx, yy, result, inside=blank)
    pressure = solver.flow.p_inf + 0.5 * solver.flow.rho_inf * (solver.flow.v_inf**2 - (u**2 + v**2))
    pressure[inside] = np.nan
    return pressure, u, v, inside


def _format_contour_axis(ax, geom, xlim, ylim, title):
    chord = geom.chord
    ax.fill(geom.XB / chord, geom.YB / chord, "k", zorder=3)
    ax.set_title(title)
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")


def plot_velocity_contours_comparison(
    porous_solver,
    porous_result,
    solid_solver,
    solid_result,
    output_dir: str | Path,
    xlim: tuple[float, float] = (-0.5, 1.5),
    ylim: tuple[float, float] = (-0.5, 0.5),
    nx: int = 1000,
    ny: int = 1000,
    save_filename: str = "velocity_contours_comparison.png",
    view_label: str = "",
) -> None:
    out = _prepare_output_dir(output_dir)
    save_path = out / save_filename
    label = f" — {view_label}" if view_label else ""
    xx, yy, xxn, yyn = _build_field_grid(porous_solver, xlim, ylim, nx, ny)
    porous_mag, _, _, _ = _velocity_magnitude_for_display(porous_solver, porous_result, xx, yy)
    solid_mag, _, _, _ = _velocity_magnitude_for_display(solid_solver, solid_result, xx, yy)
    finite = np.concatenate([porous_mag[np.isfinite(porous_mag)], solid_mag[np.isfinite(solid_mag)]])
    vmin, vmax = float(np.min(finite)), float(np.max(finite))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
    axes[0].contourf(xxn, yyn, porous_mag, levels=40, vmin=vmin, vmax=vmax)
    image = axes[1].contourf(xxn, yyn, solid_mag, levels=40, vmin=vmin, vmax=vmax)
    _format_contour_axis(axes[0], porous_solver.geom, xlim, ylim, f"Velocity Magnitude: Porous{label}")
    _format_contour_axis(axes[1], solid_solver.geom, xlim, ylim, f"Velocity Magnitude: Solid{label}")
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.95)
    colorbar.set_label("Velocity magnitude [m/s]")
    _save_figure(fig, save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot Saved] {save_path}")


def plot_pressure_contours_comparison(
    porous_solver,
    porous_result,
    solid_solver,
    solid_result,
    output_dir: str | Path,
    xlim: tuple[float, float] = (-0.2, 1.2),
    ylim: tuple[float, float] = (-0.35, 0.35),
    nx: int = 1000,
    ny: int = 1000,
    save_filename: str = "pressure_contours_comparison.png",
    view_label: str = "",
) -> None:
    out = _prepare_output_dir(output_dir)
    save_path = out / save_filename
    label = f" — {view_label}" if view_label else ""
    xx, yy, xxn, yyn = _build_field_grid(porous_solver, xlim, ylim, nx, ny)
    porous_pressure, _, _, _ = _pressure_for_display(porous_solver, porous_result, xx, yy)
    solid_pressure, _, _, _ = _pressure_for_display(solid_solver, solid_result, xx, yy)
    finite = np.concatenate([porous_pressure[np.isfinite(porous_pressure)], solid_pressure[np.isfinite(solid_pressure)]])
    vmin, vmax = float(np.min(finite)), float(np.max(finite))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    image = axes[0].contourf(xxn, yyn, porous_pressure, levels=40, vmin=vmin, vmax=vmax)
    axes[1].contourf(xxn, yyn, solid_pressure, levels=40, vmin=vmin, vmax=vmax)
    _format_contour_axis(axes[0], porous_solver.geom, xlim, ylim, f"Static Pressure: Porous{label}")
    _format_contour_axis(axes[1], solid_solver.geom, xlim, ylim, f"Static Pressure: Solid{label}")
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.95)
    colorbar.set_label("Pressure [Pa]")
    _save_figure(fig, save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot Saved] {save_path}")


# =============================================================================
# JOURNAL CL/CM POST-PROCESSING
# =============================================================================
from pathlib import Path
from matplotlib.lines import Line2D
import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# USER SETTINGS
# =============================================================================
# Set an exact sweep CSV here when possible. When None, the script searches
# recursively inside SEARCH_ROOT.
INPUT_CSV_PATH: Path | None = None

try:
    from ..config import OUTPUT_ROOT

    SEARCH_ROOT = Path(OUTPUT_ROOT)
except Exception:
    SEARCH_ROOT = Path(__file__).resolve().parents[2] / "outputs"

OUTPUT_DIR = SEARCH_ROOT / "aero_graphs_three_diameters"

# Three diameters compared in every model panel.
SELECT_PORE_DIAMETERS_M = (
    0.0080,
    0.0090,
    0.0100,
)

# Retain the original validation/reference curves.
SHOW_SOLID_REFERENCE = True
SHOW_XFOIL_REFERENCE = True

SAVE_FORMATS = ("png", "pdf", "svg")
PNG_DPI = 1000

# False gives colour curves that are also differentiated by line type.
# True makes all curves black while preserving line-style differences.
MONOCHROME_MODE = False

# Three panels in one journal-width row.
THREE_PANEL_FIGSIZE = (7.5, 3.15)

# Use a common y-axis across the three models for direct comparison.
SHARE_Y_AXIS = True


# =============================================================================
# MODEL LABELS AND PANEL ORDER
# =============================================================================
CASE_LABELS = {
    "model_1_4_chordwise": "Horizontal",
    "model_2_9_perpendicular": "Vertical",
    "model_2_4_perpendicular": "Vertical",
    "model_3_combined_independent": "Horizontal + Vertical",
    "horizontal": "Horizontal",
    "vertical": "Vertical",
    "isotropic": "Horizontal + Vertical",
    "horizontal + vertical": "Horizontal + Vertical",
    "solid": "Solid panel",
    "solid_panel": "Solid panel",
    "solid panel": "Solid panel",
    "xfoil": "XFOIL",
}

MODEL_PANELS = (
    ("Horizontal", "(a) Horizontal"),
    ("Vertical", "(b) Vertical"),
    ("Horizontal + Vertical", "(c) Horizontal + Vertical"),
)


# =============================================================================
# CURVE STYLES
# =============================================================================
# Diameter is encoded by line type only. Colour is an additional aid.
DIAMETER_LINESTYLES = ("-", "--", "-.")
DIAMETER_COLOURS = ("#0072B2", "#D55E00", "#009E73")

REFERENCE_STYLES = {
    "Solid panel": {
        "color": "black",
        "linestyle": ":",
        "marker": "",
        "linewidth": 1.65,
        "markersize": 0.0,
        "zorder": 3,
    },
    "XFOIL": {
        "color": "#666666",
        "linestyle": (0, (5, 2)),
        "marker": "",
        "linewidth": 1.45,
        "markersize": 0.0,
        "zorder": 2,
    },
}


# =============================================================================
# JOURNAL STYLE
# =============================================================================

from pathlib import Path
from matplotlib import font_manager

WINDOWS_FONT_DIR = Path("/mnt/c/Windows/Fonts")

arial_files = [
    WINDOWS_FONT_DIR / "arial.ttf",
    WINDOWS_FONT_DIR / "arialbd.ttf",
    WINDOWS_FONT_DIR / "ariali.ttf",
    WINDOWS_FONT_DIR / "arialbi.ttf",
]

for font_path in arial_files:
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))

def apply_journal_style() -> None:
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
            "axes.labelsize": 9.8,
            "axes.titlesize": 10.0,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def windows_long_path(path: Path) -> str:
    """Return a Windows-safe path string for long OneDrive paths."""
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


def save_figure_all_formats(fig: mpl.figure.Figure, output_stem: Path) -> None:
    """Save one figure as PNG, PDF and SVG."""
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    for ext in SAVE_FORMATS:
        path = output_stem.with_suffix(f".{ext}")
        kwargs = {
            "bbox_inches": "tight",
            "pad_inches": 0.035,
        }

        if ext == "png":
            kwargs["dpi"] = PNG_DPI

        fig.savefig(windows_long_path(path), **kwargs)
        print(f"[Saved] {path}")


# =============================================================================
# NAME AND DIAMETER HELPERS
# =============================================================================
def standardise_case_name(raw_name: str) -> str:
    """Convert raw model names to publication labels."""
    name = str(raw_name).strip()
    lower = name.lower()

    # Check longer/more specific names first.
    if "model_3" in lower or "isotropic" in lower or "horizontal + vertical" in lower:
        return "Horizontal + Vertical"

    if "model_1" in lower or "horizontal" in lower:
        return "Horizontal"

    if "model_2" in lower or "vertical" in lower:
        return "Vertical"

    if "solid_panel" in lower or "solid panel" in lower or lower == "solid":
        return "Solid panel"

    if "xfoil" in lower:
        return "XFOIL"

    if name in CASE_LABELS:
        return CASE_LABELS[name]

    return name


def infer_case_from_path(path: Path) -> str:
    """Infer the porous model from a file or folder path."""
    return standardise_case_name(str(path))


def infer_diameter_from_text(text: str) -> float:
    """Infer a pore diameter in metres from text such as D=9 mm or D_9p00mm."""
    normalised = str(text).replace("p", ".")

    patterns = (
        r"(?:^|[_\-\s])D(?:=|[_\-\s])?([0-9]+(?:\.[0-9]+)?)\s*mm",
        r"([0-9]+(?:\.[0-9]+)?)\s*mm",
    )

    for pattern in patterns:
        match = re.search(pattern, normalised, flags=re.IGNORECASE)
        if match:
            return float(match.group(1)) / 1000.0

    return float("nan")


def diameter_is_selected(diameter_m: float) -> bool:
    """Return True when a diameter belongs to the requested comparison set."""
    if not np.isfinite(diameter_m):
        return False

    return any(
        np.isclose(
            float(diameter_m),
            float(selected),
            rtol=0.0,
            atol=1.0e-10,
        )
        for selected in SELECT_PORE_DIAMETERS_M
    )


def diameter_label(diameter_m: float) -> str:
    """Return a compact legend label in millimetres."""
    diameter_mm = 1000.0 * float(diameter_m)

    if np.isclose(diameter_mm, round(diameter_mm), atol=1.0e-10):
        return rf"$D={int(round(diameter_mm))}$ mm"

    return rf"$D={diameter_mm:.1f}$ mm"


def diameter_style(diameter_m: float) -> dict:
    """Return the line style assigned to one selected diameter."""
    index = None

    for i, selected in enumerate(SELECT_PORE_DIAMETERS_M):
        if np.isclose(diameter_m, selected, rtol=0.0, atol=1.0e-10):
            index = i
            break

    if index is None:
        index = 0

    color = "black" if MONOCHROME_MODE else DIAMETER_COLOURS[index % len(DIAMETER_COLOURS)]

    return {
        "color": color,
        "linestyle": DIAMETER_LINESTYLES[index % len(DIAMETER_LINESTYLES)],
        "marker": "",
        "linewidth": 1.85,
        "markersize": 0.0,
        "zorder": 6 + index,
    }


# =============================================================================
# CSV DISCOVERY AND COLUMN HELPERS
# =============================================================================
def find_candidate_csv_files() -> list[Path]:
    """Find likely AoA-sweep CSV files."""
    if INPUT_CSV_PATH is not None:
        path = Path(INPUT_CSV_PATH)
        if not path.exists():
            raise FileNotFoundError(f"Input CSV does not exist: {path}")
        return [path]

    patterns = (
        "**/*aoa*sweep*.csv",
        "**/*AoA*sweep*.csv",
        "**/*sweep*.csv",
        "**/*aerodynamic*metrics*.csv",
        "**/*all*summary*.csv",
    )

    found: list[Path] = []
    seen: set[str] = set()

    for pattern in patterns:
        for path in Path(SEARCH_ROOT).glob(pattern):
            try:
                resolved = str(path.resolve())
            except OSError:
                resolved = str(path)

            # Do not read this script's own exported plotting-data CSV.
            if "aero_graphs_three_diameters" in resolved:
                continue

            if resolved not in seen:
                seen.add(resolved)
                found.append(path)

    return sorted(found)


def first_existing_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Return the first matching column name."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def extract_diameter_series(path: Path, df: pd.DataFrame) -> pd.Series:
    """Return pore diameter in metres for every row."""
    diameter_m_col = first_existing_column(
        df,
        (
            "pore_diameter_m",
            "diameter_m",
            "pore_diameter",
        ),
    )

    if diameter_m_col is not None:
        return pd.to_numeric(df[diameter_m_col], errors="coerce")

    diameter_mm_col = first_existing_column(
        df,
        (
            "pore_diameter_mm",
            "diameter_mm",
        ),
    )

    if diameter_mm_col is not None:
        return pd.to_numeric(df[diameter_mm_col], errors="coerce") / 1000.0

    inferred = infer_diameter_from_text(str(path))
    return pd.Series([inferred] * len(df), index=df.index, dtype=float)


# =============================================================================
# CSV CONVERSION
# =============================================================================
def load_wide_sweep_csv(path: Path, df: pd.DataFrame) -> pd.DataFrame:
    """Convert a wide porous-sweep CSV to a common long plotting format."""
    alpha_col = first_existing_column(
        df,
        ("alpha_deg", "aoa_deg", "AoA", "alpha"),
    )

    if alpha_col is None or "porous_CL" not in df.columns:
        return pd.DataFrame()

    model_col = first_existing_column(
        df,
        ("model_label", "model_name", "case", "label"),
    )

    if model_col is not None:
        model_series = df[model_col].astype(str).map(standardise_case_name)
    else:
        inferred_case = infer_case_from_path(path)
        model_series = pd.Series([inferred_case] * len(df), index=df.index)

    diameter_series = extract_diameter_series(path, df)
    rows: list[dict] = []

    for index, row in df.iterrows():
        alpha = pd.to_numeric(row.get(alpha_col), errors="coerce")
        diameter_m = pd.to_numeric(diameter_series.loc[index], errors="coerce")
        model = model_series.loc[index]

        if not np.isfinite(alpha):
            continue

        porous_cl = pd.to_numeric(row.get("porous_CL"), errors="coerce")
        porous_cm = pd.to_numeric(row.get("porous_CM", np.nan), errors="coerce")

        if diameter_is_selected(float(diameter_m)) and np.isfinite(porous_cl):
            rows.append(
                {
                    "alpha_deg": float(alpha),
                    "base_case": model,
                    "pore_diameter_m": float(diameter_m),
                    "CL": float(porous_cl),
                    "CM": float(porous_cm) if np.isfinite(porous_cm) else np.nan,
                    "source_file": str(path),
                }
            )

        solid_cl = pd.to_numeric(row.get("solid_panel_CL", np.nan), errors="coerce")
        solid_cm = pd.to_numeric(row.get("solid_panel_CM", np.nan), errors="coerce")

        if np.isfinite(solid_cl):
            rows.append(
                {
                    "alpha_deg": float(alpha),
                    "base_case": "Solid panel",
                    "pore_diameter_m": np.nan,
                    "CL": float(solid_cl),
                    "CM": float(solid_cm) if np.isfinite(solid_cm) else np.nan,
                    "source_file": str(path),
                }
            )

        xfoil_cl = pd.to_numeric(row.get("xfoil_CL", np.nan), errors="coerce")
        xfoil_cm = pd.to_numeric(row.get("xfoil_CM", np.nan), errors="coerce")

        if np.isfinite(xfoil_cl):
            rows.append(
                {
                    "alpha_deg": float(alpha),
                    "base_case": "XFOIL",
                    "pore_diameter_m": np.nan,
                    "CL": float(xfoil_cl),
                    "CM": float(xfoil_cm) if np.isfinite(xfoil_cm) else np.nan,
                    "source_file": str(path),
                }
            )

    return pd.DataFrame(rows)


def load_long_sweep_csv(path: Path, df: pd.DataFrame) -> pd.DataFrame:
    """Convert an already-long sweep CSV to the common plotting format."""
    alpha_col = first_existing_column(
        df,
        ("alpha_deg", "aoa_deg", "AoA", "alpha"),
    )
    case_col = first_existing_column(
        df,
        ("case", "model_label", "model_name", "label"),
    )
    cl_col = first_existing_column(
        df,
        ("CL", "C_L", "cl", "c_l"),
    )
    cm_col = first_existing_column(
        df,
        ("CM", "C_M", "cm", "c_m"),
    )

    if alpha_col is None or case_col is None or cl_col is None:
        return pd.DataFrame()

    diameter_series = extract_diameter_series(path, df)
    raw_cases = df[case_col].astype(str)

    out = pd.DataFrame(index=df.index)
    out["alpha_deg"] = pd.to_numeric(df[alpha_col], errors="coerce")
    out["base_case"] = raw_cases.map(standardise_case_name)
    out["pore_diameter_m"] = pd.to_numeric(diameter_series, errors="coerce")
    out["CL"] = pd.to_numeric(df[cl_col], errors="coerce")
    out["CM"] = (
        pd.to_numeric(df[cm_col], errors="coerce")
        if cm_col is not None
        else np.nan
    )
    out["source_file"] = str(path)

    # Try to recover diameter values embedded in case labels.
    missing_diameter = ~np.isfinite(out["pore_diameter_m"])
    if missing_diameter.any():
        recovered = raw_cases.map(infer_diameter_from_text)
        out.loc[missing_diameter, "pore_diameter_m"] = recovered.loc[missing_diameter]

    reference_mask = out["base_case"].isin(["Solid panel", "XFOIL"])
    selected_porous_mask = out["pore_diameter_m"].map(diameter_is_selected)

    out = out.loc[reference_mask | selected_porous_mask].copy()
    out.loc[reference_mask, "pore_diameter_m"] = np.nan

    return out.reset_index(drop=True)


def load_all_sweep_data() -> pd.DataFrame:
    """Load and merge all usable AoA sweep CSV files."""
    csv_files = find_candidate_csv_files()

    if not csv_files:
        raise FileNotFoundError(
            f"No AoA sweep CSV files found in: {SEARCH_ROOT}\n"
            "Set INPUT_CSV_PATH to the required sweep CSV."
        )

    loaded: list[pd.DataFrame] = []

    for path in csv_files:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"[Skip] Could not read {path}: {exc}")
            continue

        if df.empty:
            continue

        try:
            if "porous_CL" in df.columns:
                long_df = load_wide_sweep_csv(path, df)
            else:
                long_df = load_long_sweep_csv(path, df)
        except Exception as exc:
            print(f"[Skip] Could not interpret {path}: {exc}")
            continue

        if not long_df.empty:
            loaded.append(long_df)
            print(f"[Loaded] {path}")

    if not loaded:
        raise RuntimeError(
            "CSV files were found, but none contained usable CL/AoA data.\n"
            "Expected either wide columns such as alpha_deg, porous_CL, "
            "porous_CM and pore_diameter_m, or an equivalent long format."
        )

    data = pd.concat(loaded, ignore_index=True)
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["alpha_deg", "base_case", "CL"])

    # Remove exact duplicated exports. Curve-level averaging is performed only
    # during plotting to avoid repeated baseline values from multiple model rows.
    data = data.drop_duplicates(
        subset=[
            "alpha_deg",
            "base_case",
            "pore_diameter_m",
            "CL",
            "CM",
        ],
        keep="first",
    )

    return data.reset_index(drop=True)


# =============================================================================
# PLOTTING HELPERS
# =============================================================================
def average_curve_by_alpha(df_curve: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return one ordered value per AoA for a curve."""
    curve = df_curve.dropna(subset=["alpha_deg", metric]).copy()

    if curve.empty:
        return curve

    return (
        curve.groupby("alpha_deg", as_index=False)[metric]
        .mean()
        .sort_values("alpha_deg")
    )


def plot_line(
    ax: plt.Axes,
    curve: pd.DataFrame,
    metric: str,
    label: str,
    style: dict,
) -> None:
    """Plot one formatted aerodynamic curve."""
    if curve.empty:
        return

    marker = style.get("marker", "")

    ax.plot(
        curve["alpha_deg"],
        curve[metric],
        label=label,
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=style["linewidth"],
        marker=marker,
        markersize=style["markersize"],
        markerfacecolor=None,
        markeredgecolor=None,
        markeredgewidth=0.0,
        zorder=style["zorder"],
    )


def format_panel_axis(
    ax: plt.Axes,
    title: str,
    ylabel: str,
    show_ylabel: bool,
) -> None:
    """Apply common formatting to one model panel."""
    ax.set_title(title, pad=5.0, fontweight="bold")
    ax.set_xlabel(r"Angle of attack, $\alpha$ [deg]")
    ax.set_ylabel(ylabel if show_ylabel else "")

    ax.grid(False)
    ax.minorticks_on()

    ax.tick_params(
        which="major",
        direction="in",
        top=True,
        right=True,
        length=3.8,
        width=0.78,
    )
    ax.tick_params(
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=2.0,
        width=0.58,
    )

    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True, nbins=6))

    for spine in ax.spines.values():
        spine.set_linewidth(0.85)


def plot_reference_curves(
    ax: plt.Axes,
    data: pd.DataFrame,
    metric: str,
) -> None:
    """Plot solid-panel and XFOIL reference curves when requested."""
    requested_references = []

    if SHOW_SOLID_REFERENCE:
        requested_references.append("Solid panel")

    if SHOW_XFOIL_REFERENCE:
        requested_references.append("XFOIL")

    for reference in requested_references:
        curve = average_curve_by_alpha(
            data[data["base_case"] == reference],
            metric,
        )

        if curve.empty:
            continue

        style = REFERENCE_STYLES[reference].copy()
        if MONOCHROME_MODE:
            style["color"] = "black"

        plot_line(
            ax=ax,
            curve=curve,
            metric=metric,
            label=reference,
            style=style,
        )


def plot_model_diameters(
    ax: plt.Axes,
    data: pd.DataFrame,
    model_name: str,
    metric: str,
) -> None:
    """Plot all selected pore diameters for one porous model."""
    model_data = data[data["base_case"] == model_name]

    for diameter_m in SELECT_PORE_DIAMETERS_M:
        mask = np.isclose(
            model_data["pore_diameter_m"].to_numpy(dtype=float),
            float(diameter_m),
            rtol=0.0,
            atol=1.0e-10,
            equal_nan=False,
        )

        curve = average_curve_by_alpha(
            model_data.loc[mask],
            metric,
        )

        if curve.empty:
            print(
                f"[Warning] Missing {metric} curve: "
                f"{model_name}, D={1000.0 * diameter_m:.1f} mm"
            )
            continue

        plot_line(
            ax=ax,
            curve=curve,
            metric=metric,
            label=diameter_label(diameter_m),
            style=diameter_style(diameter_m),
        )


def build_shared_legend_handles(data: pd.DataFrame, metric: str) -> list[Line2D]:
    """Build a fixed, consistent legend for all three panels."""
    handles: list[Line2D] = []

    for diameter_m in SELECT_PORE_DIAMETERS_M:
        style = diameter_style(diameter_m)
        handles.append(
            Line2D(
                [0],
                [0],
                label=diameter_label(diameter_m),
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                marker="",
                markersize=0.0,
            )
        )

    reference_names = []
    if SHOW_SOLID_REFERENCE:
        reference_names.append("Solid panel")
    if SHOW_XFOIL_REFERENCE:
        reference_names.append("XFOIL")

    for reference in reference_names:
        reference_data = data[
            (data["base_case"] == reference)
            & data[metric].notna()
        ]

        if reference_data.empty:
            continue

        style = REFERENCE_STYLES[reference].copy()
        if MONOCHROME_MODE:
            style["color"] = "black"

        handles.append(
            Line2D(
                [0],
                [0],
                label=reference,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
            )
        )

    return handles


# =============================================================================
# THREE-PANEL FIGURES
# =============================================================================
def plot_three_panel_metric(
    data: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_filename: str,
) -> None:
    """Create one 1 x 3 model-comparison figure for CL or CM."""
    if metric not in data.columns or not data[metric].notna().any():
        print(f"[Skip] No usable {metric} data were found.")
        return

    fig, axes = plt.subplots(
        1,
        3,
        figsize=THREE_PANEL_FIGSIZE,
        sharex=True,
        sharey=SHARE_Y_AXIS,
    )

    for panel_index, (ax, (model_name, panel_title)) in enumerate(
        zip(axes, MODEL_PANELS)
    ):
        plot_reference_curves(
            ax=ax,
            data=data,
            metric=metric,
        )

        plot_model_diameters(
            ax=ax,
            data=data,
            model_name=model_name,
            metric=metric,
        )

        format_panel_axis(
            ax=ax,
            title=panel_title,
            ylabel=ylabel,
            show_ylabel=(panel_index == 0),
        )

    legend_handles = build_shared_legend_handles(data, metric)

    if legend_handles:
        legend = fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=len(legend_handles),
            frameon=True,
            framealpha=0.97,
            edgecolor="black",
            fancybox=False,
            handlelength=2.7,
            columnspacing=1.05,
            borderpad=0.40,
            labelspacing=0.30,
        )
        legend.get_frame().set_linewidth(0.6)

    fig.subplots_adjust(
        left=0.085,
        right=0.99,
        bottom=0.18,
        top=0.80,
        wspace=0.10,
    )

    save_figure_all_formats(
        fig=fig,
        output_stem=OUTPUT_DIR / output_filename,
    )

    plt.close(fig)


def plot_cl_and_cm_figures(data: pd.DataFrame) -> None:
    """Export the requested three-panel CL and CM figures."""
    plot_three_panel_metric(
        data=data,
        metric="CL",
        ylabel=r"Lift coefficient, $C_L$",
        output_filename="CL_vs_AoA_three_models_three_diameters",
    )

    plot_three_panel_metric(
        data=data,
        metric="CM",
        ylabel=r"Pitching moment coefficient, $C_M$",
        output_filename="CM_vs_AoA_three_models_three_diameters",
    )


# =============================================================================
# REPORTING
# =============================================================================
def print_curve_summary(data: pd.DataFrame) -> None:
    """Print the model/diameter curves found in the loaded data."""
    print()
    print("Curves requested:")

    for model_name, _ in MODEL_PANELS:
        print(f"  {model_name}")

        model_data = data[data["base_case"] == model_name]

        for diameter_m in SELECT_PORE_DIAMETERS_M:
            mask = np.isclose(
                model_data["pore_diameter_m"].to_numpy(dtype=float),
                diameter_m,
                rtol=0.0,
                atol=1.0e-10,
                equal_nan=False,
            )
            n_points = int(model_data.loc[mask, "alpha_deg"].nunique())
            print(
                f"    D={1000.0 * diameter_m:.1f} mm: "
                f"{n_points} AoA points"
            )

    for reference in ("Solid panel", "XFOIL"):
        n_points = int(
            data.loc[data["base_case"] == reference, "alpha_deg"].nunique()
        )
        print(f"  {reference}: {n_points} AoA points")


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    apply_journal_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CREATING THREE-PANEL CL AND CM FIGURES")
    print("=" * 80)
    print(f"Search root              : {SEARCH_ROOT}")
    print(f"Output folder            : {OUTPUT_DIR}")
    print(
        "Selected diameters [mm] : "
        + ", ".join(
            f"{1000.0 * diameter_m:.1f}"
            for diameter_m in SELECT_PORE_DIAMETERS_M
        )
    )
    print(f"Show solid reference     : {SHOW_SOLID_REFERENCE}")
    print(f"Show XFOIL reference     : {SHOW_XFOIL_REFERENCE}")
    print(f"Monochrome mode          : {MONOCHROME_MODE}")

    data = load_all_sweep_data()

    if data.empty:
        raise RuntimeError("No aerodynamic data are available for plotting.")

    plotting_csv = OUTPUT_DIR / "three_diameter_plotting_data.csv"
    data.to_csv(plotting_csv, index=False)
    print(f"[Saved] {plotting_csv}")

    print_curve_summary(data)
    plot_cl_and_cm_figures(data)

    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
