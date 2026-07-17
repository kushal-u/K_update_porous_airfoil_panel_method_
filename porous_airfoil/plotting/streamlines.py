"""Standalone Python streamline figures for porous-airfoil models.

This script solves one AoA and pore diameter, then exports Python streamline
figures for the selected porous layouts.

Main purpose:
    - compare porous and solid external streamlines
    - use continuous streamline integration, not Matplotlib broken streamplot
    - keep streamline colour and thickness constant everywhere
    - draw internal streamlines from exact pore inlet/outlet seed locations
    - export exact pore inlet/outlet locations and seed positions to CSV

Important:
    The panel method gives the external velocity field.
    The internal channel streamline visualisation is approximate: each channel
    is represented by a narrow corridor with velocity estimated from Q/A.
    This is not full internal CFD.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator

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
    ZOOMED_CONTOUR_XLIM,
    ZOOMED_CONTOUR_YLIM,
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
STREAMLINE_AOA_DEG = 10
STREAMLINE_PORE_DIAMETER_M = 0.0090

STREAMLINE_MODEL_NAMES = (
    "model_1_4_chordwise",
    "model_2_9_perpendicular",
    "model_3_combined_independent",
)

MODEL_LABELS = {
    "model_1_4_chordwise": "Horizontal",
    "model_2_9_perpendicular": "Vertical",
    "model_3_combined_independent": "Isotropic",
    "model_4_saved_case_1": "Saved case",
}

MODEL_FOLDER_NAMES = {
    "model_1_4_chordwise": "horizontal",
    "model_2_9_perpendicular": "vertical",
    "model_3_combined_independent": "isotropic",
    "model_4_saved_case_1": "saved_case",
}

NORMAL_XLIM = (-0.50, 1.50)
NORMAL_YLIM = (-0.50, 0.50)

CLOSE_UP_XLIM = ZOOMED_CONTOUR_XLIM
CLOSE_UP_YLIM = ZOOMED_CONTOUR_YLIM

FIELD_NX = int(CONTOUR_NX)
FIELD_NY = int(CONTOUR_NY)

STREAMLINE_OUTPUT_ROOT = Path(OUTPUT_ROOT) / "streamline_results"

SAVE_FORMATS = ("png", "pdf", "svg")
PNG_DPI = 1000

VERBOSE_COUPLING = True

# Generate external streamlines from pore inlet and outlet locations.
DRAW_PORE_MOUTH_EXTERNAL_STREAMLINES = True

# Use the same pore-adjacent seed locations on the solid comparison panel.
DRAW_PORE_MOUTH_EXTERNAL_STREAMLINES_ON_SOLID = True

# Options: "inlet", "outlet", or "both".
PORE_MOUTH_EXTERNAL_SEED_ROLES = "outlet"

# One external streamline per pore mouth.
# For three nearby streamlines per pore, use:
# (-0.0015, 0.0, 0.0015)
PORE_MOUTH_TANGENTIAL_OFFSETS_C = (0.0,)

# Prevent excessive streamline density in the isotropic model.
MAX_PORE_MOUTH_EXTERNAL_SEEDS = 64
# =============================================================================
# FIGURE CONTROLS
# =============================================================================
MAKE_STREAMLINE_PLOTS = True

DRAW_EXTERNAL_STREAMLINES = True
DRAW_INTERNAL_CHANNEL_STREAMLINES = True

# Internal streamline seeds are now exact pore-mouth seeds.
# Options:
#   "inlet"  = draw from inlet pores only
#   "outlet" = draw from outlet pores only
#   "both"   = draw from inlet and outlet pore mouths
#
# Note:
#   "both" can make the same channel look darker because inlet and outlet seeds
#   may trace the same path from opposite ends. Use "inlet" for the cleanest
#   one-line-per-channel visualisation while still exporting both pore locations.
INTERNAL_STREAMLINE_SEED_ROLES = "inlet"

EXPORT_PORE_LOCATION_CSV = True

# External streamlines: solid black
EXTERNAL_STREAMLINE_COLOR = "black"
EXTERNAL_STREAMLINE_LINEWIDTH = 0.32
EXTERNAL_STREAMLINE_ALPHA = 1.0
EXTERNAL_STREAMLINE_LINESTYLE = "-"

# Internal streamlines: dashed black
INTERNAL_STREAMLINE_COLOR = "black"
INTERNAL_STREAMLINE_LINEWIDTH = 0.32
INTERNAL_STREAMLINE_ALPHA = 1.0
INTERNAL_STREAMLINE_LINESTYLE = "--"

# Airfoil style
AIRFOIL_FILL = "#D9D9D9"   # light grey
AIRFOIL_EDGE = "black"
AIRFOIL_EDGE_WIDTH = 0.70


# =============================================================================
# CONTINUOUS STREAMLINE CONTROLS
# =============================================================================
EXTERNAL_STREAMLINE_COUNT_NORMAL = 75
EXTERNAL_STREAMLINE_COUNT_CLOSE = 25

STREAMLINE_START_X_OFFSET_C = 0.015
STREAMLINE_Y_MARGIN_C = 0.015

STREAMLINE_MAX_STEP_C = 0.004
STREAMLINE_RTOL = 1.0e-6
STREAMLINE_ATOL = 1.0e-8

STREAMLINE_MIN_POINTS = 5
STREAMLINE_MIN_LENGTH_C = 0.015

# Maximum integration distance in multiples of the plotting-window diagonal.
STREAMLINE_MAX_DISTANCE_FACTOR = 3.0


# =============================================================================
# INTERNAL CHANNEL CORRIDOR CONTROLS
# =============================================================================
# The channel corridor is used only for approximate internal streamline
# visualisation. It allows the continuous integrator to pass through the body
# along the internal hydraulic path.
INTERNAL_CHANNEL_CORRIDOR_RADIUS_FACTOR = 0.55
INTERNAL_CHANNEL_MIN_RADIUS_C = 0.002

# Extend the internal channel corridor slightly beyond the two pore mouths.
# This helps exact pore-mouth seeds connect to the internal corridor.
INTERNAL_CHANNEL_END_CAP_FACTOR = 1.20

# "plug" or "parabolic"
INTERNAL_CHANNEL_PROFILE = "plug"

INTERNAL_CHANNEL_MIN_AREA = 1.0e-14
INTERNAL_CHANNEL_FALLBACK_DIAMETER_M = STREAMLINE_PORE_DIAMETER_M


# =============================================================================
# EXTERNAL PORE-SEED CONTROLS
# =============================================================================
# These are used only in the CSV. Internal streamline seeds are placed exactly
# on the pore mouth. External-offset seed locations are also saved for reference.
PORE_EXTERNAL_SEED_OFFSET_C_NORMAL = 0.006
PORE_EXTERNAL_SEED_OFFSET_C_CLOSE = 0.004

# Streamline integration direction options:
#   "forward"  = integrate along the local velocity direction
#   "backward" = integrate opposite to the local velocity direction
#   "both"     = integrate forward and backward, then join both branches
#
# For left-boundary farfield seeds, "forward" is recommended.
FARFIELD_STREAMLINE_DIRECTION = "forward"

# For pore inlet/outlet seeds, "both" normally gives the clearest full line.
PORE_MOUTH_STREAMLINE_DIRECTION = "forward"
# =============================================================================
# PLOT STYLE
# =============================================================================
def apply_publication_style() -> None:
    """Apply a clean paper-friendly Matplotlib style."""
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
            "axes.labelsize": 10.5,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "figure.titlesize": 11.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

# =============================================================================
# STREAMLINE DIRECTION ARROWS
# =============================================================================
DRAW_STREAMLINE_ARROWS = True

# Number of arrows per streamline.
# Use 1 for clean figures. Use 2 if the streamlines are long.
STREAMLINE_ARROW_COUNT = 1

# Arrow positions along each streamline.
# Used only when STREAMLINE_ARROW_COUNT = 1.
STREAMLINE_ARROW_POSITION_FRACTION = 0.58

# Arrow size in plot coordinates.
STREAMLINE_ARROW_LENGTH_C = 0.030
STREAMLINE_ARROW_MUTATION_SCALE = 6.0

# Skip arrows on very short streamlines.
STREAMLINE_ARROW_MIN_LINE_LENGTH_C = 0.050

# =============================================================================
# INTERNAL CHANNEL FLOW-DIRECTION ARROWS
# =============================================================================
DRAW_INTERNAL_FLOW_DIRECTION_ARROWS = True

# Position of the arrow along each internal channel.
# 0.50 means centre of the internal channel.
INTERNAL_ARROW_POSITION_FRACTION = 0.50

# Arrow length in chord-normalised coordinates.
INTERNAL_ARROW_LENGTH_C = 0.040

# Use slightly darker grey than the streamlines so the direction is visible.
INTERNAL_ARROW_COLOR = "black"
INTERNAL_ARROW_LINEWIDTH = 0.50
INTERNAL_ARROW_MUTATION_SCALE = 8.0

# =============================================================================
# DATA CONTAINERS
# =============================================================================
@dataclass(frozen=True)
class StreamlineView:
    """One plotting window."""

    folder_name: str
    label: str
    xlim: tuple[float, float]
    ylim: tuple[float, float]


@dataclass
class FieldData:
    """Velocity field on a structured grid."""

    XX: np.ndarray
    YY: np.ndarray
    XXn: np.ndarray
    YYn: np.ndarray
    U: np.ndarray
    V: np.ndarray
    speed: np.ndarray
    inside: np.ndarray
    internal_channel: np.ndarray
    field_region: np.ndarray


@dataclass(frozen=True)
class SeedRecord:
    """One streamline seed record."""

    x_c: float
    y_c: float
    direction_sign: float
    role: str
    passage_id: int
    pore_name: str


# =============================================================================
# PATH AND FORMAT HELPERS
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
    """Create folder safely, including long Windows/OneDrive paths."""
    os.makedirs(windows_long_path(Path(path)), exist_ok=True)


def write_csv(path: Path, rows: list[dict] | dict) -> None:
    """Write CSV safely."""
    path = Path(path)
    safe_mkdir(path.parent)

    if isinstance(rows, dict):
        rows = [rows]

    with open(windows_long_path(path), "w", encoding="utf-8", newline="") as f:
        pd.DataFrame(rows).to_csv(f, index=False)


def save_figure_all_formats(fig: mpl.figure.Figure, output_stem: Path) -> None:
    """Save one figure as PNG, PDF and SVG."""
    output_stem = Path(output_stem)
    safe_mkdir(output_stem.parent)

    for ext in SAVE_FORMATS:
        path = output_stem.with_suffix(f".{ext}")
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}

        if ext == "png":
            kwargs["dpi"] = PNG_DPI

        fig.savefig(windows_long_path(path), **kwargs)
        print(f"[Plot Saved] {path}")


def format_float_label(value: float, decimals: int = 2) -> str:
    """Return a folder-safe signed float label."""
    sign = "p" if value >= 0.0 else "m"
    mag = f"{abs(float(value)):.{decimals}f}".replace(".", "p")
    return f"{sign}{mag}"


def diameter_label_mm(diameter_m: float) -> str:
    """Return a folder-safe diameter label in mm."""
    return f"{1000.0 * float(diameter_m):.2f}mm".replace(".", "p")


def streamline_views() -> tuple[StreamlineView, ...]:
    """Return normal and close-up views."""
    return (
        #StreamlineView("normal", "Normal view", NORMAL_XLIM, NORMAL_YLIM),
        StreamlineView("close_up", "Close-up view", CLOSE_UP_XLIM, CLOSE_UP_YLIM),
    )


def model_label(model_name: str) -> str:
    """Return publication label for a model."""
    return MODEL_LABELS.get(model_name, model_name)


def model_folder_name(model_name: str) -> str:
    """Return folder-safe label for a model."""
    return MODEL_FOLDER_NAMES.get(model_name, model_name)


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================
def airfoil_polygon_path(geom) -> MplPath:
    """Return a Matplotlib path for the airfoil polygon."""
    return MplPath(np.column_stack([geom.XB, geom.YB]))


def point_is_inside_airfoil(poly: MplPath, x: float, y: float) -> bool:
    """Return True if a point is inside the airfoil polygon."""
    return bool(poly.contains_points([[x, y]])[0])


def outward_surface_normal(geom, poly: MplPath, surface_point) -> tuple[float, float]:
    """Return a robust outward normal at a surface point."""
    chord = geom.chord

    nx = float(surface_point.nx)
    ny = float(surface_point.ny)

    norm = float(np.hypot(nx, ny))
    if norm <= 1.0e-14:
        return 0.0, 1.0

    nx /= norm
    ny /= norm

    eps = 1.0e-5 * chord
    x_try = float(surface_point.x) + eps * nx
    y_try = float(surface_point.y) + eps * ny

    if point_is_inside_airfoil(poly, x_try, y_try):
        nx *= -1.0
        ny *= -1.0

    return nx, ny

def surface_tangent_from_normal(
    nx: float,
    ny: float,
) -> tuple[float, float]:
    """Return a unit surface-tangent vector from an outward normal."""
    tx = -float(ny)
    ty = float(nx)

    magnitude = float(np.hypot(tx, ty))

    if magnitude <= 1.0e-14:
        return 1.0, 0.0

    return tx / magnitude, ty / magnitude


def unique_xy_points(
    points: list[tuple[float, float]],
    tol: float = 1.0e-8,
) -> list[tuple[float, float]]:
    """Remove duplicate streamline seed locations."""
    unique: list[tuple[float, float]] = []

    for point in points:
        duplicate = any(
            np.hypot(
                float(point[0]) - float(old[0]),
                float(point[1]) - float(old[1]),
            ) <= tol
            for old in unique
        )

        if not duplicate:
            unique.append(
                (
                    float(point[0]),
                    float(point[1]),
                )
            )

    return unique

def unique_seed_records(
    seeds: list[SeedRecord],
    tol: float,
) -> list[SeedRecord]:
    """Remove duplicate seed records using x/y/role."""
    unique: list[SeedRecord] = []

    for seed in seeds:
        duplicate = False

        for old in unique:
            same_xy = np.hypot(seed.x_c - old.x_c, seed.y_c - old.y_c) <= tol
            same_role = seed.role == old.role

            if same_xy and same_role:
                duplicate = True
                break

        if not duplicate:
            unique.append(seed)

    return unique


# =============================================================================
# FIELD COMPUTATION
# =============================================================================
def build_field_grid(
    geom,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build dimensional and nondimensional field grids."""
    chord = geom.chord

    x = np.linspace(xlim[0] * chord, xlim[1] * chord, FIELD_NX)
    y = np.linspace(ylim[0] * chord, ylim[1] * chord, FIELD_NY)

    XX, YY = np.meshgrid(x, y)

    return XX, YY, XX / chord, YY / chord


def get_passage_diameter(passage) -> float:
    """Return a representative diameter for one passage."""
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

    return float(INTERNAL_CHANNEL_FALLBACK_DIAMETER_M)


def get_passage_cross_section_area(passage) -> float:
    """Return a representative passage cross-sectional area."""
    chamber = getattr(passage, "chamber", None)

    for obj in (chamber, passage):
        if obj is None:
            continue

        for attr in ("area", "cross_section_area", "A"):
            value = getattr(obj, attr, None)

            if value is None:
                continue

            try:
                area = float(value)
            except (TypeError, ValueError):
                continue

            if np.isfinite(area) and area > INTERNAL_CHANNEL_MIN_AREA:
                return area

    diameter = get_passage_diameter(passage)
    area = np.pi * diameter**2 / 4.0

    return max(float(area), INTERNAL_CHANNEL_MIN_AREA)


def overlay_internal_channel_velocity_on_grid(
    geom,
    XX: np.ndarray,
    YY: np.ndarray,
    inside: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    network,
    passage_states,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Add approximate internal-channel velocity corridors to the plotting grid.

    Outside the airfoil:
        the external panel-method velocity is kept.

    Inside the solid airfoil body:
        velocity remains zero.

    Inside channel corridors:
        velocity is assigned from Q/A along the channel direction.
    """
    chord = geom.chord

    U_out = np.asarray(U, dtype=float).copy()
    V_out = np.asarray(V, dtype=float).copy()

    internal_channel = np.zeros_like(inside, dtype=bool)
    best_speed = np.zeros_like(XX, dtype=float)

    if network is None or passage_states is None:
        return U_out, V_out, internal_channel

    min_radius = INTERNAL_CHANNEL_MIN_RADIUS_C * chord

    for passage, state in zip(network.passages, passage_states):
        sp1 = passage.pore_surface_point(geom, passage.pore1)
        sp2 = passage.pore_surface_point(geom, passage.pore2)

        q_value = float(state.Q)
        area = get_passage_cross_section_area(passage)
        diameter = get_passage_diameter(passage)

        mean_speed = abs(q_value) / max(area, INTERNAL_CHANNEL_MIN_AREA)

        radius = max(
            INTERNAL_CHANNEL_CORRIDOR_RADIUS_FACTOR * diameter,
            min_radius,
        )

        x1 = float(sp1.x)
        y1 = float(sp1.y)
        x2 = float(sp2.x)
        y2 = float(sp2.y)

        # Direction follows solved Q.
        # Q >= 0: pore1 -> pore2
        # Q < 0 : pore2 -> pore1
        if q_value < 0.0:
            x_start, y_start = x2, y2
            x_end, y_end = x1, y1
        else:
            x_start, y_start = x1, y1
            x_end, y_end = x2, y2

        dx = x_end - x_start
        dy = y_end - y_start
        length = float(np.hypot(dx, dy))

        if length <= 1.0e-14:
            continue

        tx = dx / length
        ty = dy / length

        rx = XX - x_start
        ry = YY - y_start

        s = rx * tx + ry * ty
        n = -rx * ty + ry * tx

        end_cap = INTERNAL_CHANNEL_END_CAP_FACTOR * radius

        # The corridor is mainly inside the airfoil, but the end caps are allowed
        # to reach the exact pore-mouth seed locations.
        corridor = (
            (s >= -end_cap)
            & (s <= length + end_cap)
            & (np.abs(n) <= radius)
            & (inside | (s < 0.0) | (s > length))
        )

        if not np.any(corridor):
            continue

        if INTERNAL_CHANNEL_PROFILE.lower() == "parabolic":
            eta = np.clip(np.abs(n) / max(radius, 1.0e-16), 0.0, 1.0)
            local_speed = 1.5 * mean_speed * (1.0 - eta**2)
            local_speed = np.maximum(local_speed, 0.0)
        else:
            local_speed = np.full_like(XX, mean_speed, dtype=float)

        use_here = corridor & (local_speed > best_speed)

        U_out[use_here] = local_speed[use_here] * tx
        V_out[use_here] = local_speed[use_here] * ty

        best_speed[use_here] = local_speed[use_here]
        internal_channel[use_here] = True

    return U_out, V_out, internal_channel


def compute_streamline_field(
    aero_solver,
    aero_result,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    network=None,
    passage_states=None,
) -> FieldData:
    """Compute the velocity field used for continuous streamline integration."""
    geom = aero_solver.geom

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

    U = np.asarray(U_raw, dtype=float).copy()
    V = np.asarray(V_raw, dtype=float).copy()

    # Start with a solid body: no velocity inside the airfoil.
    U[inside] = 0.0
    V[inside] = 0.0

    internal_channel = np.zeros_like(inside, dtype=bool)

    # For porous cases, open narrow internal corridors using Q/A.
    if network is not None and passage_states is not None:
        U, V, internal_channel = overlay_internal_channel_velocity_on_grid(
            geom=geom,
            XX=XX,
            YY=YY,
            inside=inside,
            U=U,
            V=V,
            network=network,
            passage_states=passage_states,
        )

    speed = np.sqrt(U**2 + V**2)

    solid_inside = inside & (~internal_channel)

    field_region = np.zeros_like(XX, dtype=float)
    field_region[solid_inside] = 1.0
    field_region[internal_channel] = 2.0

    return FieldData(
        XX=XX,
        YY=YY,
        XXn=XXn,
        YYn=YYn,
        U=U,
        V=V,
        speed=speed,
        inside=inside,
        internal_channel=internal_channel,
        field_region=field_region,
    )


# =============================================================================
# CONTINUOUS STREAMLINE INTEGRATION
# =============================================================================
def body_mask_for_streamlines(
    field: FieldData,
    include_internal_channels: bool,
) -> np.ndarray:
    """
    Return the blocked region for streamline integration.

    If include_internal_channels is True, the channel corridor is not blocked.
    """
    if include_internal_channels:
        return field.inside & (~field.internal_channel)

    return field.inside


def make_velocity_interpolators(
    field: FieldData,
    include_internal_channels: bool,
) -> tuple[RegularGridInterpolator, RegularGridInterpolator]:
    """Create velocity interpolators for continuous streamline integration."""
    x = np.asarray(field.XXn[0, :], dtype=float)
    y = np.asarray(field.YYn[:, 0], dtype=float)

    u = np.asarray(field.U, dtype=float).copy()
    v = np.asarray(field.V, dtype=float).copy()

    blocked = body_mask_for_streamlines(
        field=field,
        include_internal_channels=include_internal_channels,
    )

    invalid = blocked | (~np.isfinite(u)) | (~np.isfinite(v))

    u[invalid] = 0.0
    v[invalid] = 0.0

    fu = RegularGridInterpolator(
        (y, x),
        u,
        bounds_error=False,
        fill_value=0.0,
        method="linear",
    )

    fv = RegularGridInterpolator(
        (y, x),
        v,
        bounds_error=False,
        fill_value=0.0,
        method="linear",
    )

    return fu, fv


def integrate_one_direction(
    fu: RegularGridInterpolator,
    fv: RegularGridInterpolator,
    start_point: tuple[float, float],
    view: StreamlineView,
    direction_sign: float,
) -> np.ndarray | None:
    """Integrate one streamline branch in one direction."""
    x0, y0 = float(start_point[0]), float(start_point[1])

    if not (
        view.xlim[0] <= x0 <= view.xlim[1]
        and view.ylim[0] <= y0 <= view.ylim[1]
    ):
        return None

    x_min, x_max = float(view.xlim[0]), float(view.xlim[1])
    y_min, y_max = float(view.ylim[0]), float(view.ylim[1])

    domain_diag = float(np.hypot(x_max - x_min, y_max - y_min))
    s_max = STREAMLINE_MAX_DISTANCE_FACTOR * domain_diag

    def velocity_at(x: float, y: float) -> tuple[float, float, float]:
        u = float(fu([[y, x]])[0])
        v = float(fv([[y, x]])[0])
        speed = float(np.hypot(u, v))
        return u, v, speed

    def rhs(_s, state):
        x, y = float(state[0]), float(state[1])
        u, v, speed = velocity_at(x, y)

        if speed <= 1.0e-12:
            return [0.0, 0.0]

        return [
            direction_sign * u / speed,
            direction_sign * v / speed,
        ]

    def hit_left(_s, state):
        return state[0] - x_min

    def hit_right(_s, state):
        return x_max - state[0]

    def hit_bottom(_s, state):
        return state[1] - y_min

    def hit_top(_s, state):
        return y_max - state[1]

    def low_speed(_s, state):
        _, _, speed = velocity_at(float(state[0]), float(state[1]))
        return speed - 1.0e-10

    for event in (hit_left, hit_right, hit_bottom, hit_top, low_speed):
        event.terminal = True
        event.direction = -1

    sol = solve_ivp(
        rhs,
        (0.0, s_max),
        [x0, y0],
        events=[hit_left, hit_right, hit_bottom, hit_top, low_speed],
        max_step=STREAMLINE_MAX_STEP_C,
        rtol=STREAMLINE_RTOL,
        atol=STREAMLINE_ATOL,
    )

    if sol.y.shape[1] < STREAMLINE_MIN_POINTS:
        return None

    line = sol.y.T

    length = float(np.sum(np.hypot(np.diff(line[:, 0]), np.diff(line[:, 1]))))

    if length < STREAMLINE_MIN_LENGTH_C:
        return None

    return line


def integrate_both_directions(
    fu: RegularGridInterpolator,
    fv: RegularGridInterpolator,
    start_point: tuple[float, float],
    view: StreamlineView,
) -> np.ndarray | None:
    """Integrate one continuous streamline forward and backward from a seed."""
    forward = integrate_one_direction(
        fu=fu,
        fv=fv,
        start_point=start_point,
        view=view,
        direction_sign=1.0,
    )

    backward = integrate_one_direction(
        fu=fu,
        fv=fv,
        start_point=start_point,
        view=view,
        direction_sign=-1.0,
    )

    if forward is None and backward is None:
        return None

    if forward is None:
        return backward

    if backward is None:
        return forward

    return np.vstack([backward[::-1], forward[1:]])

def integrate_streamline_by_direction(
    fu: RegularGridInterpolator,
    fv: RegularGridInterpolator,
    start_point: tuple[float, float],
    view: StreamlineView,
    direction_mode: str,
) -> np.ndarray | None:
    """
    Integrate a streamline using the selected direction mode.

    Parameters
    ----------
    direction_mode:
        "forward"  -> along the velocity field
        "backward" -> opposite to the velocity field
        "both"     -> full streamline through the seed
    """
    mode = str(direction_mode).strip().lower()

    if mode == "forward":
        return integrate_one_direction(
            fu=fu,
            fv=fv,
            start_point=start_point,
            view=view,
            direction_sign=1.0,
        )

    if mode == "backward":
        return integrate_one_direction(
            fu=fu,
            fv=fv,
            start_point=start_point,
            view=view,
            direction_sign=-1.0,
        )

    if mode == "both":
        return integrate_both_directions(
            fu=fu,
            fv=fv,
            start_point=start_point,
            view=view,
        )

    raise ValueError(
        "Streamline direction must be "
        "'forward', 'backward', or 'both'. "
        f"Received: {direction_mode!r}"
    )

def draw_arrows_on_streamline(
    ax: plt.Axes,
    line: np.ndarray,
    zorder: int,
) -> None:
    """
    Draw small direction arrows on an already-integrated streamline.

    The arrow direction follows the order of points in `line`.
    Therefore, make sure the line is ordered in the physical flow direction.
    """
    if not DRAW_STREAMLINE_ARROWS:
        return

    if line is None or line.shape[0] < 2:
        return

    dx = np.diff(line[:, 0])
    dy = np.diff(line[:, 1])

    segment_lengths = np.sqrt(dx**2 + dy**2)
    total_length = float(np.sum(segment_lengths))

    if total_length < STREAMLINE_ARROW_MIN_LINE_LENGTH_C:
        return

    valid_segments = np.where(segment_lengths > 1.0e-12)[0]

    if valid_segments.size == 0:
        return

    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])

    if STREAMLINE_ARROW_COUNT <= 1:
        fractions = [float(STREAMLINE_ARROW_POSITION_FRACTION)]
    else:
        fractions = np.linspace(
            0.30,
            0.75,
            int(STREAMLINE_ARROW_COUNT),
        )

    arrow_length = min(
        float(STREAMLINE_ARROW_LENGTH_C),
        0.20 * total_length,
    )

    for fraction in fractions:
        fraction = float(np.clip(fraction, 0.05, 0.95))
        target_length = fraction * total_length

        idx = int(np.searchsorted(cumulative, target_length) - 1)
        idx = max(0, min(idx, line.shape[0] - 2))

        # If the selected segment is too short, use the nearest valid segment.
        if segment_lengths[idx] <= 1.0e-12:
            idx = int(valid_segments[np.argmin(np.abs(valid_segments - idx))])

        seg_len = float(segment_lengths[idx])

        if seg_len <= 1.0e-12:
            continue

        tx = dx[idx] / seg_len
        ty = dy[idx] / seg_len

        # Interpolate arrow centre along the selected segment.
        local_s = target_length - cumulative[idx]
        local_fraction = float(np.clip(local_s / seg_len, 0.0, 1.0))

        cx = (1.0 - local_fraction) * line[idx, 0] + local_fraction * line[idx + 1, 0]
        cy = (1.0 - local_fraction) * line[idx, 1] + local_fraction * line[idx + 1, 1]

        x0 = cx - 0.5 * arrow_length * tx
        y0 = cy - 0.5 * arrow_length * ty
        x1 = cx + 0.5 * arrow_length * tx
        y1 = cy + 0.5 * arrow_length * ty

        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>",
                color=EXTERNAL_STREAMLINE_COLOR,
                linewidth=EXTERNAL_STREAMLINE_LINEWIDTH,
                alpha=EXTERNAL_STREAMLINE_ALPHA,
                mutation_scale=STREAMLINE_ARROW_MUTATION_SCALE,
                shrinkA=0.0,
                shrinkB=0.0,
            ),
            zorder=zorder + 1,
        )

def draw_streamlines_from_points(
    ax: plt.Axes,
    field: FieldData,
    view: StreamlineView,
    seeds: list[tuple[float, float]],
    include_internal_channels: bool,
    direction_mode: str,
    zorder: int,
) -> None:
    """Draw continuous external streamlines from x/y seed points."""
    if not seeds:
        return

    fu, fv = make_velocity_interpolators(
        field=field,
        include_internal_channels=include_internal_channels,
    )

    for seed in seeds:
        line = integrate_streamline_by_direction(
            fu=fu,
            fv=fv,
            start_point=seed,
            view=view,
            direction_mode=direction_mode,
        )

        if line is None:
            continue

        ax.plot(
            line[:, 0],
            line[:, 1],
            linestyle=EXTERNAL_STREAMLINE_LINESTYLE,
            color=EXTERNAL_STREAMLINE_COLOR,
            linewidth=EXTERNAL_STREAMLINE_LINEWIDTH,
            alpha=EXTERNAL_STREAMLINE_ALPHA,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=zorder,
        )

        draw_arrows_on_streamline(
            ax=ax,
            line=line,
            zorder=zorder + 1,
        )


def draw_streamlines_from_seed_records(
    ax: plt.Axes,
    field: FieldData,
    view: StreamlineView,
    seed_records: list[SeedRecord],
    include_internal_channels: bool,
    zorder: int,
) -> None:
    """
    Draw continuous streamlines from seed records.

    Each seed record has its own direction sign:
        inlet  -> +1, along the solved internal channel velocity
        outlet -> -1, opposite to solved internal channel velocity
    """
    if not seed_records:
        return

    fu, fv = make_velocity_interpolators(
        field=field,
        include_internal_channels=include_internal_channels,
    )

    for seed in seed_records:
        line = integrate_one_direction(
            fu=fu,
            fv=fv,
            start_point=(seed.x_c, seed.y_c),
            view=view,
            direction_sign=seed.direction_sign,
        )

        if line is None:
            continue

        # For outlet seeds, the numerical integration is performed backward along
        # the channel to recover the full internal path. Reverse the line before
        # plotting arrows so the arrow still points in the physical inlet-to-outlet
        # flow direction.
        if seed.role == "outlet":
            line = line[::-1]

        ax.plot(
            line[:, 0],
            line[:, 1],
            linestyle=INTERNAL_STREAMLINE_LINESTYLE,
            color=INTERNAL_STREAMLINE_COLOR,
            linewidth=INTERNAL_STREAMLINE_LINEWIDTH,
            alpha=INTERNAL_STREAMLINE_ALPHA,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=zorder,
        )


# =============================================================================
# STREAMLINE SEEDS
# =============================================================================
def build_external_streamline_seeds(view: StreamlineView) -> list[tuple[float, float]]:
    """Build uniformly spaced external seeds along the left boundary."""
    close_up = view.folder_name == "close_up"

    n_seeds = (
        EXTERNAL_STREAMLINE_COUNT_CLOSE
        if close_up
        else EXTERNAL_STREAMLINE_COUNT_NORMAL
    )

    x_seed = view.xlim[0] + STREAMLINE_START_X_OFFSET_C

    y_min = view.ylim[0] + STREAMLINE_Y_MARGIN_C
    y_max = view.ylim[1] - STREAMLINE_Y_MARGIN_C

    y_values = np.linspace(y_min, y_max, int(n_seeds))

    return [(float(x_seed), float(y)) for y in y_values]

def build_external_pore_mouth_seeds(
    geom,
    network,
    passage_states,
    view: StreamlineView,
) -> list[tuple[float, float]]:
    """
    Build streamline seeds at the exact pore inlet and outlet coordinates.

    No outward-normal displacement and no surface-tangential displacement
    are applied.
    """
    chord = float(geom.chord)

    seeds: list[tuple[float, float]] = []

    for passage, state in zip(network.passages, passage_states):
        q_value = float(state.Q)
        role1, role2 = pore_role_from_q(q_value)

        for pore, role in (
            (passage.pore1, role1),
            (passage.pore2, role2),
        ):
            if not pore_role_is_selected_for_external_seed(role):
                continue

            surface_point = passage.pore_surface_point(
                geom,
                pore,
            )

            # Exact pore location on the airfoil surface.
            x_seed_c = float(surface_point.x) / chord
            y_seed_c = float(surface_point.y) / chord

            inside_view = (
                view.xlim[0] <= x_seed_c <= view.xlim[1]
                and view.ylim[0] <= y_seed_c <= view.ylim[1]
            )

            if inside_view:
                seeds.append(
                    (
                        x_seed_c,
                        y_seed_c,
                    )
                )

    seeds = unique_xy_points(
        seeds,
        tol=1.0e-8,
    )

    if len(seeds) > MAX_PORE_MOUTH_EXTERNAL_SEEDS:
        selected_indices = np.linspace(
            0,
            len(seeds) - 1,
            int(MAX_PORE_MOUTH_EXTERNAL_SEEDS),
            dtype=int,
        )

        seeds = [
            seeds[int(index)]
            for index in selected_indices
        ]

    return seeds

def pore_role_from_q(q_value: float) -> tuple[str, str]:
    """Return pore1/pore2 role from signed Q."""
    if float(q_value) >= 0.0:
        return "inlet", "outlet"

    return "outlet", "inlet"

def pore_role_is_selected_for_external_seed(role: str) -> bool:
    """Return True when a pore role should generate an external streamline."""
    mode = str(PORE_MOUTH_EXTERNAL_SEED_ROLES).strip().lower()

    if mode not in {"inlet", "outlet", "both"}:
        raise ValueError(
            "PORE_MOUTH_EXTERNAL_SEED_ROLES must be "
            "'inlet', 'outlet', or 'both'."
        )

    if mode == "both":
        return True

    return role == mode

def seed_direction_from_role(role: str) -> float:
    """
    Return streamline integration direction from pore role.

    The internal corridor velocity is defined from inlet to outlet.
    Therefore:
        inlet seed  -> integrate forward
        outlet seed -> integrate backward
    """
    if role == "inlet":
        return 1.0

    return -1.0


def role_is_selected_for_internal_streamline(role: str) -> bool:
    """Return True if this pore role should be used as an internal seed."""
    mode = str(INTERNAL_STREAMLINE_SEED_ROLES).strip().lower()

    if mode not in {"inlet", "outlet", "both"}:
        raise ValueError(
            "INTERNAL_STREAMLINE_SEED_ROLES must be 'inlet', 'outlet', or 'both'."
        )

    if mode == "both":
        return True

    return role == mode


def collect_exact_pore_locations_and_seed_records(
    geom,
    network,
    passage_states,
    view: StreamlineView,
) -> tuple[list[dict], list[SeedRecord]]:
    """
    Find exact pore inlet/outlet locations and build exact pore-mouth seed records.

    Internal streamline seed:
        exactly at the pore-mouth coordinate:
            x_seed = x_pore
            y_seed = y_pore

    External reference seed:
        slightly outside the airfoil along the outward normal, saved to CSV only.
    """
    chord = geom.chord
    poly = airfoil_polygon_path(geom)
    close_up = view.folder_name == "close_up"

    external_offset_c = (
        PORE_EXTERNAL_SEED_OFFSET_C_CLOSE
        if close_up
        else PORE_EXTERNAL_SEED_OFFSET_C_NORMAL
    )
    external_offset = external_offset_c * chord

    rows: list[dict] = []
    seed_records: list[SeedRecord] = []

    for passage_id, (passage, state) in enumerate(zip(network.passages, passage_states)):
        q_value = float(state.Q)
        role1, role2 = pore_role_from_q(q_value)

        for pore_name, pore, role in (
            ("pore1", passage.pore1, role1),
            ("pore2", passage.pore2, role2),
        ):
            sp = passage.pore_surface_point(geom, pore)

            x_pore = float(sp.x)
            y_pore = float(sp.y)

            nx, ny = outward_surface_normal(
                geom=geom,
                poly=poly,
                surface_point=sp,
            )

            x_external_seed = x_pore + external_offset * nx
            y_external_seed = y_pore + external_offset * ny

            if point_is_inside_airfoil(poly, x_external_seed, y_external_seed):
                x_external_seed = x_pore + 3.0 * external_offset * nx
                y_external_seed = y_pore + 3.0 * external_offset * ny

            x_pore_c = x_pore / chord
            y_pore_c = y_pore / chord

            x_external_seed_c = x_external_seed / chord
            y_external_seed_c = y_external_seed / chord

            exact_seed_inside_view = (
                view.xlim[0] <= x_pore_c <= view.xlim[1]
                and view.ylim[0] <= y_pore_c <= view.ylim[1]
            )

            selected_for_internal = (
                exact_seed_inside_view
                and role_is_selected_for_internal_streamline(role)
            )

            direction_sign = seed_direction_from_role(role)

            rows.append(
                {
                    "passage_id": int(passage_id),
                    "pore_name": pore_name,
                    "role": role,
                    "Q_m3_s": q_value,
                    "x_pore_m": x_pore,
                    "y_pore_m": y_pore,
                    "x_pore_c": x_pore_c,
                    "y_pore_c": y_pore_c,
                    "nx_outward": nx,
                    "ny_outward": ny,
                    "x_internal_exact_seed_m": x_pore,
                    "y_internal_exact_seed_m": y_pore,
                    "x_internal_exact_seed_c": x_pore_c,
                    "y_internal_exact_seed_c": y_pore_c,
                    "internal_direction_sign": direction_sign,
                    "selected_for_internal_streamline": bool(selected_for_internal),
                    "x_external_reference_seed_m": x_external_seed,
                    "y_external_reference_seed_m": y_external_seed,
                    "x_external_reference_seed_c": x_external_seed_c,
                    "y_external_reference_seed_c": y_external_seed_c,
                    "external_reference_offset_c": external_offset_c,
                    "view": view.folder_name,
                }
            )

            if selected_for_internal:
                seed_records.append(
                    SeedRecord(
                        x_c=x_pore_c,
                        y_c=y_pore_c,
                        direction_sign=direction_sign,
                        role=role,
                        passage_id=int(passage_id),
                        pore_name=pore_name,
                    )
                )

    seed_records = unique_seed_records(seed_records, tol=1.0e-8)

    return rows, seed_records


# =============================================================================
# DRAWING
# =============================================================================
def add_airfoil_body(ax: plt.Axes, geom) -> None:
    """Draw the airfoil body above the external streamlines."""
    chord = geom.chord

    ax.fill(
        geom.XB / chord,
        geom.YB / chord,
        facecolor=AIRFOIL_FILL,
        edgecolor="none",
        linewidth=0.0,
        zorder=8,
    )

    ax.plot(
        geom.XB / chord,
        geom.YB / chord,
        color=AIRFOIL_EDGE,
        linewidth=AIRFOIL_EDGE_WIDTH,
        solid_joinstyle="round",
        solid_capstyle="round",
        zorder=14,
    )


def format_streamline_axis(
    ax: plt.Axes,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    """Format streamline axes consistently."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x/c$")
    ax.set_ylabel(r"$y/c$")
    ax.tick_params(direction="in", top=True, right=True, length=3.5, width=0.75)
    ax.minorticks_on()

def draw_internal_channel_flow_arrows(
    ax: plt.Axes,
    geom,
    network,
    passage_states,
) -> None:
    """
    Draw one flow-direction arrow inside each internal channel.

    Direction is based on the solved hydraulic flow rate Q:
        Q >= 0 : pore1 -> pore2
        Q < 0  : pore2 -> pore1
    """
    if not DRAW_INTERNAL_FLOW_DIRECTION_ARROWS:
        return

    chord = geom.chord

    for passage, state in zip(network.passages, passage_states):
        sp1 = passage.pore_surface_point(geom, passage.pore1)
        sp2 = passage.pore_surface_point(geom, passage.pore2)

        q_value = float(state.Q)

        x1 = float(sp1.x)
        y1 = float(sp1.y)
        x2 = float(sp2.x)
        y2 = float(sp2.y)

        # Choose arrow direction using Q.
        if q_value >= 0.0:
            x_start, y_start = x1, y1
            x_end, y_end = x2, y2
        else:
            x_start, y_start = x2, y2
            x_end, y_end = x1, y1

        dx = x_end - x_start
        dy = y_end - y_start
        length = float(np.hypot(dx, dy))

        if length <= 1.0e-14:
            continue

        tx = dx / length
        ty = dy / length

        # Arrow centre location along the internal channel.
        f = float(np.clip(INTERNAL_ARROW_POSITION_FRACTION, 0.10, 0.90))

        cx = x_start + f * dx
        cy = y_start + f * dy

        arrow_length = min(
            INTERNAL_ARROW_LENGTH_C * chord,
            0.35 * length,
        )

        xa = cx - 0.5 * arrow_length * tx
        ya = cy - 0.5 * arrow_length * ty
        xb = cx + 0.5 * arrow_length * tx
        yb = cy + 0.5 * arrow_length * ty

        ax.annotate(
            "",
            xy=(xb / chord, yb / chord),
            xytext=(xa / chord, ya / chord),
            arrowprops=dict(
                arrowstyle="-|>",
                color=EXTERNAL_STREAMLINE_COLOR,
                linewidth=EXTERNAL_STREAMLINE_LINEWIDTH,
                alpha=EXTERNAL_STREAMLINE_ALPHA,
                mutation_scale=STREAMLINE_ARROW_MUTATION_SCALE,
                shrinkA=0.0,
                shrinkB=0.0,
            ),
            zorder=25,
        )


def draw_independent_internal_passage_streamlines(
    ax: plt.Axes,
    geom,
    network,
    passage_states,
    zorder: int = 18,
) -> None:
    """
    Draw each internal passage independently.

    Internal passages are shown as dashed black lines. Horizontal and vertical
    passages may cross visually, but they remain hydraulically independent.
    """
    if not DRAW_INTERNAL_CHANNEL_STREAMLINES:
        return

    chord = float(geom.chord)

    for passage, state in zip(network.passages, passage_states):
        sp1 = passage.pore_surface_point(geom, passage.pore1)
        sp2 = passage.pore_surface_point(geom, passage.pore2)

        q_value = float(state.Q)

        # Orient the line in the solved physical flow direction.
        if q_value >= 0.0:
            x_start = float(sp1.x) / chord
            y_start = float(sp1.y) / chord
            x_end = float(sp2.x) / chord
            y_end = float(sp2.y) / chord
        else:
            x_start = float(sp2.x) / chord
            y_start = float(sp2.y) / chord
            x_end = float(sp1.x) / chord
            y_end = float(sp1.y) / chord

        line = np.column_stack(
            (
                np.linspace(x_start, x_end, 200),
                np.linspace(y_start, y_end, 200),
            )
        )

        ax.plot(
            line[:, 0],
            line[:, 1],
            linestyle=INTERNAL_STREAMLINE_LINESTYLE,
            color=INTERNAL_STREAMLINE_COLOR,
            linewidth=INTERNAL_STREAMLINE_LINEWIDTH,
            alpha=INTERNAL_STREAMLINE_ALPHA,
            dash_capstyle="round",
            dash_joinstyle="round",
            zorder=zorder,
        )

def plot_streamline_pair(
    geom,
    network,
    passage_states,
    porous_field: FieldData,
    solid_field: FieldData,
    model_title: str,
    view: StreamlineView,
    output_stem: Path,
) -> None:
    """Plot porous-versus-solid continuous streamline figures."""
    title_suffix = f" — {view.label}" if view.label else ""

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.7, 3.25),
        constrained_layout=True,
    )

    cases = (
        ("Porous", porous_field, True),
        ("Solid", solid_field, False),
    )

    # Uniform farfield seeds.
    external_seeds = (
        build_external_streamline_seeds(view)
        if DRAW_EXTERNAL_STREAMLINES
        else []
    )

    # Additional seeds immediately outside every pore inlet and outlet.
    pore_mouth_external_seeds = (
        build_external_pore_mouth_seeds(
            geom=geom,
            network=network,
            passage_states=passage_states,
            view=view,
        )
        if DRAW_PORE_MOUTH_EXTERNAL_STREAMLINES
        else []
    )

    for ax, (case_title, field, is_porous) in zip(axes, cases):
        # Regular farfield streamlines.
        if DRAW_EXTERNAL_STREAMLINES:
            draw_streamlines_from_points(
                ax=ax,
                field=field,
                view=view,
                seeds=external_seeds,
                include_internal_channels=False,
                direction_mode=FARFIELD_STREAMLINE_DIRECTION,
                zorder=5,
            )

        # Full continuous external streamlines passing close to pore mouths.
        draw_pore_seeded_lines_here = (
            DRAW_PORE_MOUTH_EXTERNAL_STREAMLINES
            and (
                is_porous
                or DRAW_PORE_MOUTH_EXTERNAL_STREAMLINES_ON_SOLID
            )
        )

        if draw_pore_seeded_lines_here:
            draw_streamlines_from_points(
                ax=ax,
                field=field,
                view=view,
                seeds=pore_mouth_external_seeds,
                include_internal_channels=False,
                direction_mode=PORE_MOUTH_STREAMLINE_DIRECTION,
                zorder=6,
            )

        # Draw the airfoil after external streamlines so the grey body masks any
        # small numerical penetration through the surface.
        add_airfoil_body(
            ax=ax,
            geom=geom,
        )

        # Independent dashed internal passages on the porous panel only.
        if is_porous and DRAW_INTERNAL_CHANNEL_STREAMLINES:
            draw_independent_internal_passage_streamlines(
                ax=ax,
                geom=geom,
                network=network,
                passage_states=passage_states,
                zorder=18,
            )

            draw_internal_channel_flow_arrows(
                ax=ax,
                geom=geom,
                network=network,
                passage_states=passage_states,
            )

        ax.set_title(case_title)

        format_streamline_axis(
            ax=ax,
            xlim=view.xlim,
            ylim=view.ylim,
        )

    fig.suptitle(
        f"{model_title}: continuous streamline visualisation"
        f"{title_suffix}",
        y=1.035,
    )

    save_figure_all_formats(
        fig,
        output_stem,
    )

    plt.close(fig)


# =============================================================================
# MODEL RUNNER
# =============================================================================
def select_streamline_models(all_models: list) -> list:
    """Return requested models in requested order."""
    models_by_name = {model.name: model for model in all_models}
    missing = [name for name in STREAMLINE_MODEL_NAMES if name not in models_by_name]

    if missing:
        raise RuntimeError(
            f"Requested streamline model(s) not found: {missing}. "
            f"Available models: {list(models_by_name)}"
        )

    return [models_by_name[name] for name in STREAMLINE_MODEL_NAMES]


def run_one_streamline_model(
    model,
    geom,
    flow,
    aero_solver,
    solid_result,
    solid_field_cache: dict[str, FieldData],
    output_root: Path,
) -> dict:
    """Solve one porous model and export its streamline figures."""
    label = model_label(model.name)
    folder_name = model_folder_name(model.name)

    print()
    print("=" * 80)
    print(f"RUNNING STREAMLINES: {label}")
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
        print("[Result] Coupling did not converge. Reporting NaN and skipping figures.")

        summary = {
            "model_name": model.name,
            "model_label": label,
            "aoa_deg": flow.aoa_deg,
            "pore_diameter_m": representative_pore_diameter(model),
            "solid_CL": float(solid_result.CL),
            "solid_CM": float(solid_result.CM),
            "porous_CL": float("nan"),
            "porous_CM": float("nan"),
            "coupling_converged": False,
            "coupling_iterations": int(coupled.iterations),
            "max_vn_m_s": float("nan"),
            "max_vn_over_vinf": float("nan"),
            "elapsed_seconds": float(elapsed),
        }

        summary_path = model_dir / "streamline_case_summary.csv"
        write_csv(summary_path, summary)
        print(f"[CSV Saved] {summary_path}")

        return summary

    print(f"[Result] Converged          : {coupled.converged}")
    print(f"[Result] Iterations         : {coupled.iterations}")
    print(f"[Result] Solid CL           : {solid_result.CL:.8f}")
    print(f"[Result] Porous CL          : {coupled.aero_result.CL:.8f}")
    print(f"[Result] Solid CM           : {solid_result.CM:.8f}")
    print(f"[Result] Porous CM          : {coupled.aero_result.CM:.8f}")
    print(f"[Result] max |vn| / Vinf    : {coupled.max_vn_over_vinf:.8f}")
    print(f"[Result] elapsed [s]        : {elapsed:.3f}")

    for view in streamline_views():
        print(f"[Field] Computing {view.label.lower()} streamline field for {label}...")

        if view.folder_name not in solid_field_cache:
            solid_field_cache[view.folder_name] = compute_streamline_field(
                aero_solver=aero_solver,
                aero_result=solid_result,
                xlim=view.xlim,
                ylim=view.ylim,
            )

        porous_field = compute_streamline_field(
            aero_solver=aero_solver,
            aero_result=coupled.aero_result,
            xlim=view.xlim,
            ylim=view.ylim,
            network=None,
            passage_states=None,
        )

        view_dir = model_dir / view.folder_name
        safe_mkdir(view_dir)

        if EXPORT_PORE_LOCATION_CSV:
            pore_rows, _seed_records = collect_exact_pore_locations_and_seed_records(
                geom=geom,
                network=network,
                passage_states=coupled.passage_states,
                view=view,
            )

            pore_csv_path = view_dir / "exact_pore_locations_and_streamline_seeds.csv"
            write_csv(pore_csv_path, pore_rows)
            print(f"[CSV Saved] {pore_csv_path}")

        if MAKE_STREAMLINE_PLOTS:
            plot_streamline_pair(
                geom=geom,
                network=network,
                passage_states=coupled.passage_states,
                porous_field=porous_field,
                solid_field=solid_field_cache[view.folder_name],
                model_title=label,
                view=view,
                output_stem=view_dir / "streamlines_porous_vs_solid",
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

    summary_path = model_dir / "streamline_case_summary.csv"
    write_csv(summary_path, summary)
    print(f"[CSV Saved] {summary_path}")

    return summary


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    apply_publication_style()

    aoa_label = format_float_label(STREAMLINE_AOA_DEG, decimals=2)
    d_label = diameter_label_mm(STREAMLINE_PORE_DIAMETER_M)

    output_root = STREAMLINE_OUTPUT_ROOT / f"AOA_{aoa_label}_D_{d_label}"
    safe_mkdir(output_root)

    print("=" * 80)
    print("RUNNING PYTHON STREAMLINE FIGURES")
    print("=" * 80)
    print(f"Airfoil                  : NACA {AIRFOIL_NAME}")
    print(f"AoA [deg]                : {STREAMLINE_AOA_DEG:.6g}")
    print(f"Reynolds number          : {REYNOLDS_EXTERNAL:.6e}")
    print(f"Panels                   : {N_PANELS}")
    print(f"Pore diameter [m]        : {STREAMLINE_PORE_DIAMETER_M:.6e}")
    print(f"Pore diameter [mm]       : {1000.0 * STREAMLINE_PORE_DIAMETER_M:.3f}")
    print(
        "Models                   : "
        + ", ".join(model_label(n) for n in STREAMLINE_MODEL_NAMES)
    )
    print(f"Field grid               : {FIELD_NX} x {FIELD_NY}")
    print(f"External streamlines     : {DRAW_EXTERNAL_STREAMLINES}")
    print(f"Internal streamlines     : {DRAW_INTERNAL_CHANNEL_STREAMLINES}")
    print(f"Internal seed roles      : {INTERNAL_STREAMLINE_SEED_ROLES}")
    print(f"External streamline colour    : {EXTERNAL_STREAMLINE_COLOR}")
    print(f"External streamline linewidth : {EXTERNAL_STREAMLINE_LINEWIDTH}")
    print(f"External streamline style     : {EXTERNAL_STREAMLINE_LINESTYLE}")

    print(f"Internal streamline colour    : {INTERNAL_STREAMLINE_COLOR}")
    print(f"Internal streamline linewidth : {INTERNAL_STREAMLINE_LINEWIDTH}")
    print(f"Internal streamline style     : {INTERNAL_STREAMLINE_LINESTYLE}")
    print(f"Output root              : {output_root}")

    print(
        f"Pore-mouth external lines     : "
        f"{DRAW_PORE_MOUTH_EXTERNAL_STREAMLINES}"
    )
    print(
        f"Pore-mouth seed roles         : "
        f"{PORE_MOUTH_EXTERNAL_SEED_ROLES}"
    )
    print(
        f"Pore-mouth lines on solid     : "
        f"{DRAW_PORE_MOUTH_EXTERNAL_STREAMLINES_ON_SOLID}"
    )
    print(
        f"Farfield integration direction : "
        f"{FARFIELD_STREAMLINE_DIRECTION}"
    )
    print(
        f"Pore-mouth integration direction: "
        f"{PORE_MOUTH_STREAMLINE_DIRECTION}"
    )



    for view in streamline_views():
        print(f"{view.label} x/c           : {view.xlim}")
        print(f"{view.label} y/c           : {view.ylim}")

    flow, geom, _, aero_solver = build_base_problem(
        airfoil_name=AIRFOIL_NAME,
        chord=CHORD,
        n_panels=int(N_PANELS),
        aoa_deg=float(STREAMLINE_AOA_DEG),
        reynolds_external=REYNOLDS_EXTERNAL,
        rho_inf=RHO_INF,
        mu_inf=MU_INF,
        p_inf=P_INF,
    )

    print(f"Freestream speed [m/s]   : {flow.v_inf:.8f}")
    print("[Setup] Solving solid baseline...")

    solid_result = aero_solver.solve()

    print(f"[Setup] Solid CL          : {solid_result.CL:.8f}")
    print(f"[Setup] Solid CD          : {solid_result.CD:.8e}")
    print(f"[Setup] Solid CM          : {solid_result.CM:.8f}")

    all_models = build_model_specs(
        geom=geom,
        diameter_m=float(STREAMLINE_PORE_DIAMETER_M),
    )

    selected_models = select_streamline_models(all_models)

    solid_field_cache: dict[str, FieldData] = {}
    summary_rows: list[dict] = []

    for model in selected_models:
        row = run_one_streamline_model(
            model=model,
            geom=geom,
            flow=flow,
            aero_solver=aero_solver,
            solid_result=solid_result,
            solid_field_cache=solid_field_cache,
            output_root=output_root,
        )
        summary_rows.append(row)

    combined_summary_path = output_root / "all_streamline_cases_summary.csv"
    write_csv(combined_summary_path, summary_rows)

    print()
    print(f"[CSV Saved] {combined_summary_path}")
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()