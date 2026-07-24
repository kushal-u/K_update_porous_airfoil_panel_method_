"""User-editable configuration and lightweight data containers.

Edit this file to change the airfoil, Reynolds number, selected porous model,
rectangular channel height, coupling iteration limit, plotting resolution, and XFOIL settings.
The numerical implementation lives in solver.py and porous_core.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shutil

import numpy as np

from .solver import CouplingConfig

# Repository root, independent of the terminal working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# XFOIL CONFIGURATION
# =============================================================================
# XFOIL is optional. The panel-method solver runs without it.
#
# Recommended GitHub-friendly setup:
#   1. Put xfoil or xfoil.exe on your PATH, OR
#   2. Set the environment variable XFOIL_EXE to the executable path.
#
USE_XFOIL = os.getenv("USE_XFOIL", "1").strip().lower() not in {"0", "false", "no"}

# Optional search folder used only when XFOIL is not found on PATH or by XFOIL_EXE.
# Recursive scanning is enabled only when this environment variable is set.
_XFOIL_FOLDER_ENV = os.getenv("XFOIL_FOLDER", "").strip()
XFOIL_FOLDER = Path(
    os.getenv(
        "XFOIL_FOLDER",
        str(PROJECT_ROOT / "xfoil"),
    )
).expanduser()

# Inviscid/incompressible comparison settings.
XFOIL_MACH = 0.0
XFOIL_TIMEOUT = 30.0


def find_xfoil_executable() -> Path | None:
    """Find an XFOIL executable in a portable, GitHub-friendly way.

    Search order:
        1. XFOIL_EXE environment variable
        2. xfoil or xfoil.exe on system PATH
        3. Common subfolders inside XFOIL_FOLDER
        4. Recursive search inside XFOIL_FOLDER

    Returns
    -------
    Path | None
        Absolute executable path when found; otherwise None.
    """
    env_xfoil = os.getenv("XFOIL_EXE", "").strip()
    if env_xfoil:
        env_path = Path(env_xfoil).expanduser()
        if env_path.exists():
            return env_path.resolve()

    for executable_name in ("xfoil", "xfoil.exe"):
        system_xfoil = shutil.which(executable_name)
        if system_xfoil is not None:
            return Path(system_xfoil).resolve()

    candidates = [
        PROJECT_ROOT / "xfoil" / "xfoil",
        PROJECT_ROOT / "xfoil" / "xfoil.exe",
        XFOIL_FOLDER / "xfoil",
        XFOIL_FOLDER / "xfoil.exe",
        XFOIL_FOLDER / "bin" / "xfoil",
        XFOIL_FOLDER / "bin" / "xfoil.exe",
        XFOIL_FOLDER / "runs" / "xfoil",
        XFOIL_FOLDER / "runs" / "xfoil.exe",
        XFOIL_FOLDER / "src" / "xfoil",
        XFOIL_FOLDER / "src" / "xfoil.exe",
        XFOIL_FOLDER / "Xfoil" / "xfoil",
        XFOIL_FOLDER / "Xfoil" / "xfoil.exe",
        XFOIL_FOLDER / "Xfoil" / "bin" / "xfoil",
        XFOIL_FOLDER / "Xfoil" / "bin" / "xfoil.exe",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    # Avoid an expensive recursive scan of the whole repository unless the
    # user explicitly configured XFOIL_FOLDER. This keeps imports fast on fresh clones.
    if _XFOIL_FOLDER_ENV and XFOIL_FOLDER.exists():
        for pattern in ("xfoil", "xfoil.exe"):
            matches = list(XFOIL_FOLDER.rglob(pattern))
            if matches:
                return matches[0].resolve()

    return None


XFOIL_EXE_PATH = find_xfoil_executable()

if USE_XFOIL and XFOIL_EXE_PATH is None:
    print("=" * 80)
    print("[XFOIL WARNING]")
    print("Could not find xfoil/xfoil.exe. XFOIL comparisons will be skipped.")
    print("Set XFOIL_EXE or XFOIL_FOLDER to enable XFOIL on this machine.")
    print(f"Checked folder: {XFOIL_FOLDER}")
    print("=" * 80)
    USE_XFOIL = False
elif USE_XFOIL:
    print(f"[XFOIL] Using executable: {XFOIL_EXE_PATH}")

# =============================================================================
# USER SETTINGS
# =============================================================================

# Output folder.

OUTPUT_ROOT = Path(
    os.getenv("POROUS_OUTPUT_ROOT", str(PROJECT_ROOT / "outputs"))
).expanduser()

# Keep XFOIL text files inside the same output tree.
XFOIL_OUTPUT_ROOT = Path(
    os.getenv("XFOIL_OUTPUT_ROOT", str(OUTPUT_ROOT / "_xfoil"))
).expanduser()

# Choose which porous model to run.
# Valid values:
#   "all"
#   "model_1_4_chordwise"
#   "model_2_9_perpendicular"
#   "model_3_combined_independent"
#   "model_4_saved_case_1"
SELECTED_MODEL = "all"

# Geometry and discretisation.
AIRFOIL_NAME = "0018"
CHORD = 1.0
N_PANELS = 1000

# External flow condition.
REYNOLDS_EXTERNAL = 5.0e5
AOA_DEG = 4.0

# Fluid properties.
RHO_INF = 1.225
MU_INF = 1.8e-5
P_INF = 101325.0

# Two-dimensional rectangular porous-channel dimensions.
#
# Each internal channel is an in-plane rectangle with:
#     L : distance between the two pore centres
#     H : channel opening measured normal to the channel centreline
#
# The hydraulic equations are written per unit span:
#     q_prime [m^2/s] = flow rate per unit span
#     A_prime [m]     = H
#     U_bar [m/s]     = q_prime / H
#     D_h [m]         = 2 H
#     R_prime         = 12 mu L / H^3       (laminar parallel-wall limit)
#
# The visible in-plane channel area is L*H, but L*H is parallel to the flow and
# is therefore not used as the flow-through area.
#
# SPAN_WIDTH is retained only for backward-compatible function calls and old
# output readers. It is not used by the active 2D hydraulic or transpiration
# equations.
SPAN_WIDTH = 0.02
CHANNEL_HEIGHT = 0.0170
DEFAULT_CHANNEL_HEIGHT = CHANNEL_HEIGHT

# Compatibility aliases. There is no independent hydraulic width in the 2D
# per-unit-span model.
CHANNEL_WIDTH = 1.0
DEFAULT_CHANNEL_WIDTH = 1.0

# Backward-compatible aliases. They now mean rectangular channel height H.
PORE_DIAMETER = CHANNEL_HEIGHT
DEFAULT_PORE_DIAMETER = DEFAULT_CHANNEL_HEIGHT

# =============================================================================
# FIXED-AOA RECTANGULAR-CHANNEL HEIGHT CASES
# =============================================================================
# These are additional fixed-AoA cases. Each channel height gets its own output folder.
MAKE_FIXED_AOA_CHANNEL_HEIGHT_OUTPUTS = True

FIXED_AOA_CHANNEL_HEIGHTS = (
    0.0080,
    0.0090,
    0.0100,
)

# Backward-compatible aliases.
MAKE_FIXED_AOA_DIAMETER_OUTPUTS = MAKE_FIXED_AOA_CHANNEL_HEIGHT_OUTPUTS
FIXED_AOA_PORE_DIAMETERS = FIXED_AOA_CHANNEL_HEIGHTS

# =============================================================================
# AOA SWEEP SETTINGS
# =============================================================================
AOA_SWEEP_START_DEG = -5.0
AOA_SWEEP_END_DEG = 15.0
AOA_SWEEP_STEP_DEG = 1.0

AOA_SWEEP_CHANNEL_HEIGHTS = (
    0.0080,
    0.0090,
    0.0100,
)

# Backward-compatible alias.
AOA_SWEEP_PORE_DIAMETERS = AOA_SWEEP_CHANNEL_HEIGHTS

# =============================================================================
# COUPLING SETTINGS
# =============================================================================
COUPLING = CouplingConfig(
    max_iter=500,
)

# =============================================================================
# PLOT SETTINGS
# =============================================================================
MAKE_CONTOUR_PLOTS = True
CONTOUR_NX = 1000
CONTOUR_NY = 1000

# Extra close-up contour plots around the airfoil.
MAKE_ZOOMED_CONTOUR_PLOTS = True
ZOOMED_CONTOUR_XLIM = (-0.05, 1.05)
ZOOMED_CONTOUR_YLIM = (-0.20, 0.20)

# =============================================================================
# PARAVIEW SETTINGS
# =============================================================================
# Fixed-AoA ParaView export.
EXPORT_PARAVIEW = False

# AoA-sweep ParaView export.
# Set False if the run is too slow or long-path errors appear.
EXPORT_AOA_SWEEP_PARAVIEW = False


# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass(frozen=True)
class GeometryLimits:
    # Allow exact LE and TE for the middle Model 1 channel.
    x_min_frac: float = 0.0
    x_max_frac: float = 1.0

    thickness_fraction_limit: float = 0.85
    min_gap: float = 0.0020
    fit_fraction: float = 0.70


    # Exact endpoint handling.
    endpoint_tol: float = 1e-8

    # Skip the zero-thickness endpoint region when checking if the internal
    # channel fits inside the airfoil body.
    endpoint_fit_margin: float = 0.02


@dataclass(frozen=True)
class Pore:
    x_frac: float
    side: str
    height: float

    def __post_init__(self) -> None:
        side = self.side.lower()
        object.__setattr__(self, "side", side)

        if side not in ("upper", "lower"):
            raise ValueError("Pore side must be 'upper' or 'lower'.")
        if not (0.0 <= self.x_frac <= 1.0):
            raise ValueError("x_frac must be in [0, 1].")
        if self.height <= 0.0:
            raise ValueError("Rectangular pore height must be positive.")

    @property
    def area(self) -> float:
        """Flow area per unit span, A' = H [m]."""
        return float(self.height)

    @property
    def hydraulic_diameter(self) -> float:
        """Parallel-wall hydraulic diameter, D_h = 2H."""
        return float(2.0 * self.height)

    @property
    def diameter(self) -> float:
        """Backward-compatible alias returning channel height H."""
        return float(self.height)


@dataclass(frozen=True)
class Chamber:
    height: float

    def __post_init__(self) -> None:
        if self.height <= 0.0:
            raise ValueError("Rectangular channel height must be positive.")

    @property
    def area(self) -> float:
        """Flow area per unit span, A' = H [m]."""
        return float(self.height)

    @property
    def flow_area_per_unit_span(self) -> float:
        return float(self.height)

    @property
    def wetted_perimeter_per_unit_span(self) -> float:
        """Two parallel walls per unit span."""
        return 2.0

    @property
    def hydraulic_diameter(self) -> float:
        """D_h = 4 A'/P' = 2H."""
        return float(2.0 * self.height)

    @property
    def aspect_ratio(self) -> float:
        """Infinite-width parallel-wall limit."""
        return float("inf")

    def planform_area(self, length: float) -> float:
        """In-plane area visible in the 2D airfoil, A_planform = L H."""
        length = float(length)
        if length <= 0.0:
            raise ValueError("Channel length must be positive.")
        return float(length * self.height)


@dataclass
class PassageState:
    Q: float
    p1: float
    p2: float
    dp_total: float
    Rs: float
    reynolds_equivalent: float
    friction_factor: float = float("nan")
    resistance_model: str = ""


@dataclass(frozen=True)
class PassageSpec:
    name: str
    x1_frac: float
    side1: str
    x2_frac: float
    side2: str
    channel_height_m: float = DEFAULT_CHANNEL_HEIGHT
    layout_kind: str = "surface_pair"

    def __post_init__(self) -> None:
        if self.channel_height_m <= 0.0:
            raise ValueError("channel_height_m must be positive.")

    @property
    def diameter_m(self) -> float:
        """Backward-compatible alias returning rectangular channel height."""
        return float(self.channel_height_m)

    @property
    def channel_width_m(self) -> float:
        """Compatibility reference width for the per-unit-span model."""
        return 1.0


@dataclass(frozen=True)
class PorousModelSpec:
    name: str
    description: str
    passages: tuple[PassageSpec, ...]


@dataclass
class MultiPassageResult:
    aero_result: object
    passage_states: list[PassageState]
    normal_transpiration: np.ndarray
    converged: bool
    iterations: int
    max_vn: float
    max_vn_over_vinf: float


# =============================================================================
# BASIC HELPERS
# =============================================================================
def reynolds_to_velocity(
    reynolds: float,
    rho: float,
    mu: float,
    chord: float,
) -> float:
    if reynolds <= 0.0:
        raise ValueError("Reynolds number must be positive.")
    if rho <= 0.0:
        raise ValueError("Density must be positive.")
    if mu <= 0.0:
        raise ValueError("Dynamic viscosity must be positive.")
    if chord <= 0.0:
        raise ValueError("Chord must be positive.")

    return reynolds * mu / (rho * chord)


def rectangular_hydraulic_diameter(
    height_or_width: float,
    height: float | None = None,
) -> float:
    """Return the 2D parallel-wall hydraulic diameter, D_h = 2H."""
    h = float(height_or_width if height is None else height)
    if h <= 0.0:
        raise ValueError("Rectangular channel height must be positive.")
    return float(2.0 * h)


def rectangular_2d_laminar_resistance(
    mu: float,
    length: float,
    height: float,
) -> tuple[float, float]:
    """Return laminar resistance and flow area per unit span.

    q' = Delta p H^3 / (12 mu L)
    R' = Delta p / q' = 12 mu L / H^3

    Returns
    -------
    tuple[float, float]
        ``(R_prime, A_prime)`` where ``A_prime = H``.
    """
    mu = float(mu)
    length = float(length)
    height = float(height)

    if mu <= 0.0:
        raise ValueError("Dynamic viscosity must be positive.")
    if length <= 0.0:
        raise ValueError("Channel length must be positive.")
    if height <= 0.0:
        raise ValueError("Rectangular channel height must be positive.")

    resistance_per_unit_span = 12.0 * mu * length / height**3
    flow_area_per_unit_span = height
    return float(resistance_per_unit_span), float(flow_area_per_unit_span)


def square_channel_laminar_resistance(
    mu: float,
    length: float,
    height: float,
    series_terms: int = 50,
) -> tuple[float, float]:
    """Deprecated compatibility alias for the active 2D resistance."""
    del series_terms
    return rectangular_2d_laminar_resistance(mu, length, height)


def rectangular_laminar_resistance(
    mu: float,
    length: float,
    width: float | None = None,
    height: float | None = None,
    series_terms: int = 50,
) -> tuple[float, float]:
    """Compatibility wrapper for the 2D per-unit-span resistance."""
    del series_terms
    if height is None:
        if width is None:
            raise ValueError("height must be provided.")
        height = float(width)
    return rectangular_2d_laminar_resistance(
        mu=float(mu),
        length=float(length),
        height=float(height),
    )
