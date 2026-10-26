# PAPER CROSS-REFERENCES -- configuration and notation
# [subsec:internal_hydraulics] H, D_h=2H, R'=12*mu*L/H^3.
# [subsec:coupled_solution] Newton/TRF, jet head and continuation controls.
# All q' values are per unit span [m^2/s]; legacy diameter aliases mean H.
# IMPORTANT MANUSCRIPT DIFFERENCE: this implementation uses a smooth
# Re_h=2000..3000 blend. INTERNAL_RE_TRANSITION=2300 is a compatibility
# constant, not an active regime switch. See methods_review.md.
# This edition changes comments only; all supplied settings are retained.

"""User-editable configuration and lightweight data containers.

All run, hydraulic, continuation, export and plotting inputs live here.
Restart the workflow after editing this file. Values use SI units unless labelled
otherwise; diameter aliases refer to rectangular channel height, not a circle.
The numerical implementation lives in solver.py and porous.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

import numpy as np

from .solver import CouplingConfig

# Repository root, independent of the terminal working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# XFOIL CONFIGURATION
# =============================================================================
# XFOIL is optional. Configure it here; environment variables do not override
# settings. Automatic discovery on PATH is still allowed.
USE_XFOIL = True
XFOIL_EXECUTABLE: Path | None = None  # e.g. Path("/usr/bin/xfoil")
XFOIL_FOLDER = PROJECT_ROOT / "xfoil"
XFOIL_RECURSIVE_SEARCH = False
XFOIL_MACH = 0.0
XFOIL_TIMEOUT = 30.0


# =============================================================================
# USER SETTINGS
# =============================================================================

# Output folder.

OUTPUT_ROOT = PROJECT_ROOT / "outputs"

# Keep XFOIL text files inside the same output tree.
XFOIL_OUTPUT_ROOT = OUTPUT_ROOT / "_xfoil"

# =============================================================================
# PIPELINE SWITCHES -- True runs the stage; False skips it
# =============================================================================
RUN_FIXED_CASES = False
RUN_AOA_SWEEP = True
RUN_AERODYNAMIC_PLOTS = True
RUN_GEOMETRY = True
RUN_CONTOURS = True
RUN_STREAMLINES = True
ENABLE_PLOTS = True                  # master switch for every figure
CONTINUE_AFTER_STAGE_ERROR = True

# Fixed-case figures and detailed CSVs (summary CSVs are always retained).
MAKE_CP_PLOTS = True
MAKE_NETWORK_PLOTS = True
MAKE_AOA_SWEEP_PLOTS = True
EXPORT_SURFACE_DATA = True
EXPORT_PASSAGE_DATA = True

# Reporting: changes are relative to the solid panel result at the SAME AoA/Re.
PRINT_AERODYNAMIC_METRICS = True
SAVE_AERODYNAMIC_METRICS = True
PRINT_METRICS_PER_CASE = False
METRICS_MIN_ABS_BASELINE = 1.0e-8     # suppress undefined relative changes
METRICS_MIN_POSITIVE_CD = 1.0e-6     # guard pressure-based CL/CD
PRINT_ELAPSED_TIME = True
SAVE_TIMING_REPORT = True
REPORT_OUTPUT_DIR = OUTPUT_ROOT / "reports"

# Choose which porous model to run.
# Valid values:
#   "all"
#   "model_1_4_chordwise"
#   "model_2_9_perpendicular"
#   "model_3_combined_independent"
# Model 4 is not enabled in the supplied model builder.
SELECTED_MODEL = "all"

# Geometry and discretisation.
# [subsec:external_panel; subsec:internal_hydraulics]
# Current run: closed-TE NACA 0018, c=1 m, N=1000; Re_c=1e6.
# build_base_problem uses V_inf=Re_c*mu/(rho*c), approximately 14.694 m/s.
# Re_c is an external input; each Re_h follows from the solved channel flow.
AIRFOIL_NAME = "0018"
CHORD = 1.0
N_PANELS = 1000

# External flow condition.
REYNOLDS_EXTERNAL = 1.0e6
AOA_DEG = 4.0
METRICS_PRINT_AOA_DEG = AOA_DEG

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

# EDIT THIS LIST ONCE. Fixed cases, sweeps and aerodynamic plots all inherit it.
# Legacy "pore diameter" names mean rectangular channel height H, in metres.
# [eq:channel_velocity; eq:hydraulic_diameter; eq:flow_scaling]
# Shared heights H=8,9,10 mm for the fixed/sweep defaults. Channel area
# per span is H; U=q'/H, D_h=2H, and the flow scale is H*V_inf.
PORE_HEIGHTS_M = (0.0080, 0.0090, 0.0100)  # 8, 9, 10 mm
CHANNEL_HEIGHT = PORE_HEIGHTS_M[0]         # main single-height calculation
DEFAULT_CHANNEL_HEIGHT = CHANNEL_HEIGHT

# Contour, streamline and geometry stages use one height/angle by default.
# Set a per-stage override below only when a different operating point is wanted.
VISUALISATION_HEIGHT_OVERRIDE_M = None
VISUALISATION_AOA_OVERRIDE_DEG = None
VISUALISATION_HEIGHT_M = (CHANNEL_HEIGHT if VISUALISATION_HEIGHT_OVERRIDE_M is None
                          else float(VISUALISATION_HEIGHT_OVERRIDE_M))
VISUALISATION_AOA_DEG = (AOA_DEG if VISUALISATION_AOA_OVERRIDE_DEG is None
                         else float(VISUALISATION_AOA_OVERRIDE_DEG))

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

FIXED_AOA_HEIGHTS_OVERRIDE_M = None  # None -> PORE_HEIGHTS_M
FIXED_AOA_CHANNEL_HEIGHTS = (PORE_HEIGHTS_M if FIXED_AOA_HEIGHTS_OVERRIDE_M is None
                            else tuple(FIXED_AOA_HEIGHTS_OVERRIDE_M))

# Backward-compatible aliases.
MAKE_FIXED_AOA_DIAMETER_OUTPUTS = MAKE_FIXED_AOA_CHANNEL_HEIGHT_OUTPUTS
FIXED_AOA_PORE_DIAMETERS = FIXED_AOA_CHANNEL_HEIGHTS

# =============================================================================
# AOA SWEEP SETTINGS
# =============================================================================
AOA_SWEEP_START_DEG = -5.0
AOA_SWEEP_END_DEG = 15.0
AOA_SWEEP_STEP_DEG = 1.0
AOA_SWEEP_HEIGHTS_OVERRIDE_M = None  # None -> PORE_HEIGHTS_M
AOA_SWEEP_CHANNEL_HEIGHTS = (PORE_HEIGHTS_M if AOA_SWEEP_HEIGHTS_OVERRIDE_M is None
                            else tuple(AOA_SWEEP_HEIGHTS_OVERRIDE_M))

# Backward-compatible alias.
AOA_SWEEP_PORE_DIAMETERS = AOA_SWEEP_CHANNEL_HEIGHTS

# =============================================================================
# COUPLING SETTINGS
# =============================================================================
# [eq:newton_system; eq:newton_update; subsec:verification_outputs]
# 500 Newton iterations; up to 24 Armijo backtracks. Ill-conditioned
# Newton systems use a least-squares linear step; TRF is a separate fallback.
# The active physical root tolerance is sqrt(machine epsilon)*max(1,||z||inf)
# inside porous.py, not HYBR_XTOL or the TRF stopping tolerances.
COUPLING = CouplingConfig(
    max_iter=500,
    max_backtracks=24,
    armijo=1.0e-4,
    condition_limit=1.0e12,
    least_squares_fallback=True,
)

# =============================================================================
# PLOT SETTINGS
# =============================================================================
MAKE_CONTOUR_PLOTS = ENABLE_PLOTS and RUN_CONTOURS
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
PARAVIEW_FIELD_NX = 300
PARAVIEW_FIELD_NY = 300


# =============================================================================
# NETWORK GEOMETRY AND MOMENT REFERENCE
# =============================================================================
# Model identifiers retain their legacy names; Model 2 currently has four channels.
AVAILABLE_MODEL_NAMES = (
    "model_1_4_chordwise",
    "model_2_9_perpendicular",
    "model_3_combined_independent",
)
# [subsec:model_overview]
# Active builders: 4 horizontal channels, 4 vertical channels, and their
# 8-channel independent union. The legacy name "model_2_9_perpendicular"
# does not specify the current channel count. Crossings are not junctions.
MODEL1_CHANNEL_COUNT = 4
MODEL1_MAX_Y_FRACTION = 0.85
MODEL1_HELPER_MAX_Y_FRACTION = 0.60  # legacy direct-helper default
MODEL2_X_START_C = 0.100
MODEL2_X_END_C = 0.890
MODEL2_CHANNEL_COUNT = 4
# Replace this expression with a tuple to prescribe nonuniform positions.
MODEL2_X_LOCATIONS = tuple(np.linspace(MODEL2_X_START_C, MODEL2_X_END_C, MODEL2_CHANNEL_COUNT))
GEOMETRY_THICKNESS_FRACTION_LIMIT = 0.85
GEOMETRY_MIN_GAP_M = 0.0020
GEOMETRY_FIT_FRACTION = 0.70
GEOMETRY_ENDPOINT_TOL = 1.0e-8
GEOMETRY_ENDPOINT_FIT_MARGIN = 0.02
MOMENT_REFERENCE_X_C = 0.25
MOMENT_REFERENCE_Y_C = 0.0

# Retained saved-case helper; Model 4 is not an active run option.
MODEL4_CHANNEL_HEIGHTS = (0.012, 0.009)
MODEL4_LOCATIONS = (
    ("m4_case1_passage_01_lower_to_upper", 0.400094, 0.933001),
    ("m4_case1_passage_02_lower_to_upper", 0.925069, 0.955118),
)

# Smooth hydraulic transition used by the uploaded implementation.
# [eq:channel_friction_factor -- differs from uploaded manuscript]
# Active f_D=(1-w)*96/Re_h + w*0.3164*Re_h^(-1/4),
# where w=0 below LOW, w=1 above HIGH, and w=xi^2*(3-2*xi) between them.
# xi=(Re_h-LOW)/(HIGH-LOW). The Jacobian differentiates this blend.
# [eq:transition_weight] in the checked manuscript defines w explicitly.
# methods_checked.tex replaces the manuscript's abrupt 2300 switch.
HYDRAULIC_TRANSITION_RE_LOW = 2000.0
HYDRAULIC_TRANSITION_RE_HIGH = 3000.0

# Numerical solver controls. COUPLING.max_iter bounds Newton iterations and
# the analytical-Jacobian TRF fallback evaluations separately. Every accepted
# solution is independently checked against the original equations. Newton
# solves J(z) delta_z=-R(z); this is not a one-shot linear physical model.
# HYBR/PAL_* budget names below are legacy compatibility settings only.
HYBR_STEP_FACTORS = (0.5, 0.2)
HYBR_XTOL = 1.0e-10
TRF_FTOL = 1.0e-12
TRF_XTOL = 1.0e-12
TRF_GTOL = 1.0e-12
TRF_MIN_EVALUATIONS = 1000
TRF_EVALUATION_MULTIPLIER = 4
PAL_HYBR_MIN_EVALUATIONS = 500
PAL_HYBR_STEP_FACTOR = 0.5
PAL_TRF_MIN_EVALUATIONS = 1500
PAL_TRF_EVALUATION_MULTIPLIER = 5
COUPLING_VERBOSE = True

# Fixed-case aerodynamic plot defaults.
NORMAL_CONTOUR_XLIM = (-0.50, 1.50)
NORMAL_CONTOUR_YLIM = (-0.50, 0.50)
AERODYNAMIC_PNG_DPI = 300
SAVE_FORMATS = ("png", "pdf", "svg")
PNG_DPI = 1000

# =============================================================================
# INTERNAL CHANNEL HYDRAULIC RESISTANCE
# =============================================================================
# Compatibility constant; active friction thresholds are the LOW/HIGH settings.
# Compatibility name only: changing this constant does not change the
# active friction law or introduce a branch-admissibility test at 2300.
INTERNAL_RE_TRANSITION = 2300.0

# Reduced interface model, not a resolved jet/mixing calculation.
# Inlet direction follows the static-pressure difference. Inlet tangential
# kinetic energy is dissipated (zero recovery). The separate bulk outlet jet
# matches exterior static pressure and carries alpha_j*rho*(q'/H)^2/2.
# Set "legacy_static" only for comparison with old results.
# [eq:channel_energy_equation]
# Active balance: |pbar1-pbar2| + <rho*Vn_in^2/2>
#                 = dp_f(|q'|) + alpha_j*rho*(|q'|/H)^2/2.
# JET_KINETIC_ENERGY_FACTOR is alpha_j=1, the uniform bulk-jet factor.
# legacy_static is an alternative compatibility model and omits both heads.
CHANNEL_ENERGY_CLOSURE = "zero_tangential_recovery_jet"
JET_KINETIC_ENERGY_FACTOR = 1.0  # uniform bulk jet; not a fitted loss coefficient

# -----------------------------------------------------------------------------
# NONLINEAR CONTINUATION SETTINGS
# -----------------------------------------------------------------------------
# These settings change only the numerical path used to reach the requested
# physical state.  Every reported solution is still checked at the requested
# AoA, Reynolds number and full coupling strength lambda=1.
ADAPTIVE_AOA_MIN_STEP_DEG = 0.05
RE_CONTINUATION_REFERENCE = 5.0e5
RE_CONTINUATION_INITIAL_DIVISIONS = 5
RE_CONTINUATION_MIN_STEP = 1.0e4
HOMOTOPY_INITIAL_STEP = 0.20
HOMOTOPY_MIN_STEP = 0.025

# Branch-boundary refinement.  These tolerances control only how accurately a
# loss-of-root boundary is bracketed; they do not relax the nonlinear root
# acceptance criterion used for physical results.
AOA_BOUNDARY_TOL_DEG = 0.01
RE_BOUNDARY_TOL = 2.0e3
HOMOTOPY_BOUNDARY_TOL = 2.0e-3

# Pseudo-arclength continuation settings.  The arclength state contains the
# dimensionless passage-flow vector plus alpha/PAL_ALPHA_SCALE_DEG.
PAL_ALPHA_SCALE_DEG = 1.0
PAL_INITIAL_DS = 0.05
PAL_MIN_DS = 0.005
PAL_MAX_DS = 0.20
PAL_MAX_STEPS = 50
PAL_CORRECTOR_TOL = 1.0e-8
PAL_TARGET_TOL_DEG = 0.02
PAL_ALPHA_LIMIT_DEG = 30.0


# =============================================================================
# CONTOURS WORKFLOW SETTINGS
# =============================================================================
# Edit these settings here; the corresponding plotting module imports them.
CONTOUR_AOA_OVERRIDE_DEG = None
PLOT_CONTOUR_FINAL_CONTOUR_AOA_DEG = VISUALISATION_AOA_DEG if CONTOUR_AOA_OVERRIDE_DEG is None else CONTOUR_AOA_OVERRIDE_DEG

CONTOUR_HEIGHT_OVERRIDE_M = None
PLOT_CONTOUR_FINAL_CONTOUR_PORE_DIAMETER_M = VISUALISATION_HEIGHT_M if CONTOUR_HEIGHT_OVERRIDE_M is None else CONTOUR_HEIGHT_OVERRIDE_M

CONTOUR_MODEL_NAMES_OVERRIDE = None  # None -> SELECTED_MODEL; or a tuple of model names
PLOT_CONTOUR_FINAL_CONTOUR_MODEL_NAMES = (
    (AVAILABLE_MODEL_NAMES if SELECTED_MODEL == "all" else (SELECTED_MODEL,))
    if CONTOUR_MODEL_NAMES_OVERRIDE is None else tuple(CONTOUR_MODEL_NAMES_OVERRIDE)
)

PLOT_CONTOUR_MODEL_LABELS = {
    "model_1_4_chordwise": "Horizontal",
    "model_2_9_perpendicular": "Vertical",
    "model_3_combined_independent": "Horizontal & Vertical",
}

PLOT_CONTOUR_MODEL_FOLDER_NAMES = {
    "model_1_4_chordwise": "horizontal",
    "model_2_9_perpendicular": "vertical",
    "model_3_combined_independent": "Horizontal & Vertical",
}

PLOT_CONTOUR_OUTPUT_SUBDIR = "difference_contours"

PLOT_CONTOUR_FINAL_CONTOUR_OUTPUT_ROOT = Path(OUTPUT_ROOT) / PLOT_CONTOUR_OUTPUT_SUBDIR

PLOT_CONTOUR_SAVE_FORMATS = SAVE_FORMATS

PLOT_CONTOUR_PNG_DPI = PNG_DPI

PLOT_CONTOUR_FIELD_NX = int(CONTOUR_NX)

PLOT_CONTOUR_FIELD_NY = int(CONTOUR_NY)

PLOT_CONTOUR_VERBOSE_COUPLING = COUPLING_VERBOSE

# This cropped view makes the airfoil much larger in the final journal figure.
# It is better than the wide far-field view for Results and Discussion figures.
PLOT_CONTOUR_JOURNAL_XLIM = (-0.08, 1.08)

PLOT_CONTOUR_JOURNAL_YLIM = (-0.20, 0.20)

# Optional wider view for supplementary material.
PLOT_CONTOUR_EXPORT_WIDE_SUPPORTING_VIEW = False

PLOT_CONTOUR_WIDE_XLIM = (-0.50, 1.50)

PLOT_CONTOUR_WIDE_YLIM = (-0.50, 0.50)

# Main outputs.
PLOT_CONTOUR_EXPORT_VELOCITY_PERCENT_DIFFERENCE = True

PLOT_CONTOUR_EXPORT_PRESSURE_PA_DIFFERENCE = True

# Additional pressure-contour set:
# gauge pressure = porous static pressure - reference pressure.
PLOT_CONTOUR_EXPORT_REFERENCE_SUBTRACTED_PRESSURE = True

PLOT_CONTOUR_REFERENCE_PRESSURE_PA = P_INF  # gauge-pressure reference

PLOT_CONTOUR_EXPORT_DELTA_CP_DIFFERENCE = True

PLOT_CONTOUR_EXPORT_VELOCITY_MPS_DIFFERENCE = True

# Recommended journal figure size.
# Portrait 3 x 1 layout gives larger airfoils on the page.
PLOT_CONTOUR_PORTRAIT_FIGSIZE = (6.9, 6.2)

PLOT_CONTOUR_PORTRAIT_FIGSIZE_WIDE = (6.9, 7.1)

# Use no figure suptitle for LaTeX papers.
# Put the explanation in the caption instead.
PLOT_CONTOUR_SHOW_FIGURE_SUPTITLE = False

# Panel labels.
PLOT_CONTOUR_PANEL_LABELS = ("(a)", "(b)", "(c)")

PLOT_CONTOUR_PANEL_LABEL_X = 0.018

PLOT_CONTOUR_PANEL_LABEL_Y = 0.875

PLOT_CONTOUR_DIFFERENCE_CMAP = "coolwarm"

PLOT_CONTOUR_DIFFERENCE_LEVELS = 101

PLOT_CONTOUR_SYMMETRIC_LIMIT_PERCENTILE = 99.3

PLOT_CONTOUR_MIN_SYMMETRIC_LIMIT = 1.0e-12

# Use one global limit across all three models for each field.
# This is important for fair comparison.
PLOT_CONTOUR_USE_COMMON_COLOUR_LIMITS = True

# Optional manual colour limits. Set to None for automatic robust limits.
PLOT_CONTOUR_MANUAL_VELOCITY_PERCENT_LIMIT = None

PLOT_CONTOUR_MANUAL_PRESSURE_PA_LIMIT = None

PLOT_CONTOUR_MANUAL_REFERENCE_SUBTRACTED_PRESSURE_LIMIT = None

PLOT_CONTOUR_MANUAL_DELTA_CP_LIMIT = None

PLOT_CONTOUR_MANUAL_VELOCITY_MPS_LIMIT = None

PLOT_CONTOUR_AIRFOIL_FACE_COLOR = "white"

PLOT_CONTOUR_AIRFOIL_EDGE_COLOR = "black"

PLOT_CONTOUR_AIRFOIL_EDGE_WIDTH = 0.90

PLOT_CONTOUR_DRAW_INTERNAL_NETWORK = True

PLOT_CONTOUR_NETWORK_CHANNEL_COLOR = "#222222"

PLOT_CONTOUR_NETWORK_CHANNEL_LINEWIDTH = 0.62

PLOT_CONTOUR_NETWORK_CHANNEL_ALPHA = 0.78

PLOT_CONTOUR_DRAW_PORE_MARKERS = False

PLOT_CONTOUR_NETWORK_PORE_FACE_COLOR = "white"

PLOT_CONTOUR_NETWORK_PORE_EDGE_COLOR = "black"

PLOT_CONTOUR_NETWORK_PORE_SIZE = 8.0

PLOT_CONTOUR_RCPARAMS = {
            "figure.dpi": 160,
            "savefig.dpi": PLOT_CONTOUR_PNG_DPI,
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

# =============================================================================
# STREAMLINES WORKFLOW SETTINGS
# =============================================================================
# Edit these settings here; the corresponding plotting module imports them.
STREAMLINE_AOA_OVERRIDE_DEG = None
PLOT_STREAMLINE_STREAMLINE_AOA_DEG = VISUALISATION_AOA_DEG if STREAMLINE_AOA_OVERRIDE_DEG is None else STREAMLINE_AOA_OVERRIDE_DEG

STREAMLINE_HEIGHT_OVERRIDE_M = None
PLOT_STREAMLINE_STREAMLINE_PORE_DIAMETER_M = VISUALISATION_HEIGHT_M if STREAMLINE_HEIGHT_OVERRIDE_M is None else STREAMLINE_HEIGHT_OVERRIDE_M

STREAMLINE_MODEL_NAMES_OVERRIDE = None  # None -> SELECTED_MODEL; or a tuple of model names
PLOT_STREAMLINE_STREAMLINE_MODEL_NAMES = (
    (AVAILABLE_MODEL_NAMES if SELECTED_MODEL == "all" else (SELECTED_MODEL,))
    if STREAMLINE_MODEL_NAMES_OVERRIDE is None else tuple(STREAMLINE_MODEL_NAMES_OVERRIDE)
)

PLOT_STREAMLINE_MODEL_LABELS = {
    "model_1_4_chordwise": "Horizontal",
    "model_2_9_perpendicular": "Vertical",
    "model_3_combined_independent": "Isotropic",
    "model_4_saved_case_1": "Saved case",
}

PLOT_STREAMLINE_MODEL_FOLDER_NAMES = {
    "model_1_4_chordwise": "horizontal",
    "model_2_9_perpendicular": "vertical",
    "model_3_combined_independent": "isotropic",
    "model_4_saved_case_1": "saved_case",
}

PLOT_STREAMLINE_EXPORT_NORMAL_VIEW = False
PLOT_STREAMLINE_EXPORT_CLOSE_UP_VIEW = True

PLOT_STREAMLINE_NORMAL_XLIM = (-0.50, 1.50)

PLOT_STREAMLINE_NORMAL_YLIM = (-0.50, 0.50)

PLOT_STREAMLINE_CLOSE_UP_XLIM = ZOOMED_CONTOUR_XLIM

PLOT_STREAMLINE_CLOSE_UP_YLIM = ZOOMED_CONTOUR_YLIM

PLOT_STREAMLINE_FIELD_NX = int(CONTOUR_NX)

PLOT_STREAMLINE_FIELD_NY = int(CONTOUR_NY)

PLOT_STREAMLINE_STREAMLINE_OUTPUT_ROOT = Path(OUTPUT_ROOT) / "streamline_results"

PLOT_STREAMLINE_SAVE_FORMATS = SAVE_FORMATS

PLOT_STREAMLINE_PNG_DPI = PNG_DPI

PLOT_STREAMLINE_VERBOSE_COUPLING = COUPLING_VERBOSE

# Generate external streamlines from pore inlet and outlet locations.
PLOT_STREAMLINE_DRAW_PORE_MOUTH_EXTERNAL_STREAMLINES = True

# Use the same pore-adjacent seed locations on the solid comparison panel.
PLOT_STREAMLINE_DRAW_PORE_MOUTH_EXTERNAL_STREAMLINES_ON_SOLID = True

# Options: "inlet", "outlet", or "both".
PLOT_STREAMLINE_PORE_MOUTH_EXTERNAL_SEED_ROLES = "outlet"

# One external streamline per pore mouth.
# For three nearby streamlines per pore, use:
# (-0.0015, 0.0, 0.0015)
PLOT_STREAMLINE_PORE_MOUTH_TANGENTIAL_OFFSETS_C = (0.0,)

# Prevent excessive streamline density in the isotropic model.
PLOT_STREAMLINE_MAX_PORE_MOUTH_EXTERNAL_SEEDS = 64

PLOT_STREAMLINE_MAKE_STREAMLINE_PLOTS = ENABLE_PLOTS and RUN_STREAMLINES

PLOT_STREAMLINE_DRAW_EXTERNAL_STREAMLINES = True

PLOT_STREAMLINE_DRAW_INTERNAL_CHANNEL_STREAMLINES = True

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
PLOT_STREAMLINE_INTERNAL_STREAMLINE_SEED_ROLES = "inlet"

PLOT_STREAMLINE_EXPORT_PORE_LOCATION_CSV = True

# External streamlines: solid black
PLOT_STREAMLINE_EXTERNAL_STREAMLINE_COLOR = "black"

PLOT_STREAMLINE_EXTERNAL_STREAMLINE_LINEWIDTH = 0.32

PLOT_STREAMLINE_EXTERNAL_STREAMLINE_ALPHA = 1.0

PLOT_STREAMLINE_EXTERNAL_STREAMLINE_LINESTYLE = "-"

# Internal streamlines: dashed black
PLOT_STREAMLINE_INTERNAL_STREAMLINE_COLOR = "black"

PLOT_STREAMLINE_INTERNAL_STREAMLINE_LINEWIDTH = 0.32

PLOT_STREAMLINE_INTERNAL_STREAMLINE_ALPHA = 1.0

PLOT_STREAMLINE_INTERNAL_STREAMLINE_LINESTYLE = "--"

# Airfoil style
PLOT_STREAMLINE_AIRFOIL_FILL = "#D9D9D9"   # light grey

PLOT_STREAMLINE_AIRFOIL_EDGE = "black"

PLOT_STREAMLINE_AIRFOIL_EDGE_WIDTH = 0.70

PLOT_STREAMLINE_EXTERNAL_STREAMLINE_COUNT_NORMAL = 75

PLOT_STREAMLINE_EXTERNAL_STREAMLINE_COUNT_CLOSE = 25

PLOT_STREAMLINE_STREAMLINE_START_X_OFFSET_C = 0.015

PLOT_STREAMLINE_STREAMLINE_Y_MARGIN_C = 0.015

PLOT_STREAMLINE_STREAMLINE_MAX_STEP_C = 0.004

PLOT_STREAMLINE_STREAMLINE_RTOL = 1.0e-6

PLOT_STREAMLINE_STREAMLINE_ATOL = 1.0e-8

PLOT_STREAMLINE_STREAMLINE_MIN_POINTS = 5

PLOT_STREAMLINE_STREAMLINE_MIN_LENGTH_C = 0.015

# Maximum integration distance in multiples of the plotting-window diagonal.
PLOT_STREAMLINE_STREAMLINE_MAX_DISTANCE_FACTOR = 3.0

# The channel corridor is used only for approximate internal streamline
# visualisation. It allows the continuous integrator to pass through the body
# along the internal hydraulic path.
PLOT_STREAMLINE_INTERNAL_CHANNEL_CORRIDOR_RADIUS_FACTOR = 0.55

PLOT_STREAMLINE_INTERNAL_CHANNEL_MIN_RADIUS_C = 0.002

# Extend the internal channel corridor slightly beyond the two pore mouths.
# This helps exact pore-mouth seeds connect to the internal corridor.
PLOT_STREAMLINE_INTERNAL_CHANNEL_END_CAP_FACTOR = 1.20

# "plug" or "parabolic"
# [subsec:model_overview; eq:laminar_channel_profile]
# This selects a plotting profile only. It does not replace the hydraulic
# resistance or solve the laminar velocity-profile differential equation.
PLOT_STREAMLINE_INTERNAL_CHANNEL_PROFILE = "plug"

PLOT_STREAMLINE_INTERNAL_CHANNEL_MIN_AREA = 1.0e-14

PLOT_STREAMLINE_INTERNAL_CHANNEL_FALLBACK_DIAMETER_M = PLOT_STREAMLINE_STREAMLINE_PORE_DIAMETER_M

# These are used only in the CSV. Internal streamline seeds are placed exactly
# on the pore mouth. External-offset seed locations are also saved for reference.
PLOT_STREAMLINE_PORE_EXTERNAL_SEED_OFFSET_C_NORMAL = 0.006

PLOT_STREAMLINE_PORE_EXTERNAL_SEED_OFFSET_C_CLOSE = 0.004

# Streamline integration direction options:
#   "forward"  = integrate along the local velocity direction
#   "backward" = integrate opposite to the local velocity direction
#   "both"     = integrate forward and backward, then join both branches
#
# For left-boundary farfield seeds, "forward" is recommended.
PLOT_STREAMLINE_FARFIELD_STREAMLINE_DIRECTION = "forward"

# For pore inlet/outlet seeds, "both" normally gives the clearest full line.
PLOT_STREAMLINE_PORE_MOUTH_STREAMLINE_DIRECTION = "forward"

PLOT_STREAMLINE_DRAW_STREAMLINE_ARROWS = True

# Number of arrows per streamline.
# Use 1 for clean figures. Use 2 if the streamlines are long.
PLOT_STREAMLINE_STREAMLINE_ARROW_COUNT = 1

# Arrow positions along each streamline.
# Used only when STREAMLINE_ARROW_COUNT = 1.
PLOT_STREAMLINE_STREAMLINE_ARROW_POSITION_FRACTION = 0.58

# Arrow size in plot coordinates.
PLOT_STREAMLINE_STREAMLINE_ARROW_LENGTH_C = 0.030

PLOT_STREAMLINE_STREAMLINE_ARROW_MUTATION_SCALE = 6.0

# Skip arrows on very short streamlines.
PLOT_STREAMLINE_STREAMLINE_ARROW_MIN_LINE_LENGTH_C = 0.050

PLOT_STREAMLINE_DRAW_INTERNAL_FLOW_DIRECTION_ARROWS = True

# Position of the arrow along each internal channel.
# 0.50 means centre of the internal channel.
PLOT_STREAMLINE_INTERNAL_ARROW_POSITION_FRACTION = 0.50

# Arrow length in chord-normalised coordinates.
PLOT_STREAMLINE_INTERNAL_ARROW_LENGTH_C = 0.040

# Use slightly darker grey than the streamlines so the direction is visible.
PLOT_STREAMLINE_INTERNAL_ARROW_COLOR = "black"

PLOT_STREAMLINE_INTERNAL_ARROW_LINEWIDTH = 0.50

PLOT_STREAMLINE_INTERNAL_ARROW_MUTATION_SCALE = 8.0

PLOT_STREAMLINE_RCPARAMS = {
            "figure.dpi": 160,
            "savefig.dpi": PLOT_STREAMLINE_PNG_DPI,
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

# =============================================================================
# GEOMETRY WORKFLOW SETTINGS
# =============================================================================
# Edit these settings here; the corresponding plotting module imports them.
GEOMETRY_AOA_OVERRIDE_DEG = None
PLOT_GEOMETRY_EXPORT_AOA_DEG = VISUALISATION_AOA_DEG if GEOMETRY_AOA_OVERRIDE_DEG is None else GEOMETRY_AOA_OVERRIDE_DEG

GEOMETRY_HEIGHT_OVERRIDE_M = None
PLOT_GEOMETRY_EXPORT_PORE_DIAMETER_M = VISUALISATION_HEIGHT_M if GEOMETRY_HEIGHT_OVERRIDE_M is None else GEOMETRY_HEIGHT_OVERRIDE_M

GEOMETRY_MODEL_NAMES_OVERRIDE = None  # None -> SELECTED_MODEL; or a tuple of model names
PLOT_GEOMETRY_MODELS_TO_EXPORT = (
    (AVAILABLE_MODEL_NAMES if SELECTED_MODEL == "all" else (SELECTED_MODEL,))
    if GEOMETRY_MODEL_NAMES_OVERRIDE is None else tuple(GEOMETRY_MODEL_NAMES_OVERRIDE)
)

PLOT_GEOMETRY_MODEL_LABELS = {
    "model_1_4_chordwise": "Model 1: Horizontal channels",
    "model_2_9_perpendicular": "Model 2: Vertical channels",
    "model_3_combined_independent": "Model 3: Isotropic channels",
}

PLOT_GEOMETRY_MODEL_SHORT_LABELS = {
    "model_1_4_chordwise": "Horizontal",
    "model_2_9_perpendicular": "Vertical",
    "model_3_combined_independent": "Isotropic",
}

PLOT_GEOMETRY_MODEL_FOLDER_NAMES = {
    "model_1_4_chordwise": "model_1_horizontal",
    "model_2_9_perpendicular": "model_2_vertical",
    "model_3_combined_independent": "model_3_isotropic",
}

PLOT_GEOMETRY_EXPORT_ROOT = Path(OUTPUT_ROOT) / "porous_models"

PLOT_GEOMETRY_SAVE_FORMATS = SAVE_FORMATS

PLOT_GEOMETRY_PNG_DPI = PNG_DPI

PLOT_GEOMETRY_FIGURE_XLIM = (-0.03, 1.03)

PLOT_GEOMETRY_FIGURE_YLIM = (-0.16, 0.16)

PLOT_GEOMETRY_AIRFOIL_FACE_COLOR = "white"

PLOT_GEOMETRY_AIRFOIL_EDGE_COLOR = "black"

PLOT_GEOMETRY_AIRFOIL_EDGE_WIDTH = 1.00

PLOT_GEOMETRY_CHANNEL_COLOR = "#606060"

PLOT_GEOMETRY_CHANNEL_WIDTH = 0.95

PLOT_GEOMETRY_CHANNEL_ALPHA = 0.95

PLOT_GEOMETRY_PORE_FACE_COLOR = "white"

PLOT_GEOMETRY_PORE_EDGE_COLOR = "black"

PLOT_GEOMETRY_PORE_EDGE_WIDTH = 0.55

PLOT_GEOMETRY_SHOW_PORE_CIRCLES = True

PLOT_GEOMETRY_SHOW_PORE_CENTRES = True

PLOT_GEOMETRY_PORE_CENTRE_MARKER_SIZE = 5.0

PLOT_GEOMETRY_SHOW_CHANNEL_LABELS = False

PLOT_GEOMETRY_SHOW_TITLES = True

PLOT_GEOMETRY_RCPARAMS = {
            "figure.dpi": 160,
            "savefig.dpi": PLOT_GEOMETRY_PNG_DPI,
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
            "figure.titlesize": 12.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }

# =============================================================================
# AERODYNAMIC WORKFLOW SETTINGS
# =============================================================================
# Edit these settings here; the corresponding plotting module imports them.
# Set an exact sweep CSV here when possible. When None, the script searches
# recursively inside SEARCH_ROOT.
PLOT_AERO_INPUT_CSV_PATH: Path | None = None

PLOT_AERO_OUTPUT_DIR = OUTPUT_ROOT / "aero_graphs_three_diameters"

# Aerodynamic comparisons inherit the actual sweep heights.
AERO_PLOT_HEIGHTS_OVERRIDE_M = None  # None -> AOA_SWEEP_CHANNEL_HEIGHTS
PLOT_AERO_SELECT_PORE_DIAMETERS_M = AOA_SWEEP_CHANNEL_HEIGHTS if AERO_PLOT_HEIGHTS_OVERRIDE_M is None else tuple(AERO_PLOT_HEIGHTS_OVERRIDE_M)

# Retain the original validation/reference curves.
PLOT_AERO_SHOW_SOLID_REFERENCE = True

PLOT_AERO_SHOW_XFOIL_REFERENCE = True

PLOT_AERO_SAVE_FORMATS = SAVE_FORMATS

PLOT_AERO_PNG_DPI = PNG_DPI

# False gives colour curves that are also differentiated by line type.
# True makes all curves black while preserving line-style differences.
PLOT_AERO_MONOCHROME_MODE = False

# Three panels in one journal-width row.
PLOT_AERO_THREE_PANEL_FIGSIZE = (7.5, 3.15)

# Use a common y-axis across the three models for direct comparison.
PLOT_AERO_SHARE_Y_AXIS = True

PLOT_AERO_CASE_LABELS = {
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

PLOT_AERO_MODEL_PANELS = tuple(
    (PLOT_AERO_CASE_LABELS[name], f"({chr(97+i)}) {PLOT_AERO_CASE_LABELS[name]}")
    for i, name in enumerate(AVAILABLE_MODEL_NAMES if SELECTED_MODEL == "all" else (SELECTED_MODEL,))
)
PLOT_AERO_METRICS = ("CL", "CD", "CM")

# Diameter is encoded by line type only. Colour is an additional aid.
PLOT_AERO_DIAMETER_LINESTYLES = ("-", "--", "-.")

PLOT_AERO_DIAMETER_COLOURS = ("#0072B2", "#D55E00", "#009E73")

PLOT_AERO_REFERENCE_STYLES = {
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

PLOT_AERO_WINDOWS_FONT_DIR = Path("/mnt/c/Windows/Fonts")

PLOT_AERO_SEARCH_ROOT = OUTPUT_ROOT
PLOT_AERO_RCPARAMS = {
            "figure.dpi": 160,
            "savefig.dpi": PLOT_AERO_PNG_DPI,
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

# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass(frozen=True)
class GeometryLimits:
    # Allow exact LE and TE for the middle Model 1 channel.
    x_min_frac: float = 0.0
    x_max_frac: float = 1.0

    thickness_fraction_limit: float = GEOMETRY_THICKNESS_FRACTION_LIMIT
    min_gap: float = GEOMETRY_MIN_GAP_M
    fit_fraction: float = GEOMETRY_FIT_FRACTION


    # Exact endpoint handling.
    endpoint_tol: float = GEOMETRY_ENDPOINT_TOL

    # Skip the zero-thickness endpoint region when checking if the internal
    # channel fits inside the airfoil body.
    endpoint_fit_margin: float = GEOMETRY_ENDPOINT_FIT_MARGIN

N_CRIT: float = 9.0  # Amplification factor threshold for e^N transition

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
# [eq:channel_flow_vector; eq:channel_reynolds]
# Q stores signed q'_k [m^2/s], NOT volumetric Q [m^3/s].
# Rs=|pbar1-pbar2|/|Q| is a static-pressure diagnostic. With kinetic heads,
# it is not generally the laminar distributed-friction resistance R'_k.
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
    coupling_status: str = "unknown"
    residual_norm: float = float("nan")
    residual_tolerance: float = float("nan")
    solver_success: bool = False
    solver_message: str = ""
    function_evaluations: int = 0
    newton_iterations: int = 0
    jacobian_condition_number: float = float("nan")
    jacobian_smallest_singular_value: float = float("nan")
    reduced_residual_norm: float = float("nan")
    original_energy_residual_max_pa: float = float("nan")


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

    # External input conversion: Re_c=rho*V_inf*c/mu.
    return reynolds * mu / (rho * chord)


# [eq:hydraulic_diameter]
# Parallel-wall limit D_h=4*A'/P'=4*H/2=2H; no finite-width correction.
def rectangular_hydraulic_diameter(
    height_or_width: float,
    height: float | None = None,
) -> float:
    """Return the 2D parallel-wall hydraulic diameter, D_h = 2H."""
    h = float(height_or_width if height is None else height)
    if h <= 0.0:
        raise ValueError("Rectangular channel height must be positive.")
    return float(2.0 * h)


# [eq:laminar_channel_momentum; eq:laminar_channel_profile;
# eq:hagen_poiseuille_channel; eq:laminar_pressure_flow; eq:laminar_resistance]
# The paper derives u(y)=dp_f*y*(H-y)/(2*mu*L) and integrates it to
# |q'|=H^3*dp_f/(12*mu*L). Only the resulting R'=12*mu*L/H^3 is
# evaluated here: no ODE or resolved internal velocity field is solved.
# The active friction law in porous.py has this same laminar limit.
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

# =============================================================================
# INPUT VALIDATION
# =============================================================================
def validate_config() -> None:
    """Catch invalid inputs before starting a potentially expensive workflow."""
    def positive(name: str, value: float) -> None:
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")

    for name in ("RUN_FIXED_CASES", "RUN_AOA_SWEEP", "RUN_AERODYNAMIC_PLOTS",
                 "RUN_GEOMETRY", "RUN_CONTOURS", "RUN_STREAMLINES", "ENABLE_PLOTS",
                 "CONTINUE_AFTER_STAGE_ERROR", "MAKE_CP_PLOTS", "MAKE_NETWORK_PLOTS",
                 "MAKE_AOA_SWEEP_PLOTS", "EXPORT_SURFACE_DATA", "EXPORT_PASSAGE_DATA",
                 "PRINT_AERODYNAMIC_METRICS", "SAVE_AERODYNAMIC_METRICS",
                 "PRINT_METRICS_PER_CASE", "PRINT_ELAPSED_TIME", "SAVE_TIMING_REPORT"):
        if not isinstance(globals()[name], bool):
            raise ValueError(f"{name} must be True or False.")
    for name in ("METRICS_MIN_ABS_BASELINE", "METRICS_MIN_POSITIVE_CD"):
        positive(name, globals()[name])
    if METRICS_PRINT_AOA_DEG is not None and not np.isfinite(METRICS_PRINT_AOA_DEG):
        raise ValueError("METRICS_PRINT_AOA_DEG must be finite or None.")
    if (not PORE_HEIGHTS_M or len(set(PORE_HEIGHTS_M)) != len(PORE_HEIGHTS_M)):
        raise ValueError("PORE_HEIGHTS_M must contain distinct positive heights.")
    for height in PORE_HEIGHTS_M:
        positive("PORE_HEIGHTS_M", height)
    if not PLOT_AERO_METRICS or any(key not in ("CL", "CD", "CM") for key in PLOT_AERO_METRICS):
        raise ValueError("PLOT_AERO_METRICS must select CL, CD and/or CM.")
    if RUN_STREAMLINES and not (PLOT_STREAMLINE_EXPORT_NORMAL_VIEW or PLOT_STREAMLINE_EXPORT_CLOSE_UP_VIEW):
        raise ValueError("Enable at least one streamline view when RUN_STREAMLINES=True.")
    if not np.isfinite(GEOMETRY_MIN_GAP_M) or GEOMETRY_MIN_GAP_M < 0.0:
        raise ValueError("GEOMETRY_MIN_GAP_M must be finite and nonnegative.")
    if not 0.0 < GEOMETRY_FIT_FRACTION <= 1.0 or not 0.0 < GEOMETRY_THICKNESS_FRACTION_LIMIT <= 1.0:
        raise ValueError("Geometry fit/thickness fractions must lie in (0, 1].")

    for name in ("CHORD", "REYNOLDS_EXTERNAL", "RHO_INF", "MU_INF",
                 "CHANNEL_HEIGHT", "SPAN_WIDTH", "XFOIL_TIMEOUT"):
        positive(name, globals()[name])
    if not isinstance(N_PANELS, int) or N_PANELS < 10 or N_PANELS % 2:
        raise ValueError("N_PANELS must be an even integer of at least 10.")
    code = str(AIRFOIL_NAME).lower().replace("naca", "").replace(" ", "").replace("-", "")
    if len(code) != 4 or not code.isdigit():
        raise ValueError("AIRFOIL_NAME must be a NACA four-digit identifier.")
    if SELECTED_MODEL not in ("all", *AVAILABLE_MODEL_NAMES):
        raise ValueError(f"Unsupported SELECTED_MODEL: {SELECTED_MODEL}")
    if not all(np.isfinite(value) for value in (AOA_DEG, P_INF,
            MOMENT_REFERENCE_X_C, MOMENT_REFERENCE_Y_C,
            AOA_SWEEP_START_DEG, AOA_SWEEP_END_DEG, AOA_SWEEP_STEP_DEG)):
        raise ValueError("Flow, reference and sweep values must be finite.")
    if AOA_SWEEP_STEP_DEG <= 0.0 or AOA_SWEEP_END_DEG < AOA_SWEEP_START_DEG:
        raise ValueError("The AoA sweep requires a positive step and end >= start.")
    for name, heights in (("FIXED_AOA_CHANNEL_HEIGHTS", FIXED_AOA_CHANNEL_HEIGHTS),
                         ("AOA_SWEEP_CHANNEL_HEIGHTS", AOA_SWEEP_CHANNEL_HEIGHTS),
                         ("PLOT_AERO_SELECT_PORE_DIAMETERS_M", PLOT_AERO_SELECT_PORE_DIAMETERS_M)):
        if not heights:
            raise ValueError(f"{name} must not be empty.")
        for height in heights:
            positive(name, height)
    for name, angle, height, models in (
        ("contours", PLOT_CONTOUR_FINAL_CONTOUR_AOA_DEG,
            PLOT_CONTOUR_FINAL_CONTOUR_PORE_DIAMETER_M, PLOT_CONTOUR_FINAL_CONTOUR_MODEL_NAMES),
        ("streamlines", PLOT_STREAMLINE_STREAMLINE_AOA_DEG,
            PLOT_STREAMLINE_STREAMLINE_PORE_DIAMETER_M, PLOT_STREAMLINE_STREAMLINE_MODEL_NAMES),
        ("geometry", PLOT_GEOMETRY_EXPORT_AOA_DEG,
            PLOT_GEOMETRY_EXPORT_PORE_DIAMETER_M, PLOT_GEOMETRY_MODELS_TO_EXPORT),
    ):
        if not np.isfinite(angle):
            raise ValueError(f"{name} AoA must be finite.")
        positive(f"{name} height", height)
        if not models or len(set(models)) != len(models) or any(m not in AVAILABLE_MODEL_NAMES for m in models):
            raise ValueError(f"{name} requires distinct supported model names.")
    if not isinstance(MODEL1_CHANNEL_COUNT, int) or MODEL1_CHANNEL_COUNT < 1:
        raise ValueError("MODEL1_CHANNEL_COUNT must be a positive integer.")
    if not 0.0 < MODEL1_MAX_Y_FRACTION < 1.0:
        raise ValueError("MODEL1_MAX_Y_FRACTION must lie between 0 and 1.")
    if not MODEL2_X_LOCATIONS or any(not np.isfinite(x) or not 0.0 < x < 1.0 for x in MODEL2_X_LOCATIONS):
        raise ValueError("MODEL2_X_LOCATIONS must contain positions strictly inside (0, 1).")
    if len(set(MODEL2_X_LOCATIONS)) != len(MODEL2_X_LOCATIONS):
        raise ValueError("MODEL2_X_LOCATIONS must be distinct.")
    if not (0.0 < HYDRAULIC_TRANSITION_RE_LOW < HYDRAULIC_TRANSITION_RE_HIGH):
        raise ValueError("Hydraulic transition limits require 0 < LOW < HIGH.")
    if CHANNEL_ENERGY_CLOSURE not in {"legacy_static", "zero_tangential_recovery_jet"}:
        raise ValueError("Unsupported CHANNEL_ENERGY_CLOSURE.")
    if not np.isfinite(JET_KINETIC_ENERGY_FACTOR) or JET_KINETIC_ENERGY_FACTOR < 1.0:
        raise ValueError("JET_KINETIC_ENERGY_FACTOR must be finite and >= 1.")
    for name, count in (("COUPLING.max_iter", COUPLING.max_iter),
                        ("CONTOUR_NX", CONTOUR_NX), ("CONTOUR_NY", CONTOUR_NY)):
        if not isinstance(count, int) or count < (2 if name.startswith("CONTOUR") else 1):
            raise ValueError(f"{name} must be a sufficiently large positive integer.")
    for name in ("TRF_FTOL", "TRF_XTOL", "TRF_GTOL"):
        positive(name, globals()[name])
    if PLOT_STREAMLINE_INTERNAL_CHANNEL_PROFILE not in {"plug", "parabolic"}:
        raise ValueError("Internal streamline profile must be 'plug' or 'parabolic'.")
    for name in ("PLOT_STREAMLINE_FARFIELD_STREAMLINE_DIRECTION", "PLOT_STREAMLINE_PORE_MOUTH_STREAMLINE_DIRECTION"):
        if globals()[name] not in {"forward", "backward", "both"}:
            raise ValueError(f"Invalid streamline direction: {name}")
    for name in ("PLOT_STREAMLINE_PORE_MOUTH_EXTERNAL_SEED_ROLES", "PLOT_STREAMLINE_INTERNAL_STREAMLINE_SEED_ROLES"):
        if globals()[name] not in {"inlet", "outlet", "both"}:
            raise ValueError(f"Invalid seed role: {name}")


validate_config()

def find_xfoil_executable() -> Path | None:
    """Find an XFOIL executable in a portable, GitHub-friendly way.

    Search order:
        1. XFOIL_EXECUTABLE configured above
        2. xfoil or xfoil.exe on system PATH
        3. Common subfolders inside XFOIL_FOLDER
        4. Recursive search inside XFOIL_FOLDER

    Returns
    -------
    Path | None
        Absolute executable path when found; otherwise None.
    """
    if XFOIL_EXECUTABLE is not None:
        explicit_path = Path(XFOIL_EXECUTABLE).expanduser()
        if not explicit_path.is_file():
            raise ValueError(f"XFOIL_EXECUTABLE is not a file: {explicit_path}")
        return explicit_path.resolve()

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
        if candidate.is_file():
            return candidate.resolve()

    # Recursive discovery is enabled only by the setting above.
    if XFOIL_RECURSIVE_SEARCH and XFOIL_FOLDER.exists():
        for pattern in ("xfoil", "xfoil.exe"):
            matches = list(XFOIL_FOLDER.rglob(pattern))
            if matches:
                return matches[0].resolve()

    return None


XFOIL_EXE_PATH = find_xfoil_executable() if USE_XFOIL else None

if USE_XFOIL and XFOIL_EXE_PATH is None:
    print("=" * 80)
    print("[XFOIL WARNING]")
    print("Could not find xfoil/xfoil.exe. XFOIL comparisons will be skipped.")
    print("Set XFOIL_EXECUTABLE or XFOIL_FOLDER in config.py to enable XFOIL.")
    print(f"Checked folder: {XFOIL_FOLDER}")
    print("=" * 80)
    USE_XFOIL = False
elif USE_XFOIL:
    print(f"[XFOIL] Using executable: {XFOIL_EXE_PATH}")

