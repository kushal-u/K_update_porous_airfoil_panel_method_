"""Porous-airfoil coupling with two-dimensional rectangular channels.

The internal passages are rectangular slots in the airfoil cross-section and
are solved per unit span.  The flow variable is therefore q [m^2/s], although
the legacy ``PassageState.Q`` attribute is retained for compatibility with the
existing workflow and plotting code.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import time

from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd

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
    AOA_SWEEP_PORE_DIAMETERS,
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
    reynolds_to_velocity,
)
from .integrations.paraview import export_aoa_sweep_collections, export_paraview_files

# =============================================================================
# TWO-DIMENSIONAL RECTANGULAR-SLOT HYDRAULICS
# =============================================================================
INTERNAL_RE_TRANSITION = 2300.0
MIN_OPENING_NORMAL_COMPONENT = 5.0e-2
OPENING_TOLERANCE = 1.0e-10


def _slot_darcy_friction_factor(reynolds: float) -> tuple[float, str]:
    """Return the equivalent Darcy friction factor for a 2D slot.

    Laminar flow is governed directly by the plane-Poiseuille equation.  The
    value ``96/Re`` is retained only as its equivalent Darcy friction factor
    for reporting and force checks.  Turbulent flow uses the smooth-slot
    Blasius approximation.
    """
    re = float(reynolds)
    if re <= 1.0e-12:
        return float("inf"), "zero_flow"
    if re < INTERNAL_RE_TRANSITION:
        return 96.0 / re, "laminar_plane_poiseuille"
    return 0.3164 / re**0.25, "turbulent_blasius_smooth_slot"

def _rectangular_slot_pressure_drop_abs(
    q_abs: float,
    rho: float,
    mu: float,
    length: float,
    height: float,
) -> tuple[float, float, float, str, float, float, float]:
    """Return pressure loss and derived quantities for ``|q|`` per unit span.

    For ``Re < 2300`` the pressure loss is evaluated directly from the
    plane-Poiseuille equation

        dp = 12 * mu * L * q / h**3.

    For ``Re >= 2300`` Darcy-Weisbach with the Blasius friction factor is used.
    """
    q_abs = abs(float(q_abs))
    rho = float(rho)
    mu = float(mu)
    height = float(height)
    length = float(length)

    if rho <= 0.0:
        raise ValueError("Fluid density must be positive.")
    if mu <= 0.0:
        raise ValueError("Dynamic viscosity must be positive.")
    if height <= 0.0:
        raise ValueError("Rectangular channel height must be positive.")
    if length <= 0.0:
        raise ValueError("Rectangular channel length must be positive.")
    if q_abs <= 0.0:
        return 0.0, 0.0, float("inf"), "zero_flow", 0.0, 0.0, 0.0

    mean_speed = q_abs / height
    hydraulic_diameter = 2.0 * height
    reynolds = rho * mean_speed * hydraulic_diameter / mu

    if reynolds < INTERNAL_RE_TRANSITION:
        # Plane Poiseuille flow between two parallel walls separated by h.
        dp_abs = 12.0 * mu * length * q_abs / height**3
        friction_factor = 96.0 / max(reynolds, 1.0e-30)
        model_name = "laminar_plane_poiseuille"
        wall_shear = 6.0 * mu * mean_speed / height
    else:
        friction_factor = 0.3164 / reynolds**0.25
        model_name = "turbulent_blasius_smooth_slot"
        dynamic_pressure = 0.5 * rho * mean_speed**2
        dp_abs = friction_factor * (length / hydraulic_diameter) * dynamic_pressure
        wall_shear = friction_factor * rho * mean_speed**2 / 8.0

    # Two wetted walls, each of length L, per unit span.
    wall_force_per_span = 2.0 * wall_shear * length

    return (
        float(dp_abs),
        float(reynolds),
        float(friction_factor),
        str(model_name),
        float(mean_speed),
        float(wall_shear),
        float(wall_force_per_span),
    )

def solve_rectangular_slot_flow(
    dp: float,
    rho: float,
    mu: float,
    length: float,
    height: float,
) -> tuple[float, float, float, float, str, float, float, float]:
    """Solve the rectangular-slot flow rate per unit span from ``p1-p2``.

    The laminar candidate is obtained directly from plane Poiseuille flow.  It
    is accepted when its Reynolds number is below 2300.  Only the turbulent
    branch requires a nonlinear bisection solve.

    Returns
    -------
    tuple
        ``q, equivalent_R, Re, f_D, model, U, tau_w, F_wall_per_span``.
    """
    dp = float(dp)
    dp_abs = abs(dp)
    rho = float(rho)
    mu = float(mu)
    length = float(length)
    height = float(height)

    if rho <= 0.0:
        raise ValueError("Fluid density must be positive.")
    if mu <= 0.0:
        raise ValueError("Dynamic viscosity must be positive.")
    if length <= 0.0:
        raise ValueError("Rectangular channel length must be positive.")
    if height <= 0.0:
        raise ValueError("Rectangular channel height must be positive.")
    if dp_abs <= 0.0:
        return (
            0.0, float("inf"), 0.0, float("inf"), "zero_flow",
            0.0, 0.0, 0.0,
        )

    sign = 1.0 if dp >= 0.0 else -1.0

    # ------------------------------------------------------------------
    # Laminar candidate: direct plane-Poiseuille solution.
    # q = dp*h^3/(12*mu*L), where q is flow per unit span [m^2/s].
    # ------------------------------------------------------------------
    poiseuille_resistance = 12.0 * mu * length / height**3
    q_poiseuille_abs = dp_abs / poiseuille_resistance
    mean_speed_poiseuille = q_poiseuille_abs / height
    hydraulic_diameter = 2.0 * height
    reynolds_poiseuille = (
        rho * mean_speed_poiseuille * hydraulic_diameter / mu
    )

    if reynolds_poiseuille < INTERNAL_RE_TRANSITION:
        friction_factor = 96.0 / max(reynolds_poiseuille, 1.0e-30)
        wall_shear = 6.0 * mu * mean_speed_poiseuille / height
        wall_force_per_span = 2.0 * wall_shear * length

        return (
            float(sign * q_poiseuille_abs),
            float(poiseuille_resistance),
            float(reynolds_poiseuille),
            float(friction_factor),
            "laminar_plane_poiseuille",
            float(sign * mean_speed_poiseuille),
            float(wall_shear),
            float(wall_force_per_span),
        )

    # ------------------------------------------------------------------
    # Turbulent branch: Darcy-Weisbach + Blasius, solved by bisection.
    # ------------------------------------------------------------------
    def turbulent_pressure_drop(q_abs: float) -> tuple[float, float, float, float, float]:
        mean_speed = q_abs / height
        reynolds = rho * mean_speed * hydraulic_diameter / mu
        friction_factor = 0.3164 / max(reynolds, 1.0e-30) ** 0.25
        dynamic_pressure = 0.5 * rho * mean_speed**2
        pressure_drop = (
            friction_factor
            * (length / hydraulic_diameter)
            * dynamic_pressure
        )
        wall_shear = friction_factor * rho * mean_speed**2 / 8.0
        wall_force = 2.0 * wall_shear * length
        return pressure_drop, reynolds, friction_factor, wall_shear, wall_force

    q_low = 0.0
    q_high = max(q_poiseuille_abs, height, 1.0e-12)

    for _ in range(100):
        dp_high, *_ = turbulent_pressure_drop(q_high)
        if dp_high >= dp_abs:
            break
        q_high *= 2.0
    else:
        raise RuntimeError(
            "Could not bracket the turbulent rectangular-channel flow solution."
        )

    for _ in range(90):
        q_mid = 0.5 * (q_low + q_high)
        dp_mid, *_ = turbulent_pressure_drop(q_mid)
        if dp_mid < dp_abs:
            q_low = q_mid
        else:
            q_high = q_mid

    q_abs = 0.5 * (q_low + q_high)
    (
        _dp_check,
        reynolds,
        friction_factor,
        wall_shear,
        wall_force_per_span,
    ) = turbulent_pressure_drop(q_abs)
    mean_speed = q_abs / height
    equivalent_resistance = dp_abs / max(q_abs, 1.0e-30)

    return (
        float(sign * q_abs),
        float(equivalent_resistance),
        float(reynolds),
        float(friction_factor),
        "turbulent_blasius_smooth_slot",
        float(sign * mean_speed),
        float(wall_shear),
        float(wall_force_per_span),
    )

def solve_internal_channel_flow(
    dp: float,
    rho: float,
    mu: float,
    length: float,
    diameter: float | None = None,
    area: float | None = None,
    height: float | None = None,
) -> tuple[float, float, float, float, str]:
    """Compatibility wrapper returning the historical five-value tuple.

    In the rectangular formulation ``area`` is the flow area per unit span and
    is numerically equal to the channel height.  ``diameter`` is interpreted as
    hydraulic diameter only when no height/area value is supplied.
    """
    if height is None:
        if area is not None:
            height = float(area)
        elif diameter is not None:
            height = 0.5 * float(diameter)
        else:
            raise ValueError("A rectangular channel height is required.")

    q, resistance, re, f_d, model, *_ = solve_rectangular_slot_flow(
        dp=dp, rho=rho, mu=mu, length=length, height=float(height)
    )
    return q, resistance, re, f_d, model


@dataclass(frozen=True)
class OpeningPatch:
    """Exact fractional panel coverage of one rectangular surface opening."""

    panel_ids: np.ndarray
    overlap_lengths_m: np.ndarray
    panel_open_fractions: np.ndarray
    opening_length_m: float
    normal_component: float
    surface_point: SurfacePoint


def _segment_distance(
    p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray
) -> float:
    """Return the minimum distance between two 2D line segments."""
    def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1.0e-30:
            return float(np.linalg.norm(p - a))
        t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
        return float(np.linalg.norm(p - (a + t * ab)))

    def orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    o1, o2 = orient(p1, p2, q1), orient(p1, p2, q2)
    o3, o4 = orient(q1, q2, p1), orient(q1, q2, p2)
    if o1 * o2 <= 0.0 and o3 * o4 <= 0.0:
        return 0.0

    return min(
        point_segment_distance(p1, q1, q2),
        point_segment_distance(p2, q1, q2),
        point_segment_distance(q1, p1, p2),
        point_segment_distance(q2, p1, p2),
    )


# =============================================================================
# ONE INDEPENDENT RECTANGULAR CHANNEL
# =============================================================================
class IndependentPassage:
    """One straight rectangular slot solved per unit span."""

    def __init__(
        self,
        name: str,
        pore1: Pore,
        pore2: Pore,
        chamber: Chamber,
        span_width: float | None = None,
        layout_kind: str = "surface_pair",
    ) -> None:
        self.name = str(name)
        self.pore1 = pore1
        self.pore2 = pore2
        self.chamber = chamber
        self.layout_kind = str(layout_kind)

        # Retained only so existing callers do not break.  The rectangular-slot
        # physics is per unit span and does not use this value.
        self.span_width = None if span_width is None else float(span_width)

    @property
    def channel_height_m(self) -> float:
        """Return the full 2D rectangular channel height."""
        for obj in (self.pore1, self.pore2, self.chamber):
            for attr in ("channel_height_m", "height_m", "height", "diameter"):
                value = getattr(obj, attr, None)
                if value is not None:
                    value = float(value)
                    if np.isfinite(value) and value > 0.0:
                        return value

        # In the current compatibility data model the chamber hydraulic
        # diameter is explicitly set to 2h.
        dh = float(getattr(self.chamber, "hydraulic_diameter", 0.0))
        if dh > 0.0:
            return 0.5 * dh
        raise ValueError(f"{self.name}: no positive channel height is available.")

    @property
    def hydraulic_diameter_m(self) -> float:
        return 2.0 * self.channel_height_m

    @property
    def area_per_unit_span_m(self) -> float:
        """Flow area per unit span; numerically equal to channel height."""
        return self.channel_height_m

    def pore_surface_point(self, geom: PanelGeometry, pore: Pore) -> SurfacePoint:
        return geom.surface_point_from_fraction(pore.x_frac, pore.side)

    def channel_endpoints(
        self, geom: PanelGeometry
    ) -> tuple[SurfacePoint, SurfacePoint, np.ndarray]:
        sp1 = self.pore_surface_point(geom, self.pore1)
        sp2 = self.pore_surface_point(geom, self.pore2)
        delta = np.array([sp2.x - sp1.x, sp2.y - sp1.y], dtype=float)
        length = float(np.linalg.norm(delta))
        if length <= 1.0e-14:
            raise ValueError(f"{self.name}: channel endpoints are coincident.")
        return sp1, sp2, delta / length

    def passage_length(self, geom: PanelGeometry) -> float:
        """Straight inlet-to-outlet channel length."""
        sp1, sp2, _ = self.channel_endpoints(geom)
        return max(float(np.hypot(sp2.x - sp1.x, sp2.y - sp1.y)), 1.0e-12)

    def _opening_length(
        self, geom: PanelGeometry, pore: Pore, surface_point: SurfacePoint
    ) -> tuple[float, float]:
        """Return the skin-intersection length of the rectangular slot."""
        _sp1, _sp2, direction = self.channel_endpoints(geom)
        normal = np.array([surface_point.nx, surface_point.ny], dtype=float)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1.0e-14:
            raise ValueError(f"{self.name}: invalid surface normal at an opening.")
        normal /= normal_norm
        normal_component = abs(float(np.dot(direction, normal)))

        if normal_component < MIN_OPENING_NORMAL_COMPONENT:
            raise ValueError(
                f"{self.name}: channel intersects the {pore.side} surface too "
                f"tangentially (|e·n|={normal_component:.4f})."
            )

        opening_length = self.channel_height_m / normal_component
        return float(opening_length), float(normal_component)

    def opening_patch(self, geom: PanelGeometry, pore: Pore) -> OpeningPatch:
        """Map the exact rectangular opening to fractional surface panels."""
        sp = self.pore_surface_point(geom, pore)
        opening_length, normal_component = self._opening_length(geom, pore, sp)

        candidate_ids = (
            np.asarray(geom.upper_panel_ids, dtype=int)
            if pore.side == "upper"
            else np.asarray(geom.lower_panel_ids, dtype=int)
        )
        candidate_ids = np.sort(candidate_ids)
        if candidate_ids.size == 0:
            raise RuntimeError(f"{self.name}: no panels found on {pore.side} surface.")

        lengths = np.asarray(geom.S[candidate_ids], dtype=float)
        starts = np.concatenate([[0.0], np.cumsum(lengths[:-1])])
        ends = starts + lengths

        matches = np.flatnonzero(candidate_ids == int(sp.panel_id))
        if matches.size:
            local_index = int(matches[0])
        else:
            distances = (geom.XC[candidate_ids] - sp.x) ** 2 + (
                geom.YC[candidate_ids] - sp.y
            ) ** 2
            local_index = int(np.argmin(distances))

        panel_id = int(candidate_ids[local_index])
        x0 = float(geom.XB[panel_id])
        y0 = float(geom.YB[panel_id])
        panel_length = max(float(geom.S[panel_id]), 1.0e-16)
        tangent = np.array(
            [geom.XB[panel_id + 1] - x0, geom.YB[panel_id + 1] - y0],
            dtype=float,
        ) / panel_length
        local_distance = float(
            np.clip(np.dot(np.array([sp.x - x0, sp.y - y0]), tangent), 0.0, panel_length)
        )
        centre_s = float(starts[local_index] + local_distance)

        interval_start = centre_s - 0.5 * opening_length
        interval_end = centre_s + 0.5 * opening_length
        overlaps = np.maximum(
            0.0, np.minimum(ends, interval_end) - np.maximum(starts, interval_start)
        )
        active = overlaps > OPENING_TOLERANCE

        ids = candidate_ids[active]
        overlap_lengths = overlaps[active]
        fractions = overlap_lengths / np.maximum(geom.S[ids], 1.0e-16)
        represented_length = float(np.sum(overlap_lengths))

        if not ids.size or not np.isclose(
            represented_length, opening_length, rtol=1.0e-8, atol=1.0e-10
        ):
            raise ValueError(
                f"{self.name}: rectangular opening of length "
                f"{opening_length:.6e} m does not fit on the {pore.side} surface "
                f"at x/c={pore.x_frac:.6f}. Represented length is "
                f"{represented_length:.6e} m."
            )

        return OpeningPatch(
            panel_ids=np.asarray(ids, dtype=int),
            overlap_lengths_m=np.asarray(overlap_lengths, dtype=float),
            panel_open_fractions=np.asarray(fractions, dtype=float),
            opening_length_m=float(opening_length),
            normal_component=float(normal_component),
            surface_point=sp,
        )

    def validate_geometry(self, geom: PanelGeometry, limits: GeometryLimits) -> None:
        """Validate a finite-height rectangular channel inside the airfoil."""
        height = self.channel_height_m
        if height <= 0.0:
            raise ValueError(f"{self.name}: channel height must be positive.")

        for label, pore in (("opening1", self.pore1), ("opening2", self.pore2)):
            if not (limits.x_min_frac <= pore.x_frac <= limits.x_max_frac):
                raise ValueError(
                    f"{self.name}:{label} x/c={pore.x_frac:.4f} is outside "
                    f"[{limits.x_min_frac}, {limits.x_max_frac}]."
                )
            self.opening_patch(geom, pore)

        sp1, sp2, direction = self.channel_endpoints(geom)
        length = self.passage_length(geom)
        if length <= height + float(limits.min_gap):
            raise ValueError(
                f"{self.name}: channel length {length:.6e} m is too short for "
                f"height {height:.6e} m."
            )

        # Check the rectangular corridor away from the two oblique skin cuts.
        # The required inset follows from the local line-plane intersection of
        # the channel side wall with the airfoil tangent.
        transverse = np.array([-direction[1], direction[0]], dtype=float)

        def opening_inset(surface_point: SurfacePoint) -> float:
            normal = np.array([surface_point.nx, surface_point.ny], dtype=float)
            normal /= max(float(np.linalg.norm(normal)), 1.0e-16)
            denominator = max(abs(float(np.dot(normal, direction))), 1.0e-12)
            geometric = (
                0.5 * height * abs(float(np.dot(normal, transverse))) / denominator
            )
            return geometric + 0.10 * height

        start_margin = opening_inset(sp1)
        end_margin = opening_inset(sp2)
        if start_margin + end_margin >= 0.90 * length:
            raise ValueError(
                f"{self.name}: the rectangular opening cuts consume too much of "
                "the available channel length."
            )

        s_values = np.linspace(start_margin, length - end_margin, 60)
        centre0 = np.array([sp1.x, sp1.y], dtype=float)
        points: list[np.ndarray] = []
        for s_value in s_values:
            centre = centre0 + s_value * direction
            for eta in (-0.499, 0.0, 0.499):
                points.append(centre + eta * height * transverse)

        polygon = MplPath(np.column_stack([geom.XB, geom.YB]), closed=True)
        inside = polygon.contains_points(np.asarray(points), radius=1.0e-12)
        if not bool(np.all(inside)):
            raise ValueError(
                f"{self.name}: the finite-height rectangular channel leaves the "
                "airfoil interior. Reduce the channel height or move the channel."
            )

    def _area_average_pressure_at_pore(
        self,
        geom: PanelGeometry,
        pore: Pore,
        panel_pressures: np.ndarray,
    ) -> float:
        """Average pressure over the exact rectangular opening footprint."""
        patch = self.opening_patch(geom, pore)
        pressure = np.asarray(panel_pressures, dtype=float)[patch.panel_ids]
        weights = patch.overlap_lengths_m
        return float(np.sum(pressure * weights) / max(np.sum(weights), 1.0e-16))

    def solve(
        self,
        geom: PanelGeometry,
        panel_pressures: np.ndarray,
        flow: FlowConfig,
    ) -> PassageState:
        """Solve one rectangular channel using exact opening pressures."""
        p1 = self._area_average_pressure_at_pore(geom, self.pore1, panel_pressures)
        p2 = self._area_average_pressure_at_pore(geom, self.pore2, panel_pressures)
        dp = p1 - p2
        length = self.passage_length(geom)
        height = self.channel_height_m

        (
            q, resistance, reynolds, friction_factor, resistance_model,
            mean_velocity, wall_shear, wall_force_per_span,
        ) = solve_rectangular_slot_flow(
            dp=dp,
            rho=flow.rho_inf,
            mu=flow.mu_inf,
            length=length,
            height=height,
        )

        state = PassageState(
            Q=float(q),  # compatibility: now q per unit span [m^2/s]
            p1=float(p1),
            p2=float(p2),
            dp_total=float(abs(dp)),
            Rs=float(resistance),
            reynolds_equivalent=float(reynolds),
            friction_factor=float(friction_factor),
            resistance_model=str(resistance_model),
        )

        # Explicit rectangular-channel fields.  PassageState is intentionally
        # extended dynamically so the existing config dataclass remains usable.
        state.flow_rate_per_unit_span_m2_s = float(q)
        state.mean_velocity_m_s = float(mean_velocity)
        state.channel_height_m = float(height)
        state.hydraulic_diameter_m = float(2.0 * height)
        state.wall_shear_pa = float(wall_shear)
        state.wetted_wall_length_per_unit_span_m = float(2.0 * length)
        state.wall_force_per_span_n_m = float(wall_force_per_span)
        state.pressure_force_per_span_n_m = float(abs(dp) * height)
        state.force_balance_error_per_span_n_m = float(
            wall_force_per_span - abs(dp) * height
        )

        sp1, sp2, direction_12 = self.channel_endpoints(geom)
        flow_direction = direction_12 if q >= 0.0 else -direction_12
        state.flow_direction_x = float(flow_direction[0])
        state.flow_direction_y = float(flow_direction[1])
        state.internal_force_x_per_span_n_m = float(
            wall_force_per_span * flow_direction[0]
        )
        state.internal_force_y_per_span_n_m = float(
            wall_force_per_span * flow_direction[1]
        )
        state.channel_midpoint_x_m = 0.5 * float(sp1.x + sp2.x)
        state.channel_midpoint_y_m = 0.5 * float(sp1.y + sp2.y)
        return state

    def transpiration_from_flow(
        self, geom: PanelGeometry, state: PassageState
    ) -> np.ndarray:
        """Map q per unit span to exact opening-normal panel velocities."""
        vn = np.zeros(geom.num_pan, dtype=float)
        q = float(state.Q)
        if abs(q) < 1.0e-30:
            return vn

        patch1 = self.opening_patch(geom, self.pore1)
        patch2 = self.opening_patch(geom, self.pore2)

        for patch, signed_q in ((patch1, -q), (patch2, q)):
            opening_velocity = signed_q / patch.opening_length_m
            vn[patch.panel_ids] += opening_velocity * patch.panel_open_fractions

        return vn

    def opening_fraction(self, geom: PanelGeometry) -> np.ndarray:
        fraction = np.zeros(geom.num_pan, dtype=float)
        for pore in (self.pore1, self.pore2):
            patch = self.opening_patch(geom, pore)
            fraction[patch.panel_ids] += patch.panel_open_fractions
        return fraction

    def normal_velocity_mean_square_from_flow(
        self, geom: PanelGeometry, state: PassageState
    ) -> np.ndarray:
        """Return the panel-area mean of local opening normal velocity squared."""
        values = np.zeros(geom.num_pan, dtype=float)
        q = float(state.Q)
        if abs(q) < 1.0e-30:
            return values
        for patch in (
            self.opening_patch(geom, self.pore1),
            self.opening_patch(geom, self.pore2),
        ):
            local_velocity = q / patch.opening_length_m
            values[patch.panel_ids] += (
                local_velocity**2 * patch.panel_open_fractions
            )
        return values


# =============================================================================
# NETWORK OF INDEPENDENT RECTANGULAR CHANNELS
# =============================================================================
class IndependentPassageNetwork:
    def __init__(self, passages: list[IndependentPassage]) -> None:
        if not passages:
            raise ValueError("At least one passage is required.")
        self.passages = passages

    def validate_geometry(self, geom: PanelGeometry, limits: GeometryLimits) -> None:
        for passage in self.passages:
            passage.validate_geometry(geom, limits)

        # Channels in the same hydraulic layer may not overlap.  Cross-layout
        # intersections in the combined model are retained as the documented
        # idealised independent-layer superposition.
        for i, passage_i in enumerate(self.passages):
            sp1_i, sp2_i, _ = passage_i.channel_endpoints(geom)
            p1 = np.array([sp1_i.x, sp1_i.y])
            p2 = np.array([sp2_i.x, sp2_i.y])
            for passage_j in self.passages[i + 1 :]:
                if passage_i.layout_kind != passage_j.layout_kind:
                    continue
                sp1_j, sp2_j, _ = passage_j.channel_endpoints(geom)
                q1 = np.array([sp1_j.x, sp1_j.y])
                q2 = np.array([sp2_j.x, sp2_j.y])
                distance = _segment_distance(p1, p2, q1, q2)
                required = (
                    0.5 * passage_i.channel_height_m
                    + 0.5 * passage_j.channel_height_m
                    + float(limits.min_gap)
                )
                if distance < required:
                    raise ValueError(
                        f"Rectangular channels {passage_i.name!r} and "
                        f"{passage_j.name!r} overlap: distance={distance:.6e} m, "
                        f"required={required:.6e} m."
                    )

        # Openings belonging to the same layout layer may not cover the same
        # part of a panel.
        by_layout: dict[str, np.ndarray] = {}
        for passage in self.passages:
            values = by_layout.setdefault(
                passage.layout_kind, np.zeros(geom.num_pan, dtype=float)
            )
            values += passage.opening_fraction(geom)
        for layout, values in by_layout.items():
            if np.any(values > 1.0 + 1.0e-8):
                raise ValueError(
                    f"Rectangular surface openings overlap in layout {layout!r}."
                )

    def opening_fraction(self, geom: PanelGeometry) -> np.ndarray:
        opening = np.zeros(geom.num_pan, dtype=float)
        for passage in self.passages:
            opening += passage.opening_fraction(geom)
        # Cross-layout overlap represents separate idealised spanwise layers and
        # is not allowed to remove more than the complete external wall.
        return np.clip(opening, 0.0, 1.0)

    def wall_fraction(self, geom: PanelGeometry) -> np.ndarray:
        return 1.0 - self.opening_fraction(geom)

    def solve(
        self, geom: PanelGeometry, panel_pressures: np.ndarray, flow: FlowConfig
    ) -> list[PassageState]:
        return [
            passage.solve(geom=geom, panel_pressures=panel_pressures, flow=flow)
            for passage in self.passages
        ]

    def transpiration_from_flow(
        self, geom: PanelGeometry, states: list[PassageState]
    ) -> np.ndarray:
        if len(states) != len(self.passages):
            raise ValueError("One passage state is required for every passage.")
        vn = np.zeros(geom.num_pan, dtype=float)
        for passage, state in zip(self.passages, states):
            vn += passage.transpiration_from_flow(geom=geom, state=state)
        return vn

    def normal_velocity_mean_square_from_flow(
        self, geom: PanelGeometry, states: list[PassageState]
    ) -> np.ndarray:
        if len(states) != len(self.passages):
            raise ValueError("One passage state is required for every passage.")
        values = np.zeros(geom.num_pan, dtype=float)
        for passage, state in zip(self.passages, states):
            values += passage.normal_velocity_mean_square_from_flow(geom, state)
        return values


def _add_internal_channel_forces(
    aero_result,
    network: IndependentPassageNetwork,
    states: list[PassageState],
    aero_solver: SourceVortexPanelMethod,
):
    """Add internal wall-shear force and moment to the external-wall result."""
    flow = aero_solver.flow
    geom = aero_solver.geom
    ref = aero_solver.ref

    force_x = 0.0
    force_y = 0.0
    moment = 0.0

    for state in states:
        fx = float(getattr(state, "internal_force_x_per_span_n_m", 0.0))
        fy = float(getattr(state, "internal_force_y_per_span_n_m", 0.0))
        x_mid = float(getattr(state, "channel_midpoint_x_m", ref.x_ref))
        y_mid = float(getattr(state, "channel_midpoint_y_m", ref.y_ref))
        m = (x_mid - ref.x_ref) * fy - (y_mid - ref.y_ref) * fx
        state.internal_moment_per_span_n = float(m)
        force_x += fx
        force_y += fy
        moment += m

    q_inf = 0.5 * flow.rho_inf * flow.v_inf**2
    force_scale = max(q_inf * geom.chord, 1.0e-30)
    moment_scale = max(q_inf * geom.chord**2, 1.0e-30)

    internal_Cx = force_x / force_scale
    internal_Cy = force_y / force_scale
    alpha = np.deg2rad(flow.aoa_deg)
    internal_CL = internal_Cy * np.cos(alpha) - internal_Cx * np.sin(alpha)
    internal_CD = internal_Cx * np.cos(alpha) + internal_Cy * np.sin(alpha)
    internal_CM = moment / moment_scale

    total_Cx = aero_result.external_wall_Cx + internal_Cx
    total_Cy = aero_result.external_wall_Cy + internal_Cy
    total_CL = aero_result.external_wall_CL + internal_CL
    total_CD = aero_result.external_wall_CD + internal_CD
    total_CM = aero_result.external_wall_CM + internal_CM

    return replace(
        aero_result,
        Cx=float(total_Cx),
        Cy=float(total_Cy),
        CL=float(total_CL),
        CD=float(total_CD),
        CM=float(total_CM),
        internal_channel_Cx=float(internal_Cx),
        internal_channel_Cy=float(internal_Cy),
        internal_channel_CL=float(internal_CL),
        internal_channel_CD=float(internal_CD),
        internal_channel_CM=float(internal_CM),
        total_Cx=float(total_Cx),
        total_Cy=float(total_Cy),
        total_CL=float(total_CL),
        total_CD=float(total_CD),
        total_CM=float(total_CM),
    )


# =============================================================================
# COUPLED PANEL + RECTANGULAR-CHANNEL SOLVER
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

    def solve(self, verbose: bool = False) -> MultiPassageResult:
        geom = self.aero_solver.geom
        flow = self.aero_solver.flow
        self.network.validate_geometry(geom, self.limits)

        opening_fraction = self.network.opening_fraction(geom)
        wall_fraction = 1.0 - opening_fraction
        vn = np.zeros(geom.num_pan, dtype=float)
        vn_mean_square = np.zeros(geom.num_pan, dtype=float)
        previous_q: list[float] | None = None
        converged = False
        final_iteration = 0

        for iteration in range(1, self.coupling.max_iter + 1):
            aero_result = self.aero_solver.solve(
                normal_transpiration=vn,
                wall_fraction=wall_fraction,
                normal_velocity_mean_square=vn_mean_square,
            )
            panel_pressures = self.aero_solver.panel_pressures(aero_result)
            states = self.network.solve(geom, panel_pressures, flow)
            vn_target = self.network.transpiration_from_flow(geom, states)
            vn_mean_square_target = (
                self.network.normal_velocity_mean_square_from_flow(geom, states)
            )
            vn_new = (
                (1.0 - self.coupling.relaxation) * vn
                + self.coupling.relaxation * vn_target
            )
            vn_mean_square_new = (
                (1.0 - self.coupling.relaxation) * vn_mean_square
                + self.coupling.relaxation * vn_mean_square_target
            )

            current_q = [float(state.Q) for state in states]
            dq = (
                np.inf
                if previous_q is None
                else max(
                    abs(q_new - q_old)
                    for q_new, q_old in zip(current_q, previous_q)
                )
            )
            dvn = float(np.max(np.abs(vn_new - vn)))

            if verbose:
                max_q = max((abs(q) for q in current_q), default=0.0)
                boundary_flux = float(np.sum(vn_new * geom.S))
                print(
                    f"[Coupling] iter={iteration:03d} | "
                    f"external CL={aero_result.external_wall_CL: .8f} | "
                    f"max|q|={max_q: .6e} m2/s | "
                    f"dvn={dvn: .3e} | dq={dq: .3e} | "
                    f"net flux={boundary_flux: .3e} m2/s"
                )

            vn = vn_new
            vn_mean_square = vn_mean_square_new
            previous_q = current_q
            final_iteration = iteration
            if dvn < self.coupling.tol_vn and dq < self.coupling.tol_q:
                converged = True
                break

        aero_external = self.aero_solver.solve(
            normal_transpiration=vn,
            wall_fraction=wall_fraction,
            normal_velocity_mean_square=vn_mean_square,
        )
        panel_pressures = self.aero_solver.panel_pressures(aero_external)
        states = self.network.solve(geom, panel_pressures, flow)
        aero_total = _add_internal_channel_forces(
            aero_result=aero_external,
            network=self.network,
            states=states,
            aero_solver=self.aero_solver,
        )

        max_vn = float(np.max(np.abs(vn)))
        result = MultiPassageResult(
            aero_result=aero_total,
            passage_states=states,
            normal_transpiration=vn,
            converged=converged,
            iterations=final_iteration,
            max_vn=max_vn,
            max_vn_over_vinf=max_vn / max(flow.v_inf, 1.0e-16),
        )

        # Additional decomposition and diagnostics without breaking the existing
        # lightweight result dataclass in config.py.
        result.opening_fraction = opening_fraction
        result.wall_fraction = wall_fraction
        result.normal_velocity_mean_square = vn_mean_square
        result.boundary_flux_per_unit_span_m2_s = float(np.sum(vn * geom.S))
        result.external_wall_CL = float(aero_total.external_wall_CL)
        result.external_wall_CD = float(aero_total.external_wall_CD)
        result.external_wall_CM = float(aero_total.external_wall_CM)
        result.internal_channel_CL = float(aero_total.internal_channel_CL)
        result.internal_channel_CD = float(aero_total.internal_channel_CD)
        result.internal_channel_CM = float(aero_total.internal_channel_CM)
        result.total_CL = float(aero_total.total_CL)
        result.total_CD = float(aero_total.total_CD)
        result.total_CM = float(aero_total.total_CM)
        return result


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
    diameter_m: float,
) -> list[PassageSpec]:
    """
    Model 1:
        4 horizontal straight internal channels.

    Each channel is generated from a constant y/c level.  The two pore
    positions are the left and right intersections of that horizontal line
    with the airfoil surface, so the plotted/exported internal passage is a
    straight horizontal line.
    """
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
                diameter_m=diameter_m,
                layout_kind="internal_horizontal",
            )
        )

    return passages

def build_vertical_passages(
    x_values: np.ndarray,
    diameter_m: float,
    prefix: str,
) -> list[PassageSpec]:
    """
    Build lower-to-upper perpendicular channels.
    """
    passages: list[PassageSpec] = []

    for i, x in enumerate(x_values, start=1):
        passages.append(
            PassageSpec(
                name=f"{prefix}_{i:02d}_perpendicular",
                x1_frac=float(x),
                side1="lower",
                x2_frac=float(x),
                side2="upper",
                diameter_m=diameter_m,
                layout_kind="perpendicular",
            )
        )

    return passages


def _copy_passages_with_prefix(
    passages: list[PassageSpec],
    prefix: str,
    label: str,
) -> list[PassageSpec]:
    """
    Copy passage definitions and rename them.

    Used by Model 3 so it can reuse the exact Model 1 channel geometry.
    """
    copied: list[PassageSpec] = []

    for i, passage in enumerate(passages, start=1):
        copied.append(
            PassageSpec(
                name=f"{prefix}_{i:02d}_{label}",
                x1_frac=passage.x1_frac,
                side1=passage.side1,
                x2_frac=passage.x2_frac,
                side2=passage.side2,
                diameter_m=passage.diameter_m,
                layout_kind=passage.layout_kind,
            )
        )

    return copied


# =============================================================================
# MODEL 4: SAVED CASE 1 FROM OLD run_test.py
# =============================================================================
def build_model4_saved_case1(diameter_m: float | None = None) -> list[PassageSpec]:
    """
    Model 4:
        Saved case_1 from the old run_test.py validation script.

    The legacy ``diameter_m`` input is interpreted as rectangular channel height.
    This keeps existing workflow calls compatible.
    """
    if diameter_m is None:
        d1 = 1.2e-02
        d2 = 0.9e-02
    else:
        d1 = float(diameter_m)
        d2 = float(diameter_m)

    return [
        PassageSpec(
            name="m4_case1_passage_01_lower_to_upper",
            x1_frac=0.400094,
            side1="lower",
            x2_frac=0.933001,
            side2="upper",
            diameter_m=d1,
            layout_kind="saved_case_1",
        ),
        PassageSpec(
            name="m4_case1_passage_02_lower_to_upper",
            x1_frac=0.925069,
            side1="lower",
            x2_frac=0.955118,
            side2="upper",
            diameter_m=d2,
            layout_kind="saved_case_1",
        ),
    ]


def build_model_specs(
    geom: PanelGeometry,
    diameter_m: float,
) -> list[PorousModelSpec]:
    """
    Build the fixed porous models.

    The legacy ``diameter_m`` argument is the rectangular channel height for all
    generated passages.
    """
    model_1_chordwise = build_model1_chordwise(
        geom=geom,
        diameter_m=diameter_m,
    )

    model_2_perpendicular = build_vertical_passages(
        x_values=np.linspace(0.100, 0.900, 4),
        diameter_m=diameter_m,
        prefix="m2",
    )

    model_3_chordwise = _copy_passages_with_prefix(
        passages=model_1_chordwise,
        prefix="m3c",
        label="LE_to_TE_internal_channel",
    )

    model_3_perpendicular = build_vertical_passages(
        x_values=np.linspace(0.100, 0.900, 4),
        diameter_m=diameter_m,
        prefix="m3v",
    )

    model_3_combined = model_3_chordwise + model_3_perpendicular

    
    model_4_saved_case1 = build_model4_saved_case1(diameter_m=diameter_m)

    return [
        PorousModelSpec(
            name="model_1_4_chordwise",
            description=(
                "4 straight horizontal rectangular channels at constant y/c levels. "
                "Each channel is solved per unit span."
            ),
            passages=tuple(model_1_chordwise),
        ),
        PorousModelSpec(
            name="model_2_9_perpendicular",
            description="4 independent lower-to-upper rectangular channels.",
            passages=tuple(model_2_perpendicular),
        ),
        PorousModelSpec(
            name="model_3_combined_independent",
            description=(
                "Idealised independent-layer combination of 4 horizontal and "
                "4 vertical rectangular channels. Crossings are not hydraulic junctions."
            ),
            passages=tuple(model_3_combined),
        ),
        PorousModelSpec(
            name="model_4_saved_case_1",
            description=(
                "Saved two-passage rectangular-channel case with fixed surface "
                "locations. The legacy diameter value is interpreted as channel height."
            ),
            passages=tuple(model_4_saved_case1),
        ),
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


def _spec_channel_height(spec: PassageSpec) -> float:
    """Read rectangular channel height with legacy ``diameter_m`` fallback."""
    for attr in ("channel_height_m", "height_m", "diameter_m"):
        value = getattr(spec, attr, None)
        if value is not None:
            value = float(value)
            if np.isfinite(value) and value > 0.0:
                return value
    raise ValueError(f"Passage {getattr(spec, 'name', '<unnamed>')!r} has no height.")


def build_network_from_model(
    model: PorousModelSpec,
    span_width: float | None = None,
) -> IndependentPassageNetwork:
    """Build a per-unit-span rectangular-channel network.

    ``span_width`` is accepted only for compatibility with existing workflows
    and is not used by the 2D rectangular-slot equations.
    """
    passages: list[IndependentPassage] = []

    for spec in model.passages:
        height = _spec_channel_height(spec)
        passages.append(
            IndependentPassage(
                name=spec.name,
                pore1=Pore(spec.x1_frac, spec.side1, height),
                pore2=Pore(spec.x2_frac, spec.side2, height),
                chamber=Chamber(
                    hydraulic_diameter=2.0 * height,
                    area=height,  # flow area per unit span
                ),
                span_width=span_width,
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
    """Build the per-passage rectangular-channel result table."""
    rows: list[dict] = []

    for passage_id, (passage, state) in enumerate(
        zip(network.passages, result.passage_states), start=1
    ):
        sp1 = passage.pore_surface_point(geom, passage.pore1)
        sp2 = passage.pore_surface_point(geom, passage.pore2)
        patch1 = passage.opening_patch(geom, passage.pore1)
        patch2 = passage.opening_patch(geom, passage.pore2)
        height = passage.channel_height_m
        q = float(state.Q)

        rows.append(
            {
                "model_name": model.name,
                "passage_id": passage_id,
                "passage_name": passage.name,
                "opening1_side": passage.pore1.side,
                "opening2_side": passage.pore2.side,
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
                "channel_height_m": height,
                "hydraulic_diameter_m": 2.0 * height,
                "passage_length_m": passage.passage_length(geom),
                "opening1_length_m": patch1.opening_length_m,
                "opening2_length_m": patch2.opening_length_m,
                "opening1_normal_component": patch1.normal_component,
                "opening2_normal_component": patch2.normal_component,
                "flow_rate_per_unit_span_m2_s": q,
                "abs_flow_rate_per_unit_span_m2_s": abs(q),
                "mean_velocity_m_s": getattr(state, "mean_velocity_m_s", np.nan),
                "equivalent_resistance_Pa_s_per_m2": state.Rs,
                "dp_total_Pa": state.dp_total,
                "reynolds_internal": state.reynolds_equivalent,
                "p1_Pa": state.p1,
                "p2_Pa": state.p2,
                "friction_factor_darcy": state.friction_factor,
                "resistance_model": state.resistance_model,
                "wall_shear_Pa": getattr(state, "wall_shear_pa", np.nan),
                "wall_force_per_span_N_per_m": getattr(
                    state, "wall_force_per_span_n_m", np.nan
                ),
                "pressure_force_per_span_N_per_m": getattr(
                    state, "pressure_force_per_span_n_m", np.nan
                ),
                "force_balance_error_per_span_N_per_m": getattr(
                    state, "force_balance_error_per_span_n_m", np.nan
                ),
                "internal_force_x_per_span_N_per_m": getattr(
                    state, "internal_force_x_per_span_n_m", np.nan
                ),
                "internal_force_y_per_span_N_per_m": getattr(
                    state, "internal_force_y_per_span_n_m", np.nan
                ),
                "internal_moment_per_span_N": getattr(
                    state, "internal_moment_per_span_n", np.nan
                ),
                # Compatibility aliases for existing post-processing.
                "pore_diameter_m": height,
                "Q_m3_s": q,
                "abs_Q_m3_s": abs(q),
                "Rs_Pa_s_per_m3": state.Rs,
                "reynolds_internal_equivalent": state.reynolds_equivalent,
            }
        )

    return pd.DataFrame(rows)


def representative_channel_height(model: PorousModelSpec) -> float:
    heights = [_spec_channel_height(spec) for spec in model.passages]
    return float(np.mean(heights)) if heights else float("nan")


def representative_pore_diameter(model: PorousModelSpec) -> float:
    """Compatibility alias: legacy diameter now means channel height."""
    return representative_channel_height(model)


def nan_if_not_converged(value: float, converged: bool) -> float:
    return float(value) if converged else float("nan")


def make_nonconverged_summary_row(
    model: PorousModelSpec,
    flow: FlowConfig,
    solid_result: object,
    porous_result: MultiPassageResult,
    elapsed_seconds: float,
) -> dict:
    nan = float("nan")
    height = representative_channel_height(model)
    return {
        "model_name": model.name,
        "description": model.description,
        "channel_height_m": height,
        "pore_diameter_m": height,  # compatibility
        "num_independent_passages": len(model.passages),
        "reynolds_external": REYNOLDS_EXTERNAL,
        "v_inf_m_s": flow.v_inf,
        "aoa_deg": flow.aoa_deg,
        "solid_CL": solid_result.CL,
        "solid_CD": solid_result.CD,
        "solid_CM": solid_result.CM,
        "external_wall_CL": nan,
        "external_wall_CD": nan,
        "external_wall_CM": nan,
        "internal_channel_CL": nan,
        "internal_channel_CD": nan,
        "internal_channel_CM": nan,
        "porous_CL": nan,
        "porous_CD": nan,
        "porous_CM": nan,
        "delta_CL": nan,
        "delta_CD": nan,
        "delta_CM": nan,
        "gain_percent_vs_solid_panel": nan,
        "max_abs_flow_rate_per_unit_span_m2_s": nan,
        "sum_abs_flow_rate_per_unit_span_m2_s": nan,
        "max_abs_Q_m3_s": nan,  # compatibility
        "sum_abs_Q_m3_s": nan,
        "max_dp_total_Pa": nan,
        "max_internal_reynolds": nan,
        "max_internal_reynolds_equivalent": nan,
        "max_wall_shear_Pa": nan,
        "max_vn_m_s": nan,
        "max_vn_over_vinf": nan,
        "boundary_flux_per_unit_span_m2_s": nan,
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
    opening_fraction = np.asarray(
        getattr(porous_result, "opening_fraction", np.zeros(geom.num_pan)),
        dtype=float,
    )
    wall_fraction = np.asarray(
        getattr(porous_result, "wall_fraction", np.ones(geom.num_pan)),
        dtype=float,
    )

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
            "porous_normal_velocity_m_s": aero.normal_velocity,
            "porous_Cp": aero.Cp,
            "solid_Vt_m_s": solid_result.Vt,
            "solid_Cp": solid_result.Cp,
            "normal_transpiration_m_s": porous_result.normal_transpiration,
            "opening_fraction": opening_fraction,
            "solid_wall_fraction": wall_fraction,
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
            model, flow, solid_result, porous_result, elapsed_seconds
        )

    aero = porous_result.aero_result
    q_abs = [abs(float(state.Q)) for state in porous_result.passage_states]
    dp_values = [float(state.dp_total) for state in porous_result.passage_states]
    re_values = [
        float(state.reynolds_equivalent) for state in porous_result.passage_states
    ]
    shear_values = [
        float(getattr(state, "wall_shear_pa", np.nan))
        for state in porous_result.passage_states
    ]
    height = representative_channel_height(model)

    return {
        "model_name": model.name,
        "description": model.description,
        "channel_height_m": height,
        "pore_diameter_m": height,  # compatibility
        "num_independent_passages": len(model.passages),
        "reynolds_external": REYNOLDS_EXTERNAL,
        "v_inf_m_s": flow.v_inf,
        "aoa_deg": flow.aoa_deg,
        "solid_CL": solid_result.CL,
        "solid_CD": solid_result.CD,
        "solid_CM": solid_result.CM,
        "external_wall_CL": aero.external_wall_CL,
        "external_wall_CD": aero.external_wall_CD,
        "external_wall_CM": aero.external_wall_CM,
        "internal_channel_CL": aero.internal_channel_CL,
        "internal_channel_CD": aero.internal_channel_CD,
        "internal_channel_CM": aero.internal_channel_CM,
        "porous_CL": aero.CL,
        "porous_CD": aero.CD,
        "porous_CM": aero.CM,
        "delta_CL": aero.CL - solid_result.CL,
        "delta_CD": aero.CD - solid_result.CD,
        "delta_CM": aero.CM - solid_result.CM,
        "gain_percent_vs_solid_panel": 100.0 * (aero.CL / solid_result.CL - 1.0),
        "max_abs_flow_rate_per_unit_span_m2_s": max(q_abs) if q_abs else 0.0,
        "sum_abs_flow_rate_per_unit_span_m2_s": sum(q_abs),
        "max_abs_Q_m3_s": max(q_abs) if q_abs else 0.0,  # compatibility
        "sum_abs_Q_m3_s": sum(q_abs),
        "max_dp_total_Pa": max(dp_values) if dp_values else 0.0,
        "max_internal_reynolds": max(re_values) if re_values else 0.0,
        "max_internal_reynolds_equivalent": max(re_values) if re_values else 0.0,
        "max_wall_shear_Pa": np.nanmax(shear_values) if shear_values else 0.0,
        "max_vn_m_s": porous_result.max_vn,
        "max_vn_over_vinf": porous_result.max_vn_over_vinf,
        "boundary_flux_per_unit_span_m2_s": getattr(
            porous_result, "boundary_flux_per_unit_span_m2_s", np.nan
        ),
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

        coupled_i = MultiPassageCoupledSolver(
            aero_solver=aero_solver_i,
            network=network_i,
            coupling=coupling,
            limits=limits,
        ).solve(verbose=False)

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

        channel_height = representative_channel_height(model)
        if coupled_i.converged:
            aero_i = coupled_i.aero_result
            q_values = [abs(float(state.Q)) for state in coupled_i.passage_states]
            row = {
                "alpha_deg": alpha,
                "channel_height_m": channel_height,
                "pore_diameter_m": channel_height,  # compatibility
                "external_wall_CL": float(aero_i.external_wall_CL),
                "external_wall_CD": float(aero_i.external_wall_CD),
                "external_wall_CM": float(aero_i.external_wall_CM),
                "internal_channel_CL": float(aero_i.internal_channel_CL),
                "internal_channel_CD": float(aero_i.internal_channel_CD),
                "internal_channel_CM": float(aero_i.internal_channel_CM),
                "porous_CL": float(aero_i.CL),
                "porous_CD": float(aero_i.CD),
                "porous_CM": float(aero_i.CM),
                "solid_panel_CL": float(solid_i.CL),
                "solid_panel_CD": float(solid_i.CD),
                "solid_panel_CM": float(solid_i.CM),
                "delta_CL": float(aero_i.CL - solid_i.CL),
                "delta_CD": float(aero_i.CD - solid_i.CD),
                "delta_CM": float(aero_i.CM - solid_i.CM),
                "max_abs_flow_rate_per_unit_span_m2_s": max(q_values),
                "sum_abs_flow_rate_per_unit_span_m2_s": sum(q_values),
                "max_abs_Q_m3_s": max(q_values),  # compatibility
                "sum_abs_Q_m3_s": sum(q_values),
                "max_dp_total_Pa": max(
                    float(state.dp_total) for state in coupled_i.passage_states
                ),
                "max_internal_reynolds": max(
                    float(state.reynolds_equivalent)
                    for state in coupled_i.passage_states
                ),
                "max_internal_reynolds_equivalent": max(
                    float(state.reynolds_equivalent)
                    for state in coupled_i.passage_states
                ),
                "max_vn_m_s": float(coupled_i.max_vn),
                "boundary_flux_per_unit_span_m2_s": float(
                    coupled_i.boundary_flux_per_unit_span_m2_s
                ),
                "coupling_converged": True,
                "coupling_iterations": int(coupled_i.iterations),
                "max_vn_over_vinf": float(coupled_i.max_vn_over_vinf),
            }
        else:
            nan = float("nan")
            row = {
                "alpha_deg": alpha,
                "channel_height_m": channel_height,
                "pore_diameter_m": channel_height,
                "external_wall_CL": nan,
                "external_wall_CD": nan,
                "external_wall_CM": nan,
                "internal_channel_CL": nan,
                "internal_channel_CD": nan,
                "internal_channel_CM": nan,
                "porous_CL": nan,
                "porous_CD": nan,
                "porous_CM": nan,
                "solid_panel_CL": float(solid_i.CL),
                "solid_panel_CD": float(solid_i.CD),
                "solid_panel_CM": float(solid_i.CM),
                "delta_CL": nan,
                "delta_CD": nan,
                "delta_CM": nan,
                "max_abs_flow_rate_per_unit_span_m2_s": nan,
                "sum_abs_flow_rate_per_unit_span_m2_s": nan,
                "max_abs_Q_m3_s": nan,
                "sum_abs_Q_m3_s": nan,
                "max_dp_total_Pa": nan,
                "max_internal_reynolds": nan,
                "max_internal_reynolds_equivalent": nan,
                "max_vn_m_s": nan,
                "boundary_flux_per_unit_span_m2_s": nan,
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
        aoa_sweep_diameters = tuple(float(d) for d in AOA_SWEEP_PORE_DIAMETERS)

    # Existing configuration names are retained, but these values are rectangular
    # channel heights rather than circular pore diameters.
    # Build all sweep models first.
    sweep_models = []

    for diameter_m in aoa_sweep_diameters:
        diameter_m = float(diameter_m)

        matching = [
            candidate
            for candidate in build_model_specs(geom=geom, diameter_m=diameter_m)
            if candidate.name == model.name
        ]

        if not matching:
            raise RuntimeError(
                f"Could not build {model.name} for channel height {diameter_m:g} m"
            )

        sweep_models.append(matching[0])


    # ParaView AoA sweep export is written for only one channel height.
    paraview_diameter_m = (
        float(aoa_sweep_diameters[0])
        if aoa_sweep_diameters
        else representative_pore_diameter(model)
    )

    print("[DEBUG] sweep models to run:")
    for i, sm in enumerate(sweep_models, start=1):
        print(
            f"  {i}: {sm.name}, "
            f"h={representative_channel_height(sm):.6e} m"
        )


    # Now run the sweep models only once.
    sweep_frames: list[pd.DataFrame] = []

    for sweep_model in sweep_models:
        diameter_m = representative_channel_height(sweep_model)
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
