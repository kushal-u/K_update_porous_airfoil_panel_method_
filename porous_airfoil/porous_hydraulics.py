"""Internal channel hydraulics, surface openings and passage networks.

Extracted from the supplied porous.py; numerical definitions are unchanged.
"""
from __future__ import annotations

import numpy as np
from .solver import FlowConfig, PanelGeometry, SurfacePoint
from .config import GeometryLimits, Chamber, Pore, PassageState
from . import config as settings

# w=0 for Re<=LOW; w=xi^2*(3-2*xi) for LOW<Re<HIGH; w=1 for Re>=HIGH.
# Defaults are LOW=2000, HIGH=3000, read from config by the caller.
def _smooth_transition_weight(
    reynolds: float,
    re_low: float = 2000.0,
    re_high: float = 3000.0,
) -> float:
    """Return a C1 smooth transition weight from laminar to turbulent flow."""
    if reynolds <= re_low:
        return 0.0
    if reynolds >= re_high:
        return 1.0
    xi = (reynolds - re_low) / (re_high - re_low)
    return float(xi * xi * (3.0 - 2.0 * xi))


def _rectangular_2d_pressure_drop_abs(
    mean_velocity: float,
    rho: float,
    mu: float,
    length: float,
    height: float,
) -> tuple[float, float, float, str]:
    """Pressure drop for a 2D parallel-wall channel at a given mean velocity.

    The hydraulic diameter is ``D_h = 2H``. The laminar Darcy factor
    ``f = 96/Re_Dh`` reproduces exactly ``Delta p = 12 mu L U/H^2`` and hence
    ``R' = 12 mu L/H^3``. A smooth blend to Blasius is used from Re=2000 to
    Re=3000 so the pressure-flow law remains continuous during coupling.
    """
    u = abs(float(mean_velocity))
    if u <= 0.0:
        return 0.0, 0.0, float("inf"), "zero_flow"

    # [eq:channel_velocity; eq:hydraulic_diameter; eq:channel_reynolds]
    # u=|q'|/H; D_h=2H; Re_h=rho*u*(2H)/mu=2*rho*|q'|/mu.
    hydraulic_diameter = 2.0 * float(height)
    reynolds = rho * u * hydraulic_diameter / max(mu, 1.0e-30)

    # [eq:channel_friction_factor; eq:turbulent_pressure_flow]
    # Laminar and turbulent endpoint correlations match the paper. Their smooth
    # interpolation below differs from its abrupt-switch piecewise equation.
    f_laminar = 96.0 / max(reynolds, 1.0e-30)
    f_turbulent = 0.3164 / max(reynolds, 1.0e-30) ** 0.25
    re_low = settings.HYDRAULIC_TRANSITION_RE_LOW
    re_high = settings.HYDRAULIC_TRANSITION_RE_HIGH
    blend = _smooth_transition_weight(reynolds, re_low, re_high)
    friction_factor = (1.0 - blend) * f_laminar + blend * f_turbulent

    # [eq:friction_drop; eq:laminar_pressure_flow; eq:laminar_resistance]
    # dp_f=f_D*(L/D_h)*(rho/2)*u^2 >=0. At f_D=96/Re_h this reduces to
    # dp_f=12*mu*L*|q'|/H^3. The earlier zero-flow return avoids 96/0.
    pressure_drop = (
        friction_factor
        * (length / hydraulic_diameter)
        * 0.5
        * rho
        * u**2
    )

    if reynolds <= re_low:
        model_name = "laminar_2d_rectangular_12muL_over_H3"
    elif reynolds < re_high:
        model_name = "transition_2d_rectangular_smooth_blend"
    else:
        model_name = "turbulent_2d_rectangular_blasius"

    return (
        float(pressure_drop),
        float(reynolds),
        float(friction_factor),
        model_name,
    )


def _channel_head_slope(q_abs, flow, length, height, kinetic_factor=0.0):
    """Analytical d(loss+jet head)/d|q'|, including transition derivatives.

    The laminar limit is evaluated directly at zero; neither 96/Re nor a
    numerical difference is used there.
    """
    q = abs(float(q_abs))
    # [eq:residual_jacobian; eq:laminar_resistance]
    # Hydraulic inverse derivative uses g'(Q), Q=|q'|, where
    # g(Q)=dp_f(Q)+alpha_j*rho*Q^2/(2H^2). Laminar dp_f'=R'.
    # Turbulent dp_f=K*Q^(7/4), so dp_f'=(7/4)*K*Q^(3/4).
    # Within the transition interval, differentiate the cubic blend too:
    # dp_f'=(1-w)*R' + w*dp_t' + w'*(dp_t-R'*Q).
    resistance = 12.0*flow.mu_inf*length/height**3
    re_slope = 2.0*flow.rho_inf/flow.mu_inf
    re = re_slope*q
    low = settings.HYDRAULIC_TRANSITION_RE_LOW
    high = settings.HYDRAULIC_TRANSITION_RE_HIGH
    if re <= low:
        slope = resistance
    else:
        # Turbulent head is K q^(7/4). Avoid singular f_D at zero.
        coefficient = (0.3164*re_slope**(-0.25)*length/(2.0*height)
                       * 0.5*flow.rho_inf/height**2)
        turbulent = coefficient*q**1.75
        turbulent_slope = 1.75*coefficient*q**0.75
        if re >= high:
            slope = turbulent_slope
        else:
            xi = (re-low)/(high-low)
            w = xi*xi*(3.0-2.0*xi)
            dw = 6.0*xi*(1.0-xi)*re_slope/(high-low)
            slope = ((1.0-w)*resistance + w*turbulent_slope
                     + dw*(turbulent-resistance*q))
    # [eq:channel_energy_equation; eq:residual_jacobian]
    # Add derivative of the outlet jet head: alpha_j*rho*Q/H^2.
    return float(slope + kinetic_factor*flow.rho_inf*q/height**2)


def solve_rectangular_channel_flow(
    dp: float,
    rho: float,
    mu: float,
    length: float,
    height: float,
    outlet_kinetic_factor: float = 0.0,
) -> tuple[float, float, float, float, str]:
    """Solve the 2D rectangular-channel flow rate per unit span.

    Returns
    -------
    tuple
        ``q_prime, R_prime_equivalent, Re_Dh, f_D, model_name``.

    Notes
    -----
    ``q_prime`` has units m^2/s. The flow-through area per unit span is H,
    therefore ``U_bar = q_prime/H``. The visible in-plane area ``L*H`` is not
    used as the flow-through area.
    When outlet_kinetic_factor > 0, dp is the available inlet-to-outlet
    head in pressure units, not just the distributed friction drop. The
    scalar equation includes outlet_kinetic_factor*rho*U_bar**2/2. This
    kinetic energy leaves with the jet; it is not a friction loss.
    """
    dp = float(dp)
    outlet_kinetic_factor = float(outlet_kinetic_factor)
    if not np.isfinite(outlet_kinetic_factor) or outlet_kinetic_factor < 0.0:
        raise ValueError("outlet_kinetic_factor must be finite and nonnegative.")
    rho = float(rho)
    mu = float(mu)
    length = float(length)
    height = float(height)

    if not all(np.isfinite(value) for value in (dp, rho, mu, length, height)):
        raise FloatingPointError("Non-finite rectangular-channel input.")
    if rho <= 0.0 or mu <= 0.0 or length <= 0.0 or height <= 0.0:
        raise ValueError("Density, viscosity, length and channel height must be positive.")

    dp_abs = abs(dp)
    if dp_abs <= 1.0e-30:
        return 0.0, float("inf"), 0.0, float("inf"), "zero_flow"

    sign = 1.0 if dp >= 0.0 else -1.0

    # [eq:channel_energy_equation]
    # Scalar hydraulic inversion at FIXED available head: find u>=0 satisfying
    # dp_f(H*u)+alpha_j*rho*u^2/2=|dp|, then q'=sign(dp)*H*u.
    # Here the argument dp already includes the inlet normal head if enabled.
    # This bracketing/bisection is separate from the simultaneous outer Newton
    # solve for all channel flows. Friction/Re are recomputed at each u.
    # Bracket and solve the monotonic pressure-drop law in mean velocity.
    u_low = 0.0
    u_high = 1.0
    for _ in range(100):
        dp_high, _, _, _ = _rectangular_2d_pressure_drop_abs(
            mean_velocity=u_high,
            rho=rho,
            mu=mu,
            length=length,
            height=height,
        )
        dp_high += outlet_kinetic_factor * 0.5 * rho * u_high**2
        if dp_high >= dp_abs:
            break
        u_high *= 2.0
    else:
        raise RuntimeError("Could not bracket the rectangular-channel flow solution.")

    for _ in range(80):
        u_mid = 0.5 * (u_low + u_high)
        dp_mid, _, _, _ = _rectangular_2d_pressure_drop_abs(
            mean_velocity=u_mid,
            rho=rho,
            mu=mu,
            length=length,
            height=height,
        )
        dp_mid += outlet_kinetic_factor * 0.5 * rho * u_mid**2
        if dp_mid < dp_abs:
            u_low = u_mid
        else:
            u_high = u_mid

    mean_velocity = 0.5 * (u_low + u_high)
    _, reynolds, friction_factor, model_name = _rectangular_2d_pressure_drop_abs(
        mean_velocity=mean_velocity,
        rho=rho,
        mu=mu,
        length=length,
        height=height,
    )

    q_prime_abs = mean_velocity * height
    if outlet_kinetic_factor > 0.0:
        model_name += "+outlet_jet_kinetic_head"
    # This returned equivalent resistance is available_head/|q'|.
    # It includes jet head when enabled; it is not generally the friction R'.
    equivalent_resistance = dp_abs / max(q_prime_abs, 1.0e-30)

    if not all(
        np.isfinite(value)
        for value in (q_prime_abs, equivalent_resistance, reynolds, friction_factor)
    ):
        raise FloatingPointError("Non-finite rectangular-channel result.")

    return (
        float(sign * q_prime_abs),
        float(equivalent_resistance),
        float(reynolds),
        float(friction_factor),
        str(model_name),
    )


# =============================================================================
# ONE INDEPENDENT CHANNEL
# =============================================================================
class IndependentPassage:
    def __init__(
        self,
        name: str,
        pore1: Pore,
        pore2: Pore,
        chamber: Chamber,
        span_width: float,
        layout_kind: str = "surface_pair",
    ) -> None:
        self.name = name
        self.pore1 = pore1
        self.pore2 = pore2
        self.chamber = chamber
        self.span_width = float(span_width)
        self.layout_kind = str(layout_kind)

    def pore_surface_point(self, geom: PanelGeometry, pore: Pore) -> SurfacePoint:
        return geom.surface_point_from_fraction(pore.x_frac, pore.side)

    # [subsec:internal_hydraulics]
    # For all active horizontal/perpendicular layouts, L is the direct distance
    # between surface endpoints. Other legacy layout kinds add half-thickness
    # corrections; those helpers are not part of the three active paper models.
    def passage_length(self, geom: PanelGeometry) -> float:
        sp1 = self.pore_surface_point(geom, self.pore1)
        sp2 = self.pore_surface_point(geom, self.pore2)

        def safe_thickness(x_frac: float) -> float:
            if x_frac <= 1e-8 or x_frac >= 1.0 - 1e-8:
                return 0.0
            return geom.local_thickness_at_fraction(x_frac)

        t1 = safe_thickness(self.pore1.x_frac)
        t2 = safe_thickness(self.pore2.x_frac)

        straight = np.hypot(sp2.x - sp1.x, sp2.y - sp1.y)

        # Straight in-plane channels use the direct surface-to-surface centreline
        # distance. For a lower-to-upper perpendicular passage on a symmetric
        # airfoil, this is exactly the local thickness; adding t1/2 + t2/2 would
        # double-count the same geometric distance.
        if self.layout_kind in {
            "internal_horizontal",
            "internal_chordwise_horizontal",
            "perpendicular",
        }:
            return max(float(straight), 1e-6)

        return max(float(straight + 0.5 * t1 + 0.5 * t2), 1e-6)


    def validate_geometry(self, geom: PanelGeometry, limits: GeometryLimits) -> None:
        for label, pore in (("pore1", self.pore1), ("pore2", self.pore2)):
            if not (limits.x_min_frac <= pore.x_frac <= limits.x_max_frac):
                raise ValueError(
                    f"{self.name}:{label} x/c={pore.x_frac:.4f} is outside "
                    f"[{limits.x_min_frac}, {limits.x_max_frac}]"
                )

            is_endpoint_pore = (
                abs(pore.x_frac - 0.0) <= limits.endpoint_tol
                or abs(pore.x_frac - 1.0) <= limits.endpoint_tol
            )

            # Exact LE/TE surface points have zero local thickness in 2D.
            # They are allowed only as endpoint pores.
            if not is_endpoint_pore:
                thickness = geom.local_thickness_at_fraction(pore.x_frac)

                if pore.height > limits.thickness_fraction_limit * thickness:
                    raise ValueError(
                        f"{self.name}:{label} height={pore.height:.4e} is too large "
                        f"for local thickness={thickness:.4e}"
                    )

        sp1 = self.pore_surface_point(geom, self.pore1)
        sp2 = self.pore_surface_point(geom, self.pore2)

        dist = float(np.hypot(sp2.x - sp1.x, sp2.y - sp1.y))
        required = 0.5 * (self.pore1.height + self.pore2.height) + limits.min_gap

        if dist < required:
            raise ValueError(
                f"{self.name} pore centers are too close: distance={dist:.4e}, "
                f"required={required:.4e}"
            )

        x_min = min(self.pore1.x_frac, self.pore2.x_frac)
        x_max = max(self.pore1.x_frac, self.pore2.x_frac)

        # Skip exact LE/TE zero-thickness locations for chamber fit checking.
        check_x_min = max(x_min, limits.endpoint_fit_margin)
        check_x_max = min(x_max, 1.0 - limits.endpoint_fit_margin)

        if check_x_min <= check_x_max:
            if abs(check_x_max - check_x_min) <= 1e-12:
                xs = np.array([check_x_min])
            else:
                xs = np.linspace(check_x_min, check_x_max, 40)

            t_min = min(geom.local_thickness_at_fraction(float(x)) for x in xs)

            if self.chamber.height > limits.fit_fraction * t_min:
                raise ValueError(
                    f"{self.name} rectangular channel is too tall for the minimum local thickness. "
                    f"height={self.chamber.height:.4e}, t_min={t_min:.4e}"
                )

    # [eq:channel_velocity; eq:hydraulic_diameter; eq:channel_reynolds]
    # q_abs is a per-span flow. Re_h=2*rho*|q'|/mu: explicit H cancels for
    # fixed q', although the solved q' itself depends on channel height.
    def equivalent_reynolds(self, q_abs: float, flow: FlowConfig) -> float:
        """Return Re based on q' per unit span and D_h=2H."""
        u_internal = q_abs / max(self.chamber.height, 1e-16)
        return (
            flow.rho_inf
            * abs(u_internal)
            * (2.0 * self.chamber.height)
            / max(flow.mu_inf, 1e-16)
        )


    # [subsec:interface_transfer; eq:opening_length]
    # Intersect the strip |(x-origin).transverse|<=H/2 with each straight panel;
    # return exact polygonal overlap fractions t0,t1. Actual overlap length is
    # ell_ik=(t1-t0)*S_i, not H or the whole panel length.
    def _opening_segments(
        self, geom: PanelGeometry, pore: Pore,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Intersect the channel strip with the polygonal airfoil surface.

        H is measured perpendicular to the centreline. Each returned segment
        is [t0,t1] within its original panel, with 0 <= t0 < t1 <= 1.
        Select the connected opening containing this end of the centreline;
        do not confuse it with the opening at the other end of the channel.
        Geometry is exact for the supplied straight-panel boundary.
        """
        cache = getattr(self, "_opening_cache", None)
        if cache is None or cache[0] is not geom:
            cache = (geom, {})
            self._opening_cache = cache
        key = (pore, self.pore1, self.pore2, self.chamber.height)
        if key in cache[1]:
            return cache[1][key]
        if not np.isclose(pore.height, self.chamber.height):
            raise ValueError("Opening and channel heights must agree.")
        a = self.pore_surface_point(geom, self.pore1)
        b = self.pore_surface_point(geom, self.pore2)
        origin = np.array([a.x, a.y])
        direction = np.array([b.x - a.x, b.y - a.y])
        length = np.linalg.norm(direction)
        if length <= 0.0:
            raise ValueError("Channel endpoints must be distinct.")
        direction /= length
        transverse = np.array([-direction[1], direction[0]])
        points = np.column_stack((geom.XB, geom.YB))
        z = (points - origin) @ transverse
        half_height = 0.5 * self.chamber.height
        dz = np.diff(z)
        lo = np.zeros(geom.num_pan)
        hi = np.ones(geom.num_pan)
        nonparallel = np.abs(dz) > 1e-14 * max(geom.chord, self.chamber.height)
        t_a = np.zeros_like(lo)
        t_b = np.zeros_like(lo)
        t_a[nonparallel] = (-half_height-z[:-1][nonparallel])/dz[nonparallel]
        t_b[nonparallel] = ( half_height-z[:-1][nonparallel])/dz[nonparallel]
        lo[nonparallel] = np.maximum(0.0, np.minimum(t_a, t_b)[nonparallel])
        hi[nonparallel] = np.minimum(1.0, np.maximum(t_a, t_b)[nonparallel])
        hi[~nonparallel & (np.abs(z[:-1]) > half_height)] = -1.0
        active = (hi-lo)*geom.S > 1e-14*geom.chord
        ids = np.flatnonzero(active)
        if not ids.size:
            raise ValueError(f"{self.name}: channel does not open on the boundary.")
        # Components in cyclic panel order, joining only across covered vertices.
        groups = [[int(ids[0])]]
        for i in ids[1:]:
            prev = groups[-1][-1]
            if i == prev+1 and hi[prev] >= 1.0-1e-12 and lo[i] <= 1e-12:
                groups[-1].append(int(i))
            else:
                groups.append([int(i)])
        if (len(groups) > 1 and groups[0][0] == 0
                and groups[-1][-1] == geom.num_pan-1
                and lo[0] <= 1e-12 and hi[-1] >= 1.0-1e-12):
            groups[0] = groups[-1]+groups[0]
            groups.pop()
        target = self.pore_surface_point(geom, pore)
        target_xy = np.array([target.x, target.y])
        def distance(group):
            ii = np.asarray(group)
            edges = points[ii+1]-points[ii]
            t = np.sum((target_xy-points[ii])*edges, axis=1)/geom.S[ii]**2
            t = np.clip(t, lo[ii], hi[ii])
            return np.min(np.linalg.norm(points[ii]+t[:, None]*edges-target_xy, axis=1))
        selected = min(groups, key=distance)
        if distance(selected) > 1e-5*geom.chord:
            raise ValueError(f"{self.name}: no opening contains the specified endpoint.")
        ii = np.asarray(selected, dtype=int)
        # A physical end must lie on its declared surface. Openings wrapping
        # around LE/TE need explicit geometry, rather than silently shifting them.
        side_ids = geom.upper_panel_ids if pore.side == "upper" else geom.lower_panel_ids
        if not np.all(np.isin(ii, side_ids)):
            raise ValueError(f"{self.name}: opening wraps around LE/TE; revise channel geometry.")
        result = (ii, lo[ii].copy(), hi[ii].copy())
        cache[1][key] = result
        return result

    def _surface_patch_weights(
        self, geom: PanelGeometry, pore: Pore,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compatibility API: actual panel intersection lengths, not fitted weights.

        Used only for conservative transfer of channel flow to panel velocities.
        Pressure is integrated independently by _area_average_pressure_at_pore.
        """
        ids, t0, t1 = self._opening_segments(geom, pore)
        return ids, (t1-t0)*geom.S[ids]

    # [eq:opening_pressure_average]
    # pbar=integral_opening p(s) ds / ell. Reconstruction is linear through
    # panel centres on the same surface and constant beyond endpoint centres.
    # The same routine averages already-squared panel velocities for kinetic
    # heads; it does not square the opening-average velocity.
    def _area_average_pressure_at_pore(
        self, geom: PanelGeometry, pore: Pore,
        panel_pressures: np.ndarray, min_panels: int = 0,
    ) -> float:
        """Integrate reconstructed p(s) over the geometrically resolved opening.

        p(s) is piecewise linear through surface panel-centre pressures, with
        constant continuation over the half-panels at the surface endpoints.
        Integration splits at every interpolation knot, so it is exact for
        this reconstruction. Accuracy of physical pressure still depends on
        panel resolution and the aerodynamic model. No overlap-weighted mean
        of piecewise-constant panel pressures is used.
        """
        del min_panels
        pressures = np.asarray(panel_pressures, dtype=float)
        if pressures.shape != (geom.num_pan,) or not np.all(np.isfinite(pressures)):
            raise ValueError("Panel pressures must be finite and match the panel count.")
        ids, t0, t1 = self._opening_segments(geom, pore)
        edges = np.concatenate(([0.0], np.cumsum(geom.S)))
        centres = 0.5*(edges[:-1]+edges[1:])
        side = geom.upper_panel_ids if pore.side == "upper" else geom.lower_panel_ids
        side = np.sort(side)
        knots = centres[side]
        values = pressures[side]
        integral = 0.0
        opening_length = 0.0
        for i, left, right in zip(ids, t0, t1):
            start = edges[i]+left*geom.S[i]
            end = edges[i]+right*geom.S[i]
            x = np.concatenate(([start], knots[(knots > start) & (knots < end)], [end]))
            y = np.interp(x, knots, values)
            integral += float(np.sum(0.5*(y[:-1]+y[1:])*np.diff(x)))
            opening_length += end-start
        return integral/opening_length

    # [eq:opening_pressure_average; subsec:coupled_solution]
    # Assemble W so pbar=W@p equals the independent reconstruction integral.
    # These interpolation/integration weights are distinct from geometric
    # overlaps used to conserve q'. Reduced quadratic pressures use W.
    def _pressure_average_weights(self, geom, pore):
        """Exact linear weights for the existing piecewise-linear p(s) integral.

        This is a coefficient assembly only. The original integration above
        remains independent and is used to verify the final coupled state.
        """
        cache = getattr(self, "_pressure_weight_cache", None)
        if cache is None or cache[0] is not geom:
            cache = (geom, {})
            self._pressure_weight_cache = cache
        key = (pore, self.pore1, self.pore2, self.chamber.height)
        if key in cache[1]:
            return cache[1][key]
        ids, t0, t1 = self._opening_segments(geom, pore)
        edges = np.r_[0.0, np.cumsum(geom.S)]
        centres = 0.5*(edges[:-1]+edges[1:])
        side = np.sort(geom.upper_panel_ids if pore.side == "upper"
                       else geom.lower_panel_ids)
        knots = centres[side]
        weights = np.zeros(geom.num_pan)
        opening_length = 0.0
        for i, left, right in zip(ids, t0, t1):
            start, end = edges[i]+left*geom.S[i], edges[i]+right*geom.S[i]
            x = np.r_[start, knots[(knots > start) & (knots < end)], end]
            node_weight = np.zeros(x.size)
            node_weight[:-1] += 0.5*np.diff(x)
            node_weight[1:] += 0.5*np.diff(x)
            for location, weight in zip(x, node_weight):
                j = int(np.searchsorted(knots, location))
                if j == 0:
                    weights[side[0]] += weight
                elif j == len(knots):
                    weights[side[-1]] += weight
                else:
                    fraction = (location-knots[j-1])/(knots[j]-knots[j-1])
                    weights[side[j-1]] += weight*(1.0-fraction)
                    weights[side[j]] += weight*fraction
            opening_length += end-start
        weights /= opening_length
        cache[1][key] = weights
        return weights


    def solve(
        self,
        geom: PanelGeometry,
        panel_pressures: np.ndarray,
        flow: FlowConfig,
        aero_result: object | None = None,
    ) -> PassageState:
        """Static-pressure-selected inlet, zero tangential recovery, separate jet.

        p_in + <rho*Vn_in^2/2> - p_out
            = dp_friction(q') + alpha_j*rho*(q'/H)^2/2.

        Inlet tangential kinetic energy is removed before entering the channel.
        Outlet static pressure matches the exterior, but its velocity/total
        pressure are those of the bulk jet, not the surrounding potential flow.
        This is a conservative reduced closure hypothesis; jet mixing and
        tangential recovery are not resolved or calibrated.
        """
        # [eq:channel_energy_equation]
        # dp=pbar1-pbar2 sets inlet/direction. The kinetic-head closure is enabled
        # by zero_tangential_recovery_jet; legacy_static uses static dp alone.
        p1 = self._area_average_pressure_at_pore(geom, self.pore1, panel_pressures)
        p2 = self._area_average_pressure_at_pore(geom, self.pore2, panel_pressures)
        dp = p1 - p2
        length = self.passage_length(geom)
        closure = settings.CHANNEL_ENERGY_CLOSURE
        if closure not in {"legacy_static", "zero_tangential_recovery_jet"}:
            raise ValueError(f"Unknown CHANNEL_ENERGY_CLOSURE: {closure}")
        normal_head = tangential_head = 0.0
        kinetic_factor = 0.0
        if closure == "zero_tangential_recovery_jet":
            if aero_result is None or getattr(aero_result, "Vn", None) is None:
                raise ValueError("Install the matching solver.py and pass aero_result "
                                 "to network.solve for the inlet/jet energy closure.")
            vn = np.asarray(aero_result.Vn, dtype=float)
            vt = np.asarray(aero_result.Vt, dtype=float)
            if vn.shape != (geom.num_pan,) or vt.shape != (geom.num_pan,):
                raise ValueError("Invalid aerodynamic velocity arrays for channel energy closure.")
            # Same linear averaging operator for pressure and kinetic heads:
            # <p> + <rho*(Vt^2+Vn^2)/2> = p0_inf, up to roundoff.
            inlet = self.pore1 if dp >= 0.0 else self.pore2
            # [eq:channel_energy_equation; eq:pressure_coefficient]
            # Average panel values rho*Vn_i^2/2 with the SAME linear operator as p.
            # Vn_i is the panel-average boundary velocity, not q'/ell on the open
            # fraction and not the bulk internal/jet speed q'/H. This distinction
            # matters where a mouth cuts a panel. Tangential head is diagnostic only.
            normal_head = self._area_average_pressure_at_pore(
                geom, inlet, 0.5*flow.rho_inf*vn**2)
            tangential_head = self._area_average_pressure_at_pore(
                geom, inlet, 0.5*flow.rho_inf*vt**2)
            kinetic_factor = float(settings.JET_KINETIC_ENERGY_FACTOR)
            if not np.isfinite(kinetic_factor) or kinetic_factor < 1.0:
                raise ValueError("JET_KINETIC_ENERGY_FACTOR must be finite and >= 1.")

        # This reduced pressure-driven model excludes inertia-driven flow against
        # the static-pressure gradient. At an exact tie it selects zero flow.
        # [eq:channel_energy_equation -- pressure-tie convention]
        # For nonzero dp: available=|dp|+normal_head, signed by static dp.
        # At dp==0 the implementation explicitly imposes q_hyd=0 even if the
        # normal head is nonzero; the energy equation needs this separate tie rule.
        direction = float(np.sign(dp))
        available_head = abs(dp) + normal_head if direction != 0.0 else 0.0
        q, _, reynolds, friction_factor, resistance_model = solve_rectangular_channel_flow(
            dp=direction*available_head,
            rho=flow.rho_inf,
            mu=flow.mu_inf,
            length=length,
            height=self.chamber.height,
            outlet_kinetic_factor=kinetic_factor,
        )
        # [eq:channel_velocity; eq:friction_drop; eq:channel_energy_equation]
        # Bulk speed U=|q'|/H; outlet head alpha_j*rho*U^2/2 is distinct from
        # distributed friction. The following diagnostics are in Pa or W/m span.
        speed = abs(q)/self.chamber.height
        friction_head = _rectangular_2d_pressure_drop_abs(
            speed, flow.rho_inf, flow.mu_inf, length, self.chamber.height)[0]
        jet_head = kinetic_factor*0.5*flow.rho_inf*speed**2
        # Rs retains the original static-pressure-difference / flow meaning.
        rs = abs(dp)/abs(q) if q != 0.0 else float("inf")
        state = PassageState(
            Q=float(q), p1=p1, p2=p2, dp_total=float(abs(dp)), Rs=float(rs),
            reynolds_equivalent=float(reynolds), friction_factor=float(friction_factor),
            resistance_model=str(resistance_model),
        )
        p_in, p_out = (p1, p2) if dp >= 0.0 else (p2, p1)
        active = direction != 0.0
        energy_residual = available_head-friction_head-jet_head
        state.energy_diagnostics = {
            "energy_closure": closure,
            "energy_balance_active": bool(active),
            "inlet_pore": 1 if dp > 0.0 else (2 if dp < 0.0 else 0),
            "inlet_normal_kinetic_head_Pa": float(normal_head),
            "inlet_tangential_head_removed_Pa": float(tangential_head),
            "available_channel_head_Pa": float(available_head),
            "channel_friction_loss_Pa": float(friction_head),
            "outlet_jet_kinetic_head_Pa": float(jet_head),
            "outlet_jet_speed_m_s": float(speed),
            "channel_inlet_total_pressure_Pa": float(p_in+normal_head) if active else float("nan"),
            "outlet_jet_total_pressure_Pa": float(p_out+jet_head) if active else float("nan"),
            "energy_balance_residual_Pa": float(energy_residual),
            "friction_power_W_per_m_span": float(abs(q)*friction_head),
            "inlet_turning_power_W_per_m_span": float(abs(q)*tangential_head),
            "outlet_kinetic_energy_flux_W_per_m_span": float(abs(q)*jet_head),
        }
        if abs(energy_residual) > 1e-10*max(1.0, available_head):
            raise FloatingPointError("Channel energy equation failed its residual check.")
        return state


    def transpiration_from_flow(
        self,
        geom: PanelGeometry,
        state: PassageState,
    ) -> np.ndarray:
        """Map q' [m^2/s] over the actual channel/surface intersection.

        The panel-average values conserve flow exactly:

            sum(v_n,i S_i) = +/- q'

        No out-of-plane span width appears because both the internal and
        external coupling are formulated per unit span.
        """
        vn = np.zeros(geom.num_pan, dtype=float)

        q_prime = float(state.Q)
        if abs(q_prime) < 1.0e-30:
            return vn

        ids1, overlap1 = self._surface_patch_weights(geom, self.pore1)
        ids2, overlap2 = self._surface_patch_weights(geom, self.pore2)

        # [eq:opening_length; eq:opening_velocity;
        # eq:inlet_flow_conservation; eq:outlet_flow_conservation]
        # ell_1=sum_i overlap1_i; ell_2=sum_i overlap2_i.
        # Vn_i^(1)=-q'*ell_i1/(ell_1*S_i), Vn_i^(2)=+q'*ell_i2/(ell_2*S_i).
        # Hence each signed opening sum(Vn_i*S_i) is exactly -q' or +q'.
        opening1 = max(float(np.sum(overlap1)), 1.0e-16)
        opening2 = max(float(np.sum(overlap2)), 1.0e-16)

        # Uniform physical velocity over the opening, represented as a
        # panel-average when an opening cuts through a partial panel.
        vn[ids1] -= q_prime * overlap1 / (opening1 * geom.S[ids1])
        vn[ids2] += q_prime * overlap2 / (opening2 * geom.S[ids2])

        return vn

# =============================================================================
# NETWORK OF INDEPENDENT CHANNELS
# =============================================================================
# [subsec:model_overview; eq:normal_velocity_map]
# Channels have independent hydraulics (no junction pressure unknowns).
# Their Vn contributions superpose; their flows couple through exterior p.
class IndependentPassageNetwork:
    def __init__(self, passages: list[IndependentPassage]) -> None:
        if not passages:
            raise ValueError("At least one passage is required.")
        self.passages = passages

    def validate_geometry(self, geom: PanelGeometry, limits: GeometryLimits) -> None:
        for passage in self.passages:
            passage.validate_geometry(geom, limits)

        # Check real opening intervals, including partial panels. Centre
        # distances alone cannot detect oblique rectangular-opening overlap.
        openings = []
        for passage in self.passages:
            for label, pore in (("pore1", passage.pore1), ("pore2", passage.pore2)):
                ids, t0, t1 = passage._opening_segments(geom, pore)
                intervals = {int(i): (float(a), float(b)) for i, a, b in zip(ids, t0, t1)}
                for previous_name, previous in openings:
                    for i in intervals.keys() & previous.keys():
                        overlap = min(intervals[i][1], previous[i][1])-max(intervals[i][0], previous[i][0])
                        if overlap*geom.S[i] > 1e-12*geom.chord:
                            raise ValueError(
                                f"Actual rectangular openings overlap: {previous_name} "
                                f"and {passage.name}:{label}. Revise the channel layout."
                            )
                openings.append((f"{passage.name}:{label}", intervals))

        pore_refs: list[tuple[str, Pore, SurfacePoint]] = []
        for passage in self.passages:
            pore_refs.append(
                (
                    f"{passage.name}:pore1",
                    passage.pore1,
                    passage.pore_surface_point(geom, passage.pore1),
                )
            )
            pore_refs.append(
                (
                    f"{passage.name}:pore2",
                    passage.pore2,
                    passage.pore_surface_point(geom, passage.pore2),
                )
            )

        for i in range(len(pore_refs)):
            name_i, pore_i, sp_i = pore_refs[i]

            for j in range(i + 1, len(pore_refs)):
                name_j, pore_j, sp_j = pore_refs[j]

                dist = float(np.hypot(sp_i.x - sp_j.x, sp_i.y - sp_j.y))
                required = 0.5 * (pore_i.height + pore_j.height) + limits.min_gap

                if dist < required:
                    raise ValueError(
                        f"Pore overlap between {name_i} and {name_j}: "
                        f"distance={dist:.4e}, required={required:.4e}"
                    )

    def solve(
        self,
        geom: PanelGeometry,
        panel_pressures: np.ndarray,
        flow: FlowConfig,
        aero_result: object | None = None,
    ) -> list[PassageState]:
        return [
            passage.solve(
                geom=geom,
                panel_pressures=panel_pressures,
                aero_result=aero_result,
                flow=flow,
            )
            for passage in self.passages
        ]

    def transpiration_from_flow(
        self,
        geom: PanelGeometry,
        states: list[PassageState],
    ) -> np.ndarray:
        vn = np.zeros(geom.num_pan, dtype=float)

        for passage, state in zip(self.passages, states):
            vn += passage.transpiration_from_flow(
                geom=geom,
                state=state,
            )

        return vn



# =============================================================================
# COUPLED PANEL + POROUS SOLVER
# =============================================================================
