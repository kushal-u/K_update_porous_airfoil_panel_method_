"""Reduced aerodynamic coupling and damped Newton solution.

Extracted from the supplied porous.py; numerical definitions are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from .config import PassageState, MultiPassageResult
from . import config as settings
from .porous_hydraulics import (
    _channel_head_slope,
    _rectangular_2d_pressure_drop_abs,
    solve_rectangular_channel_flow,
)

class ReducedFlowProblem:
    """Exact static condensation of the discretised aerodynamic equations.

    z_k=q'_k/(H_k V_inf), u=[cos(alpha), sin(alpha), z]. Precompute
    Vt=T u, Vn=N u and opening pressure perturbations u^T P_l u.
    The unknown flow vector remains nonlinear; only the Newton corrections
    are solved with a linear matrix. No Taylor truncation of Bernoulli is made.

    The original flow-mismatch residual is retained. Its Jacobian differentiates
    the scalar hydraulic inverse implicitly, including f_D(Re) and transition.
    Static-pressure inlet selection is piecewise smooth; derivatives are valid
    on the selected branch, not across a finite-head pressure-tie discontinuity.
    """

    def __init__(self, aero_solver, network, coupling_strength=1.0):
        self.aero_solver, self.network = aero_solver, network
        self.geom, self.flow = aero_solver.geom, aero_solver.flow
        self.strength = float(coupling_strength)
        if not np.isfinite(self.strength) or not 0.0 <= self.strength <= 1.0:
            raise ValueError("coupling_strength must lie in [0, 1].")
        self.closure = settings.CHANNEL_ENERGY_CLOSURE
        if self.closure not in {"legacy_static", "zero_tangential_recovery_jet"}:
            raise ValueError("Unsupported CHANNEL_ENERGY_CLOSURE.")
        self.kinetic_factor = (float(settings.JET_KINETIC_ENERGY_FACTOR)
                               if self.closure != "legacy_static" else 0.0)
        if self.closure != "legacy_static" and (not np.isfinite(self.kinetic_factor)
                                                or self.kinetic_factor < 1.0):
            raise ValueError("JET_KINETIC_ENERGY_FACTOR must be finite and >= 1.")
        self.heights = np.array([p.chamber.height for p in network.passages])
        self.lengths = np.array([p.passage_length(self.geom) for p in network.passages])
        # [eq:channel_flow_vector; eq:flow_scaling]
        # q_scale=diag(S), S=V_inf*diag(H_k); q=S*z and z_k=q'_k/(H_k*V_inf).
        self.q_scale = self.heights*self.flow.v_inf
        self.size = len(self.heights)
        # [eq:normal_velocity_map; eq:reduced_normal_velocity]
        # Each column is the Vn response to q'_k=1, giving Vn=M*q.
        # During homotopy only, Vn=chi*M*q with chi=coupling_strength; physical
        # results use chi=1. chi is unrelated to source strengths lambda_i.
        self.normal_map = np.column_stack([
            p.transpiration_from_flow(self.geom, PassageState(1.0, 0, 0, 0, 0, 0))
            for p in network.passages
        ])
        # [eq:precomputed_aerodynamic_responses; eq:strength_reconstruction;
        # eq:reduced_tangential_velocity]
        # strengths_q=X_q, tangent_q=T_q; freestream columns produce x_0,Vt_0.
        # The following basis multiplies u=[cos(alpha),sin(alpha),z]. At chi=1,
        # x=x_0+X_q*q, Vt=Vt_0+T_q*q, and Vn=M*q exactly.
        strengths_q, tangent_q = aero_solver.linear_flow_response(self.normal_map)
        strengths_inf, tangent_inf = aero_solver.freestream_response()
        self.strength_basis = np.column_stack((
            strengths_inf, strengths_q*(self.strength*self.q_scale)[None, :]))
        tangent = np.column_stack((
            tangent_inf, tangent_q*(self.strength*self.q_scale)[None, :]))
        normal = np.column_stack((np.zeros((self.geom.num_pan, 2)),
                                  self.normal_map*(self.strength*self.q_scale)[None, :]))
        weights = np.array([
            p._pressure_average_weights(self.geom, pore)
            for p in network.passages for pore in (p.pore1, p.pore2)
        ])
        # [eq:pressure_coefficient; eq:panel_pressure; eq:opening_pressure_average]
        # If Vt=T*u, Vn=N*u, define symmetric P_l=-rho/2*sum_i W_li
        # *(T_i^T*T_i+N_i^T*N_i). Then pbar_l=p_inf+rho*V_inf^2/2+u^T*P_l*u.
        # Normal-head tensors use +rho/2*sum_i W_li*N_i^T*N_i.
        # This preserves all quadratic terms; no Bernoulli Taylor truncation.
        rho_half = 0.5*self.flow.rho_inf
        self.pressure_quadratic = -rho_half*(
            np.einsum("li,ij,ik->ljk", weights, tangent, tangent, optimize=True)
            + np.einsum("li,ij,ik->ljk", weights, normal, normal, optimize=True))
        self.normal_quadratic = rho_half*np.einsum(
            "li,ij,ik->ljk", weights, normal, normal, optimize=True)
        self._last_key = None
        self.evaluation_count = 0

    def evaluate(self, z, alpha_deg=None):
        """Return (R, dR/dz, dR/dalpha_radian) without a panel solve.

        At dp=0 the original model sets Q_hyd=0. If normal inlet head is also
        zero, use its continuous laminar derivative. Otherwise the tie is
        discontinuous: use a zero hydraulic branch derivative at the tie and
        let the line search and original-residual acceptance handle the switch.
        """
        z = np.asarray(z, dtype=float)
        if z.shape != (self.size,) or not np.all(np.isfinite(z)):
            raise FloatingPointError("Invalid reduced flow state.")
        alpha = self.flow.aoa_deg if alpha_deg is None else float(alpha_deg)
        if not np.isfinite(alpha):
            raise FloatingPointError("Invalid angle of attack.")
        key = (z.tobytes(), alpha)
        if key == self._last_key:
            return self._last_value
        angle = np.deg2rad(alpha)
        u = np.r_[np.cos(angle), np.sin(angle), z]
        du_da = np.r_[-np.sin(angle), np.cos(angle), np.zeros(self.size)]
        # Gauge pressures omit the common p_inf+rho*V_inf^2/2. This avoids
        # cancellation of atmospheric pressure in the reduced residual.
        p = np.einsum("i,lij,j->l", u, self.pressure_quadratic, u)
        # [eq:residual_jacobian]
        # For a symmetric quadratic tensor, grad_u(u^T*P_l*u)=2*P_l*u.
        # pg and hg are exact opening-pressure and normal-head derivatives.
        pg = 2.0*np.einsum("lij,j->li", self.pressure_quadratic, u)
        head = np.einsum("i,lij,j->l", u, self.normal_quadratic, u)
        hg = 2.0*np.einsum("lij,j->li", self.normal_quadratic, u)
        residual = z.copy()
        jacobian = np.eye(self.size)
        derivative_alpha = np.zeros(self.size)
        self.q_hydraulic = np.zeros(self.size)
        self.static_dp = p[::2]-p[1::2]
        self.available_head = np.zeros(self.size)
        self.pressure_ties = []
        for k in range(self.size):
            dp = self.static_dp[k]
            dp_gradient = pg[2*k]-pg[2*k+1]
            direction = float(np.sign(dp))
            inlet = 2*k if direction >= 0.0 else 2*k+1
            normal_head = head[inlet] if self.closure != "legacy_static" else 0.0
            normal_gradient = hg[inlet] if self.closure != "legacy_static" else np.zeros(u.size)
            available = abs(dp)+normal_head if direction != 0.0 else 0.0
            q_hyd = solve_rectangular_channel_flow(
                direction*available, self.flow.rho_inf, self.flow.mu_inf,
                self.lengths[k], self.heights[k], self.kinetic_factor)[0]
            slope = _channel_head_slope(
                abs(q_hyd), self.flow, self.lengths[k], self.heights[k], self.kinetic_factor)
            if not np.isfinite(slope) or slope <= 0.0:
                raise FloatingPointError("Hydraulic inverse requires a positive finite slope.")
            if direction != 0.0:
                # [eq:residual_jacobian]
                # On a fixed inlet branch s=sign(dp), g(Q)=|dp|+h_n, q_hyd=s*Q:
                # dq_hyd/du = s*(s*d(dp)/du + dh_n/du)/g'(Q).
                # g' includes the active cubic friction blend and the jet-head derivative.
                q_gradient = direction*(direction*dp_gradient+normal_gradient)/slope
            elif (self.closure == "legacy_static" or
                  max(head[2*k], head[2*k+1]) <= 64.0*np.finfo(float).eps
                  *max(1.0, self.flow.rho_inf*self.flow.v_inf**2)):
                q_gradient = dp_gradient/slope
            else:
                self.pressure_ties.append(k)
                q_gradient = np.zeros(u.size)
            # [eq:coupled_residual; eq:residual_jacobian]
            # R=z-S^-1*q_hyd; J_R=I-S^-1*(dq_hyd/dz).
            # The alpha derivative below is per RADIAN for pseudo-arclength use.
            residual[k] -= q_hyd/self.q_scale[k]
            jacobian[k] -= q_gradient[2:]/self.q_scale[k]
            derivative_alpha[k] = -np.dot(q_gradient, du_da)/self.q_scale[k]
            self.q_hydraulic[k] = q_hyd
            self.available_head[k] = available
        if not all(np.all(np.isfinite(a)) for a in (residual, jacobian, derivative_alpha)):
            raise FloatingPointError("Non-finite reduced residual or Jacobian.")
        self.evaluation_count += 1
        self._last_key = key
        self._last_value = (residual, jacobian, derivative_alpha)
        return self._last_value

    def residual_and_jacobian(self, z):
        residual, jacobian, _ = self.evaluate(z)
        return residual, jacobian


@dataclass
class _NewtonOutcome:
    x: np.ndarray
    residual_norm: float
    converged: bool
    iterations: int
    evaluations: int
    condition_number: float
    smallest_singular_value: float
    method: str
    message: str


def _damped_newton(evaluate, initial, coupling, tolerance=None, verbose=False):
    """Small dense Newton system with line search and analytical-Jacobian TRF.

    evaluate(x) supplies the ORIGINAL residual and its consistent Jacobian.
    A least-squares stationary point never passes without a small residual.
    """
    x = np.asarray(initial, dtype=float).copy()
    best_x = x.copy()
    best_norm = float("inf")
    calls, iterations = 0, 0
    condition, smallest = float("nan"), float("nan")
    method, message = "reduced_newton", "iteration limit reached"
    last_x, last_value = None, None

    def checked(value):
        nonlocal calls, best_x, best_norm, last_x, last_value
        if last_x is not None and np.array_equal(value, last_x):
            return last_value
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            r, j = evaluate(value)
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(j)):
            raise FloatingPointError("Non-finite nonlinear residual or Jacobian.")
        calls += 1
        norm = float(np.linalg.norm(r, np.inf))
        if norm < best_norm:
            best_x, best_norm = np.array(value, copy=True), norm
        last_x, last_value = np.array(value, copy=True), (r, j)
        return r, j

    # [subsec:verification_outputs; eq:flow_acceptance_tolerance]
    # Default physical flow-residual threshold:
    # tau=sqrt(machine epsilon)*max(1,||z||inf), approximately 1.49e-8
    # when ||z||inf<=1. PAL correctors instead pass their own explicit tolerance.
    def threshold(value):
        return (float(tolerance) if tolerance is not None else
                float(np.sqrt(np.finfo(float).eps)*max(1.0, np.linalg.norm(value, np.inf))))

    try:
        for iterations in range(1, coupling.max_iter+1):
            r, j = checked(x)
            singular = np.linalg.svd(j, compute_uv=False)
            smallest = float(singular[-1])
            condition = float(singular[0]/smallest) if smallest > 0.0 else float("inf")
            if np.linalg.norm(r, np.inf) <= threshold(x):
                message = "original residual criterion satisfied"
                break
            # [eq:newton_system]
            # Solve J_R*delta_z=-R. For ill-conditioned J_R, np.linalg.lstsq supplies
            # a truncated least-squares linear step. A failed descent check switches
            # to a residual-merit gradient direction.
            if condition < coupling.condition_limit:
                step = np.linalg.solve(j, -r)
            else:
                step = np.linalg.lstsq(j, -r, rcond=1.0/coupling.condition_limit)[0]
            gradient = j.T @ r
            descent = float(gradient @ step)
            if not np.isfinite(descent) or descent >= 0.0:
                step = -gradient/max(1.0, np.linalg.norm(gradient))
                descent = float(gradient @ step)
            if descent >= 0.0:
                message = "no residual descent direction"
                break
            # [eq:newton_update]
            # Armijo merit phi=||R||_2^2/2; require
            # phi(z+eta*step)<=phi(z)+armijo*eta*grad(phi).step.
            # Start eta=1 and halve at each backtrack; trial residuals are nonlinear.
            merit = 0.5*float(r @ r)
            damping, accepted = 1.0, False
            for _ in range(coupling.max_backtracks):
                trial = x+damping*step
                try:
                    rt, _ = checked(trial)
                    if (0.5*float(rt @ rt) <= merit+coupling.armijo*damping*descent
                            or np.linalg.norm(rt, np.inf) <= threshold(trial)):
                        accepted = True
                        break
                except (FloatingPointError, OverflowError, RuntimeError):
                    pass
                damping *= 0.5
            if not accepted:
                message = "line search exhausted"
                break
            # [eq:newton_update]
            # Accept z_next=z+eta*delta_z (or the safeguarded descent step).
            x = trial
            if verbose:
                print(f"[Newton] iteration={iterations:03d}, R={np.linalg.norm(rt, np.inf):.3e}, "
                      f"step={damping:.3g}, cond(J)={condition:.3e}")
    except (FloatingPointError, OverflowError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        message = f"Newton stopped: {type(exc).__name__}: {exc}"

    # [subsec:coupled_solution]
    # Trust-region reflective nonlinear least squares uses the same original
    # residual and analytical Jacobian; a solver success flag alone cannot
    # replace the independent residual acceptance criterion.
    if best_norm > threshold(best_x) and coupling.least_squares_fallback:
        try:
            lsq = least_squares(
                lambda v: checked(v)[0], best_x, jac=lambda v: checked(v)[1],
                method="trf", x_scale="jac", ftol=settings.TRF_FTOL,
                xtol=settings.TRF_XTOL, gtol=settings.TRF_GTOL,
                max_nfev=int(coupling.max_iter))
            checked(lsq.x)
            method = "reduced_newton+analytical_TRF"
            message += "; TRF: "+str(lsq.message).replace("\n", " ")
        except (FloatingPointError, OverflowError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            message += f"; TRF stopped: {type(exc).__name__}: {exc}"
    try:
        r, j = checked(best_x)
        singular = np.linalg.svd(j, compute_uv=False)
        smallest = float(singular[-1])
        condition = float(singular[0]/smallest) if smallest > 0.0 else float("inf")
    except (FloatingPointError, OverflowError, ValueError, RuntimeError, np.linalg.LinAlgError):
        pass
    return _NewtonOutcome(best_x, best_norm, best_norm <= threshold(best_x),
                          iterations, calls, condition, smallest, method, message)


class MultiPassageCoupledSolver:
    def __init__(self, aero_solver, network, coupling, limits):
        self.aero_solver, self.network = aero_solver, network
        self.coupling, self.limits = coupling, limits

    def solve(self, verbose=False, initial_vn=None, initial_q=None, coupling_strength=1.0):
        """Exact panel condensation followed by damped Newton in channel flows.

        Interfaces/seeds/homotopy match the previous solver. The final candidate
        is checked by an independent full panel solve and the original hydraulic
        inversion. Returned channel states use Q_trial, the same flows used to
        build the returned aerodynamic field, and retain original residuals.
        """
        geom, flow = self.aero_solver.geom, self.aero_solver.flow
        if initial_q is not None and initial_vn is not None:
            raise ValueError("Provide initial_q or initial_vn, not both.")
        if not np.isfinite(coupling_strength) or not 0.0 <= coupling_strength <= 1.0:
            raise ValueError("coupling_strength must lie in [0, 1].")
        try:
            self.network.validate_geometry(geom, self.limits)
        except ValueError as exc:
            zero = np.zeros(geom.num_pan)
            return MultiPassageResult(
                self.aero_solver.solve(zero), [], zero, False, 0,
                float("nan"), float("nan"), coupling_status="invalid_geometry",
                solver_message=str(exc))
        problem = ReducedFlowProblem(self.aero_solver, self.network, coupling_strength)
        q_scale = problem.q_scale
        if initial_q is not None:
            seed = np.asarray(initial_q, dtype=float)
            if seed.shape != q_scale.shape or not np.all(np.isfinite(seed)):
                raise ValueError(f"initial_q must be finite with shape {q_scale.shape}.")
            z0 = seed/q_scale
        elif initial_vn is not None:
            vn = np.asarray(initial_vn, dtype=float)
            if vn.shape != (geom.num_pan,) or not np.all(np.isfinite(vn)):
                raise ValueError(f"initial_vn must be finite with shape ({geom.num_pan},).")
            seed_aero = self.aero_solver.solve(vn)
            states = self.network.solve(geom, self.aero_solver.panel_pressures(seed_aero), flow, seed_aero)
            z0 = np.array([state.Q for state in states])/q_scale
        else:
            problem.evaluate(np.zeros(problem.size))
            z0 = problem.q_hydraulic.copy()/q_scale
        outcome = _damped_newton(problem.residual_and_jacobian, z0, self.coupling, verbose=verbose)
        # [eq:flow_scaling; subsec:verification_outputs]
        # Recover q=S*z, then rebuild Vn and solve the full aerodynamic equations.
        # Final checks below do not use the condensed pressure tensors.
        q_trial = outcome.x*q_scale
        vn = coupling_strength*(problem.normal_map @ q_trial)
        # Independent, original evaluator. No condensed pressure coefficients
        # participate in this final acceptance check.
        aero = self.aero_solver.solve(vn)
        pressures = self.aero_solver.panel_pressures(aero)
        states = self.network.solve(geom, pressures, flow, aero)
        q_hyd = np.array([state.Q for state in states])
        # [eq:coupled_residual; subsec:verification_outputs]
        # Independent original residual ||S^-1*(q_trial-q_hyd(full panel p))||inf.
        # Returned states are subsequently set to q_trial so fields and stored flows
        # refer to the same candidate, not to a separate hydraulic prediction.
        original_norm = float(np.linalg.norm((q_trial-q_hyd)/q_scale, np.inf))
        tolerance = float(np.sqrt(np.finfo(float).eps)*max(1.0, np.linalg.norm(outcome.x, np.inf)))
        energy_norm = 0.0
        for k, (passage, state) in enumerate(zip(self.network.passages, states)):
            hydraulic_q = state.Q
            state.Q = float(q_trial[k])
            speed = abs(state.Q)/passage.chamber.height
            friction, re, factor, model = _rectangular_2d_pressure_drop_abs(
                speed, flow.rho_inf, flow.mu_inf, problem.lengths[k], problem.heights[k])
            jet = problem.kinetic_factor*0.5*flow.rho_inf*speed**2
            diagnostics = state.energy_diagnostics
            available = diagnostics["available_channel_head_Pa"]
            balance = available-friction-jet
            energy_norm = max(energy_norm, abs(balance))
            diagnostics.update({
                "hydraulic_predicted_q_per_unit_span_m2_s": float(hydraulic_q),
                "channel_friction_loss_Pa": float(friction),
                "outlet_jet_kinetic_head_Pa": float(jet),
                "outlet_jet_speed_m_s": float(speed),
                "energy_balance_residual_Pa": float(balance),
                "friction_power_W_per_m_span": float(abs(state.Q)*friction),
                "inlet_turning_power_W_per_m_span": float(abs(state.Q)*diagnostics["inlet_tangential_head_removed_Pa"]),
                "outlet_kinetic_energy_flux_W_per_m_span": float(abs(state.Q)*jet),
            })
            if diagnostics["energy_balance_active"]:
                outlet_pressure = state.p2 if diagnostics["inlet_pore"] == 1 else state.p1
                diagnostics["outlet_jet_total_pressure_Pa"] = float(outlet_pressure+jet)
            state.reynolds_equivalent, state.friction_factor = re, factor
            state.resistance_model = model+("+outlet_jet_kinetic_head" if problem.kinetic_factor else "")
            state.Rs = abs(state.p1-state.p2)/abs(state.Q) if state.Q else float("inf")
        # The flow residual is authoritative. Also check the equivalent energy
        # balance, scaled by its local hydraulic slope, and flow direction.
        # [subsec:verification_outputs; eq:channel_energy_equation;
        # eq:energy_acceptance_check]
        # For each channel, e_k=available_head-dp_f(|q_trial|)-jet_head.
        # Require max_k |e_k|/[g'_k(max(|q_trial|,|q_hyd|))*q_scale_k]<=2*tau,
        # plus flow mismatch <=tau and direction consistency (with residual-sized
        # roundoff allowance). No separate hard Re_h=2300 branch check is present.
        energy_flow_norm = max((
            abs(s.energy_diagnostics["energy_balance_residual_Pa"])
            /max(_channel_head_slope(max(abs(s.Q), abs(q_hyd[k])), flow,
                                     problem.lengths[k], problem.heights[k], problem.kinetic_factor)
                 *q_scale[k], np.finfo(float).tiny)
            for k, s in enumerate(states)), default=0.0)
        direction_ok = all(q == 0.0 or np.sign(q) == np.sign(s.p1-s.p2)
                           for q, s in zip(q_trial, states))
        # At pressure ties the original closure requires zero flow; roundoff
        # deviations below the flow-residual tolerance are allowed as before.
        direction_ok = direction_ok or bool(np.max(np.abs(q_trial-q_hyd)/q_scale) <= tolerance)
        accepted = bool(outcome.converged and original_norm <= tolerance
                        and energy_flow_norm <= 2.0*tolerance and direction_ok)
        max_vn = float(np.max(np.abs(vn)))
        return MultiPassageResult(
            aero, states, vn, accepted, outcome.iterations, max_vn, max_vn/flow.v_inf,
            coupling_status="converged" if accepted else "not_converged",
            residual_norm=original_norm, residual_tolerance=tolerance,
            solver_success=accepted,
            solver_message=(f"lambda={coupling_strength:.3f}; selected={outcome.method}; "
                            f"{outcome.message}; original_R={original_norm:.3e}; "
                            f"energy_R_Pa={energy_norm:.3e}; cond(J)={outcome.condition_number:.3e}"),
            function_evaluations=problem.evaluation_count,
            newton_iterations=outcome.iterations,
            jacobian_condition_number=outcome.condition_number,
            jacobian_smallest_singular_value=outcome.smallest_singular_value,
            reduced_residual_norm=outcome.residual_norm,
            original_energy_residual_max_pa=energy_norm)


