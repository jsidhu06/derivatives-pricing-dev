"""Vanilla PDE engine: backward solver, grid-greek mixin, valuation classes.

``_fd_core`` runs the backward finite-difference solve for vanilla and
custom-payoff (PayoffSpec) options; ``_FDGridGreeksMixin`` extracts
delta/gamma/theta from a solved grid; ``_FDValuationBase`` and the
European/American subclasses plug into ``OptionValuation``.
"""

from __future__ import annotations
from collections.abc import Callable
from typing import TYPE_CHECKING

import logging
import threading
import warnings

import numpy as np

from ...enums import (
    ExerciseType,
    PDEEarlyExercise,
    PDEMethod,
    PDESpaceGrid,
    OptionType,
)
from ...rates import DiscountCurve
from ...utils import log_timing
from ..contracts import (
    BarrierSpec,
    PayoffSpec,
    PayoffBoundaryModel,
    VanillaSpec,
)
from ..params import PDEParams
from ._kernels import (
    _apply_dividend_jump,
    _boundary_values,
    _build_log_grid,
    _build_tau_grid,
    _build_time_step_schedule,
    _check_explicit_spot_stability,
    _dividend_tau_schedule,
    _explicit_step,
    _fit_affine_boundary_model,
    _implicit_cn_step,
    _log_operator_coeffs,
    _scaled_operator_coeffs,
    _spot_operator_coeffs,
    _validate_fd_inputs,
)

if TYPE_CHECKING:
    from ..core import OptionValuation, UnderlyingData


logger = logging.getLogger(__name__)


def _fd_core(
    *,
    spot: float,
    strike: float | None,
    time_to_maturity: float,
    volatility: float,
    discount_curve: DiscountCurve,
    dividend_curve: DiscountCurve | None,
    dividend_schedule: list[tuple[float, float]] | None,
    option_type: OptionType | None,
    smax_mult: float,
    spot_steps: int,
    time_steps: int,
    early_exercise: bool,
    method: PDEMethod,
    rannacher_steps: int,
    space_grid: PDESpaceGrid,
    american_solver: PDEEarlyExercise | None = None,
    omega: float | None = None,
    tol: float | None = None,
    max_iter: int | None = None,
    payoff_fn: Callable | None = None,
    payoff_boundary_model: PayoffBoundaryModel | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    """Core finite-difference solver for option valuation.

    Supports vanilla CALL/PUT (via *option_type* and *strike*) and
    arbitrary payoffs (via *payoff_fn*).  When *payoff_fn* is provided
    it takes precedence and *option_type*/*strike* may be ``None``.

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray, np.ndarray, float]
        ``(price, spot_grid, V_final, V_prev, last_dtau)``
        where *V_prev* is the value slice one time step before pricing
        time and *last_dtau* is the size of that step (both needed for
        theta extraction from the grid).
    """
    _validate_fd_inputs(
        option_type=option_type,
        time_to_maturity=time_to_maturity,
        spot_steps=spot_steps,
        time_steps=time_steps,
        volatility=volatility,
        discount_curve=discount_curve,
        early_exercise=early_exercise,
        method=method,
        american_solver=american_solver,
        omega=omega,
        tol=tol,
        max_iter=max_iter,
        payoff_fn=payoff_fn,
    )

    # For grid sizing, use strike when available, otherwise use spot.
    # After the grid is built, normalize smin/smax to the ACTUAL grid
    # boundaries so all downstream code uses a single consistent meaning.
    ref_price = max(spot, strike) if strike is not None else spot
    smax = float(smax_mult * ref_price)
    if space_grid is PDESpaceGrid.SPOT:
        grid = np.linspace(0.0, smax, spot_steps + 1)
        S = grid
        dS = smax / spot_steps
    else:
        grid, S, dz = _build_log_grid(
            spot=spot,
            strike=strike if strike is not None else spot,
            time_to_maturity=time_to_maturity,
            volatility=volatility,
            smax_mult=smax_mult,
            spot_steps=spot_steps,
            time_steps=time_steps,
            method=method,
        )

    smin = float(S[0])
    smax = float(S[-1])

    j = np.arange(1, spot_steps)  # interior indices 1..M-1

    # Terminal payoff at maturity
    if payoff_fn is not None:
        payoff = np.asarray(payoff_fn(S), dtype=float)
    elif option_type is OptionType.PUT:
        payoff = np.maximum(strike - S, 0.0)
    else:
        payoff = np.maximum(S - strike, 0.0)

    # Resolve affine wing boundary models once (used for boundary
    # conditions on every time step). Prefer explicit metadata; fall back
    # to a local affine fit on the actual PDE boundary nodes.
    if payoff_fn is not None:
        if payoff_boundary_model is None:
            payoff_boundary_model = PayoffBoundaryModel(
                left=_fit_affine_boundary_model(payoff_fn, wing="left", spot_samples=S[:4]),
                right=_fit_affine_boundary_model(payoff_fn, wing="right", spot_samples=S[-4:]),
            )
        elif space_grid is PDESpaceGrid.LOG_SPOT:
            warnings.warn(
                "Explicit PayoffBoundaryModel with LOG_SPOT grid is interpreted as an affine "
                "boundary model on the finite truncated PDE domain, not as a true payoff tail "
                "asymptote. Ensure the supplied boundary model is appropriate at the actual grid "
                "boundaries.",
                stacklevel=2,
            )

    V = payoff.copy()  # V at tau=0 (maturity)
    intrinsic = payoff if early_exercise else None

    schedule = dividend_schedule or []
    # Round keys to 12dp to absorb float arithmetic noise; lookups must also round.
    dividend_map = {round(tau, 12): amount for tau, amount in schedule}

    # Maturity-date dividend (tau=0): apply as an immediate jump
    # V(S,T⁻)=V(S-D,T⁺) right after setting the terminal condition.
    mat_div = dividend_map.pop(0.0, None)
    if mat_div is not None:
        _apply_dividend_jump(V, grid, mat_div, space_grid=space_grid)
        if early_exercise:
            V[:] = np.maximum(V, payoff)

    # Pricing-date dividend (tau=ttm): will be applied as a spot shift
    # at interpolation time.  Remove from the map so it's not applied
    # as a mid-grid jump during time-stepping.
    ttm_key = round(time_to_maturity, 12)
    pricing_div = dividend_map.pop(ttm_key, None)

    dividend_taus = list(dividend_map.keys())
    tau_grid = _build_tau_grid(time_to_maturity, time_steps, dividend_taus)

    if method in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL) and space_grid is PDESpaceGrid.SPOT:
        _check_explicit_spot_stability(
            tau_grid=tau_grid,
            volatility=volatility,
            smax=smax,
            dS=dS,
            time_to_maturity=time_to_maturity,
            discount_curve=discount_curve,
            dividend_curve=dividend_curve,
            implicit_discounting=method is PDEMethod.EXPLICIT_HULL,
        )

    # March forward in tau: 0 -> T (equivalently backward in calendar time)
    df_0T = float(discount_curve.df(time_to_maturity))  # P(0,T)
    if dividend_curve is not None:
        dq_0T = float(dividend_curve.df(time_to_maturity))  # Dq(0,T)
    else:
        dq_0T = None

    psor_steps = 0
    psor_total_iters = 0
    psor_max_iters = 0
    psor_not_converged = 0

    steps = _build_time_step_schedule(tau_grid, method, rannacher_steps)

    V_prev = V.copy()
    last_dtau = 0.0

    for tau_prev, tau_curr, method_used in steps:
        d_tau = tau_curr - tau_prev
        t_prev = time_to_maturity - tau_prev
        t_curr = time_to_maturity - tau_curr

        r = float(discount_curve.forward_rate(t_curr, t_prev))
        if dividend_curve is not None:
            q = float(dividend_curve.forward_rate(t_curr, t_prev))
        else:
            q = 0.0

        implicit_discounting = method_used is PDEMethod.EXPLICIT_HULL

        if space_grid is PDESpaceGrid.SPOT:
            gamma, beta, alpha = _spot_operator_coeffs(
                spot_values=S[1:-1],
                dS=dS,
                risk_free_rate=r,
                dividend_rate=q,
                volatility=volatility,
                implicit_discounting=implicit_discounting,
            )
        else:
            gamma, beta, alpha = _log_operator_coeffs(
                dz=dz,
                risk_free_rate=r,
                dividend_rate=q,
                volatility=volatility,
                implicit_discounting=implicit_discounting,
                size=spot_steps - 1,
            )

        df_0t = float(discount_curve.df(t_curr))
        df_tT: float = df_0T / df_0t
        if dividend_curve is not None:
            dq_0t = float(dividend_curve.df(t_curr))
            dq_tT: float = dq_0T / dq_0t  # type: ignore[operator]
        else:
            dq_tT = 1.0

        left, right = _boundary_values(
            option_type=option_type,
            strike=strike,
            smin=smin,
            smax=smax,
            df_tT=df_tT,
            dq_tT=dq_tT,
            early_exercise=early_exercise,
            payoff_fn=payoff_fn,
            payoff_boundary_model=payoff_boundary_model,
        )

        V_prev = V.copy()
        last_dtau = d_tau

        a, b, c = _scaled_operator_coeffs(gamma=gamma, beta=beta, alpha=alpha, d_tau=d_tau)

        intrinsic_for_step = intrinsic if early_exercise else None

        if method_used in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL):
            V = _explicit_step(
                V_prev,
                j,
                a,
                b,
                c,
                left,
                right,
                intrinsic_for_step,
                r_dt=r * d_tau if implicit_discounting else 0.0,
            )
        else:
            V, psor_iters = _implicit_cn_step(
                V_prev,
                V,
                j,
                a,
                b,
                c,
                left,
                right,
                method_used,
                intrinsic_for_step,
                american_solver,
                omega,
                tol,
                max_iter,
            )
            if psor_iters is not None:
                psor_steps += 1
                psor_total_iters += psor_iters
                psor_max_iters = max(psor_max_iters, psor_iters)
                if psor_iters == int(max_iter):
                    psor_not_converged += 1

        # Apply discrete dividend jump at tau if needed
        if dividend_map:
            amount = dividend_map.get(round(tau_curr, 12))
            if amount is not None:
                _apply_dividend_jump(V, grid, amount, space_grid=space_grid)
                if early_exercise:
                    V[:] = np.maximum(V, intrinsic)

    if psor_steps > 0:
        avg_iters = psor_total_iters / psor_steps
        logger.debug(
            "PDE PSOR steps=%d avg_iters=%.2f max_iters=%d not_converged=%d",
            psor_steps,
            avg_iters,
            psor_max_iters,
            psor_not_converged,
        )

    # Apply pricing-date dividend: the input spot is cum-dividend, so
    # interpolate at S₀ − D to get the ex-dividend option value.
    interp_spot = spot - pricing_div if pricing_div is not None else spot
    price = np.interp(interp_spot, S, V)
    return price, S, V, V_prev, last_dtau


class _FDGridGreeksMixin:
    """Mixin providing delta/gamma/theta extracted from the PDE solution grid.

    Subclasses must define ``_solve()`` returning
    ``(price, S, V, V_prev, last_dtau)``.
    """

    valuation_ctx: OptionValuation
    underlying: UnderlyingData

    @staticmethod
    def _spot_grid_index(S: np.ndarray, spot: float) -> int:
        """Return the nearest interior grid index to the current spot."""
        j = int(np.searchsorted(S, spot))
        return max(1, min(j, len(S) - 2))

    @staticmethod
    def _grid_gamma_at_index(S: np.ndarray, V: np.ndarray, j: int) -> float:
        """Return the non-uniform three-point gamma stencil at index ``j``."""
        h_up = S[j + 1] - S[j]
        h_dn = S[j] - S[j - 1]
        return float(
            2.0
            * (V[j + 1] * h_dn + V[j - 1] * h_up - V[j] * (h_up + h_dn))
            / (h_up * h_dn * (h_up + h_dn))
        )

    @staticmethod
    def _grid_delta_at_spot(S: np.ndarray, V: np.ndarray, j: int, spot: float) -> float:
        """Parabolic-Lagrange first derivative evaluated exactly at ``spot``.

        Differentiates the quadratic interpolant through nodes
        ``(S[j-1], S[j], S[j+1])`` at the actual spot rather than at the
        nearest node ``S[j]``. This removes an O(spot - S[j]) discretization
        bias that shows up whenever the query spot doesn't coincide with a
        grid node.

        This is especially important for KI parity as the vanilla and KO
        surfaces live on different grids.
        """
        x0, x1, x2 = S[j - 1], S[j], S[j + 1]
        v0, v1, v2 = V[j - 1], V[j], V[j + 1]
        return float(
            v0 * (2.0 * spot - x1 - x2) / ((x0 - x1) * (x0 - x2))
            + v1 * (2.0 * spot - x0 - x2) / ((x1 - x0) * (x1 - x2))
            + v2 * (2.0 * spot - x0 - x1) / ((x2 - x0) * (x2 - x1))
        )

    @staticmethod
    def _grid_gamma_at_spot(S: np.ndarray, V: np.ndarray, j: int, spot: float) -> float:
        """Cubic-Lagrange second derivative evaluated exactly at ``spot``.

        A 3-point parabolic fit gives a *constant* second derivative
        (``f''(x) = 2a``), so a parabolic at-spot evaluation is identical
        to the at-index value. A 4-point cubic Lagrange yields
        ``f''(x) = 6ax + 2b`` (linear in ``x``), so evaluating at exactly
        ``spot`` decouples the result from the local node placement and
        removes an O(spot - S[j]) bias whenever the query spot doesn't
        coincide with a grid node.

        Especially important for KI parity: vanilla and KO live on
        different grids, so both gammas need to be referenced to the same
        physical ``spot`` rather than to differently-placed ``S[j]`` nodes.

        Uses nodes ``(S[j-1], S[j], S[j+1], S[j+2])``; the index ``j`` is
        clamped so all four neighbours exist.
        """
        n = len(S)
        jc = max(1, min(j, n - 3))
        x0, x1, x2, x3 = S[jc - 1], S[jc], S[jc + 1], S[jc + 2]
        v0, v1, v2, v3 = V[jc - 1], V[jc], V[jc + 1], V[jc + 2]
        # For the cubic Lagrange basis L_i(x) = prod_{k!=i}(x - x_k) / D_i,
        # L_i''(x) = (6x - 2 * sum_{k!=i} x_k) / D_i.
        s0 = x1 + x2 + x3
        s1 = x0 + x2 + x3
        s2 = x0 + x1 + x3
        s3 = x0 + x1 + x2
        d0 = (x0 - x1) * (x0 - x2) * (x0 - x3)
        d1 = (x1 - x0) * (x1 - x2) * (x1 - x3)
        d2 = (x2 - x0) * (x2 - x1) * (x2 - x3)
        d3 = (x3 - x0) * (x3 - x1) * (x3 - x2)
        return float(
            v0 * (6.0 * spot - 2.0 * s0) / d0
            + v1 * (6.0 * spot - 2.0 * s1) / d1
            + v2 * (6.0 * spot - 2.0 * s2) / d2
            + v3 * (6.0 * spot - 2.0 * s3) / d3
        )

    # Relative tolerance for the cubic-vs-parabolic gamma agreement check;
    # if cubic disagrees with parabolic by more than this fraction (of the
    # larger magnitude), we treat the cubic as polluted by PDE noise and
    # fall back to the parabolic value.
    _GAMMA_CUBIC_PARABOLIC_REL_TOL: float = 0.5

    @staticmethod
    def _grid_gamma_safe(S: np.ndarray, V: np.ndarray, j: int, spot: float) -> float:
        """Cubic-at-spot gamma with parabolic-at-index fallback.

        The cubic 4-point stencil is more accurate where ``V`` is locally
        smooth, but it has wider reach and larger basis-function weights
        than the parabolic 3-point stencil, so it amplifies PDE noise more
        aggressively (e.g. PSOR oscillations near American exercise
        boundaries, or noise leaking back from the KI coupling step in
        the two-surface solver). When the two stencils disagree by more
        than ``_GAMMA_CUBIC_PARABOLIC_REL_TOL``, we treat the cubic as
        polluted and fall back to the parabolic value.

        For the well-behaved scenarios cubic and parabolic agree closely
        and the cubic is returned (slightly more accurate); for the
        pathological scenarios the parabolic safety net catches the
        amplified noise.
        """
        parabolic = _FDGridGreeksMixin._grid_gamma_at_index(S, V, j)
        cubic = _FDGridGreeksMixin._grid_gamma_at_spot(S, V, j, spot)
        scale = max(abs(parabolic), abs(cubic), 1e-6)
        if abs(cubic - parabolic) > _FDGridGreeksMixin._GAMMA_CUBIC_PARABOLIC_REL_TOL * scale:
            return parabolic
        return cubic

    def _solve(
        self,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]: ...

    def _grid_greeks_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, float]:
        """Run the PDE solve and locate the spot node.

        Returns
        -------
        S, V, V_prev, last_dtau, j, spot
            The spot grid, value vector, previous-step value vector,
            last time-step size, the spot-grid index closest to the
            current spot, and the spot itself.
        """
        _, S, V, V_prev, last_dtau = self._solve()
        spot = float(self.underlying.initial_value)
        j = self._spot_grid_index(S, spot)
        return S, V, V_prev, last_dtau, j, spot

    def _intrinsic_short_circuit_greeks(
        self, S: np.ndarray, V: np.ndarray, j: int
    ) -> tuple[float, float] | None:
        """Return ``(delta, gamma)`` if the spot node sits in the
        early-exercise region of an American option, else ``None``.

        When ``V[j]`` equals the intrinsic value at ``S[j]``, the node
        lies in the early-exercise region and the local value function is
        ``V(s) = max(K - s, 0)`` (put) or ``V(s) = max(s - K, 0)`` (call).
        Local greeks are then exact (``delta = ±1``, ``gamma = 0``) and
        the PDE stencil extraction is unreliable due to PSOR oscillations
        near the exercise boundary — both parabolic and cubic stencils
        can produce wildly wrong values because the noise is in the
        underlying ``V`` samples, not in the stencil.

        Only fires for American options on a CALL/PUT spec with strictly
        positive intrinsic value at ``S[j]``.
        """
        spec = self.valuation_ctx.spec
        if not isinstance(spec, (VanillaSpec, BarrierSpec)):
            return None
        if spec.exercise_type is not ExerciseType.AMERICAN:
            return None
        strike = float(spec.strike)
        s_node = float(S[j])
        if spec.option_type is OptionType.CALL:
            intrinsic = s_node - strike
            sign = 1.0
        else:
            intrinsic = strike - s_node
            sign = -1.0
        if intrinsic <= 0.0:
            return None
        # PV at the node should equal intrinsic when in the exercise region.
        # PSOR convergence is exact at the exercise constraint up to the
        # tolerance ``tol``; allow a small relative slack.
        if abs(V[j] - intrinsic) > max(1e-8, 1e-6 * intrinsic):
            return None
        return (sign, 0.0)

    def delta(self) -> float:
        r"""Grid delta via parabolic-Lagrange first derivative at exactly
        ``spot`` (not at the nearest node).

        Collapses to the standard central difference on a uniform grid
        when ``spot`` coincides with a node.

        Short-circuits to ``±1`` when the option is American and the
        spot node sits in the early-exercise region (PV equals intrinsic);
        the PDE stencil is unreliable there because of PSOR oscillations.
        """
        S, V, _, _, j, spot = self._grid_greeks_data()
        short_circuit = self._intrinsic_short_circuit_greeks(S, V, j)
        if short_circuit is not None:
            return short_circuit[0]
        return self._grid_delta_at_spot(S, V, j, spot)

    def gamma(self) -> float:
        r"""Grid gamma via cubic-Lagrange second derivative at exactly
        ``spot``, with a parabolic fallback when the cubic stencil
        appears polluted by PDE noise (see ``_grid_gamma_safe``).

        Short-circuits to ``0`` when the option is American and the
        spot node sits in the early-exercise region (PV equals intrinsic).
        """
        S, V, _, _, j, spot = self._grid_greeks_data()
        short_circuit = self._intrinsic_short_circuit_greeks(S, V, j)
        if short_circuit is not None:
            return short_circuit[1]
        return self._grid_gamma_safe(S, V, j, spot)

    def _grid_theta_bs_identity(
        self,
        S: np.ndarray,
        V: np.ndarray,
        j: int,
        spot: float,
        last_dtau: float,
    ) -> float:
        r"""Return per-calendar-day theta via the Black-Scholes PDE identity.

        .. math::

            \Theta = r V - (r - q) S \Delta - \tfrac{1}{2} \sigma^{2} S^{2} \Gamma

        ``V`` is interpolated to exact ``spot`` with the same parabolic
        Lagrange stencil used by :meth:`_grid_delta_at_spot`; ``r`` and
        ``q`` are the forward rates over the first PDE step (exact for
        flat curves, first-order local otherwise).  Result is divided by
        365 to match the calendar-day convention used elsewhere.
        """
        # PV at exactly spot (parabolic Lagrange through the delta stencil).
        x0, x1, x2 = S[j - 1], S[j], S[j + 1]
        v0, v1, v2 = V[j - 1], V[j], V[j + 1]
        pv_at_spot = float(
            v0 * (spot - x1) * (spot - x2) / ((x0 - x1) * (x0 - x2))
            + v1 * (spot - x0) * (spot - x2) / ((x1 - x0) * (x1 - x2))
            + v2 * (spot - x0) * (spot - x1) / ((x2 - x0) * (x2 - x1))
        )
        delta = self._grid_delta_at_spot(S, V, j, spot)
        gamma = self._grid_gamma_safe(S, V, j, spot)

        sigma = float(self.underlying.volatility)
        dt_probe = max(last_dtau, 1.0e-8)
        r = float(self.valuation_ctx.discount_curve.forward_rate(0.0, dt_probe))
        q = 0.0
        if self.underlying.dividend_curve is not None:
            q = float(self.underlying.dividend_curve.forward_rate(0.0, dt_probe))

        theta_annual = (
            r * pv_at_spot - (r - q) * spot * delta - 0.5 * sigma * sigma * spot * spot * gamma
        )
        return float(theta_annual / 365.0)

    def theta(self) -> float:
        r"""Grid theta via the Black-Scholes PDE identity.

        Uses :meth:`_grid_theta_bs_identity` so theta attains the same
        order of accuracy as the grid delta and gamma — notably better
        near steep spatial gradients such as a barrier, where a
        first-order backward time-difference amplifies the local error
        in ``V``.  Returned per **calendar day**.

        Short-circuits to ``0`` when the option is American and the spot
        node sits in the early-exercise region; the Black-Scholes PDE
        becomes an inequality there and the identity no longer holds.
        """
        S, V, _, last_dtau, j, spot = self._grid_greeks_data()
        short_circuit = self._intrinsic_short_circuit_greeks(S, V, j)
        if short_circuit is not None:
            return 0.0
        if last_dtau <= 0.0:
            return 0.0
        return self._grid_theta_bs_identity(S, V, j, spot, last_dtau)


class _FDValuationBase(_FDGridGreeksMixin):
    """Base class for European/American FD valuation."""

    _early_exercise: bool = False

    def __init__(self, valuation_ctx: OptionValuation) -> None:
        self.valuation_ctx = valuation_ctx
        self.underlying = valuation_ctx.underlying  # type: ignore[assignment]
        assert isinstance(valuation_ctx.params, PDEParams)
        self.pde_params = valuation_ctx.params
        # Lazy PDE-solve cache.  Populated on first ``_solve`` call and
        # shared by every subsequent PV / grid-greek access on this
        # instance, so ``delta`` / ``gamma`` / ``theta`` after a
        # ``present_value`` call are O(1) grid lookups.  Double-checked
        # locking keeps the fast path lock-free on cache hits while
        # guaranteeing at most one expensive solve under concurrent access.
        self._solve_result: tuple[float, np.ndarray, np.ndarray, np.ndarray, float] | None = None
        # ``RLock`` so nested caching calls within the same thread don't
        # deadlock (e.g. an engine that dispatches to another cached
        # helper while holding the solve lock).
        self._solve_lock = threading.RLock()

    def solve(self) -> tuple[float, np.ndarray, np.ndarray]:
        """Compute the full FD solution on the spot grid at pricing time."""
        pv, S, V, *_ = self._solve()
        return pv, S, V

    def _solve(self) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
        """Memoised PDE solve result (see ``_compute_solve`` for the real work)."""
        if self._solve_result is not None:
            return self._solve_result
        with self._solve_lock:
            if self._solve_result is not None:
                return self._solve_result
            self._solve_result = self._compute_solve()
            return self._solve_result

    def _compute_solve(self) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
        """Run the PDE finite-difference solve."""
        params = self.pde_params

        if self._early_exercise:
            logger.debug(
                "PDE American method=%s grid=%s solver=%s spot_steps=%d time_steps=%d",
                params.method.value,
                params.space_grid.value,
                params.american_solver.value,
                params.spot_steps,
                params.time_steps,
            )
        else:
            logger.debug(
                "PDE European method=%s grid=%s spot_steps=%d time_steps=%d",
                params.method.value,
                params.space_grid.value,
                params.spot_steps,
                params.time_steps,
            )

        spot = float(self.underlying.initial_value)
        strike = self.valuation_ctx.strike
        volatility = float(self.underlying.volatility)
        discount_curve = self.valuation_ctx.discount_curve
        dividend_curve = self.underlying.dividend_curve
        discrete_dividends = self.underlying.discrete_dividends

        time_to_maturity = self.valuation_ctx._maturity_year_fraction()

        dividend_schedule = _dividend_tau_schedule(
            discrete_dividends=discrete_dividends,
            pricing_date=self.valuation_ctx.pricing_date,
            maturity=self.valuation_ctx.maturity,
            day_count_convention=self.valuation_ctx.day_count_convention,
        )

        smax_mult = float(params.smax_mult)
        spot_steps = int(params.spot_steps)
        time_steps = int(params.time_steps)

        # Custom payoff support: extract payoff callable and boundary model from PayoffSpec
        spec = self.valuation_ctx.spec
        if isinstance(spec, PayoffSpec):
            custom_payoff = spec.payoff
            custom_boundary_model = spec.boundary_model
        else:
            custom_payoff = None
            custom_boundary_model = None

        return _fd_core(
            spot=spot,
            strike=float(strike) if strike is not None else None,
            time_to_maturity=float(time_to_maturity),
            volatility=volatility,
            discount_curve=discount_curve,
            dividend_curve=dividend_curve,
            dividend_schedule=dividend_schedule,
            option_type=self.valuation_ctx.option_type,
            smax_mult=smax_mult,
            spot_steps=spot_steps,
            time_steps=time_steps,
            early_exercise=self._early_exercise,
            method=params.method,
            rannacher_steps=int(params.rannacher_steps),
            space_grid=params.space_grid,
            american_solver=params.american_solver if self._early_exercise else None,
            omega=float(params.omega) if self._early_exercise else None,
            tol=float(params.tol) if self._early_exercise else None,
            max_iter=int(params.max_iter) if self._early_exercise else None,
            payoff_fn=custom_payoff,
            payoff_boundary_model=custom_boundary_model,
        )

    def present_value(self) -> float:
        """Return present value from the PDE solve."""
        label = "PDE American" if self._early_exercise else "PDE European"
        with log_timing(logger, f"{label} present_value", self.pde_params.log_timings):
            pv, *_ = self._solve()
        return float(pv)


class _FDEuropeanValuation(_FDValuationBase):
    """European option valuation using PDE finite differences."""

    _early_exercise = False


class _FDAmericanValuation(_FDValuationBase):
    """American option valuation using PDE finite differences."""

    _early_exercise = True
