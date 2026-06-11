"""Barrier PDE core solvers.

Knock-out cores (single and double barrier; continuous via truncated
Dirichlet grids, discrete via full-grid resets at monitoring dates) and the
two-surface coupled knock-in cores used for American KI.  Each returns the
``(price, S, V, V_prev, last_dtau)`` 5-tuple consumed by the valuation
classes in :mod:`._barrier_valuation`.
"""

from __future__ import annotations
import logging

import numpy as np

from ...enums import (
    BarrierDirection,
    BarrierMonitoring,
    PDEEarlyExercise,
    PDEMethod,
    PDESpaceGrid,
    OptionType,
    RebateTiming,
)
from ...rates import DiscountCurve
from ...exceptions import ConfigurationError
from ._kernels import (
    _apply_dividend_jump,
    _boundary_values,
    _build_log_grid,
    _build_spot_grid,
    _build_tau_grid,
    _build_time_step_schedule,
    _check_explicit_spot_stability,
    _explicit_step,
    _implicit_cn_step,
    _log_operator_coeffs,
    _scaled_operator_coeffs,
    _solve_tridiagonal_thomas,
    _spot_operator_coeffs,
    _validate_fd_inputs,
)
from ._barrier_grids import (
    _build_double_barrier_continuous_log_grid,
    _build_double_barrier_discrete_grid,
    _build_double_barrier_full_grid,
    _build_ko_continuous_log_grid,
)

logger = logging.getLogger(__name__)


def _fd_barrier_ko_core(
    *,
    spot: float,
    strike: float,
    time_to_maturity: float,
    volatility: float,
    discount_curve: DiscountCurve,
    dividend_curve: DiscountCurve | None,
    dividend_schedule: list[tuple[float, float]] | None,
    option_type: OptionType,
    barrier: float,
    direction: BarrierDirection,
    monitoring: BarrierMonitoring,
    rebate: float,
    rebate_timing: RebateTiming,
    monitoring_taus: list[float] | None,  # required (not None) for DISCRETE
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
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    """Core finite-difference solver for knock-out barrier options.

    For continuous knock-out barriers, the grid is truncated at the barrier
    level so the barrier becomes a domain boundary.

    For discrete knock-out barriers, the full grid is used and barrier
    resets are applied at monitoring dates (analogous to dividend jumps).

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray, np.ndarray, float]
        ``(price, spot_grid, V_final, V_prev, last_dtau)``
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
    )

    continuous = monitoring is BarrierMonitoring.CONTINUOUS

    if not continuous and monitoring_taus is None:
        raise ConfigurationError("monitoring_taus is required for discrete barrier monitoring.")

    # ── Grid construction ─────────────────────────────────────────────
    # For continuous KO, truncate the grid at the barrier.
    # For discrete KO, use the standard full grid.
    ref_price = max(spot, strike)

    if continuous:
        if direction is BarrierDirection.DOWN:
            smin_target = barrier
            smax_target = smax_mult * ref_price
        else:
            smin_target = max(ref_price / smax_mult, 1.0e-8)
            smax_target = barrier
    else:
        smin_target = None
        smax_target = None

    if space_grid is PDESpaceGrid.LOG_SPOT:
        if continuous:
            grid, S, dz = _build_ko_continuous_log_grid(
                smin_target=smin_target,
                smax_target=smax_target,
                ref_price=ref_price,
                smax_mult=smax_mult,
                direction=direction,
                volatility=volatility,
                time_to_maturity=time_to_maturity,
                spot_steps=spot_steps,
                time_steps=time_steps,
                method=method,
            )
        else:
            grid, S, dz = _build_log_grid(
                spot=spot,
                strike=strike,
                time_to_maturity=time_to_maturity,
                volatility=volatility,
                smax_mult=smax_mult,
                spot_steps=spot_steps,
                time_steps=time_steps,
                method=method,
                anchor_spot=barrier,
                anchor_half_step=True,
            )
    else:
        # Spot grid
        if continuous:
            grid = np.linspace(smin_target, smax_target, spot_steps + 1)
            S = grid
            dS = (smax_target - smin_target) / spot_steps
        else:
            smax = float(smax_mult * ref_price)
            grid, S, dS = _build_spot_grid(
                smin=0.0,
                smax=smax,
                spot_steps=spot_steps,
                anchor_spot=barrier,
                anchor_half_step=True,
            )

    smin = float(S[0])
    smax = float(S[-1])

    j = np.arange(1, spot_steps)  # interior indices

    # ── Terminal payoff ───────────────────────────────────────────────
    if option_type is OptionType.PUT:
        payoff = np.maximum(strike - S, 0.0)
    else:
        payoff = np.maximum(S - strike, 0.0)

    # American intrinsic = vanilla payoff at any in-life moment, regardless of
    # KO-zone modifications applied at maturity below.  Discrete monitoring
    # needs this: between obs dates the holder can exercise even sitting in
    # the KO zone.  (For continuous KO the grid is truncated at the barrier,
    # so the alive-side array is the only one that exists — same logic holds.)
    intrinsic = payoff.copy() if early_exercise else None

    # For continuous KO: payoff is zero on the barrier side (enforced
    # by grid truncation since the barrier is at the boundary).
    # For discrete KO: zero out the payoff on the knocked-out side at maturity
    # only if maturity is a monitoring date (which it typically is).
    if not continuous:
        assert monitoring_taus is not None  # validated above
        # tau=0 is maturity; if it's a monitoring tau, apply the reset.
        # Rebate-aware: at tau=0 df_tT=1 so both AT_HIT and AT_EXPIRY
        # rebate timings give the undiscounted rebate as the terminal
        # value in the KO zone
        if any(abs(tau) < 1e-12 for tau in monitoring_taus):
            if direction is BarrierDirection.DOWN:
                payoff[S <= barrier] = rebate
            else:
                payoff[S >= barrier] = rebate

    V = payoff.copy()

    # ── Dividend schedule ─────────────────────────────────────────────
    schedule = dividend_schedule or []
    # Round keys to 12dp to absorb float arithmetic noise; lookups must also round.
    dividend_map = {round(tau, 12): amount for tau, amount in schedule}
    mat_div = dividend_map.pop(0.0, None)
    if mat_div is not None:
        _apply_dividend_jump(V, grid, mat_div, space_grid=space_grid)
        if early_exercise:
            V[:] = np.maximum(V, payoff)

    ttm_key = round(time_to_maturity, 12)
    pricing_div = dividend_map.pop(ttm_key, None)

    # ── Merge monitoring taus into grid ───────────────────────────────
    dividend_taus = list(dividend_map.keys())
    extra_taus = dividend_taus.copy()
    monitoring_tau_set: set[float] | None = None
    if not continuous:
        assert monitoring_taus is not None  # validated above
        extra_taus.extend(monitoring_taus)
        monitoring_tau_set = {round(t, 12) for t in monitoring_taus}

    tau_grid = _build_tau_grid(time_to_maturity, time_steps, extra_taus)

    if method in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL) and space_grid is PDESpaceGrid.SPOT:
        _check_explicit_spot_stability(
            tau_grid=tau_grid,
            volatility=volatility,
            smax=smax,
            dS=dS if space_grid is PDESpaceGrid.SPOT else (smax - smin) / spot_steps,
            time_to_maturity=time_to_maturity,
            discount_curve=discount_curve,
            dividend_curve=dividend_curve,
            implicit_discounting=method is PDEMethod.EXPLICIT_HULL,
        )

    # ── Time-stepping ─────────────────────────────────────────────────
    df_0T = float(discount_curve.df(time_to_maturity))
    if dividend_curve is not None:
        dq_0T = float(dividend_curve.df(time_to_maturity))
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
                dS=(smax - smin) / spot_steps,
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

        # ── Barrier boundary conditions ───────────────────────────────
        df_0t = float(discount_curve.df(t_curr))
        df_tT: float = df_0T / df_0t
        if dividend_curve is not None:
            dq_0t = float(dividend_curve.df(t_curr))
            dq_tT: float = dq_0T / dq_0t  # type: ignore[operator]
        else:
            dq_tT = 1.0

        if continuous:
            # Barrier is at a grid boundary; set its value
            if rebate == 0.0:
                barrier_bv = 0.0
            elif rebate_timing is RebateTiming.AT_HIT:
                barrier_bv = rebate
            else:
                # AT_EXPIRY: discounted from current time to maturity
                barrier_bv = rebate * df_tT

            if direction is BarrierDirection.DOWN:
                # Barrier at left boundary, vanilla far-field at right
                left = barrier_bv
                if option_type is OptionType.PUT:
                    right = 0.0
                else:
                    continuation = smax * dq_tT - strike * df_tT
                    intrinsic_bv = max(smax - strike, 0.0)
                    right = (
                        max(continuation, intrinsic_bv)
                        if early_exercise
                        else max(continuation, 0.0)
                    )
            else:
                # Barrier at right boundary, vanilla far-field at left
                right = barrier_bv
                if option_type is OptionType.PUT:
                    intrinsic_bv = max(strike - smin, 0.0)
                    continuation = strike * df_tT - smin * dq_tT
                    left = max(continuation, intrinsic_bv) if early_exercise else continuation
                else:
                    left = 0.0
        else:
            # Discrete monitoring: standard vanilla boundaries
            left, right = _boundary_values(
                option_type=option_type,
                strike=strike,
                smin=smin,
                smax=smax,
                df_tT=df_tT,
                dq_tT=dq_tT,
                early_exercise=early_exercise,
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

        # ── Discrete barrier reset ────────────────────────────────────
        # Imposed on the ex-div surface at the observation time; the
        # dividend jump below then carries the knocked-out state back
        # to the pre-div surface.  Standard market convention is
        # ex-div first, then monitor — so cum-div spots in (H, H+D]
        # that fall through the barrier once the cash dividend goes
        # ex are correctly mapped to the reset value.  Order matches
        # ``_fd_barrier_ki_core``.
        if monitoring_tau_set is not None:
            tau_key = round(tau_curr, 12)
            if tau_key in monitoring_tau_set:
                if rebate == 0.0:
                    reset_val = 0.0
                elif rebate_timing is RebateTiming.AT_HIT:
                    reset_val = rebate
                else:
                    reset_val = rebate * df_tT
                if direction is BarrierDirection.DOWN:
                    V[S <= barrier] = reset_val
                else:
                    V[S >= barrier] = reset_val
                # Re-enforce early exercise on surviving nodes
                if early_exercise and intrinsic is not None:
                    alive = S > barrier if direction is BarrierDirection.DOWN else S < barrier
                    mask = alive & (intrinsic > V)
                    V[mask] = intrinsic[mask]

        # ── Discrete dividend jump ────────────────────────────────────
        if dividend_map:
            amount = dividend_map.get(round(tau_curr, 12))
            if amount is not None:
                _apply_dividend_jump(V, grid, amount, space_grid=space_grid)
                if early_exercise:
                    V[:] = np.maximum(V, intrinsic)

    if psor_steps > 0:
        avg_iters = psor_total_iters / psor_steps
        logger.debug(
            "PDE barrier PSOR steps=%d avg_iters=%.2f max_iters=%d not_converged=%d",
            psor_steps,
            avg_iters,
            psor_max_iters,
            psor_not_converged,
        )

    interp_spot = spot - pricing_div if pricing_div is not None else spot
    price = float(np.interp(interp_spot, S, V))
    return price, S, V, V_prev, last_dtau


def _fd_double_barrier_ko_core(
    *,
    spot: float,
    strike: float,
    time_to_maturity: float,
    volatility: float,
    discount_curve: DiscountCurve,
    dividend_curve: DiscountCurve | None,
    dividend_schedule: list[tuple[float, float]] | None,
    option_type: OptionType,
    lower_barrier: float,
    upper_barrier: float,
    monitoring: BarrierMonitoring,
    rebate: float,
    rebate_timing: RebateTiming,
    monitoring_taus: list[float] | None,  # required (not None) for DISCRETE
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
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    """Double knock-out finite-difference core (Boyle-Tian).

    **Continuous monitoring** — the grid is truncated at both barriers, so each
    barrier is a Dirichlet boundary holding the rebate-aware reset value (0
    with no rebate; the undiscounted rebate AT_HIT; the discounted rebate
    AT_EXPIRY).  Both boundaries carry the same value — the option dies
    whichever barrier it breaches.

    **Discrete monitoring** — the grid is the *full* Boyle-Tian half-step grid
    (:func:`_build_double_barrier_discrete_grid`) with both barriers placed
    midway between nodes; standard vanilla far-field boundaries apply and the
    knock-out reset (``S ≤ L`` / ``S ≥ U`` → rebate-aware value) is imposed
    only at the monitoring dates, analogous to the single-barrier discrete KO.

    European and American exercise are both supported via the shared explicit /
    implicit / Crank-Nicolson steppers.

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray, np.ndarray, float]
        ``(price, spot_grid, V_final, V_prev, last_dtau)`` — same shape as
        the single-barrier cores, so :class:`_FDGridGreeksMixin` can extract
        grid greeks unchanged.
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
    )

    continuous = monitoring is BarrierMonitoring.CONTINUOUS
    if not continuous and monitoring_taus is None:
        raise ConfigurationError("monitoring_taus is required for discrete barrier monitoring.")

    # ── Grid construction ─────────────────────────────────────────────
    # Continuous: truncate at both barriers (each is a Dirichlet boundary).
    # Discrete: full half-step grid with both barriers midway between nodes.
    if continuous:
        if space_grid is PDESpaceGrid.LOG_SPOT:
            grid, S, dz = _build_double_barrier_continuous_log_grid(
                lower_barrier=lower_barrier,
                upper_barrier=upper_barrier,
                volatility=volatility,
                time_to_maturity=time_to_maturity,
                spot_steps=spot_steps,
                time_steps=time_steps,
                method=method,
            )
        else:
            # Linear partition pinned to the two barriers.
            grid = np.linspace(lower_barrier, upper_barrier, spot_steps + 1)
            S = grid
            dS = (upper_barrier - lower_barrier) / spot_steps
    else:
        grid, S, step = _build_double_barrier_discrete_grid(
            lower_barrier=lower_barrier,
            upper_barrier=upper_barrier,
            spot=spot,
            strike=strike,
            volatility=volatility,
            time_to_maturity=time_to_maturity,
            smax_mult=smax_mult,
            spot_steps=spot_steps,
            time_steps=time_steps,
            method=method,
            log=space_grid is PDESpaceGrid.LOG_SPOT,
        )
        if space_grid is PDESpaceGrid.LOG_SPOT:
            dz = step
        else:
            dS = step

    smin = float(S[0])
    smax = float(S[-1])
    n_intervals = S.size - 1
    j = np.arange(1, n_intervals)  # interior indices

    # ── Terminal payoff ───────────────────────────────────────────────
    if option_type is OptionType.PUT:
        payoff = np.maximum(strike - S, 0.0)
    else:
        payoff = np.maximum(S - strike, 0.0)
    intrinsic = payoff.copy() if early_exercise else None

    # Discrete KO: if maturity is a monitoring date, knock out the payoff
    # outside the corridor (rebate-aware; at tau=0 df_tT=1 so AT_HIT and
    # AT_EXPIRY both give the undiscounted rebate).  Continuous KO needs no
    # such reset — the grid is truncated at the barriers.
    if not continuous:
        assert monitoring_taus is not None  # validated above
        if any(abs(tau) < 1e-12 for tau in monitoring_taus):
            payoff[(S <= lower_barrier) | (S >= upper_barrier)] = rebate

    V = payoff.copy()

    # ── Dividend schedule ─────────────────────────────────────────────
    schedule = dividend_schedule or []
    dividend_map = {round(tau, 12): amount for tau, amount in schedule}
    mat_div = dividend_map.pop(0.0, None)
    if mat_div is not None:
        _apply_dividend_jump(V, grid, mat_div, space_grid=space_grid)
        if early_exercise:
            V[:] = np.maximum(V, payoff)
    ttm_key = round(time_to_maturity, 12)
    pricing_div = dividend_map.pop(ttm_key, None)

    # ── Merge monitoring taus into grid ───────────────────────────────
    extra_taus = list(dividend_map.keys())
    monitoring_tau_set: set[float] | None = None
    if not continuous:
        assert monitoring_taus is not None  # validated above
        extra_taus.extend(monitoring_taus)
        monitoring_tau_set = {round(t, 12) for t in monitoring_taus}

    tau_grid = _build_tau_grid(time_to_maturity, time_steps, extra_taus)

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

    df_0T = float(discount_curve.df(time_to_maturity))

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
        q = (
            float(dividend_curve.forward_rate(t_curr, t_prev))
            if dividend_curve is not None
            else 0.0
        )

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
                size=n_intervals - 1,
            )

        df_0t = float(discount_curve.df(t_curr))
        df_tT = df_0T / df_0t
        if continuous:
            # ── Both barriers are Dirichlet boundaries (rebate-aware reset) ──
            if rebate == 0.0:
                barrier_bv = 0.0
            elif rebate_timing is RebateTiming.AT_HIT:
                barrier_bv = rebate
            else:
                barrier_bv = rebate * df_tT
            left = right = barrier_bv
        else:
            # ── Discrete monitoring: standard vanilla far-field boundaries ──
            if dividend_curve is not None:
                dq_0T = float(dividend_curve.df(time_to_maturity))
                dq_0t = float(dividend_curve.df(t_curr))
                dq_tT = dq_0T / dq_0t
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

        # ── Discrete barrier reset ────────────────────────────────────
        # Imposed on the ex-div surface at the observation time; the
        # dividend jump below then carries the knocked-out state back to the
        # pre-div surface (ex-div first, then monitor).  Order matches the
        # single-barrier KO / KI cores.  Half-step grid placement means no
        # node sits on a barrier, so the ``S ≤ L`` / ``S ≥ U`` masks are exact.
        if monitoring_tau_set is not None:
            tau_key = round(tau_curr, 12)
            if tau_key in monitoring_tau_set:
                if rebate == 0.0:
                    reset_val = 0.0
                elif rebate_timing is RebateTiming.AT_HIT:
                    reset_val = rebate
                else:
                    reset_val = rebate * df_tT
                ko_zone = (S <= lower_barrier) | (S >= upper_barrier)
                V[ko_zone] = reset_val
                # Re-enforce early exercise on surviving (corridor) nodes.
                if early_exercise and intrinsic is not None:
                    alive = ~ko_zone
                    mask = alive & (intrinsic > V)
                    V[mask] = intrinsic[mask]

        # ── Discrete dividend jump ────────────────────────────────────
        if dividend_map:
            amount = dividend_map.get(round(tau_curr, 12))
            if amount is not None:
                _apply_dividend_jump(V, grid, amount, space_grid=space_grid)
                if early_exercise:
                    V[:] = np.maximum(V, intrinsic)

    if psor_steps > 0:
        avg_iters = psor_total_iters / psor_steps
        logger.debug(
            "PDE double-barrier PSOR steps=%d avg_iters=%.2f max_iters=%d not_converged=%d",
            psor_steps,
            avg_iters,
            psor_max_iters,
            psor_not_converged,
        )

    interp_spot = spot - pricing_div if pricing_div is not None else spot
    price = float(np.interp(interp_spot, S, V))
    return price, S, V, V_prev, last_dtau


def _subgrid_pde_step(
    V: np.ndarray,
    V_prev: np.ndarray,
    j_sub: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    left: float,
    right: float,
    method: PDEMethod,
    r_dt: float,
) -> np.ndarray:
    """One PDE time-step on a sub-grid. Used for the inactive-surface solve in a coupled KI PDE.

    Parameters
    ----------
    V, V_prev : full-length value arrays (current / previous tau layer).
    j_sub : node indices of the sub-grid interior.
    a, b, c : scaled operator coefficients for the **full** interior
        (``j=1..M-1``); this helper slices them to ``j_sub``.
    left, right : Dirichlet boundary values for the sub-grid.
    method : PDE stepping scheme.
    r_dt : ``r * d_tau`` for Hull explicit discounting (0 otherwise).
    """
    V = V.copy()
    ci: np.ndarray = j_sub - 1  # coefficient indices into full-interior arrays
    a_s, b_s, c_s = a[ci], b[ci], c[ci]

    if method in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL):
        interior = -a_s * V_prev[j_sub - 1] + (1.0 - b_s) * V_prev[j_sub] - c_s * V_prev[j_sub + 1]
        V[j_sub] = interior / (1.0 + r_dt)
        V[j_sub[0] - 1] = left
        V[j_sub[-1] + 1] = right
    elif method is PDEMethod.IMPLICIT:
        diag = 1.0 + b_s
        rhs = V_prev[j_sub].copy()
        rhs[0] -= a_s[0] * left
        rhs[-1] -= c_s[-1] * right
        V[j_sub] = _solve_tridiagonal_thomas(a_s[1:], diag, c_s[:-1], rhs)
    else:
        a_h, b_h, c_h = a_s * 0.5, b_s * 0.5, c_s * 0.5
        diag = 1.0 + b_h
        rhs = -a_h * V_prev[j_sub - 1] + (1.0 - b_h) * V_prev[j_sub] - c_h * V_prev[j_sub + 1]
        rhs[0] -= a_h[0] * left
        rhs[-1] -= c_h[-1] * right
        V[j_sub] = _solve_tridiagonal_thomas(a_h[1:], diag, c_h[:-1], rhs)
    return V


def _fd_barrier_ki_core(
    *,
    spot: float,
    strike: float,
    time_to_maturity: float,
    volatility: float,
    discount_curve: DiscountCurve,
    dividend_curve: DiscountCurve | None,
    dividend_schedule: list[tuple[float, float]] | None,
    option_type: OptionType,
    barrier: float,
    direction: BarrierDirection,
    monitoring: BarrierMonitoring,
    rebate: float,
    rebate_timing: RebateTiming,  # unused for KI but kept for signature consistency with KO core
    monitoring_taus: list[float] | None,
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
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    """Two-surface coupled PDE solver for knock-in barrier options.

    Maintains two value surfaces that are stepped backward in time:

    * **Active** (``V_act``): the barrier has been hit; behaves as a standard
      option (with early-exercise projection when ``early_exercise=True``).
    * **Inactive** (``V_inact``): the barrier has not yet been hit; pure
      continuation PDE with no exercise allowed.

    At the barrier the inactive surface is coupled to the active surface
    (state transition, not absorption).

    The option starts in the inactive state, so the price is read from
    ``V_inact`` at the spot level.

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray, np.ndarray, float]
        ``(price, spot_grid, V_inact_final, V_inact_prev, last_dtau)``
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
    )

    continuous = monitoring is BarrierMonitoring.CONTINUOUS

    if not continuous and monitoring_taus is None:
        raise ConfigurationError("monitoring_taus is required for discrete barrier monitoring.")

    # ── Grid construction (full grid, not truncated) ──────────────────
    ref_price = max(spot, strike)

    if space_grid is PDESpaceGrid.LOG_SPOT:
        grid, S, dz = _build_log_grid(
            spot=spot,
            strike=strike,
            time_to_maturity=time_to_maturity,
            volatility=volatility,
            smax_mult=smax_mult,
            spot_steps=spot_steps,
            time_steps=time_steps,
            method=method,
            anchor_spot=barrier,
            anchor_half_step=not continuous,
        )
    else:
        smax = float(smax_mult * ref_price)
        grid, S, dS = _build_spot_grid(
            smin=0.0,
            smax=smax,
            spot_steps=spot_steps,
            anchor_spot=barrier,
            anchor_half_step=not continuous,
        )

    smin = float(S[0])
    smax = float(S[-1])

    j = np.arange(1, spot_steps)  # interior indices

    # ── Terminal payoff ───────────────────────────────────────────────
    if option_type is OptionType.PUT:
        payoff = np.maximum(strike - S, 0.0)
    else:
        payoff = np.maximum(S - strike, 0.0)

    intrinsic = payoff if early_exercise else None

    df_0T = float(discount_curve.df(time_to_maturity))

    # Active surface: vanilla payoff at maturity
    V_act = payoff.copy()
    # Inactive surface: rebate PV at maturity (0 if no rebate)
    # KI rebate is always AT_EXPIRY (validated by BarrierSpec), so terminal
    # value for inactive paths = undiscounted rebate (we are at maturity).
    V_inact = np.full_like(payoff, rebate)

    # ── Dividend schedule ─────────────────────────────────────────────
    schedule = dividend_schedule or []
    # Round keys to 12dp to absorb float arithmetic noise; lookups must also round.
    dividend_map = {round(tau, 12): amount for tau, amount in schedule}

    mat_div = dividend_map.pop(0.0, None)
    if mat_div is not None:
        _apply_dividend_jump(V_act, grid, mat_div, space_grid=space_grid)
        if early_exercise:
            V_act[:] = np.maximum(V_act, payoff)
        # The inactive terminal surface is spot-independent (equal to the
        # maturity rebate everywhere), so the maturity-date dividend jump is
        # a no-op for V_inact.

    ttm_key = round(time_to_maturity, 12)
    pricing_div = dividend_map.pop(ttm_key, None)

    # ── Merge monitoring taus into time grid ──────────────────────────
    dividend_taus = list(dividend_map.keys())
    extra_taus = dividend_taus.copy()
    monitoring_tau_set: set[float] | None = None
    if not continuous:
        assert monitoring_taus is not None
        extra_taus.extend(monitoring_taus)
        monitoring_tau_set = {round(t, 12) for t in monitoring_taus}

    tau_grid = _build_tau_grid(time_to_maturity, time_steps, extra_taus)

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

    # ── Barrier index for sub-grid coupling ──────────────────────────
    # Find the grid node closest to the barrier level.
    j_H = int(np.argmin(np.abs(S - barrier)))

    # Terminal coupling: if the barrier is already triggered at maturity,
    # the inactive state must transition immediately to the active payoff.
    if continuous or (monitoring_tau_set is not None and 0.0 in monitoring_tau_set):
        if direction is BarrierDirection.DOWN:
            terminal_hit_mask = S <= barrier
        else:
            terminal_hit_mask = S >= barrier
        V_inact[terminal_hit_mask] = V_act[terminal_hit_mask]

    # For continuous monitoring, the inactive surface PDE is solved only
    # on the far side of the barrier (above H for down-in, below H for
    # up-in), with V_act[j_H] as the inner Dirichlet boundary.  This
    # makes the coupling implicit in the solve and avoids the operator-
    # splitting error that arises from full-grid solve + post-hoc
    # coupling.
    if continuous:
        if direction is BarrierDirection.DOWN:
            # Solve above barrier: interior nodes j_H+1 .. spot_steps-1
            j_inact = np.arange(j_H + 1, spot_steps)
        else:
            # Solve below barrier: interior nodes 1 .. j_H-1
            j_inact = np.arange(1, j_H)
    else:
        j_inact = j  # full interior for discrete (coupling at monitoring dates)

    # ── Time-stepping ────────────────────────────────────────────────
    if dividend_curve is not None:
        dq_0T = float(dividend_curve.df(time_to_maturity))
    else:
        dq_0T = None

    psor_steps = 0
    psor_total_iters = 0
    psor_max_iters = 0
    psor_not_converged = 0

    steps = _build_time_step_schedule(tau_grid, method, rannacher_steps)

    V_act_prev = V_act.copy()
    V_inact_prev = V_inact.copy()
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

        # Discount factors for boundary conditions
        df_0t = float(discount_curve.df(t_curr))
        df_tT: float = df_0T / df_0t
        if dividend_curve is not None:
            dq_0t = float(dividend_curve.df(t_curr))
            dq_tT: float = dq_0T / dq_0t  # type: ignore[operator]
        else:
            dq_tT = 1.0

        # Vanilla boundary conditions — used for active surface
        left, right = _boundary_values(
            option_type=option_type,
            strike=strike,
            smin=smin,
            smax=smax,
            df_tT=df_tT,
            dq_tT=dq_tT,
            early_exercise=early_exercise,
        )

        V_act_prev = V_act.copy()
        V_inact_prev = V_inact.copy()
        last_dtau = d_tau

        a, b, c = _scaled_operator_coeffs(gamma=gamma, beta=beta, alpha=alpha, d_tau=d_tau)

        # ── Step A: PDE step for active surface (with American exercise) ──
        if method_used in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL):
            V_act = _explicit_step(
                V_act_prev,
                j,
                a,
                b,
                c,
                left,
                right,
                intrinsic,
                r_dt=r * d_tau if implicit_discounting else 0.0,
            )
        else:
            V_act, psor_iters = _implicit_cn_step(
                V_act_prev,
                V_act,
                j,
                a,
                b,
                c,
                left,
                right,
                method_used,
                intrinsic,
                american_solver,
                omega,
                tol,
                max_iter,
            )
            if psor_iters is not None:
                psor_steps += 1
                psor_total_iters += psor_iters
                psor_max_iters = max(psor_max_iters, psor_iters)
                if psor_iters == int(max_iter):  # type: ignore[arg-type]
                    psor_not_converged += 1

        # ── Step B: PDE step for inactive surface (no exercise) ──────────
        # For continuous monitoring the inactive surface is solved on a
        # sub-grid restricted to nodes on the far side of the barrier,
        # with V_act[j_H] as the inner Dirichlet boundary.
        if continuous:
            # Inner BC: V_act at barrier; outer BC: rebate PV (0 if no rebate)
            rebate_bv = rebate * df_tT
            if direction is BarrierDirection.DOWN:
                left_inact = float(V_act[j_H])  # inner (barrier side)
                right_inact = rebate_bv  # far side → 0 if no rebate
            else:
                left_inact = rebate_bv  # far side → 0 if no rebate
                right_inact = float(V_act[j_H])  # inner (barrier side)

            if j_inact.size > 0:
                V_inact = _subgrid_pde_step(
                    V_inact,
                    V_inact_prev,
                    j_inact,
                    a,
                    b,
                    c,
                    left_inact,
                    right_inact,
                    method_used,
                    r_dt=r * d_tau if implicit_discounting else 0.0,
                )

            # The sub-grid solve fills only the continuation-region interior.
            # The assignments below complete the current V_inact slice over
            # the full spatial grid by imposing the hit-side coupling region
            # and the far-field boundary.
            if direction is BarrierDirection.DOWN:
                V_inact[: j_H + 1] = V_act[: j_H + 1]
                V_inact[-1] = rebate_bv  # far-field boundary
            else:
                V_inact[j_H:] = V_act[j_H:]
                V_inact[0] = rebate_bv
        else:
            # Discrete monitoring: full grid solve, coupling at monitoring dates
            # For an inactive KI, on the safe far side its asymptotic value is
            # the no-hit value (rebate PV, or 0 if no rebate). On the risky
            # side, immediate coupling to the current active boundary is too
            # aggressive between monitoring dates because activation cannot
            # occur until the next observation. Use a one-step look-ahead
            # proxy: current active boundary on monitoring dates, otherwise the
            # next-time-slice active boundary carried in V_act_prev. This is a
            # pragmatic discrete-monitoring closure, not an exact asymptotic
            # boundary condition.
            rebate_bv = rebate * df_tT
            tau_key = round(tau_curr, 12)
            is_monitoring_step = monitoring_tau_set is not None and tau_key in monitoring_tau_set
            if direction is BarrierDirection.DOWN:
                left_inact = float(V_act[0] if is_monitoring_step else V_act_prev[0])
                right_inact = rebate_bv
            else:
                left_inact = rebate_bv
                right_inact = float(V_act[-1] if is_monitoring_step else V_act_prev[-1])
            if method_used in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL):
                V_inact = _explicit_step(
                    V_inact_prev,
                    j,
                    a,
                    b,
                    c,
                    left_inact,
                    right_inact,
                    None,
                    r_dt=r * d_tau if implicit_discounting else 0.0,
                )
            else:
                V_inact, _ = _implicit_cn_step(
                    V_inact_prev,
                    V_inact,
                    j,
                    a,
                    b,
                    c,
                    left_inact,
                    right_inact,
                    method_used,
                    None,
                    american_solver,
                    omega,
                    tol,
                    max_iter,
                )

            # Discrete barrier coupling at monitoring dates.
            # This is imposed on the ex-div surface at the observation time;
            # the dividend jump below then carries that knocked-in state back
            # to the pre-div surface (for example, cum-div spots that fall
            # through the barrier once the cash dividend goes ex).
            if monitoring_tau_set is not None:
                if tau_key in monitoring_tau_set:
                    if direction is BarrierDirection.DOWN:
                        mask = S <= barrier
                    else:
                        mask = S >= barrier
                    V_inact[mask] = V_act[mask]

        # ── Discrete dividend jumps (both surfaces) ─────────────────────
        if dividend_map:
            amount = dividend_map.get(round(tau_curr, 12))
            if amount is not None:
                _apply_dividend_jump(V_act, grid, amount, space_grid=space_grid)
                if early_exercise:
                    V_act[:] = np.maximum(V_act, intrinsic)
                _apply_dividend_jump(V_inact, grid, amount, space_grid=space_grid)
                if continuous:
                    if direction is BarrierDirection.DOWN:
                        V_inact[: j_H + 1] = V_act[: j_H + 1]
                    else:
                        V_inact[j_H:] = V_act[j_H:]

    if psor_steps > 0:
        avg_iters = psor_total_iters / psor_steps
        logger.debug(
            "PDE barrier KI PSOR steps=%d avg_iters=%.2f max_iters=%d not_converged=%d",
            psor_steps,
            avg_iters,
            psor_max_iters,
            psor_not_converged,
        )

    # Price from inactive surface (option starts inactive)
    interp_spot = spot - pricing_div if pricing_div is not None else spot
    price = float(np.interp(interp_spot, S, V_inact))
    return price, S, V_inact, V_inact_prev, last_dtau


def _fd_double_barrier_ki_core(
    *,
    spot: float,
    strike: float,
    time_to_maturity: float,
    volatility: float,
    discount_curve: DiscountCurve,
    dividend_curve: DiscountCurve | None,
    dividend_schedule: list[tuple[float, float]] | None,
    option_type: OptionType,
    lower_barrier: float,
    upper_barrier: float,
    monitoring: BarrierMonitoring,
    rebate: float,
    rebate_timing: RebateTiming,  # unused (KI rebate is AT_EXPIRY); kept for signature parity
    monitoring_taus: list[float] | None,  # required (not None) for DISCRETE
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
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    """Two-surface coupled solver for double knock-in options.

    Direct generalisation of :func:`_fd_barrier_ki_core` to two barriers:

    * **Active** (``V_act``): full-grid vanilla (with early-exercise projection
      when ``early_exercise=True``) — the post-knock-in state.
    * **Inactive** (``V_inact``): the not-yet-knocked-in state.

    **Continuous monitoring** — the inactive surface is solved on the corridor
    interior only, with ``V_act`` coupled in as the Dirichlet boundary at
    *both* barriers (``V_act[j_L]`` and below, ``V_act[j_U]`` and above; barriers
    sit on nodes via :func:`_build_double_barrier_full_grid`).  Outside the corridor
    the inactive surface equals the active surface (already knocked in).

    **Discrete monitoring** — both barriers are placed midway between nodes
    (:func:`_build_double_barrier_discrete_grid`).  Both surfaces are solved on
    the full interior each step; the knock-in coupling ``V_inact[S≤L | S≥U] =
    V_act`` is imposed only at the monitoring dates.  Between observations the
    inactive surface evolves as a pure continuation PDE, with the active
    surface as a one-step look-ahead proxy at the two far-field boundaries
    (both lie in a knock-in zone).  Mirrors the single-barrier discrete KI
    closure in :func:`_fd_barrier_ki_core`.

    The option starts inactive, so the price is read from ``V_inact`` at spot.

    Returns ``(price, S, V_inact_final, V_inact_prev, last_dtau)``.
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
    )

    continuous = monitoring is BarrierMonitoring.CONTINUOUS
    if not continuous and monitoring_taus is None:
        raise ConfigurationError("monitoring_taus is required for discrete barrier monitoring.")

    log_grid = space_grid is PDESpaceGrid.LOG_SPOT
    if continuous:
        # Barriers on nodes → exact Dirichlet coupling at each barrier.
        grid, S, step, j_L, j_U = _build_double_barrier_full_grid(
            lower_barrier=lower_barrier,
            upper_barrier=upper_barrier,
            spot=spot,
            strike=strike,
            volatility=volatility,
            time_to_maturity=time_to_maturity,
            smax_mult=smax_mult,
            spot_steps=spot_steps,
            time_steps=time_steps,
            method=method,
            log=log_grid,
        )
    else:
        # Both barriers midway between nodes → clean S≤L / S≥U knock-in masks.
        grid, S, step = _build_double_barrier_discrete_grid(
            lower_barrier=lower_barrier,
            upper_barrier=upper_barrier,
            spot=spot,
            strike=strike,
            volatility=volatility,
            time_to_maturity=time_to_maturity,
            smax_mult=smax_mult,
            spot_steps=spot_steps,
            time_steps=time_steps,
            method=method,
            log=log_grid,
        )

    smin = float(S[0])
    smax = float(S[-1])
    n_nodes = len(S)
    j = np.arange(1, n_nodes - 1)  # full interior

    if option_type is OptionType.PUT:
        payoff = np.maximum(strike - S, 0.0)
    else:
        payoff = np.maximum(S - strike, 0.0)

    intrinsic = payoff if early_exercise else None

    df_0T = float(discount_curve.df(time_to_maturity))

    V_act = payoff.copy()
    V_inact = np.full_like(payoff, rebate)

    # Dividend schedule.
    schedule = dividend_schedule or []
    dividend_map = {round(tau, 12): amount for tau, amount in schedule}
    mat_div = dividend_map.pop(0.0, None)
    if mat_div is not None:
        _apply_dividend_jump(V_act, grid, mat_div, space_grid=space_grid)
        if early_exercise:
            V_act[:] = np.maximum(V_act, payoff)
    ttm_key = round(time_to_maturity, 12)
    pricing_div = dividend_map.pop(ttm_key, None)

    # Merge monitoring taus into the time grid for discrete monitoring.
    extra_taus = list(dividend_map.keys())
    monitoring_tau_set: set[float] | None = None
    if not continuous:
        assert monitoring_taus is not None
        extra_taus.extend(monitoring_taus)
        monitoring_tau_set = {round(t, 12) for t in monitoring_taus}
    tau_grid = _build_tau_grid(time_to_maturity, time_steps, extra_taus)

    if method in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL) and not log_grid:
        _check_explicit_spot_stability(
            tau_grid=tau_grid,
            volatility=volatility,
            smax=smax,
            dS=step,  # spot-grid branch: step is dS here
            time_to_maturity=time_to_maturity,
            discount_curve=discount_curve,
            dividend_curve=dividend_curve,
            implicit_discounting=method is PDEMethod.EXPLICIT_HULL,
        )

    # Terminal coupling: corridor breached at maturity → already knocked in.
    # (Discrete: only if maturity is itself a monitoring date.)
    hit_mask = (S <= lower_barrier) | (S >= upper_barrier)
    if continuous or (monitoring_tau_set is not None and 0.0 in monitoring_tau_set):
        V_inact[hit_mask] = V_act[hit_mask]

    # Inactive surface interior: corridor only (continuous) or full grid
    # (discrete, with coupling imposed at the monitoring dates).
    j_inact = np.arange(j_L + 1, j_U) if continuous else j

    if dividend_curve is not None:
        dq_0T: float | None = float(dividend_curve.df(time_to_maturity))
    else:
        dq_0T = None

    psor_steps = 0
    psor_total_iters = 0
    psor_max_iters = 0
    psor_not_converged = 0

    steps = _build_time_step_schedule(tau_grid, method, rannacher_steps)
    V_act_prev = V_act.copy()
    V_inact_prev = V_inact.copy()
    last_dtau = 0.0

    for tau_prev, tau_curr, method_used in steps:
        d_tau = tau_curr - tau_prev
        t_prev = time_to_maturity - tau_prev
        t_curr = time_to_maturity - tau_curr

        r = float(discount_curve.forward_rate(t_curr, t_prev))
        q = (
            float(dividend_curve.forward_rate(t_curr, t_prev))
            if dividend_curve is not None
            else 0.0
        )
        implicit_discounting = method_used is PDEMethod.EXPLICIT_HULL

        if log_grid:
            gamma, beta, alpha = _log_operator_coeffs(
                dz=step,
                risk_free_rate=r,
                dividend_rate=q,
                volatility=volatility,
                implicit_discounting=implicit_discounting,
                size=n_nodes - 2,
            )
        else:
            gamma, beta, alpha = _spot_operator_coeffs(
                spot_values=S[1:-1],
                dS=step,
                risk_free_rate=r,
                dividend_rate=q,
                volatility=volatility,
                implicit_discounting=implicit_discounting,
            )

        df_0t = float(discount_curve.df(t_curr))
        df_tT = df_0T / df_0t
        if dividend_curve is not None:
            dq_tT: float = dq_0T / float(dividend_curve.df(t_curr))  # type: ignore[operator]
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
        )

        V_act_prev = V_act.copy()
        V_inact_prev = V_inact.copy()
        last_dtau = d_tau
        a, b, c = _scaled_operator_coeffs(gamma=gamma, beta=beta, alpha=alpha, d_tau=d_tau)

        # ── Active surface: full-grid vanilla (with American exercise) ──
        if method_used in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL):
            V_act = _explicit_step(
                V_act_prev,
                j,
                a,
                b,
                c,
                left,
                right,
                intrinsic,
                r_dt=r * d_tau if implicit_discounting else 0.0,
            )
        else:
            V_act, psor_iters = _implicit_cn_step(
                V_act_prev,
                V_act,
                j,
                a,
                b,
                c,
                left,
                right,
                method_used,
                intrinsic,
                american_solver,
                omega,
                tol,
                max_iter,
            )
            if psor_iters is not None:
                psor_steps += 1
                psor_total_iters += psor_iters
                psor_max_iters = max(psor_max_iters, psor_iters)
                if psor_iters == int(max_iter):  # type: ignore[arg-type]
                    psor_not_converged += 1

        # ── Inactive surface ──
        if continuous:
            # Corridor sub-grid, coupled to V_act at both barriers.
            if j_inact.size > 0:
                V_inact = _subgrid_pde_step(
                    V_inact,
                    V_inact_prev,
                    j_inact,
                    a,
                    b,
                    c,
                    float(V_act[j_L]),  # lower-barrier coupling
                    float(V_act[j_U]),  # upper-barrier coupling
                    method_used,
                    r * d_tau if implicit_discounting else 0.0,
                )
            # Outside the corridor the option is knocked in → equals active.
            V_inact[: j_L + 1] = V_act[: j_L + 1]
            V_inact[j_U:] = V_act[j_U:]
        else:
            # Discrete: full-grid continuation solve.  Both far ends lie in a
            # knock-in zone, so use the active surface as a one-step look-ahead
            # proxy boundary (current V_act on a monitoring step, otherwise the
            # next-slice V_act_prev).  Coupling to V_act is imposed at the
            # monitoring dates only.
            tau_key = round(tau_curr, 12)
            is_monitoring_step = monitoring_tau_set is not None and tau_key in monitoring_tau_set
            left_inact = float(V_act[0] if is_monitoring_step else V_act_prev[0])
            right_inact = float(V_act[-1] if is_monitoring_step else V_act_prev[-1])
            if method_used in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL):
                V_inact = _explicit_step(
                    V_inact_prev,
                    j,
                    a,
                    b,
                    c,
                    left_inact,
                    right_inact,
                    None,
                    r_dt=r * d_tau if implicit_discounting else 0.0,
                )
            else:
                V_inact, _ = _implicit_cn_step(
                    V_inact_prev,
                    V_inact,
                    j,
                    a,
                    b,
                    c,
                    left_inact,
                    right_inact,
                    method_used,
                    None,
                    american_solver,
                    omega,
                    tol,
                    max_iter,
                )
            # Knock-in coupling on the ex-div surface at the observation time;
            # the dividend jump below carries the knocked-in state to the
            # pre-div surface (ex-div first, then monitor).
            if is_monitoring_step:
                V_inact[hit_mask] = V_act[hit_mask]

        # ── Discrete dividend jumps (both surfaces) ──
        if dividend_map:
            amount = dividend_map.get(round(tau_curr, 12))
            if amount is not None:
                _apply_dividend_jump(V_act, grid, amount, space_grid=space_grid)
                if early_exercise:
                    V_act[:] = np.maximum(V_act, intrinsic)
                _apply_dividend_jump(V_inact, grid, amount, space_grid=space_grid)
                if continuous:
                    V_inact[: j_L + 1] = V_act[: j_L + 1]
                    V_inact[j_U:] = V_act[j_U:]

    if psor_steps > 0:
        avg_iters = psor_total_iters / psor_steps
        logger.debug(
            "PDE double-barrier KI PSOR steps=%d avg_iters=%.2f max_iters=%d not_converged=%d",
            psor_steps,
            avg_iters,
            psor_max_iters,
            psor_not_converged,
        )

    interp_spot = spot - pricing_div if pricing_div is not None else spot
    price = float(np.interp(interp_spot, S, V_inact))
    return price, S, V_inact, V_inact_prev, last_dtau
