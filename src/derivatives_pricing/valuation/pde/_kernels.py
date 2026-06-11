"""Low-level finite-difference kernels shared by every PDE engine.

Tridiagonal (Thomas) solver, tau-grid and dividend-schedule construction,
payoff boundary models, the vanilla grid builders (log-spot and spot),
spatial operator coefficients, the explicit / PSOR / implicit-CN time-step
kernels, and input validation / stability checks.  Everything here is
barrier-agnostic and consumed by the engine cores in :mod:`._core` and
:mod:`._barrier_cores`.
"""

from __future__ import annotations
from collections.abc import Callable, Sequence

import logging
import math
import datetime as dt

import numpy as np

# optional acceleration via numba if available
try:
    from numba import njit as _njit
except ModuleNotFoundError:  # pragma: no cover

    def _njit(*args, **kwargs):  # type: ignore[misc]
        """Identity decorator when numba is not installed."""
        if args and callable(args[0]):
            return args[0]
        return lambda fn: fn


from ...enums import (
    DayCountConvention,
    PDEEarlyExercise,
    PDEMethod,
    PDESpaceGrid,
    OptionType,
)
from ...rates import DiscountCurve
from ...utils import calculate_year_fraction
from ...exceptions import (
    ConfigurationError,
    StabilityError,
    UnsupportedFeatureError,
    ValidationError,
)
from ..contracts import PayoffBoundaryModel, WingBoundary

logger = logging.getLogger(__name__)


def _solve_tridiagonal_thomas(
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve a tridiagonal system Ax = rhs via the Thomas algorithm.

    A has:
      - lower: subdiagonal (length n-1)  -> A[i, i-1]
      - diag:  main diagonal (length n)  -> A[i, i]
      - upper: superdiagonal (length n-1)-> A[i, i+1]

    Parameters
    ----------
    lower
        Subdiagonal coefficients.
    diag
        Main diagonal coefficients.
    upper
        Superdiagonal coefficients.
    rhs
        Right-hand-side vector.

    Returns
    -------
    np.ndarray
        Solution vector of the tridiagonal system.
    """
    n = diag.size
    if rhs.size != n:
        raise ValidationError("rhs length must match diag length")
    if lower.size != n - 1 or upper.size != n - 1:
        raise ValidationError("lower/upper must have length n-1")

    # Copy to avoid mutating inputs
    a: np.ndarray = lower.astype(float, copy=True)
    d: np.ndarray = diag.astype(float, copy=True)
    c: np.ndarray = upper.astype(float, copy=True)
    y: np.ndarray = rhs.astype(float, copy=True)

    # Forward elimination
    for i in range(1, n):
        w = a[i - 1] / d[i - 1]
        d[i] -= w * c[i - 1]
        y[i] -= w * y[i - 1]

    # Back substitution
    x: np.ndarray = np.empty(n, dtype=float)
    x[-1] = y[-1] / d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (y[i] - c[i] * x[i + 1]) / d[i]
    return x


def _build_tau_grid(
    time_to_maturity: float,
    time_steps: int,
    extra_taus: list[float],
) -> np.ndarray:
    """Build a tau grid that snaps to dividend and monitoring dates."""
    base = np.linspace(0.0, time_to_maturity, time_steps + 1)
    if not extra_taus:
        return base
    # Round to 12dp on both inputs so np.unique can collapse near-duplicates.
    # Extras already arrive at 12dp from the upstream tau converters
    # (_barrier_monitoring_taus, _dividend_tau_schedule); the linspace base
    # is bit-fresh.  Without the round, an extra that lands on the same
    # physical time as a base point survives as a sub-1e-12 neighbour, then
    # `T - tau` collapses both to the same float and downstream forward-rate
    # calls raise on dt == 0.
    grid = np.unique(np.round(np.concatenate([base, np.array(extra_taus, dtype=float)]), 12))
    return grid


def _dividend_tau_schedule(
    *,
    discrete_dividends: Sequence[tuple[dt.datetime, float]],
    pricing_date: dt.datetime,
    maturity: dt.datetime,
    day_count_convention: DayCountConvention,
) -> list[tuple[float, float]]:
    """Return list of (tau, amount) for dividends between pricing_date and maturity.

    The range is closed: ``0.0 <= tau <= ttm``.  Boundary values (tau=0 for
    maturity-date dividends, tau=ttm for pricing-date dividends) are included
    so that ``_fd_core`` can apply them as special-case jumps.
    """
    if not discrete_dividends:
        return []

    ttm = calculate_year_fraction(
        pricing_date,
        maturity,
        day_count_convention=day_count_convention,
    )
    schedule: dict[float, float] = {}
    for ex_date, amount in discrete_dividends:
        if pricing_date <= ex_date <= maturity:
            t = calculate_year_fraction(
                pricing_date,
                ex_date,
                day_count_convention=day_count_convention,
            )
            tau = ttm - t
            key = round(float(tau), 12)
            schedule[key] = schedule.get(key, 0.0) + float(amount)
    return sorted(schedule.items())


def _apply_dividend_jump(
    values: np.ndarray,
    grid: np.ndarray,
    amount: float,
    *,
    space_grid: PDESpaceGrid,
) -> None:
    """Apply the cash dividend jump condition V(S,t^-)=V(S-D,t^+)."""
    if amount == 0.0:
        return
    if space_grid is PDESpaceGrid.LOG_SPOT:
        spot_grid = np.exp(grid)
    else:
        spot_grid = grid
    shifted = np.interp(
        spot_grid - amount,
        spot_grid,
        values,
        left=values[0],
        right=values[-1],
    )
    values[:] = shifted


# ---------------------------------------------------------------------------
# Affine boundary-model helpers for custom-payoff boundary conditions
# ---------------------------------------------------------------------------


def _fit_affine_boundary_model(
    payoff_fn: Callable,
    *,
    wing: str,
    spot_samples: np.ndarray,
) -> WingBoundary:
    """Fit affine boundary model ``payoff(S) ~ slope * S + intercept``.

    Parameters
    ----------
    payoff_fn
        Vectorized payoff callable.
    wing
        ``"left"`` or ``"right"`` boundary label for logging.
    spot_samples
        Spot samples taken directly from the PDE grid near the relevant
        boundary. Using the actual grid nodes makes the fitted affine model
        consistent with the truncated domain the PDE solver uses.

    Returns
    -------
    WingBoundary
        Fitted slope / intercept pair.
    """
    if wing not in {"left", "right"}:
        raise ConfigurationError(f"wing must be 'left' or 'right', got {wing!r}")

    x = np.asarray(spot_samples, dtype=float)
    if x.ndim != 1 or x.size < 3:
        raise ConfigurationError("spot_samples must be a 1D array with at least three points")
    if np.any(np.diff(x) <= 0.0):
        raise ConfigurationError("spot_samples must be strictly increasing")

    y = np.asarray(payoff_fn(x), dtype=float)

    # Least-squares fit: y ≈ slope * x + intercept
    A = np.column_stack([x, np.ones_like(x)])
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

    # Warn if the affine fit is poor (payoff is genuinely nonlinear on the wing).
    residuals = y - (slope * x + intercept)
    ss_res = float(np.dot(residuals, residuals))
    ss_tot = float(np.dot(y - y.mean(), y - y.mean()))
    if ss_tot > 1e-30:
        r_squared = 1.0 - ss_res / ss_tot
        if r_squared < 0.99:
            logger.warning(
                "Affine boundary fit on %s wing has R²=%.4f; boundary values "
                "may be inaccurate. Consider providing explicit PayoffBoundaryModel.",
                wing,
                r_squared,
            )

    return WingBoundary(slope=float(slope), intercept=float(intercept))


def _continuation_from_affine_boundary_model(
    *,
    spot: float,
    slope: float,
    intercept: float,
    df_tT: float,
    dq_tT: float,
) -> float:
    """Continuation value implied by an affine boundary model.

    If the boundary payoff is approximated by

        payoff(S_T) ~ slope * S_T + intercept

    then under risk-neutral pricing:

        V(S, t) ~ slope * S * dq_tT + intercept * df_tT

    where *dq_tT* is the dividend discount factor and *df_tT* is the risk-free
    discount factor from *t* to *T*.
    """
    return float(slope * spot * dq_tT + intercept * df_tT)


def _boundary_values(
    *,
    option_type: OptionType | None,
    strike: float | None,
    smin: float,
    smax: float,
    df_tT: float,
    dq_tT: float,
    early_exercise: bool,
    payoff_fn: Callable | None = None,
    payoff_boundary_model: PayoffBoundaryModel | None = None,
) -> tuple[float, float]:
    """Dirichlet boundary values for PDE at S=smin (left) and S=smax (right).

    For vanilla call/put, uses standard analytical asymptotic boundary
    conditions.

    For custom payoffs, uses affine wing boundary models::

        payoff(S) ~ slope * S + intercept
        => V(S, t) ~ slope * S * dq_tT + intercept * df_tT

    For custom payoffs, the affine boundary wings are interpreted at the
    actual finite grid boundaries ``smin`` and ``smax`` of the truncated PDE
    domain rather than as true infinite-domain asymptotics.

    For American exercise the boundary is
    ``max(continuation, intrinsic)`` where intrinsic is evaluated directly
    via the payoff callable (not the boundary model).
    """
    # ------------------------------------------------------------------
    # Custom payoff branch
    # ------------------------------------------------------------------
    if payoff_fn is not None:
        if payoff_boundary_model is None:
            raise ConfigurationError(
                "_boundary_values requires a resolved payoff_boundary_model for custom payoffs"
            )

        left_cont = _continuation_from_affine_boundary_model(
            spot=smin,
            slope=payoff_boundary_model.left.slope,
            intercept=payoff_boundary_model.left.intercept,
            df_tT=df_tT,
            dq_tT=dq_tT,
        )
        right_cont = _continuation_from_affine_boundary_model(
            spot=smax,
            slope=payoff_boundary_model.right.slope,
            intercept=payoff_boundary_model.right.intercept,
            df_tT=df_tT,
            dq_tT=dq_tT,
        )

        if early_exercise:
            intrinsic = np.asarray(payoff_fn(np.array([smin, smax], dtype=float)), dtype=float)
            left = max(left_cont, float(intrinsic[0]))
            right = max(right_cont, float(intrinsic[1]))
        else:
            left = left_cont
            right = right_cont

        return float(left), float(right)

    # ------------------------------------------------------------------
    # Vanilla branch
    # ------------------------------------------------------------------
    assert option_type is not None and strike is not None
    if option_type is OptionType.PUT:
        intrinsic = max(strike - smin, 0.0)
        continuation = strike * df_tT - smin * dq_tT
        left = max(continuation, intrinsic) if early_exercise else continuation
        right = 0.0
    else:
        left = 0.0
        continuation = smax * dq_tT - strike * df_tT
        intrinsic = max(smax - strike, 0.0)
        right = max(continuation, intrinsic) if early_exercise else max(continuation, 0.0)
    return float(left), float(right)


def _build_log_grid(
    *,
    spot: float,
    strike: float,
    time_to_maturity: float,
    volatility: float,
    smax_mult: float,
    spot_steps: int,
    time_steps: int,
    method: PDEMethod,
    anchor_spot: float | None = None,
    anchor_half_step: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build a log-spot grid.

    ``dz`` selection by scheme:

    - **Explicit family** (``EXPLICIT``, ``EXPLICIT_HULL``): targets Hull's
      stability scale ``dz_hull = vol * sqrt(3 * dt)`` — the trinomial-
      equivalent spacing with up/mid/down probabilities ``1/6, 2/3, 1/6``.
      Falls back to ``(zmax_target - zmin_target) / spot_steps`` if Hull's
      grid is too narrow to cover the target span.
    - **Unconditionally stable** (``IMPLICIT``, ``CRANK_NICOLSON``):
      ``dz = (zmax_target - zmin_target) / spot_steps`` directly.

    When ``anchor_spot`` is provided, the grid is sized so the anchor sits
    exactly on an interior node (or halfway between two nodes when
    ``anchor_half_step=True``).  The resulting domain is a (possibly slight)
    superset of ``[zmin_target, zmax_target]``.  CN/IMPLICIT grows ``dz`` on
    the binding half (the side of the anchor that needs the larger ``dz``
    to cover its target half); explicit schemes keep ``dz`` fixed by
    stability and shift the grid in place instead.
    """
    if anchor_half_step and anchor_spot is None:
        raise ValidationError("anchor_half_step requires anchor_spot to be provided")

    smax = float(smax_mult * max(spot, strike))
    smin = float(max(max(spot, strike) / smax_mult, 1.0e-8))
    zmin_target = np.log(smin)
    zmax_target = np.log(smax)

    if method in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL):
        d_tau = time_to_maturity / time_steps
        dz_hull = volatility * np.sqrt(3.0 * d_tau)
        grid_width = spot_steps * dz_hull
        if (zmax_target - zmin_target) > grid_width:
            dz = (zmax_target - zmin_target) / spot_steps
            zmin = zmin_target
            zmax = zmax_target
        else:
            dz = dz_hull
            center = np.log(spot)
            zmin = center - 0.5 * grid_width
            zmax = center + 0.5 * grid_width
            if zmin > zmin_target:
                shift = zmin_target - zmin
                zmin += shift
                zmax += shift
            if zmax < zmax_target:
                shift = zmax_target - zmax
                zmin += shift
                zmax += shift
    else:
        # Unconditionally stable schemes: honor spot_steps directly.
        dz = (zmax_target - zmin_target) / spot_steps
        zmin = zmin_target
        zmax = zmax_target

    if anchor_spot is None:
        Z = zmin + dz * np.arange(spot_steps + 1, dtype=float)
        return Z, np.exp(Z), dz

    if anchor_spot <= 0.0:
        raise ValidationError("anchor_spot must be positive for log-spot grids")

    z_anchor = float(np.log(anchor_spot))
    if not (zmin_target <= z_anchor <= zmax_target):
        raise ValidationError("anchor_spot must lie within the log-grid target domain")

    anchor_offset = 0.5 if anchor_half_step else 0.0

    if method in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL):
        # Explicit: ``dz`` is fixed by Hull's stability heuristic, so
        # shift the grid in place while preserving cover of
        # ``[zmin_target, zmax_target]``.  If the target span is binding
        # (already tight to ``spot_steps * dz``), exact anchoring is only
        # feasible when the anchor lies on the fixed-dz grid — or, when
        # ``anchor_half_step=True``, halfway between two fixed-dz nodes.
        j_min = max(
            0,
            int(math.ceil((z_anchor - zmin_target) / dz - anchor_offset - 1.0e-12)),
        )
        j_max = min(
            spot_steps - 1 if anchor_half_step else spot_steps,
            int(math.floor(spot_steps - (zmax_target - z_anchor) / dz - anchor_offset + 1.0e-12)),
        )
        if j_min > j_max:
            raise StabilityError("Unable to align anchor_spot on the log grid with current setup")
        preferred_index = int(round((z_anchor - zmin) / dz - anchor_offset))
        j_anchor = min(max(preferred_index, j_min), j_max)
    else:
        # CN/IMPLICIT: ``dz`` is free, so *grow* it on the binding half
        # rather than shift a fixed-dz grid (which has an unsatisfiable
        # strict-cover constraint when ``dz`` exactly tiles the target).
        # Pick the integer node closest to where the anchor falls, then
        # take the larger of the left/right ``dz`` needed to cover each
        # half.  The resulting grid:
        #   - places the anchor exactly on an interior node,
        #   - is strictly tight to the target on the binding side,
        #   - has up to one cell of slack outside the target on the
        #     other side (slight superset — never under-covers),
        #   - costs at most ~1/(spot_steps - 1) extra ``dz`` vs the
        #     bare-minimum tile of the target span.
        span = zmax_target - zmin_target
        j_opt = int(round(spot_steps * (z_anchor - zmin_target) / span - anchor_offset))
        j_anchor = max(0 if anchor_half_step else 1, min(spot_steps - 1, j_opt))
        left_dz = (z_anchor - zmin_target) / (j_anchor + anchor_offset)
        right_dz = (zmax_target - z_anchor) / (spot_steps - j_anchor - anchor_offset)
        dz = max(left_dz, right_dz)

    zmin = z_anchor - (j_anchor + anchor_offset) * dz
    Z = zmin + dz * np.arange(spot_steps + 1, dtype=float)
    if not anchor_half_step:
        Z[j_anchor] = z_anchor
    return Z, np.exp(Z), dz


def _build_spot_grid(
    *,
    smin: float,
    smax: float,
    spot_steps: int,
    anchor_spot: float | None = None,
    anchor_half_step: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build a uniform spot grid, optionally aligning an anchor to a node or midpoint."""
    if anchor_spot is None:
        grid = np.linspace(smin, smax, spot_steps + 1)
        dS = (smax - smin) / spot_steps
        return grid, grid, dS

    if not (smin < anchor_spot < smax):
        raise ValidationError("anchor_spot must lie strictly inside the spot-grid domain")

    ratio = (anchor_spot - smin) / (smax - smin)
    if anchor_half_step:
        j_max = min(spot_steps - 1, int(math.floor(spot_steps * ratio - 0.5 + 1.0e-12)))
        if j_max < 0:
            raise StabilityError("Unable to align anchor_spot on the spot grid with current setup")
        preferred_index = int(round(spot_steps * ratio - 0.5))
        j_anchor = min(max(preferred_index, 0), j_max)
        dS = (anchor_spot - smin) / (j_anchor + 0.5)
        grid = smin + dS * np.arange(spot_steps + 1, dtype=float)
        return grid, grid, dS

    j_max = min(spot_steps - 1, int(math.floor(spot_steps * ratio + 1.0e-12)))
    if j_max < 1:
        raise StabilityError("Unable to align anchor_spot on the spot grid with current setup")

    preferred_index = int(round(spot_steps * ratio))
    j_anchor = min(max(preferred_index, 1), j_max)
    dS = (anchor_spot - smin) / j_anchor
    grid = smin + dS * np.arange(spot_steps + 1, dtype=float)
    grid[j_anchor] = anchor_spot
    return grid, grid, dS


def _spot_operator_coeffs(
    *,
    spot_values: np.ndarray,
    dS: float,
    risk_free_rate: float,
    dividend_rate: float,
    volatility: float,
    implicit_discounting: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spatial operator coefficients on the spot grid.

    When *implicit_discounting* is True (Hull's explicit scheme), the rV
    term is excluded from beta and instead applied as an implicit
    divisor ``1 / (1 + r * dt)`` in the time-step function.
    """
    diffusion = (volatility**2) * (spot_values**2) / (dS**2)
    drift = (risk_free_rate - dividend_rate) * spot_values / dS
    gamma = 0.5 * (diffusion - drift)
    beta = -diffusion if implicit_discounting else -(diffusion + risk_free_rate)
    alpha = 0.5 * (diffusion + drift)
    return gamma, beta, alpha


def _log_operator_coeffs(
    *,
    dz: float,
    risk_free_rate: float,
    dividend_rate: float,
    volatility: float,
    implicit_discounting: bool = False,
    size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spatial operator coefficients on the log-spot grid.

    Returns constant (Toeplitz) arrays of length *size*, matching the
    signature of ``_spot_operator_coeffs`` so callers can treat both
    grids uniformly.

    When *implicit_discounting* is True (Hull's explicit scheme), r is
    excluded from beta.
    """
    mu = risk_free_rate - dividend_rate - 0.5 * volatility**2
    diffusion = (volatility**2) / (dz**2)
    drift = mu / dz
    gamma = np.full(size, 0.5 * (diffusion - drift))
    beta = np.full(size, -diffusion if implicit_discounting else -(diffusion + risk_free_rate))
    alpha = np.full(size, 0.5 * (diffusion + drift))
    return gamma, beta, alpha


def _scaled_operator_coeffs(
    *,
    gamma: np.ndarray,
    beta: np.ndarray,
    alpha: np.ndarray,
    d_tau: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = -d_tau * gamma
    b = -d_tau * beta
    c = -d_tau * alpha
    return a, b, c


# ---------------------------------------------------------------------------
# Time-step helpers (extracted from _fd_core for readability)
# ---------------------------------------------------------------------------


def _explicit_step(
    V_prev: np.ndarray,
    j: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    left: float,
    right: float,
    intrinsic: np.ndarray | None,
    *,
    r_dt: float = 0.0,
) -> np.ndarray:
    """Explicit (forward-Euler) time step.

    When *r_dt* > 0 (Hull's explicit scheme), the interior update is
    divided by ``(1 + r_dt)`` to apply implicit discounting of the rV
    term.
    """
    V_new = V_prev.copy()
    interior = -a * V_prev[j - 1] + (1.0 - b) * V_prev[j] - c * V_prev[j + 1]
    V_new[j] = interior / (1.0 + r_dt)
    V_new[0] = left
    V_new[-1] = right
    if intrinsic is not None:
        V_new[:] = np.maximum(V_new, intrinsic)
    return V_new


@_njit(cache=True)
def _psor_core(
    x: np.ndarray,
    exercise_j: np.ndarray,
    rhs: np.ndarray,
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    V_left: float,
    V_right: float,
    omega: float,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, int]:
    """Numba-accelerated Projected SOR (Gauss-Seidel with overrelaxation)."""
    n = x.shape[0]
    for k in range(n):
        if x[k] < exercise_j[k]:
            x[k] = exercise_j[k]
    iter_used = max_iter
    for iter_idx in range(max_iter):
        max_diff = 0.0
        for k in range(n):
            left_val = x[k - 1] if k > 0 else V_left
            right_val = x[k + 1] if k < n - 1 else V_right
            gs = (rhs[k] - lower[k] * left_val - upper[k] * right_val) / diag[k]
            old = x[k]
            new = old + omega * (gs - old)
            if new < exercise_j[k]:
                new = exercise_j[k]
            x[k] = new
            diff = new - old if new > old else old - new
            if diff > max_diff:
                max_diff = diff
        if max_diff < tol:
            iter_used = iter_idx + 1
            break
    return x, iter_used


def _psor_solve(
    x: np.ndarray,
    exercise_j: np.ndarray,
    rhs: np.ndarray,
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    V_left: float | np.floating,
    V_right: float | np.floating,
    omega: float,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, int]:
    """Projected SOR (Gauss-Seidel with overrelaxation) for American exercise.

    Returns the updated interior values and the number of iterations used.
    Delegates to a Numba-JIT compiled inner loop when numba is available.
    """
    return _psor_core(
        x,
        exercise_j,
        rhs,
        lower,
        diag,
        upper,
        float(V_left),
        float(V_right),
        omega,
        tol,
        max_iter,
    )


def _implicit_cn_step(
    V_prev: np.ndarray,
    V: np.ndarray,
    j: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    left: float,
    right: float,
    method: PDEMethod,
    intrinsic: np.ndarray | None,
    american_solver: PDEEarlyExercise | None,
    omega: float | None,
    tol: float | None,
    max_iter: int | None,
) -> tuple[np.ndarray, int | None]:
    """One implicit or Crank-Nicolson time step (with optional early exercise).

    Returns the updated V array and the PSOR iteration count (None if PSOR was not used).
    """
    if method is PDEMethod.CRANK_NICOLSON:
        a = a * 0.5
        b = b * 0.5
        c = c * 0.5

    # Tridiagonal system (I - θ·dt·L)x = rhs, diagonals are (a, 1+b, c)
    diag = 1.0 + b

    if method is PDEMethod.IMPLICIT:
        rhs = V_prev[j].copy()
    else:
        rhs = -a * V_prev[j - 1] + (1.0 - b) * V_prev[j] - c * V_prev[j + 1]

    V[0] = left
    V[-1] = right

    rhs_adj = rhs.copy()
    rhs_adj[0] -= a[0] * V[0]
    rhs_adj[-1] -= c[-1] * V[-1]

    x = _solve_tridiagonal_thomas(a[1:], diag, c[:-1], rhs_adj)
    psor_iters: int | None = None

    if intrinsic is None:
        V[j] = x
    else:
        exercise_j = intrinsic[j]
        if american_solver is PDEEarlyExercise.INTRINSIC:
            V[j] = np.maximum(x, exercise_j)
        else:
            x, psor_iters = _psor_solve(
                x,
                exercise_j,
                rhs,
                a,
                diag,
                c,
                float(V[0]),
                float(V[-1]),
                float(omega),
                float(tol),
                int(max_iter),
            )
            V[j] = x

    return V, psor_iters


def _validate_fd_inputs(
    *,
    option_type: OptionType | None,
    time_to_maturity: float,
    spot_steps: int,
    time_steps: int,
    volatility: float,
    discount_curve: DiscountCurve,
    early_exercise: bool,
    method: PDEMethod,
    american_solver: PDEEarlyExercise | None,
    omega: float | None,
    tol: float | None,
    max_iter: int | None,
    payoff_fn: Callable | None = None,
) -> None:
    """Validate FD PDE inputs before grid construction."""
    if payoff_fn is None and option_type not in (OptionType.CALL, OptionType.PUT):
        raise UnsupportedFeatureError("FD PDE valuation supports only vanilla CALL/PUT.")
    if time_to_maturity <= 0:
        raise ValidationError("time_to_maturity must be positive")
    if spot_steps < 3:
        raise ValidationError("spot_steps must be >= 3")
    if time_steps < 1:
        raise ValidationError("time_steps must be >= 1")
    if volatility <= 0:
        raise ValidationError("volatility must be positive")
    if discount_curve is None:
        raise ValidationError("discount_curve is required for PDE valuation")
    if early_exercise and american_solver is None:
        raise ValidationError("american_solver is required when early_exercise=True")
    if early_exercise and american_solver is PDEEarlyExercise.GAUSS_SEIDEL:
        if omega is None or tol is None or max_iter is None:
            raise ValidationError(
                "PSOR params (omega/tol/max_iter) are required for early exercise"
            )
    if method in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL) and (
        american_solver is PDEEarlyExercise.GAUSS_SEIDEL
    ):
        raise UnsupportedFeatureError("GAUSS_SEIDEL is not supported with explicit time stepping")


def _check_explicit_spot_stability(
    *,
    tau_grid: np.ndarray,
    volatility: float,
    smax: float,
    dS: float,
    time_to_maturity: float,
    discount_curve: DiscountCurve,
    dividend_curve: DiscountCurve | None,
    implicit_discounting: bool,
) -> None:
    r"""CFL-style stability checks for an explicit scheme on a uniform spot grid.

    The dominant restriction comes from the diffusion term::

        dt <= dS² / (σ² S_max²)

    When *implicit_discounting* is ``False`` (pure explicit), the reaction
    term :math:`-rV` is discretised explicitly too, giving the tighter
    bound::

        dt <= 1 / (σ² S_max² / dS² + r_max)

    With Hull implicit discounting, ``r`` is handled implicitly via the
    :math:`1/(1 + r \Delta t)` divisor so only diffusion matters.

    A secondary check enforces the central-difference stencil monotonicity
    condition :math:`\alpha, \gamma \ge 0`, i.e. diffusion dominates drift
    at the worst-case grid node (``S_max``).

    Raises ``StabilityError`` if any condition is violated.
    """
    if tau_grid.size < 2:
        return

    dt_steps = np.diff(tau_grid).astype(float)
    max_dt = float(np.max(dt_steps))
    if max_dt <= 0.0:
        return

    # Forward rates for each tau step (calendar times: t = T - tau)
    t_prev = time_to_maturity - tau_grid[:-1]
    t_next = time_to_maturity - tau_grid[1:]
    r_steps = np.array(
        [discount_curve.forward_rate(t1, t0) for t0, t1 in zip(t_prev, t_next)],
        dtype=float,
    )
    if dividend_curve is not None:
        q_steps = np.array(
            [dividend_curve.forward_rate(t1, t0) for t0, t1 in zip(t_prev, t_next)],
            dtype=float,
        )
    else:
        q_steps = np.zeros_like(r_steps)

    rq_abs_max = float(np.max(np.abs(r_steps - q_steps)))

    # (A) Diffusion CFL bound
    diffusion_max = (volatility**2) * (smax**2) / (dS**2)

    if implicit_discounting:
        # r handled implicitly — only diffusion constrains dt
        dt_max = 1.0 / diffusion_max
        mode = "Hull implicit discounting"
    else:
        # Pure explicit: tighten for the reaction term -rV
        r_pos = max(float(np.max(r_steps)), 0.0)
        dt_max = (
            1.0 / (diffusion_max + r_pos) if (diffusion_max + r_pos) > 0.0 else 1.0 / diffusion_max
        )
        mode = "pure explicit discounting"

    if max_dt > dt_max:
        min_steps = int(math.ceil(time_to_maturity / dt_max))
        raise StabilityError(
            f"Explicit spot-grid scheme likely unstable (CFL violation, {mode}). "
            f"max_dt={max_dt:.4g} exceeds dt_max={dt_max:.4g}. "
            f"Increase time_steps to >= {min_steps}, or use log-spot/implicit/CN."
        )

    # (B) Drift monotonicity: central-difference stencil requires
    #     diffusion >= |drift| at worst-case S for alpha, gamma >= 0.
    drift_max = rq_abs_max * smax / dS
    if drift_max > diffusion_max:
        raise StabilityError(
            "Explicit spot-grid scheme likely unstable/oscillatory: drift dominates "
            "diffusion in the central-difference stencil. "
            f"diffusion_max={diffusion_max:.4g}, drift_max={drift_max:.4g}. "
            "Refine the spot grid (smaller dS), reduce smax, or use log-spot/implicit/CN."
        )


def _build_time_step_schedule(
    tau_grid: np.ndarray,
    method: PDEMethod,
    rannacher_steps: int,
) -> list[tuple[float, float, PDEMethod]]:
    """Build the time-step schedule from the tau grid.

    For Crank-Nicolson with Rannacher smoothing (Pooley-Vetzal-Forsyth 2003),
    the first *rannacher_steps* intervals are each replaced by two implicit
    (backward Euler) half-steps of size d_tau/2.  This damps payoff
    non-smoothness while preserving the overall time-grid structure.
    For all other methods the schedule is a straightforward pass-through.
    """
    steps: list[tuple[float, float, PDEMethod]] = []
    for n in range(1, tau_grid.size):
        tau_start = float(tau_grid[n - 1])
        tau_end = float(tau_grid[n])
        if method is PDEMethod.CRANK_NICOLSON and rannacher_steps > 0 and n <= rannacher_steps:
            tau_mid = 0.5 * (tau_start + tau_end)
            steps.append((tau_start, tau_mid, PDEMethod.IMPLICIT))
            steps.append((tau_mid, tau_end, PDEMethod.IMPLICIT))
        else:
            steps.append((tau_start, tau_end, method))
    return steps
