"""Barrier-specific grid builders and ``spot_steps`` auto-resolution.

Single-barrier continuous (truncated, barrier-on-node) and discrete
(Boyle-Tian half-step) grids, their double-barrier generalisations, and
``_resolve_pde_spot_steps`` which resolves ``PDEParams.spot_steps=None``
("auto") at ``OptionValuation`` construction time.
"""

from __future__ import annotations
from collections.abc import Sequence

import math
import datetime as dt

import numpy as np

from ...enums import (
    BarrierDirection,
    BarrierMonitoring,
    DayCountConvention,
    PDEMethod,
    PDESpaceGrid,
)
from ...utils import calculate_year_fraction
from ...exceptions import StabilityError
from ..contracts import DoubleBarrierSpec, _BaseBarrierSpec
from ..params import PDEParams


def _barrier_monitoring_taus(
    *,
    monitoring_dates: Sequence[dt.datetime],
    pricing_date: dt.datetime,
    maturity: dt.datetime,
    day_count_convention: DayCountConvention,
) -> list[float]:
    """Convert monitoring datetime schedule to tau-space values.

    Returns sorted list of taus (time remaining from maturity perspective).
    """
    ttm = calculate_year_fraction(
        pricing_date,
        maturity,
        day_count_convention=day_count_convention,
    )
    taus: list[float] = []
    for d in monitoring_dates:
        if pricing_date <= d <= maturity:
            t = calculate_year_fraction(
                pricing_date,
                d,
                day_count_convention=day_count_convention,
            )
            taus.append(round(ttm - t, 12))
    taus.sort()
    return taus


def _build_ko_continuous_log_grid(
    *,
    smin_target: float,
    smax_target: float,
    ref_price: float,
    smax_mult: float,
    direction: BarrierDirection,
    volatility: float,
    time_to_maturity: float,
    spot_steps: int,
    time_steps: int,
    method: PDEMethod,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build a log-spot grid with the barrier as a boundary node.

    For explicit-family schemes, preserves Hull's dz scale;
    CN/IMPLICIT honor spot_steps directly.
    """
    explicit_scheme = method in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL)
    if direction is BarrierDirection.DOWN:
        zmin = np.log(smin_target)
        zmax_default = np.log(smax_target)
        if explicit_scheme:
            dz_hull = volatility * np.sqrt(3.0 * (time_to_maturity / time_steps))
            grid_width = spot_steps * dz_hull
            if (zmax_default - zmin) > grid_width:
                dz = (zmax_default - zmin) / spot_steps
            else:
                dz = dz_hull
                zmax_default = zmin + grid_width
        else:
            dz = (zmax_default - zmin) / spot_steps
        Z = np.linspace(zmin, zmax_default, spot_steps + 1)
    else:
        zmax = np.log(smax_target)
        zmin_default = np.log(max(ref_price / smax_mult, 1.0e-8))
        if explicit_scheme:
            dz_hull = volatility * np.sqrt(3.0 * (time_to_maturity / time_steps))
            grid_width = spot_steps * dz_hull
            if (zmax - zmin_default) > grid_width:
                dz = (zmax - zmin_default) / spot_steps
            else:
                dz = dz_hull
                zmin_default = zmax - grid_width
        else:
            dz = (zmax - zmin_default) / spot_steps
        Z = np.linspace(zmin_default, zmax, spot_steps + 1)
    S = np.exp(Z)
    return Z, S, dz


def _build_double_barrier_continuous_log_grid(
    *,
    lower_barrier: float,
    upper_barrier: float,
    volatility: float,
    time_to_maturity: float,
    spot_steps: int,
    time_steps: int,
    method: PDEMethod,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build a log-spot grid truncated at *both* barriers (Boyle-Tian).

    Both barriers land exactly on grid boundary nodes: ``S[0] == lower_barrier``
    and ``S[-1] == upper_barrier``.  The corridor ``[log B_d, log B_u]`` is
    pinned at both ends, so we simply divide it into ``spot_steps`` intervals:
    ``dz = (log B_u − log B_d) / spot_steps``.

    This is exactly Boyle-Tian's rescaled step ``Δy* = λ*σ√Δt = corridor / N₀``
    (their eq. 16) with ``N₀ = spot_steps`` — the explicit ``λ*`` never needs
    computing, since ``corridor / spot_steps`` already *is* the rescaled step
    that places both barriers on the grid.

    For the explicit family the log-spot stencil requires
    ``λ = dz / (σ√Δt) ≥ 1`` for a non-negative middle transition probability
    (``p_m = 1 − σ²Δt/dz²``).  A narrow corridor with many ``spot_steps`` drives
    ``λ < 1``, where the explicit scheme blows up (negative probabilities →
    NaN).  We raise :class:`StabilityError` in that regime — consistent with
    ``_check_explicit_spot_stability`` for explicit spot grids — rather than
    return garbage.  CN / IMPLICIT are unconditionally stable and unaffected.

    Returns
    -------
    (Z, S, dz) : log grid, spot grid, log-step.  ``len(S) - 1 == spot_steps``.
    """
    z_d = float(np.log(lower_barrier))
    z_u = float(np.log(upper_barrier))
    corridor = z_u - z_d
    dz = corridor / spot_steps

    if method in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL):
        sigma_sqrt_dt = volatility * np.sqrt(time_to_maturity / time_steps)
        lam = dz / sigma_sqrt_dt
        if lam < 1.0:
            max_stable_steps = int(corridor / sigma_sqrt_dt)
            raise StabilityError(
                f"Explicit double-barrier scheme is unstable: "
                f"lambda = dz/(sigma*sqrt(dt)) = {lam:.3f} < 1 "
                f"(corridor {corridor:.4f} over {spot_steps} steps gives negative "
                f"transition probabilities). Reduce spot_steps to <= {max_stable_steps}, "
                f"set spot_steps=None to auto-size to Hull's stable trinomial step, "
                f"or use CRANK_NICOLSON / IMPLICIT (the default for barriers)."
            )

    Z = np.linspace(z_d, z_u, spot_steps + 1)
    S = np.exp(Z)
    return Z, S, dz


_DEFAULT_AUTO_SPOT_STEPS = 200
"""Resolution for ``spot_steps=None`` under an unconditionally-stable scheme.

CN/IMPLICIT have no physical ``dz`` scale (no CFL constraint), so ``None``
falls back to the library default rather than deriving from ``dz_hull`` (which
is an *explicit*-stability quantity and, being ``∝ √Δt``, would perversely give
the coarse-time CN grid *fewer* nodes than the fine-time explicit grid).  Pass
an explicit int to dial CN accuracy/speed; the barrier factories already do.
"""


def _resolve_pde_spot_steps(
    *,
    spec: object,
    spot: float,
    strike: float,
    volatility: float,
    time_to_maturity: float,
    params: PDEParams,
) -> int:
    """Resolve ``PDEParams.spot_steps`` when the caller left it ``None`` ("auto").

    ``None`` means "size the spatial grid for me".  Resolution only does real
    work for the **explicit family** with a log spatial grid, where stability
    fixes the log-step to Hull's trinomial spacing ``dz_hull = σ·√(3·Δt)``
    (the ``p_m = 2/3`` stencil).  What that pins depends on which boundary is free:

    * **Continuous double barrier (log grid)** — *both* ends are pinned at the
      barriers, so the width ``corridor = ln(U / L)`` is fixed and the **count**
      is solved for::

          spot_steps = round(corridor / dz_hull)      # ⇒ λ ≈ √3, p_m ≈ 2/3

      This removes the foot-gun where a hand-picked ``spot_steps`` drives
      ``λ = dz / (σ√Δt) < 1`` and the explicit scheme produces negative
      transition probabilities.

    * **Free far-field (vanilla / single barrier / double barrier American KI /
        discrete double)**
      — at least one boundary floats, so cover the target log-span at ``dz_hull``::

          spot_steps = ceil(span / dz_hull)

    Unconditionally-stable schemes (CN/IMPLICIT) and spot-space explicit grids
    fall back to :data:`_DEFAULT_AUTO_SPOT_STEPS`.  A non-``None`` ``spot_steps``
    is always honored verbatim.

    Called once, at ``OptionValuation`` construction, so the resolved value is
    frozen across every bump-and-revalue solve.
    """
    if params.spot_steps is not None:
        return params.spot_steps

    if params.method not in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL):
        return _DEFAULT_AUTO_SPOT_STEPS

    dz_hull = float(volatility * math.sqrt(3.0 * time_to_maturity / params.time_steps))
    if dz_hull <= 0.0:
        return _DEFAULT_AUTO_SPOT_STEPS

    # Pinned: continuous double barrier on a log grid — width = corridor.
    if (
        isinstance(spec, DoubleBarrierSpec)
        and spec.monitoring is BarrierMonitoring.CONTINUOUS
        and params.space_grid is PDESpaceGrid.LOG_SPOT
    ):
        corridor = float(np.log(spec.upper_barrier) - np.log(spec.lower_barrier))
        return max(3, int(round(corridor / dz_hull)))

    # Free far-field: cover the target log-span at Hull spacing.
    if params.space_grid is PDESpaceGrid.LOG_SPOT:
        ref_hi = max(spot, strike)
        ref_lo = min(spot, strike)
        if isinstance(spec, _BaseBarrierSpec):
            levels = [
                float(getattr(spec, attr))
                for attr in ("lower_barrier", "upper_barrier", "barrier")
                if getattr(spec, attr, None) is not None
            ]
            if levels:
                ref_hi = max(ref_hi, *levels)
                ref_lo = min(ref_lo, *levels)
        hi_target = np.log(params.smax_mult * ref_hi)
        lo_target = np.log(max(ref_lo / params.smax_mult, 1.0e-8))
        return max(3, int(math.ceil((hi_target - lo_target) / dz_hull)))

    # Spot (non-log) explicit grids have no clean dz_hull pin — default.
    return _DEFAULT_AUTO_SPOT_STEPS


def _build_double_barrier_discrete_grid(
    *,
    lower_barrier: float,
    upper_barrier: float,
    spot: float,
    strike: float,
    volatility: float,
    time_to_maturity: float,
    smax_mult: float,
    spot_steps: int,
    time_steps: int,
    method: PDEMethod,
    log: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build a *full* grid for DISCRETE double-barrier monitoring with **both**
    barriers placed half-way between adjacent nodes (Boyle-Tian).

    Discrete monitoring keeps the full domain (the option only knocks out when
    spot is past a barrier *on a monitoring date*), so — unlike the continuous
    core — we do not truncate at the barriers.  Instead we follow Boyle-Tian's
    half-step idea, generalised to *two* barriers.

    The single-barrier half-step trick (``anchor_half_step`` in
    :func:`_build_log_grid`) only guarantees *one* barrier sits midway between
    nodes.  Boyle-Tian note that with two barriers the upper one must be
    half-way too, which they secure by setting ``λ = λ*`` so the node spacing
    equals their rescaled ``Δy* = λ*σ√Δt``.  Here that is simply
    ``step = corridor / N`` with integer ``N`` (the number of intervals
    spanning the corridor): the corridor is then an exact multiple of the
    step, so once the lower barrier is anchored half-way the upper barrier
    lands half-way automatically.  ``zmin`` is set accordingly.

    Because no grid node coincides with either barrier, the monitoring reset
    (``S ≤ L`` / ``S ≥ U``) is unambiguous and free of the odd/even
    oscillation that on-node placement produces.

    For the explicit family the log stencil needs ``λ = step/(σ√Δt) ≥ 1`` for
    a non-negative middle transition probability; we raise
    :class:`StabilityError` otherwise (CN / IMPLICIT are unaffected).

    Returns
    -------
    (axis, S, step) : log-axis (``log=True``) or spot-axis grid, the spot
    grid, and the node spacing ``dz``/``dS``.
    """
    if log:
        lower_barrier_ax, upper_barrier_ax = (
            float(np.log(lower_barrier)),
            float(np.log(upper_barrier)),
        )
        ref_hi = max(spot, strike, upper_barrier)
        ref_lo = min(spot, strike, lower_barrier)
        hi_target = float(np.log(smax_mult * ref_hi))
        lo_target = float(np.log(max(ref_lo / smax_mult, 1.0e-8)))
    else:
        lower_barrier_ax, upper_barrier_ax = float(lower_barrier), float(upper_barrier)
        ref_hi = max(spot, strike, upper_barrier)
        hi_target = float(smax_mult * ref_hi)
        lo_target = 0.0

    corridor = upper_barrier_ax - lower_barrier_ax
    total = hi_target - lo_target
    explicit = method in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL)

    # Both regimes return exactly ``spot_steps + 1`` nodes (``spot_steps``
    # intervals), matching ``_build_log_grid``.  They differ only in how the
    # log-step ``dz`` is chosen:
    #
    # 1. ``EXPLICIT + log``: dz is fixed by stability to Hull's trinomial
    #    spacing ``σ√(3Δt)`` (kept ≥ ``σ√Δt`` for non-negative transition
    #    probabilities), which pins the corridor count ``m``.  The wings then
    #    absorb the remaining budget around that fixed dz: shrunk
    #    proportionally if the coverage wings + ``m`` overflow ``spot_steps``,
    #    or padded symmetrically — pushing the artificial far-field boundary
    #    further out — if they underflow.
    # 2. **All other schemes** (CN, IMPLICIT, or any spot-grid explicit): dz is
    #    free, so it is chosen to honor ``spot_steps`` exactly.  Pick the
    #    corridor interval count ``m`` near ``corridor * spot_steps / total``,
    #    reduce it until minimum wings + ``m`` ≤ ``spot_steps - 1`` (monotone
    #    in feasibility: enlarging dz shrinks both the corridor count and the
    #    required wings), then distribute leftover intervals between the wings.
    #
    # In both regimes the corridor is held to an *integer* number of
    # intervals (``step = corridor / m``); anchoring the lower barrier
    # half-way between nodes then puts the upper barrier half-way
    # automatically (Boyle-Tian rescaled Δy*).
    target_sum = spot_steps - 1  # = n_below + m + n_above (gives spot_steps intervals)

    def _min_wings(step_try: float) -> tuple[int, int]:
        """Minimum wing counts to cover [lo_target, hi_target] given step."""
        nb = max(1, int(math.ceil((lower_barrier_ax - lo_target) / step_try - 0.5 - 1.0e-12)))
        na = max(1, int(math.ceil((hi_target - upper_barrier_ax) / step_try - 0.5 - 1.0e-12)))
        return nb, na

    if explicit and log:
        # Hull-spacing regime: dz fixed by stability.  spot_steps is honored
        # exactly; if Hull's m plus the wings needed to cover ``smax_mult *
        # max(spot, strike, U)`` would overflow the budget, shrink wings
        # proportionally; if it would underflow, pad wings symmetrically so
        # the artificial far-field boundary sits further out (free reduction
        # of Dirichlet BC leakage).
        sigma_sqrt_dt = volatility * np.sqrt(time_to_maturity / time_steps)
        hull_step = volatility * np.sqrt(3.0 * time_to_maturity / time_steps)
        m = max(1, int(round(corridor / hull_step)))
        step = corridor / m
        lam = step / sigma_sqrt_dt
        if lam < 1.0:
            raise StabilityError(
                f"Explicit discrete double-barrier scheme is unstable: "
                f"lambda = step/(sigma*sqrt(dt)) = {lam:.3f} < 1 "
                f"(corridor {corridor:.4f} forces {m} intervals). "
                f"Increase time_steps, or use CRANK_NICOLSON / IMPLICIT."
            )
        nb_nat = max(1, int(round((lower_barrier_ax - lo_target) / step)))
        na_nat = max(1, int(round((hi_target - upper_barrier_ax) / step)))
        if nb_nat + m + na_nat > target_sum:
            # Cap-shrink wings proportionally; preserves Hull spacing.
            wing_budget = max(2, target_sum - m)
            wing = nb_nat + na_nat
            n_below = max(1, int(round(wing_budget * nb_nat / wing)))
            n_above = max(1, wing_budget - n_below)
        else:
            # Pad wings symmetrically with the leftover budget.
            extras = target_sum - (nb_nat + m + na_nat)
            n_below = nb_nat + extras // 2
            n_above = na_nat + (extras - extras // 2)
    else:
        # Variable-dz regime: honor spot_steps exactly.  Pick m near
        # ``corridor * spot_steps / total``, reduce if minimum wings won't
        # fit, then distribute leftover intervals between the wings.
        step_target = total / spot_steps
        m = max(1, int(round(corridor / step_target)))
        # Monotonicity: ``min_wings(m) + m`` is non-decreasing in ``m``
        # (smaller dz → more wing intervals AND more corridor intervals),
        # so reducing m strictly shrinks the required sum.  Loop is bounded
        # by ~1-2 iterations in practice for sensible parameter ranges.
        while m > 1:
            nb_min, na_min = _min_wings(corridor / m)
            if nb_min + m + na_min <= target_sum:
                break
            m -= 1
        step = corridor / m
        nb_min, na_min = _min_wings(step)
        extras = target_sum - (nb_min + m + na_min)
        if extras >= 0:
            # Distribute extras symmetrically (asymmetric by 1 if odd).
            n_below = nb_min + extras // 2
            n_above = na_min + (extras - extras // 2)
        else:
            # Pathological edge case (m == 1 still doesn't fit) — shrink
            # wings proportionally, sacrificing far-field coverage.
            wing_budget = max(2, target_sum - m)
            denom = max(1, nb_min + na_min)
            n_below = max(1, int(round(wing_budget * nb_min / denom)))
            n_above = max(1, wing_budget - n_below)

    if not log:
        # Spot grids have a hard floor at 0; cap the lower wing so
        # ``axis_min = lower_barrier_ax - (n_below + 0.5) * step > 0``.  Donate
        # any slack to the upper wing so the total interval count is preserved.
        max_nb = max(1, int(math.floor(lower_barrier_ax / step - 0.5 - 1.0e-12)))
        if n_below > max_nb:
            n_above += n_below - max_nb
            n_below = max_nb

    # Anchor the lower barrier half-way between nodes ``n_below`` and
    # ``n_below + 1``: ``lower_barrier_ax = axis_min + (n_below + 0.5) * step``.
    # The corridor is an exact multiple of the step, so the upper barrier lands
    # half-way too.
    axis_min = lower_barrier_ax - (n_below + 0.5) * step
    n_intervals = n_below + m + n_above + 1
    axis = axis_min + step * np.arange(n_intervals + 1, dtype=float)
    S = np.exp(axis) if log else axis
    return axis, S, step


def _build_double_barrier_full_grid(
    *,
    lower_barrier: float,
    upper_barrier: float,
    spot: float,
    strike: float,
    volatility: float,
    time_to_maturity: float,
    smax_mult: float,
    spot_steps: int,
    time_steps: int,
    method: PDEMethod,
    log: bool,
) -> tuple[np.ndarray, np.ndarray, float, int, int]:
    """Build a *full* grid (extending beyond both barriers) with both barriers
    landing exactly on nodes — used by the two-surface double-KI solver.

    Unlike the KO core's corridor-truncated grid, the active (knocked-in)
    surface is a full vanilla solve, so the grid must reach the standard
    ``smax_mult`` far field on both sides while still placing ``lower_barrier``
    and ``upper_barrier`` on nodes (so the inactive↔active coupling at each
    barrier is exact).

    For the explicit family the log stencil needs ``λ = step/(σ√Δt) ≥ 1`` for a
    non-negative middle transition probability; we raise :class:`StabilityError`
    otherwise — matching the KO core (:func:`_build_double_barrier_continuous_log_grid`)
    and the discrete builder — so an unstable explicit KI solve fails fast at
    grid construction. CN / IMPLICIT are unconditionally stable and
    unaffected.

    ``spot_steps`` is honored exactly: ``len(axis) == spot_steps + 1`` (i.e.
    ``spot_steps`` intervals).  The corridor receives an integer share ``m``
    near its proportional weight ``corridor / total``; ``m`` is monotonically
    reduced if the minimum wings needed to cover ``[lo_target, hi_target]``
    won't fit in the budget; leftover budget is then distributed between the
    two wings.

    Returns ``(grid, S, step, j_L, j_U)`` where ``grid`` is the log-axis
    (LOG_SPOT) or spot-axis (SPOT), ``step`` is ``dz``/``dS``, and
    ``j_L``/``j_U`` are the lower/upper barrier node indices (so
    ``axis[j_L] == log(lower_barrier)`` etc. — barriers land *on* nodes
    here, not half-step between them).
    """
    if log:
        lower_barrier_ax, upper_barrier_ax = (
            float(np.log(lower_barrier)),
            float(np.log(upper_barrier)),
        )
        ref_hi = max(spot, strike, upper_barrier)
        ref_lo = min(spot, strike, lower_barrier)
        hi_target = float(np.log(smax_mult * ref_hi))
        lo_target = float(np.log(max(ref_lo / smax_mult, 1.0e-8)))
    else:
        lower_barrier_ax, upper_barrier_ax = float(lower_barrier), float(upper_barrier)
        ref_hi = max(spot, strike, upper_barrier)
        hi_target = float(smax_mult * ref_hi)
        lo_target = 0.0

    corridor = upper_barrier_ax - lower_barrier_ax
    total = hi_target - lo_target

    # Constraint: ``n_below + m + n_above == spot_steps``
    # (barriers sit on nodes, so intervals = wings + corridor exactly).
    target_sum = spot_steps

    def _min_wings(step_try: float) -> tuple[int, int]:
        """Minimum wings to cover [lo_target, hi_target] given ``step``."""
        # n_below s.t. lower_barrier_ax - n_below*step ≤ lo_target
        # → n_below ≥ (lower_barrier_ax - lo_target)/step
        nb = max(1, int(math.ceil((lower_barrier_ax - lo_target) / step_try - 1.0e-12)))
        na = max(1, int(math.ceil((hi_target - upper_barrier_ax) / step_try - 1.0e-12)))
        return nb, na

    # Initial corridor count: proportional share of the budget.  Floor at 2 so
    # there's at least one interior corridor node between the two barriers
    # (the KI inactive surface needs at least one strictly-interior data point).
    m = max(2, int(round(spot_steps * corridor / total)))

    # Reduce ``m`` monotonically until ``min_wings + m`` fits the budget.
    # Reducing m enlarges the step (dz = corridor/m), which monotonically
    # shrinks the required wings, so the feasibility check is monotone-
    # decreasing in m.  Bounded by a handful of iterations in practice.
    while m > 2:
        nb_min, na_min = _min_wings(corridor / m)
        if nb_min + m + na_min <= target_sum:
            break
        m -= 1

    step = corridor / m

    # Explicit-family log stencil needs λ = step/(σ√Δt) ≥ 1 for a non-negative
    # middle transition probability. Mirror the KO core's guard so the KI full
    # grid fails fast.
    if method in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL) and log:
        sigma_sqrt_dt = volatility * math.sqrt(time_to_maturity / time_steps)
        lam = step / sigma_sqrt_dt
        if lam < 1.0:
            raise StabilityError(
                f"Explicit double knock-in scheme is unstable: "
                f"lambda = step/(sigma*sqrt(dt)) = {lam:.3f} < 1 "
                f"(corridor {corridor:.4f} over {m} intervals gives negative "
                f"transition probabilities). Reduce spot_steps, increase "
                f"time_steps, or use CRANK_NICOLSON / IMPLICIT."
            )

    nb_min, na_min = _min_wings(step)

    extras = target_sum - (nb_min + m + na_min)
    if extras >= 0:
        # Distribute leftover budget to wings symmetrically (asymmetric by 1
        # if odd).  Pushes the far-field boundary further out — free
        # reduction of Dirichlet BC leakage into the live region.
        n_below = nb_min + extras // 2
        n_above = na_min + (extras - extras // 2)
    else:
        # Pathological edge case: even m=2 with minimum wings overflows the
        # budget.  Shrink wings proportionally — domain coverage suffers but
        # spot_steps and on-node barrier placement are preserved.
        wing_budget = max(2, target_sum - m)
        denom = max(1, nb_min + na_min)
        n_below = max(1, int(round(wing_budget * nb_min / denom)))
        n_above = max(1, wing_budget - n_below)

    if not log:
        # Spot grid has a hard floor at 0.  Lower wing must keep
        # ``axis_min = lower_barrier_ax - n_below * step > 0``.  Donate any slack
        # to the upper wing so the total interval count is preserved.
        max_nb = max(1, int(math.floor(lower_barrier_ax / step - 1.0e-12)))
        if n_below > max_nb:
            n_above += n_below - max_nb
            n_below = max_nb

    # Build the grid.  ``lower_barrier_ax`` sits on node ``n_below`` exactly;
    # ``upper_barrier_ax`` on node ``n_below + m`` exactly (corridor = m * step).
    axis_min = lower_barrier_ax - n_below * step
    n_intervals = n_below + m + n_above  # = spot_steps
    axis = axis_min + step * np.arange(n_intervals + 1, dtype=float)
    S = np.exp(axis) if log else axis
    return axis, S, step, n_below, n_below + m
