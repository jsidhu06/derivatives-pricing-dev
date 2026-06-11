"""Barrier valuation classes plugging into ``OptionValuation``.

``_FDBarrierValuationBase`` holds the solve/greek/memoisation skeleton
shared by single- and double-barrier FD valuation (KO direct solve,
American KI two-surface solve, European KI in-out parity);
``_FDBarrierValuation`` and ``_FDDoubleBarrierValuation`` supply the
flavour-specific hooks.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import logging
import threading

import numpy as np

from ...enums import (
    BarrierAction,
    BarrierMonitoring,
    ExerciseType,
    RebateTiming,
)
from ...utils import log_timing
from ...exceptions import ConfigurationError
from ..contracts import BarrierSpec, DoubleBarrierSpec, _BaseBarrierSpec
from ..params import PDEParams
from ._kernels import _build_tau_grid, _dividend_tau_schedule
from ._core import _FDGridGreeksMixin, _fd_core
from ._barrier_grids import _barrier_monitoring_taus
from ._barrier_cores import (
    _fd_barrier_ki_core,
    _fd_barrier_ko_core,
    _fd_double_barrier_ki_core,
    _fd_double_barrier_ko_core,
)

if TYPE_CHECKING:
    from ..core import OptionValuation


logger = logging.getLogger(__name__)


class _FDBarrierValuationBase(_FDGridGreeksMixin):
    """Shared PDE finite-difference scaffolding for single- and double-barrier
    options.

    Both barrier flavours share an identical solve/greek/memoisation skeleton —
    knock-out runs a single backward solve, American knock-in runs a
    two-surface coupled solve, and European knock-in is reconstructed via
    in-out parity (``V_KI = V_vanilla + R·df_T − V_KO``).  Grid greeks come from
    :class:`_FDGridGreeksMixin` (KO / American KI) or native-surface parity
    (European KI).  Everything here is barrier-agnostic; subclasses supply only:

    - :attr:`_engine_label` — the engine name used in timing logs;
    - :meth:`_ko_core` / :meth:`_ki_core` — the knock-out / American-knock-in
      core solvers;
    - :meth:`_barrier_solve_args` — the barrier-specific kwargs (``barrier`` /
      ``direction`` for single, ``lower_barrier`` / ``upper_barrier`` for
      double) merged into the shared solve arguments.
    """

    #: Engine name for timing logs; overridden per subclass.
    _engine_label: str = "barrier"

    def __init__(self, valuation_ctx: OptionValuation) -> None:
        self.valuation_ctx = valuation_ctx
        self.underlying = valuation_ctx.underlying  # type: ignore[assignment]
        self._spec: _BaseBarrierSpec = valuation_ctx.spec  # type: ignore[assignment]
        assert isinstance(valuation_ctx.params, PDEParams)
        self.pde_params = valuation_ctx.params
        # Lazy PDE-solve caches.  ``_solve_result`` holds the full 5-tuple
        # returned by the backward PDE solve (KO directly; EU KI via parity; AM
        # KI via the two-surface coupled solver).  ``_ki_components_result``
        # holds the raw KO + vanilla solves used by European-KI parity greeks.
        # Both are populated on first demand and shared across every subsequent
        # PV / greek call on this instance.
        self._solve_result: tuple[float, np.ndarray, np.ndarray, np.ndarray, float] | None = None
        self._ki_components_result: (
            tuple[
                tuple[float, np.ndarray, np.ndarray, np.ndarray, float],
                tuple[float, np.ndarray, np.ndarray, np.ndarray, float],
            ]
            | None
        ) = None
        # Separate re-entrant locks for the two independent caches (European KI
        # ``_compute_solve`` calls ``_ki_components`` which takes the other lock
        # — no deadlock — but ``RLock`` future-proofs nested same-thread calls).
        self._solve_lock = threading.RLock()
        self._ki_components_lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # Subclass hooks                                                     #
    # ------------------------------------------------------------------ #
    def _ko_core(self, **kwargs) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
        """Run the knock-out core solver for this barrier flavour."""
        raise NotImplementedError

    def _ki_core(self, **kwargs) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
        """Run the American two-surface knock-in core solver."""
        raise NotImplementedError

    def _barrier_solve_args(self) -> dict:
        """Return the barrier-specific kwargs merged into :meth:`_base_solve_args`."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Shared resolution helpers                                          #
    # ------------------------------------------------------------------ #
    def _resolved_knock_out_value(self) -> float | None:
        if (
            self._spec.action is not BarrierAction.OUT
            or not self.valuation_ctx._barrier_triggered_at_inception()
        ):
            return None

        if self._spec.rebate <= 0.0:
            return 0.0
        if self._spec.rebate_timing is RebateTiming.AT_HIT:
            return float(self._spec.rebate)

        ttm = self.valuation_ctx._maturity_year_fraction()
        return float(self._spec.rebate) * float(self.valuation_ctx.discount_curve.df(ttm))

    def _last_dtau(self) -> float:
        solve_args = self._base_solve_args()
        time_to_maturity = float(solve_args["time_to_maturity"])
        extra_taus = [
            tau
            for tau, _ in solve_args["dividend_schedule"] or []
            if 1.0e-12 < tau < time_to_maturity - 1.0e-12
        ]
        if solve_args["monitoring_taus"] is not None:
            extra_taus.extend(solve_args["monitoring_taus"])
        tau_grid = _build_tau_grid(time_to_maturity, int(solve_args["time_steps"]), extra_taus)
        if tau_grid.size < 2:
            return 0.0
        return float(tau_grid[-1] - tau_grid[-2])

    def _discounted_rebate_theta(self, last_dtau: float) -> float:
        """Per-day theta of the AT_EXPIRY rebate leg ``R·df(0,T)`` (0 otherwise)."""
        if (
            self._spec.rebate <= 0.0
            or self._spec.rebate_timing is not RebateTiming.AT_EXPIRY
            or last_dtau <= 0.0
        ):
            return 0.0

        ttm = self.valuation_ctx._maturity_year_fraction()
        discount_curve = self.valuation_ctx.discount_curve
        current_value = float(self._spec.rebate) * float(discount_curve.df(ttm))
        previous_value = float(self._spec.rebate) * float(
            discount_curve.df(max(ttm - last_dtau, 0.0))
        )
        return float((previous_value - current_value) / last_dtau / 365.0)

    def _resolved_knock_out_theta(self) -> float:
        return self._discounted_rebate_theta(self._last_dtau())

    @staticmethod
    def _grid_delta_from_result(
        result: tuple[float, np.ndarray, np.ndarray, np.ndarray, float],
        spot: float,
    ) -> float:
        _, S, V, _, _ = result
        j = _FDGridGreeksMixin._spot_grid_index(S, spot)
        return _FDGridGreeksMixin._grid_delta_at_spot(S, V, j, spot)

    @staticmethod
    def _grid_gamma_from_result(
        result: tuple[float, np.ndarray, np.ndarray, np.ndarray, float],
        spot: float,
    ) -> float:
        _, S, V, _, _ = result
        j = _FDGridGreeksMixin._spot_grid_index(S, spot)
        return _FDGridGreeksMixin._grid_gamma_safe(S, V, j, spot)

    def _grid_theta_from_result(
        self,
        result: tuple[float, np.ndarray, np.ndarray, np.ndarray, float],
        spot: float,
    ) -> float:
        _, S, V, _, last_dtau = result
        if last_dtau <= 0.0:
            return 0.0
        j = _FDGridGreeksMixin._spot_grid_index(S, spot)
        return self._grid_theta_bs_identity(S, V, j, spot, last_dtau)

    def _base_solve_args(self) -> dict:
        """Build keyword arguments shared by both KO and KI solvers.

        The barrier-agnostic kwargs are assembled here; the barrier-specific
        keys come from :meth:`_barrier_solve_args`.
        """
        params = self.pde_params
        spec = self._spec
        ctx = self.valuation_ctx
        early_exercise = spec.exercise_type is ExerciseType.AMERICAN

        time_to_maturity = ctx._maturity_year_fraction()

        dividend_schedule = _dividend_tau_schedule(
            discrete_dividends=self.underlying.discrete_dividends,
            pricing_date=ctx.pricing_date,
            maturity=ctx.maturity,
            day_count_convention=ctx.day_count_convention,
        )
        # Resolve monitoring dates to taus for discrete monitoring.
        monitoring_taus: list[float] | None = None
        if spec.monitoring is BarrierMonitoring.DISCRETE:
            mon_dates = ctx._barrier_monitoring_dates()
            monitoring_taus = _barrier_monitoring_taus(
                monitoring_dates=mon_dates,
                pricing_date=ctx.pricing_date,
                maturity=ctx.maturity,
                day_count_convention=ctx.day_count_convention,
            )

        args = dict(
            spot=float(self.underlying.initial_value),
            strike=float(spec.strike),
            time_to_maturity=float(time_to_maturity),
            volatility=float(self.underlying.volatility),
            discount_curve=ctx.discount_curve,
            dividend_curve=self.underlying.dividend_curve,
            dividend_schedule=dividend_schedule,
            option_type=spec.option_type,
            monitoring=spec.monitoring,
            rebate=float(spec.rebate),
            rebate_timing=spec.rebate_timing,
            monitoring_taus=monitoring_taus,
            smax_mult=float(params.smax_mult),
            spot_steps=int(params.spot_steps),
            time_steps=int(params.time_steps),
            early_exercise=early_exercise,
            method=params.method,
            rannacher_steps=int(params.rannacher_steps),
            space_grid=params.space_grid,
            american_solver=params.american_solver if early_exercise else None,
            omega=float(params.omega) if early_exercise else None,
            tol=float(params.tol) if early_exercise else None,
            max_iter=int(params.max_iter) if early_exercise else None,
        )
        args.update(self._barrier_solve_args())
        return args

    # ------------------------------------------------------------------ #
    # European-KI parity components (memoised)                          #
    # ------------------------------------------------------------------ #
    def _ki_components(
        self,
    ) -> tuple[
        tuple[float, np.ndarray, np.ndarray, np.ndarray, float],
        tuple[float, np.ndarray, np.ndarray, np.ndarray, float],
    ]:
        """Return the native KO and vanilla solves used by European KI parity.

        Memoised on the instance: the first call runs both solves; every
        subsequent call (from :meth:`delta`, :meth:`gamma`, :meth:`theta`, or
        internally from :meth:`_compute_solve`) is O(1).
        """
        if self._ki_components_result is not None:
            return self._ki_components_result
        with self._ki_components_lock:
            if self._ki_components_result is not None:
                return self._ki_components_result
            self._ki_components_result = self._compute_ki_components()
            return self._ki_components_result

    def _compute_ki_components(
        self,
    ) -> tuple[
        tuple[float, np.ndarray, np.ndarray, np.ndarray, float],
        tuple[float, np.ndarray, np.ndarray, np.ndarray, float],
    ]:
        solve_args = self._base_solve_args()
        ko_result = self._ko_core(**solve_args)
        van_result = _fd_core(
            spot=solve_args["spot"],
            strike=solve_args["strike"],
            time_to_maturity=solve_args["time_to_maturity"],
            volatility=solve_args["volatility"],
            discount_curve=solve_args["discount_curve"],
            dividend_curve=solve_args["dividend_curve"],
            dividend_schedule=solve_args["dividend_schedule"],
            option_type=self._spec.option_type,
            smax_mult=solve_args["smax_mult"],
            spot_steps=solve_args["spot_steps"],
            time_steps=solve_args["time_steps"],
            early_exercise=False,
            method=solve_args["method"],
            rannacher_steps=solve_args["rannacher_steps"],
            space_grid=solve_args["space_grid"],
        )
        return ko_result, van_result

    def _is_european_ki(self) -> bool:
        """True iff this spec is a European knock-in — the only case that needs
        native-surface parity (``V_KI = V_vanilla − V_KO + rebate leg``) rather
        than the mixin's direct grid extraction."""
        spec = self._spec
        return spec.action is BarrierAction.IN and spec.exercise_type is ExerciseType.EUROPEAN

    # ------------------------------------------------------------------ #
    # Greeks                                                            #
    # ------------------------------------------------------------------ #
    def delta(self) -> float:
        if self.valuation_ctx._barrier_triggered_at_inception():
            # OUT triggered → dead (delta 0); IN triggered → vanilla.
            if self._spec.action is BarrierAction.OUT:
                return 0.0
            return self.valuation_ctx._vanilla_equivalent_valuation().delta()

        if self._is_european_ki():
            ko_result, van_result = self._ki_components()
            spot = float(self.underlying.initial_value)
            return self._grid_delta_from_result(van_result, spot) - self._grid_delta_from_result(
                ko_result, spot
            )

        return super().delta()

    def gamma(self) -> float:
        """Return grid gamma, using native-surface parity for European KI barriers."""
        if self.valuation_ctx._barrier_triggered_at_inception():
            if self._spec.action is BarrierAction.OUT:
                return 0.0
            return self.valuation_ctx._vanilla_equivalent_valuation().gamma()

        if self._is_european_ki():
            ko_result, van_result = self._ki_components()
            spot = float(self.underlying.initial_value)
            return self._grid_gamma_from_result(van_result, spot) - self._grid_gamma_from_result(
                ko_result, spot
            )

        return super().gamma()

    def theta(self) -> float:
        if self.valuation_ctx._barrier_triggered_at_inception():
            # OUT triggered → only an AT_EXPIRY rebate accretes; IN → vanilla.
            if self._spec.action is BarrierAction.OUT:
                return self._resolved_knock_out_theta()
            return self.valuation_ctx._vanilla_equivalent_valuation().theta()

        if self._is_european_ki():
            ko_result, van_result = self._ki_components()
            spot = float(self.underlying.initial_value)
            ko_theta = self._grid_theta_from_result(ko_result, spot)
            vanilla_theta = self._grid_theta_from_result(van_result, spot)
            rebate_theta = self._discounted_rebate_theta(ko_result[-1])
            return vanilla_theta + rebate_theta - ko_theta

        return super().theta()

    # ------------------------------------------------------------------ #
    # Solve / PV                                                        #
    # ------------------------------------------------------------------ #
    def _solve(self) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
        """Memoised PDE solve result.

        The first call runs the backward solve (KO directly, American KI via
        the coupled two-surface solver, European KI via parity on the cached KO
        + vanilla components).  Every subsequent call on the same instance —
        including those triggered transparently by the grid-greek mixin's
        :meth:`delta`, :meth:`gamma`, :meth:`theta` — is an O(1) tuple lookup.
        """
        if self._solve_result is not None:
            return self._solve_result
        with self._solve_lock:
            if self._solve_result is not None:
                return self._solve_result
            self._solve_result = self._compute_solve()
            return self._solve_result

    def _compute_solve(self) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
        """Run the PDE solve, handling KI via parity or coupled PDE."""
        spec = self._spec

        if spec.action is BarrierAction.OUT:
            return self._ko_core(**self._base_solve_args())

        # American knock-in: two-surface coupled PDE (no parity, since
        # American KI ≠ vanilla − American KO).
        if spec.exercise_type is ExerciseType.AMERICAN:
            return self._ki_core(**self._base_solve_args())

        # European knock-in via parity: V_KI = V_vanilla + R·df_T − V_KO.
        # When R=0 this reduces to V_vanilla − V_KO.
        ko_result, van_result = self._ki_components()
        ko_price, S_ko, V_ko, V_ko_prev, last_dtau_ko = ko_result
        van_price, S_van, V_van, V_van_prev, _ = van_result

        ttm = self.valuation_ctx._maturity_year_fraction()
        discount_curve = self.valuation_ctx.discount_curve
        df_T = float(discount_curve.df(ttm))
        ki_price = van_price + float(spec.rebate) * df_T - ko_price

        if last_dtau_ko > 0.0:
            rebate_prev = float(spec.rebate) * float(
                discount_curve.df(max(ttm - last_dtau_ko, 0.0))
            )
        else:
            rebate_prev = float(spec.rebate) * df_T

        # Reconstruct the KI surface on the KO grid (with rebate PV)
        # so the grid-greek mixin can extract delta/gamma/theta at spot.
        V_ki = np.interp(S_ko, S_van, V_van) + float(spec.rebate) * df_T - V_ko
        V_ki_prev = np.interp(S_ko, S_van, V_van_prev) + rebate_prev - V_ko_prev
        return ki_price, S_ko, V_ki, V_ki_prev, last_dtau_ko

    def solve(self) -> tuple[float, np.ndarray, np.ndarray]:
        """Compute the full FD solution."""
        pv, S, V, *_ = self._solve()
        return pv, S, V

    def present_value(self) -> float:
        """Return present value from the PDE barrier solve."""
        if self.valuation_ctx._barrier_triggered_at_inception():
            if self._spec.action is BarrierAction.OUT:
                triggered_value = self._resolved_knock_out_value()
                if triggered_value is None:
                    raise ConfigurationError("Resolved knock-out state unexpectedly unavailable")
                return triggered_value
            return self.valuation_ctx._vanilla_equivalent_valuation().present_value()
        spec = self._spec
        exercise = "American" if spec.exercise_type is ExerciseType.AMERICAN else "European"
        label = f"PDE {self._engine_label} {exercise}"
        with log_timing(logger, f"{label} present_value", self.pde_params.log_timings):
            pv, *_ = self._solve()
        return float(pv)


class _FDBarrierValuation(_FDBarrierValuationBase):
    """PDE finite-difference valuation for single-barrier options.

    Supports:
    - Continuous and discrete knock-out (European and American)
    - Continuous and discrete knock-in via in-out parity (European only)
    - American knock-in via two-surface coupled PDE
    - Rebates (at-hit and at-expiry)
    """

    _spec: BarrierSpec  # narrows the base annotation for barrier-specific attrs
    _engine_label = "barrier"

    def _ko_core(self, **kwargs) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
        return _fd_barrier_ko_core(**kwargs)

    def _ki_core(self, **kwargs) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
        return _fd_barrier_ki_core(**kwargs)

    def _barrier_solve_args(self) -> dict:
        return dict(barrier=float(self._spec.barrier), direction=self._spec.direction)


class _FDDoubleBarrierValuation(_FDBarrierValuationBase):
    """PDE finite-difference valuation for double-barrier options.

    - **Double knock-out** (European or American): truncate the log-spot grid
      at both barriers so each is a Dirichlet boundary (Boyle-Tian).
    - **European double knock-in**: in-out parity,
      ``V_DKI = V_vanilla + R·df_T − V_DKO`` (with ``R=0`` this is just
      ``V_vanilla − V_DKO``).  The reconstructed KI surface is returned so
      :class:`_FDGridGreeksMixin` can extract grid greeks directly.

    **Discrete monitoring** is supported across the board: knock-out
    (European/American), European knock-in (via parity), and American knock-in
    (the two-surface coupled solver).  Both barriers are placed midway between
    grid nodes (Boyle-Tian half-step) and the knock-out reset / knock-in
    coupling is imposed at the monitoring dates.
    """

    _spec: DoubleBarrierSpec  # narrows the base annotation for barrier-specific attrs
    _engine_label = "double-barrier"

    def _ko_core(self, **kwargs) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
        return _fd_double_barrier_ko_core(**kwargs)

    def _ki_core(self, **kwargs) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
        return _fd_double_barrier_ki_core(**kwargs)

    def _barrier_solve_args(self) -> dict:
        return dict(
            lower_barrier=float(self._spec.lower_barrier),
            upper_barrier=float(self._spec.upper_barrier),
        )
