"""Tests for the two layers of PV/greek result caching.

Phase 1 — ``OptionValuation`` output cache.  Repeated accessor calls with
identical kwargs reuse the previously returned value without touching the
engine impl.

Phase 2 — engine-impl solve cache.  Grid (PDE_FD) and tree (Binomial)
engines memoise the expensive backward solve on the instance so that the
first accessor call (PV or any native greek) pays for the solve and every
subsequent grid-/tree-native greek is an O(1) lookup.
"""

from __future__ import annotations

import datetime as dt
from unittest.mock import patch

from derivatives_pricing.enums import (
    BarrierAction,
    BarrierDirection,
    BarrierMonitoring,
    DayCountConvention,
    ExerciseType,
    OptionType,
    PricingMethod,
)
from derivatives_pricing.market_environment import MarketData
from derivatives_pricing.rates import DiscountCurve
from derivatives_pricing.valuation import (
    BarrierSpec,
    OptionValuation,
    UnderlyingData,
    VanillaSpec,
)


PRICING_DATE = dt.datetime(2025, 1, 1)
MATURITY = dt.datetime(2026, 1, 1)


def _underlying() -> UnderlyingData:
    md = MarketData(
        PRICING_DATE,
        DiscountCurve.flat(0.05, 2.0),
        currency="USD",
        day_count_convention=DayCountConvention.ACT_365F,
    )
    return UnderlyingData(initial_value=100.0, volatility=0.20, market_data=md)


def _am_doc_spec() -> BarrierSpec:
    return BarrierSpec(
        option_type=OptionType.CALL,
        exercise_type=ExerciseType.AMERICAN,
        strike=100.0,
        maturity=MATURITY,
        barrier=85.0,
        direction=BarrierDirection.DOWN,
        action=BarrierAction.OUT,
        monitoring=BarrierMonitoring.CONTINUOUS,
    )


def _eu_dic_spec() -> BarrierSpec:
    return BarrierSpec(
        option_type=OptionType.CALL,
        exercise_type=ExerciseType.EUROPEAN,
        strike=100.0,
        maturity=MATURITY,
        barrier=85.0,
        direction=BarrierDirection.DOWN,
        action=BarrierAction.IN,
        monitoring=BarrierMonitoring.CONTINUOUS,
    )


def _call_all_native_greeks(ov: OptionValuation) -> None:
    """Touch every grid/tree-native accessor exactly once."""
    ov.present_value()
    ov.delta()
    ov.gamma()
    ov.theta()


class TestOptionValuationOutputCache:
    """Phase 1 — OV-level output cache (keyed on accessor name + kwargs)."""

    def test_repeated_present_value_call_skips_impl(self):
        """Second ``present_value()`` hits the OV cache without touching the impl."""
        ov = OptionValuation(
            _underlying(),
            VanillaSpec(
                option_type=OptionType.CALL,
                exercise_type=ExerciseType.EUROPEAN,
                strike=100.0,
                maturity=MATURITY,
            ),
            PricingMethod.BSM,
        )
        ov.present_value()  # warm cache
        with patch.object(ov._impl, "present_value", wraps=ov._impl.present_value) as impl_pv:
            ov.present_value()
            ov.present_value()
        assert impl_pv.call_count == 0

    def test_distinct_kwargs_do_not_share_cache(self):
        """``delta(epsilon=a)`` and ``delta(epsilon=b)`` cache separately."""
        ov = OptionValuation(
            _underlying(),
            VanillaSpec(
                option_type=OptionType.CALL,
                exercise_type=ExerciseType.EUROPEAN,
                strike=100.0,
                maturity=MATURITY,
            ),
            PricingMethod.BSM,
        )
        d1 = ov.delta(epsilon=0.5)
        d2 = ov.delta(epsilon=1.0)
        assert d1 == ov.delta(epsilon=0.5)  # cache hit
        assert d2 == ov.delta(epsilon=1.0)  # cache hit
        assert ("delta", ("epsilon", 0.5)) in ov._cache
        assert ("delta", ("epsilon", 1.0)) in ov._cache


class TestEngineLevelCaching:
    """Phase 2 — engine-impl solve cache shared across PV + native greeks."""

    def test_fd_barrier_ko_solves_once_across_pv_and_greeks(self):
        """PDE_FD KO: ``_compute_solve`` fires once for PV + delta + gamma + theta."""
        ov = OptionValuation(_underlying(), _am_doc_spec(), PricingMethod.PDE_FD)
        with patch.object(
            ov._impl, "_compute_solve", wraps=ov._impl._compute_solve
        ) as compute_solve:
            _call_all_native_greeks(ov)
        assert compute_solve.call_count == 1

    def test_fd_barrier_eu_ki_components_solved_once_across_greeks(self):
        """European KI parity: ``_compute_european_ki_components`` fires once."""
        ov = OptionValuation(_underlying(), _eu_dic_spec(), PricingMethod.PDE_FD)
        with patch.object(
            ov._impl,
            "_compute_european_ki_components",
            wraps=ov._impl._compute_european_ki_components,
        ) as compute_components:
            _call_all_native_greeks(ov)
        assert compute_components.call_count == 1

    def test_binomial_american_backward_solves_once_across_pv_and_greeks(self):
        """Binomial American vanilla: ``_compute_solve_backward`` fires once."""
        ov = OptionValuation(
            _underlying(),
            VanillaSpec(
                option_type=OptionType.PUT,
                exercise_type=ExerciseType.AMERICAN,
                strike=100.0,
                maturity=MATURITY,
            ),
            PricingMethod.BINOMIAL,
        )
        with patch.object(
            ov._impl,
            "_compute_solve_backward",
            wraps=ov._impl._compute_solve_backward,
        ) as compute_backward:
            _call_all_native_greeks(ov)
        assert compute_backward.call_count == 1

    def test_binomial_barrier_ko_solves_once_across_pv_and_greeks(self):
        """Binomial barrier KO: ``_compute_knock_out`` fires once for PV + 3 greeks."""
        ov = OptionValuation(_underlying(), _am_doc_spec(), PricingMethod.BINOMIAL)
        with patch.object(
            ov._impl, "_compute_knock_out", wraps=ov._impl._compute_knock_out
        ) as compute_ko:
            _call_all_native_greeks(ov)
        assert compute_ko.call_count == 1
