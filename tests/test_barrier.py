"""Tests for barrier option pricing.

Covers: BarrierSpec validation, in/out parity, analytical pricing,
initial-state handling, rebate pricing, discrete monitoring (BG correction),
and Greeks (NUMERICAL only).
"""

import datetime as dt

import numpy as np
import pytest

from derivatives_pricing.enums import (
    BarrierAction,
    BarrierDirection,
    BarrierMonitoring,
    ExerciseType,
    GreekCalculationMethod,
    OptionType,
    PricingMethod,
    RebateTiming,
)
from derivatives_pricing.exceptions import (
    ConfigurationError,
    UnsupportedFeatureError,
    ValidationError,
)
from derivatives_pricing.market_environment import MarketData
from derivatives_pricing.rates import DiscountCurve
from derivatives_pricing.valuation import (
    BarrierSpec,
    OptionValuation,
    UnderlyingData,
    VanillaSpec,
)
from derivatives_pricing.valuation.params import BinomialParams

from helpers import flat_curve, PRICING_DATE, MATURITY, CURRENCY, SPOT, STRIKE, RATE, VOL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _market_data(
    pricing_date: dt.datetime = PRICING_DATE,
    rate: float = RATE,
    maturity: dt.datetime = MATURITY,
) -> MarketData:
    curve = flat_curve(pricing_date, maturity, rate)
    return MarketData(pricing_date, curve, currency=CURRENCY)


def _underlying(
    spot: float = SPOT,
    vol: float = VOL,
    market_data: MarketData | None = None,
    dividend_curve: DiscountCurve | None = None,
) -> UnderlyingData:
    if market_data is None:
        market_data = _market_data()
    return UnderlyingData(
        initial_value=spot,
        volatility=vol,
        market_data=market_data,
        dividend_curve=dividend_curve,
    )


def _barrier_spec(
    option_type: OptionType = OptionType.CALL,
    exercise_type: ExerciseType = ExerciseType.EUROPEAN,
    strike: float = STRIKE,
    maturity: dt.datetime = MATURITY,
    barrier: float = 120.0,
    direction: BarrierDirection = BarrierDirection.UP,
    action: BarrierAction = BarrierAction.OUT,
    monitoring: BarrierMonitoring = BarrierMonitoring.CONTINUOUS,
    rebate: float = 0.0,
    rebate_timing: RebateTiming = RebateTiming.AT_HIT,
    num_observations: int | None = None,
    monitoring_dates=None,
) -> BarrierSpec:
    return BarrierSpec(
        option_type=option_type,
        exercise_type=exercise_type,
        strike=strike,
        maturity=maturity,
        barrier=barrier,
        direction=direction,
        action=action,
        monitoring=monitoring,
        rebate=rebate,
        rebate_timing=rebate_timing,
        num_observations=num_observations,
        monitoring_dates=monitoring_dates,
    )


def _price(
    underlying: UnderlyingData | None = None,
    spec: BarrierSpec | None = None,
    **spec_kw,
) -> float:
    if underlying is None:
        underlying = _underlying()
    if spec is None:
        spec = _barrier_spec(**spec_kw)
    return OptionValuation(underlying, spec, PricingMethod.BSM).present_value()


# ===========================================================================
# BarrierSpec validation
# ===========================================================================


class TestBarrierSpecValidation:
    """Test BarrierSpec __post_init__ validation."""

    def test_valid_construction(self):
        spec = _barrier_spec()
        assert spec.option_type is OptionType.CALL
        assert spec.direction is BarrierDirection.UP
        assert spec.action is BarrierAction.OUT
        assert spec.monitoring is BarrierMonitoring.CONTINUOUS
        assert spec.rebate == 0.0
        assert spec.strike == float(STRIKE)
        assert spec.barrier == 120.0

    def test_strike_coerced_to_float(self):
        spec = _barrier_spec(strike=100)
        assert isinstance(spec.strike, float)

    def test_barrier_coerced_to_float(self):
        spec = _barrier_spec(barrier=120)
        assert isinstance(spec.barrier, float)

    def test_invalid_option_type(self):
        with pytest.raises(ConfigurationError, match="option_type"):
            BarrierSpec(
                option_type="call",  # type: ignore
                exercise_type=ExerciseType.EUROPEAN,
                strike=100,
                maturity=MATURITY,
                barrier=120,
                direction=BarrierDirection.UP,
                action=BarrierAction.OUT,
            )

    def test_invalid_direction(self):
        with pytest.raises(ConfigurationError, match="direction"):
            BarrierSpec(
                option_type=OptionType.CALL,
                exercise_type=ExerciseType.EUROPEAN,
                strike=100,
                maturity=MATURITY,
                barrier=120,
                direction="up",  # type: ignore
                action=BarrierAction.OUT,
            )

    def test_invalid_action(self):
        with pytest.raises(ConfigurationError, match="action"):
            BarrierSpec(
                option_type=OptionType.CALL,
                exercise_type=ExerciseType.EUROPEAN,
                strike=100,
                maturity=MATURITY,
                barrier=120,
                direction=BarrierDirection.UP,
                action="out",  # type: ignore
            )

    def test_negative_strike(self):
        with pytest.raises(ValidationError, match="strike.*>= 0"):
            _barrier_spec(strike=-1.0)

    def test_non_finite_strike(self):
        with pytest.raises(ValidationError, match="strike.*finite"):
            _barrier_spec(strike=float("inf"))

    def test_barrier_must_be_positive(self):
        with pytest.raises(ValidationError, match="barrier.*> 0"):
            _barrier_spec(barrier=0.0)

    def test_negative_barrier(self):
        with pytest.raises(ValidationError, match="barrier.*> 0"):
            _barrier_spec(barrier=-10.0)

    def test_negative_rebate(self):
        with pytest.raises(ValidationError, match="rebate.*>= 0"):
            _barrier_spec(rebate=-1.0)

    def test_knock_in_rebate_at_hit_rejected(self):
        with pytest.raises(ValidationError, match="Knock-in rebate"):
            _barrier_spec(
                action=BarrierAction.IN,
                rebate=5.0,
                rebate_timing=RebateTiming.AT_HIT,
            )

    def test_knock_in_rebate_at_expiry_ok(self):
        spec = _barrier_spec(
            action=BarrierAction.IN,
            rebate=5.0,
            rebate_timing=RebateTiming.AT_EXPIRY,
        )
        assert spec.rebate == 5.0

    def test_continuous_rejects_num_observations(self):
        with pytest.raises(ValidationError, match="CONTINUOUS"):
            _barrier_spec(
                monitoring=BarrierMonitoring.CONTINUOUS,
                num_observations=50,
            )

    def test_continuous_rejects_monitoring_dates(self):
        dates = [PRICING_DATE + dt.timedelta(days=i * 30) for i in range(1, 5)]
        with pytest.raises(ValidationError, match="CONTINUOUS"):
            _barrier_spec(
                monitoring=BarrierMonitoring.CONTINUOUS,
                monitoring_dates=dates,
            )

    def test_discrete_requires_schedule_source(self):
        with pytest.raises(ValidationError, match="exactly one"):
            _barrier_spec(monitoring=BarrierMonitoring.DISCRETE)

    def test_discrete_rejects_both_sources(self):
        dates = [PRICING_DATE + dt.timedelta(days=i * 30) for i in range(1, 5)]
        with pytest.raises(ValidationError, match="exactly one"):
            _barrier_spec(
                monitoring=BarrierMonitoring.DISCRETE,
                num_observations=50,
                monitoring_dates=dates,
            )

    def test_discrete_num_observations_valid(self):
        spec = _barrier_spec(
            monitoring=BarrierMonitoring.DISCRETE,
            num_observations=50,
        )
        assert spec.num_observations == 50

    def test_discrete_num_observations_too_small(self):
        with pytest.raises(ValidationError, match="num_observations"):
            _barrier_spec(
                monitoring=BarrierMonitoring.DISCRETE,
                num_observations=1,
            )

    def test_discrete_monitoring_dates_valid(self):
        dates = [PRICING_DATE + dt.timedelta(days=i * 30) for i in range(1, 10)]
        spec = _barrier_spec(
            monitoring=BarrierMonitoring.DISCRETE,
            monitoring_dates=dates,
        )
        assert len(spec.monitoring_dates) == 9

    def test_discrete_monitoring_dates_beyond_maturity(self):
        dates = [MATURITY + dt.timedelta(days=10)]
        with pytest.raises(ValidationError, match="beyond maturity"):
            _barrier_spec(
                monitoring=BarrierMonitoring.DISCRETE,
                monitoring_dates=dates,
            )

    def test_discrete_monitoring_dates_not_ascending(self):
        dates = [
            PRICING_DATE + dt.timedelta(days=60),
            PRICING_DATE + dt.timedelta(days=30),
        ]
        with pytest.raises(ValidationError, match="ascending"):
            _barrier_spec(
                monitoring=BarrierMonitoring.DISCRETE,
                monitoring_dates=dates,
            )


# ===========================================================================
# In/Out parity: knock_in + knock_out == vanilla
# ===========================================================================


class TestBarrierInOutParity:
    """The strongest correctness check: barrier_in + barrier_out == vanilla."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.underlying = _underlying()

    @pytest.mark.parametrize(
        "option_type,direction,barrier",
        [
            # Down barriers
            (OptionType.CALL, BarrierDirection.DOWN, 80.0),
            (OptionType.CALL, BarrierDirection.DOWN, 110.0),
            (OptionType.PUT, BarrierDirection.DOWN, 80.0),
            (OptionType.PUT, BarrierDirection.DOWN, 110.0),
            # Up barriers
            (OptionType.CALL, BarrierDirection.UP, 90.0),
            (OptionType.CALL, BarrierDirection.UP, 120.0),
            (OptionType.PUT, BarrierDirection.UP, 90.0),
            (OptionType.PUT, BarrierDirection.UP, 120.0),
        ],
    )
    def test_in_out_parity(self, option_type, direction, barrier):
        """knock_in + knock_out == vanilla for all barrier type combinations."""
        common = dict(
            option_type=option_type,
            direction=direction,
            barrier=barrier,
            strike=STRIKE,
            maturity=MATURITY,
        )
        pv_in = _price(self.underlying, _barrier_spec(**common, action=BarrierAction.IN))
        pv_out = _price(self.underlying, _barrier_spec(**common, action=BarrierAction.OUT))

        vanilla_spec = VanillaSpec(
            option_type=option_type,
            exercise_type=ExerciseType.EUROPEAN,
            strike=STRIKE,
            maturity=MATURITY,
        )
        pv_vanilla = OptionValuation(
            self.underlying, vanilla_spec, PricingMethod.BSM
        ).present_value()

        assert np.isclose(pv_in + pv_out, pv_vanilla, rtol=1e-10), (
            f"In/out parity violated: in={pv_in:.10f} + out={pv_out:.10f} "
            f"= {pv_in + pv_out:.10f} vs vanilla={pv_vanilla:.10f}"
        )

    @pytest.mark.parametrize("vol", [0.10, 0.30, 0.50])
    def test_in_out_parity_various_vols(self, vol):
        """Parity holds across different volatility levels."""
        u = _underlying(vol=vol)
        barrier = 120.0

        pv_in = _price(
            u,
            _barrier_spec(direction=BarrierDirection.UP, action=BarrierAction.IN, barrier=barrier),
        )
        pv_out = _price(
            u,
            _barrier_spec(direction=BarrierDirection.UP, action=BarrierAction.OUT, barrier=barrier),
        )

        vanilla_spec = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=STRIKE,
            maturity=MATURITY,
        )
        pv_vanilla = OptionValuation(u, vanilla_spec, PricingMethod.BSM).present_value()

        assert np.isclose(pv_in + pv_out, pv_vanilla, rtol=1e-10)


# ===========================================================================
# Analytical pricing — known / directional checks
# ===========================================================================


class TestBarrierAnalyticalPricing:
    """Test barrier option values for directional correctness and known properties."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.underlying = _underlying()

    def test_down_and_out_call_barrier_below_strike(self):
        """DOC with H < K: standard result, positive price < vanilla."""
        barrier = 80.0
        pv = _price(
            self.underlying,
            direction=BarrierDirection.DOWN,
            action=BarrierAction.OUT,
            barrier=barrier,
        )
        vanilla_spec = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=STRIKE,
            maturity=MATURITY,
        )
        pv_vanilla = OptionValuation(
            self.underlying, vanilla_spec, PricingMethod.BSM
        ).present_value()

        assert pv > 0.0
        assert pv < pv_vanilla

    def test_up_and_out_call_worthless_when_barrier_below_strike(self):
        """UOC with H <= K: always worthless (barrier kills before payoff triggers)."""
        pv = _price(
            self.underlying,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=90.0,  # H < K=100
        )
        assert pv == 0.0

    def test_down_and_out_put_worthless_when_barrier_above_strike(self):
        """DOP with H >= K: always worthless."""
        pv = _price(
            self.underlying,
            option_type=OptionType.PUT,
            direction=BarrierDirection.DOWN,
            action=BarrierAction.OUT,
            barrier=110.0,  # H > K=100
        )
        assert pv == 0.0

    def test_up_and_in_call_equals_vanilla_when_barrier_below_strike(self):
        """UIC with H <= K: equals vanilla (barrier always hit if option has value)."""
        pv = _price(
            self.underlying,
            direction=BarrierDirection.UP,
            action=BarrierAction.IN,
            barrier=90.0,  # H < K=100
        )
        vanilla_spec = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=STRIKE,
            maturity=MATURITY,
        )
        pv_vanilla = OptionValuation(
            self.underlying, vanilla_spec, PricingMethod.BSM
        ).present_value()

        assert np.isclose(pv, pv_vanilla, rtol=1e-10)

    def test_barrier_price_positive(self):
        """Down-and-in put with H < K produces a positive price."""
        pv = _price(
            self.underlying,
            option_type=OptionType.PUT,
            direction=BarrierDirection.DOWN,
            action=BarrierAction.IN,
            barrier=80.0,
        )
        assert pv > 0.0

    def test_barrier_price_increases_with_volatility(self):
        """Higher vol → higher knock-in option prices (more likely to hit barrier)."""
        u_lo = _underlying(vol=0.15)
        u_hi = _underlying(vol=0.35)

        pv_lo = _price(
            u_lo,
            direction=BarrierDirection.DOWN,
            action=BarrierAction.IN,
            barrier=80.0,
            option_type=OptionType.PUT,
        )
        pv_hi = _price(
            u_hi,
            direction=BarrierDirection.DOWN,
            action=BarrierAction.IN,
            barrier=80.0,
            option_type=OptionType.PUT,
        )
        assert pv_hi > pv_lo

    def test_nonflat_dividend_curve(self):
        """Barrier pricing works with a non-flat dividend curve."""
        from derivatives_pricing.utils import calculate_year_fraction

        T = calculate_year_fraction(PRICING_DATE, MATURITY)
        div_curve = DiscountCurve.flat(0.02, end_time=T)
        u = _underlying(dividend_curve=div_curve)

        pv_in = _price(
            u,
            direction=BarrierDirection.UP,
            action=BarrierAction.IN,
            barrier=120.0,
        )
        pv_out = _price(
            u,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
        )

        vanilla_spec = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=STRIKE,
            maturity=MATURITY,
        )
        pv_vanilla = OptionValuation(u, vanilla_spec, PricingMethod.BSM).present_value()

        # In/out parity should hold with non-flat curves too
        assert np.isclose(pv_in + pv_out, pv_vanilla, rtol=1e-10)


# ===========================================================================
# Initial barrier state (barrier breached at inception)
# ===========================================================================


class TestBarrierInitialState:
    """Test behavior when the barrier is already triggered at time zero."""

    def test_knock_out_up_triggered_returns_zero_no_rebate(self):
        """UOC with S >= H → knocked out at inception, PV = 0."""
        pv = _price(
            _underlying(spot=120.0),
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
        )
        assert pv == 0.0

    def test_knock_out_down_triggered_returns_zero_no_rebate(self):
        """DOP with S <= H → knocked out at inception, PV = 0."""
        pv = _price(
            _underlying(spot=80.0),
            option_type=OptionType.PUT,
            direction=BarrierDirection.DOWN,
            action=BarrierAction.OUT,
            barrier=80.0,
        )
        assert pv == 0.0

    def test_knock_out_triggered_rebate_at_hit(self):
        """Knocked out at inception with rebate AT_HIT → PV = rebate (immediate)."""
        pv = _price(
            _underlying(spot=120.0),
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
            rebate=5.0,
            rebate_timing=RebateTiming.AT_HIT,
        )
        assert pv == 5.0

    def test_knock_out_triggered_rebate_at_expiry(self):
        """Knocked out at inception with rebate AT_EXPIRY → PV = rebate × df."""
        from derivatives_pricing.utils import calculate_year_fraction

        T = calculate_year_fraction(PRICING_DATE, MATURITY)
        df_r = float(flat_curve(PRICING_DATE, MATURITY, RATE).df(T))

        pv = _price(
            _underlying(spot=120.0),
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
            rebate=5.0,
            rebate_timing=RebateTiming.AT_EXPIRY,
        )
        assert np.isclose(pv, 5.0 * df_r, rtol=1e-10)

    def test_knock_in_triggered_equals_vanilla(self):
        """Knock-in already triggered at inception → price as vanilla."""
        u = _underlying(spot=120.0)
        pv = _price(
            u,
            direction=BarrierDirection.UP,
            action=BarrierAction.IN,
            barrier=120.0,
        )
        vanilla_spec = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=STRIKE,
            maturity=MATURITY,
        )
        pv_vanilla = OptionValuation(u, vanilla_spec, PricingMethod.BSM).present_value()
        assert np.isclose(pv, pv_vanilla, rtol=1e-10)


# ===========================================================================
# Rebate pricing
# ===========================================================================


class TestBarrierRebate:
    """Test rebate legs for barrier options."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.underlying = _underlying()

    def test_zero_rebate_has_no_effect(self):
        """Rebate = 0 should not change the price."""
        pv_no_rebate = _price(
            self.underlying,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
            rebate=0.0,
        )
        pv_with_rebate = _price(
            self.underlying,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
            rebate=0.0,
        )
        assert pv_no_rebate == pv_with_rebate

    def test_knock_out_at_hit_rebate_increases_price(self):
        """Positive rebate should increase knock-out option price."""
        pv_no_rebate = _price(
            self.underlying,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
            rebate=0.0,
        )
        pv_with_rebate = _price(
            self.underlying,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
            rebate=5.0,
            rebate_timing=RebateTiming.AT_HIT,
        )
        assert pv_with_rebate > pv_no_rebate

    def test_knock_out_at_expiry_rebate_increases_price(self):
        """AT_EXPIRY rebate also increases price."""
        pv_no_rebate = _price(
            self.underlying,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
            rebate=0.0,
        )
        pv_with_rebate = _price(
            self.underlying,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
            rebate=5.0,
            rebate_timing=RebateTiming.AT_EXPIRY,
        )
        assert pv_with_rebate > pv_no_rebate

    def test_knock_in_rebate_at_expiry_increases_price(self):
        """Knock-in rebate (paid at expiry if never hit) increases price."""
        pv_no_rebate = _price(
            self.underlying,
            direction=BarrierDirection.DOWN,
            action=BarrierAction.IN,
            barrier=80.0,
            rebate=0.0,
            rebate_timing=RebateTiming.AT_EXPIRY,
        )
        pv_with_rebate = _price(
            self.underlying,
            direction=BarrierDirection.DOWN,
            action=BarrierAction.IN,
            barrier=80.0,
            rebate=5.0,
            rebate_timing=RebateTiming.AT_EXPIRY,
        )
        assert pv_with_rebate > pv_no_rebate

    def test_at_hit_and_at_expiry_rebate_differ(self):
        """AT_HIT and AT_EXPIRY rebate legs produce different PVs."""
        pv_at_hit = _price(
            self.underlying,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
            rebate=10.0,
            rebate_timing=RebateTiming.AT_HIT,
        )
        pv_at_expiry = _price(
            self.underlying,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
            rebate=10.0,
            rebate_timing=RebateTiming.AT_EXPIRY,
        )
        assert pv_at_hit != pv_at_expiry
        # Both should exceed the no-rebate price
        pv_base = _price(
            self.underlying,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
            rebate=0.0,
        )
        assert pv_at_hit > pv_base
        assert pv_at_expiry > pv_base


# ===========================================================================
# Discrete monitoring (Broadie-Glasserman)
# ===========================================================================


class TestBarrierDiscreteMonitoring:
    """Test Broadie-Glasserman-Kou continuity correction."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.underlying = _underlying()

    def test_discrete_converges_to_continuous(self):
        """As m → ∞, discrete price → continuous price."""
        continuous_pv = _price(
            self.underlying,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
            monitoring=BarrierMonitoring.CONTINUOUS,
        )

        # Many observations should be close to continuous
        discrete_pv = _price(
            self.underlying,
            _barrier_spec(
                direction=BarrierDirection.UP,
                action=BarrierAction.OUT,
                barrier=120.0,
                monitoring=BarrierMonitoring.DISCRETE,
                num_observations=50000,
            ),
        )
        assert np.isclose(discrete_pv, continuous_pv, rtol=0.02)

    def test_bg_adjustment_direction_up(self):
        """UP barrier: BG shifts barrier up → knock-out price increases
        (harder to knock out with higher barrier)."""
        continuous_pv = _price(
            self.underlying,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
            monitoring=BarrierMonitoring.CONTINUOUS,
        )
        discrete_pv = _price(
            self.underlying,
            _barrier_spec(
                direction=BarrierDirection.UP,
                action=BarrierAction.OUT,
                barrier=120.0,
                monitoring=BarrierMonitoring.DISCRETE,
                num_observations=12,
            ),
        )
        # Discrete monitoring with few observations makes knock-out worth more
        # (BG shifts barrier up — harder to knock out)
        assert discrete_pv > continuous_pv

    def test_bg_adjustment_direction_down(self):
        """DOWN barrier: BG shifts barrier down → knock-out price increases."""
        continuous_pv = _price(
            self.underlying,
            option_type=OptionType.PUT,
            direction=BarrierDirection.DOWN,
            action=BarrierAction.OUT,
            barrier=80.0,
            monitoring=BarrierMonitoring.CONTINUOUS,
        )
        discrete_pv = _price(
            self.underlying,
            _barrier_spec(
                option_type=OptionType.PUT,
                direction=BarrierDirection.DOWN,
                action=BarrierAction.OUT,
                barrier=80.0,
                monitoring=BarrierMonitoring.DISCRETE,
                num_observations=12,
            ),
        )
        assert discrete_pv > continuous_pv

    def test_discrete_monitoring_dates_raises_for_analytical(self):
        """Analytical engine rejects monitoring_dates (only num_observations)."""
        dates = [PRICING_DATE + dt.timedelta(days=i * 30) for i in range(1, 10)]
        spec = _barrier_spec(
            monitoring=BarrierMonitoring.DISCRETE,
            monitoring_dates=dates,
        )
        with pytest.raises(UnsupportedFeatureError, match="num_observations"):
            OptionValuation(_underlying(), spec, PricingMethod.BSM).present_value()

    def test_in_out_parity_discrete(self):
        """In/out parity holds under BG-adjusted discrete monitoring."""
        u = self.underlying
        common = dict(
            barrier=120.0,
            direction=BarrierDirection.UP,
            monitoring=BarrierMonitoring.DISCRETE,
            num_observations=50,
        )

        pv_in = _price(u, _barrier_spec(**common, action=BarrierAction.IN))
        pv_out = _price(u, _barrier_spec(**common, action=BarrierAction.OUT))

        vanilla_spec = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=STRIKE,
            maturity=MATURITY,
        )
        pv_vanilla = OptionValuation(u, vanilla_spec, PricingMethod.BSM).present_value()

        assert np.isclose(pv_in + pv_out, pv_vanilla, rtol=1e-10)


# ===========================================================================
# Greeks (NUMERICAL only)
# ===========================================================================


class TestBarrierGreeks:
    """Test Greek computation on barrier options — NUMERICAL only."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.underlying = _underlying()
        self.spec = _barrier_spec(
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
        )
        self.val = OptionValuation(self.underlying, self.spec, PricingMethod.BSM)

    def test_delta_numerical(self):
        delta = self.val.delta()
        # Up-and-out call delta can be negative (higher spot → more likely to knock out)
        assert np.isfinite(delta)

    def test_gamma_numerical(self):
        gamma = self.val.gamma()
        assert np.isfinite(gamma)

    def test_vega_numerical(self):
        vega = self.val.vega()
        assert np.isfinite(vega)

    def test_theta_numerical(self):
        theta = self.val.theta()
        assert np.isfinite(theta)

    def test_rho_numerical(self):
        rho = self.val.rho()
        assert np.isfinite(rho)

    def test_analytical_greek_method_rejected(self):
        with pytest.raises(UnsupportedFeatureError, match="Barrier"):
            self.val.delta(greek_calc_method=GreekCalculationMethod.ANALYTICAL)

    def test_auto_selects_numerical(self):
        """Without explicit method, barrier Greeks auto-select NUMERICAL."""
        # This should not raise
        delta = self.val.delta()
        assert np.isfinite(delta)


# ===========================================================================
# Binomial barrier NUMERICAL guard (Boyle-Lau retopology)
# ===========================================================================
# Bumping spot/vol/T on a binomial barrier valuation re-runs Boyle-Lau
# alignment with new inputs, which can pick a different num_steps for each
# bumped tree. The resulting central difference compares two unrelated tree
# topologies, so we explicitly block NUMERICAL bump-and-revalue greeks on
# binomial barriers (rho excepted — the Boyle-Lau formula has no r term).


class TestBinomialBarrierNumericalGuard:
    """Regression tests for the binomial-barrier NUMERICAL guard policy."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.underlying = _underlying()
        self.spec = _barrier_spec(
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
        )
        self.val = OptionValuation(
            self.underlying,
            self.spec,
            PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=200),
        )

    @pytest.mark.parametrize("greek_name", ["delta", "gamma", "theta", "vega"])
    def test_explicit_numerical_blocked(self, greek_name):
        """Explicit NUMERICAL on delta/gamma/theta/vega raises."""
        fn = getattr(self.val, greek_name)
        with pytest.raises(UnsupportedFeatureError, match="NUMERICAL"):
            fn(greek_calc_method=GreekCalculationMethod.NUMERICAL)

    def test_explicit_numerical_rho_allowed(self):
        """Explicit NUMERICAL on rho is allowed (rate bumps don't enter
        the Boyle-Lau formula, so the bumped trees share the same topology)."""
        rho = self.val.rho(greek_calc_method=GreekCalculationMethod.NUMERICAL)
        assert np.isfinite(rho)

    def test_auto_select_vega_blocked(self):
        """vega() with no explicit method routes to NUMERICAL (no tree-
        native path) and the guard fires from the auto-select branch."""
        with pytest.raises(UnsupportedFeatureError, match="NUMERICAL"):
            self.val.vega()

    @pytest.mark.parametrize("greek_name", ["delta", "gamma", "theta"])
    def test_auto_select_picks_tree(self, greek_name):
        """delta/gamma/theta with no explicit method auto-select TREE on
        binomial, bypassing the NUMERICAL guard entirely."""
        fn = getattr(self.val, greek_name)
        value = fn()
        assert np.isfinite(value)

    def test_auto_select_rho_uses_numerical(self):
        """rho() has no tree-native path so it auto-selects NUMERICAL,
        but rho is exempt from the guard so this works."""
        rho = self.val.rho()
        assert np.isfinite(rho)


# ===========================================================================
# Unsupported feature errors
# ===========================================================================


class TestBarrierUnsupported:
    """Test that unsupported configurations raise appropriate errors."""

    def test_american_exercise_not_supported(self):
        spec = _barrier_spec(exercise_type=ExerciseType.AMERICAN)
        with pytest.raises(UnsupportedFeatureError, match="AMERICAN"):
            OptionValuation(_underlying(), spec, PricingMethod.BSM)

    def test_discrete_dividends_rejected(self):
        divs = [(PRICING_DATE + dt.timedelta(days=90), 2.0)]
        u = UnderlyingData(
            initial_value=SPOT,
            volatility=VOL,
            market_data=_market_data(),
            discrete_dividends=divs,
        )
        spec = _barrier_spec()
        with pytest.raises(UnsupportedFeatureError, match="discrete dividends"):
            OptionValuation(u, spec, PricingMethod.BSM).present_value()
