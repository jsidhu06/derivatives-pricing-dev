"""Tests for barrier option pricing.

Covers: BarrierSpec validation, in/out parity, analytical pricing,
initial-state handling, rebate pricing, discrete monitoring (BG correction),
and Greeks (NUMERICAL only).
"""

import datetime as dt
import warnings

import numpy as np
import pytest

from derivatives_pricing.enums import (
    BarrierAction,
    BarrierDirection,
    BarrierMonitoring,
    ExerciseType,
    GreekCalculationMethod,
    OptionType,
    PDESpaceGrid,
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
from derivatives_pricing.stochastic_processes import (
    GBMParams,
    GBMProcess,
    SimulationConfig,
)
from derivatives_pricing.valuation.params import BinomialParams, MonteCarloParams, PDEParams

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

    @pytest.mark.parametrize(
        "option_type,direction,barrier",
        [
            (OptionType.CALL, BarrierDirection.DOWN, 80.0),
            (OptionType.CALL, BarrierDirection.UP, 120.0),
            (OptionType.PUT, BarrierDirection.DOWN, 80.0),
            (OptionType.PUT, BarrierDirection.UP, 120.0),
        ],
    )
    @pytest.mark.parametrize(
        "pricing_method,params,rtol",
        [
            pytest.param(
                PricingMethod.PDE_FD,
                PDEParams(spot_steps=400, time_steps=400, space_grid=PDESpaceGrid.LOG_SPOT),
                2e-3,
                id="pde_fd",
            ),
            pytest.param(
                PricingMethod.BINOMIAL,
                BinomialParams(num_steps=600),
                5e-3,
                id="binomial",
            ),
        ],
    )
    def test_in_out_parity_numerical_engines(
        self, option_type, direction, barrier, pricing_method, params, rtol
    ):
        """knock_in + knock_out == vanilla on PDE_FD and Binomial engines.

        BSM parity holds exactly because of in/out complement formulas; the
        numerical engines have discretisation residual, so tolerances are
        looser than the analytical BSM parity test above.
        """
        common = dict(
            option_type=option_type,
            direction=direction,
            barrier=barrier,
            strike=STRIKE,
            maturity=MATURITY,
        )
        pv_in = OptionValuation(
            self.underlying,
            _barrier_spec(**common, action=BarrierAction.IN),
            pricing_method,
            params=params,
        ).present_value()
        pv_out = OptionValuation(
            self.underlying,
            _barrier_spec(**common, action=BarrierAction.OUT),
            pricing_method,
            params=params,
        ).present_value()

        vanilla_spec = VanillaSpec(
            option_type=option_type,
            exercise_type=ExerciseType.EUROPEAN,
            strike=STRIKE,
            maturity=MATURITY,
        )
        pv_vanilla = OptionValuation(
            self.underlying, vanilla_spec, pricing_method, params=params
        ).present_value()

        assert np.isclose(pv_in + pv_out, pv_vanilla, rtol=rtol, atol=1e-4), (
            f"In/out parity violated on {pricing_method.name}: "
            f"in={pv_in:.6f} + out={pv_out:.6f} = {pv_in + pv_out:.6f} "
            f"vs vanilla={pv_vanilla:.6f}"
        )


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
# Discrete monitoring — BG-corrected analytical vs pathwise MC cross-check
# ===========================================================================
# The BSM discrete-monitoring path applies the Broadie-Glasserman-Kou
# continuity correction to the continuous closed-form (shift the effective
# barrier by β·σ·√Δt). The MC path explicitly checks the barrier at each
# monitoring date on simulated paths — structurally unrelated to BG. If the
# BG formula is wrong, these two will disagree.


def _mc_gbm(
    spot: float = SPOT,
    vol: float = VOL,
    market_data: MarketData | None = None,
    dividend_curve: DiscountCurve | None = None,
    paths: int = 200_000,
    num_steps: int = 252,
) -> GBMProcess:
    if market_data is None:
        market_data = _market_data()
    return GBMProcess(
        market_data,
        GBMParams(initial_value=spot, volatility=vol, dividend_curve=dividend_curve),
        SimulationConfig(paths=paths, num_steps=num_steps, end_date=MATURITY),
    )


class TestBarrierDiscreteBGvsMC:
    """Cross-validate the BG continuity correction against pathwise MC."""

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "option_type,direction,action,barrier,num_observations",
        [
            pytest.param(
                OptionType.CALL,
                BarrierDirection.UP,
                BarrierAction.OUT,
                120.0,
                12,
                id="up_out_call_monthly",
            ),
            pytest.param(
                OptionType.PUT,
                BarrierDirection.DOWN,
                BarrierAction.OUT,
                80.0,
                12,
                id="down_out_put_monthly",
            ),
            pytest.param(
                OptionType.CALL,
                BarrierDirection.UP,
                BarrierAction.IN,
                120.0,
                24,
                id="up_in_call_biweekly",
            ),
        ],
    )
    def test_bg_analytical_matches_mc_pathwise(
        self, option_type, direction, action, barrier, num_observations
    ):
        """BSM (BG-corrected analytical) ≈ MC (pathwise checks) for
        discrete-monitoring barriers. The two implementations are
        structurally unrelated — agreement validates the BG formula."""
        spec = _barrier_spec(
            option_type=option_type,
            direction=direction,
            action=action,
            barrier=barrier,
            monitoring=BarrierMonitoring.DISCRETE,
            num_observations=num_observations,
        )
        pv_bsm = OptionValuation(_underlying(), spec, PricingMethod.BSM).present_value()

        gbm = _mc_gbm(num_steps=max(200, num_observations * 10))
        pv_mc = OptionValuation(
            gbm,
            spec,
            PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=42, log_timings=True),
        ).present_value()

        # MC noise + BG approximation residual. BG is O((Δt_obs)^(3/2))
        # accurate — for monthly monitoring with σ=0.20 the residual can
        # reach a few percent on its own, with MC noise on top. ~5% is a
        # meaningful bound: a broken BG formula would miss by >10%.
        assert np.isclose(pv_bsm, pv_mc, rtol=0.05, atol=5e-3), (
            f"BG vs MC mismatch: BSM={pv_bsm:.6f} MC={pv_mc:.6f}"
        )


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
# Binomial barrier coverage
# ===========================================================================


class TestBinomialBarrierCoverage:
    """Cover binomial barrier paths not exercised elsewhere."""

    def test_discrete_monitoring_dates_explicit(self):
        """Binomial barrier with explicit monitoring_dates (not num_observations)."""
        dates = [PRICING_DATE + dt.timedelta(days=d) for d in (30, 60, 90, 120, 150)]
        spec = _barrier_spec(
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
            monitoring=BarrierMonitoring.DISCRETE,
            monitoring_dates=dates,
        )
        val = OptionValuation(
            _underlying(), spec, PricingMethod.BINOMIAL, params=BinomialParams(num_steps=200)
        )
        pv = val.present_value()
        assert pv > 0

    def test_boyle_lau_cap_bind_warning(self):
        """When barrier is very close to spot, Boyle-Lau cap-bind warning fires."""
        spec = _barrier_spec(
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=100.01,
            monitoring=BarrierMonitoring.CONTINUOUS,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            val = OptionValuation(
                _underlying(spot=100.0),
                spec,
                PricingMethod.BINOMIAL,
                params=BinomialParams(num_steps=50),
            )
            val.present_value()
        assert any("Boyle-Lau step alignment" in str(warning.message) for warning in w)

    def test_knock_in_triggered_at_inception_binomial_pv(self):
        """KI triggered at inception via binomial → matches vanilla binomial."""
        u = _underlying(spot=120.0)
        params = BinomialParams(num_steps=200)

        ki_spec = _barrier_spec(
            direction=BarrierDirection.UP,
            action=BarrierAction.IN,
            barrier=120.0,
        )
        pv_ki = OptionValuation(u, ki_spec, PricingMethod.BINOMIAL, params=params).present_value()

        vanilla_spec = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=STRIKE,
            maturity=MATURITY,
        )
        pv_vanilla = OptionValuation(
            u, vanilla_spec, PricingMethod.BINOMIAL, params=params
        ).present_value()
        assert np.isclose(pv_ki, pv_vanilla, rtol=1e-10)

    def test_knock_out_triggered_at_inception_rebate_at_hit_binomial(self):
        """KO triggered at inception with AT_HIT rebate via binomial."""
        spec = _barrier_spec(
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
            rebate=5.0,
            rebate_timing=RebateTiming.AT_HIT,
        )
        pv = OptionValuation(
            _underlying(spot=120.0), spec, PricingMethod.BINOMIAL, params=BinomialParams()
        ).present_value()
        assert pv == 5.0


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


# ===========================================================================
# Monte Carlo barrier coverage
# ===========================================================================

_MC_SEED = MonteCarloParams(random_seed=42)
_MC_PATHS = 50_000
_MC_STEPS = 200


def _mc_gbm(
    spot: float = SPOT,
    vol: float = VOL,
    dividend_curve: DiscountCurve | None = None,
) -> GBMProcess:
    md = _market_data()
    return GBMProcess(
        md,
        GBMParams(initial_value=spot, volatility=vol, dividend_curve=dividend_curve),
        SimulationConfig(paths=_MC_PATHS, end_date=MATURITY, num_steps=_MC_STEPS),
    )


def _mc_price(
    gbm: GBMProcess | None = None,
    spec: BarrierSpec | None = None,
    params: MonteCarloParams = _MC_SEED,
    **spec_kw,
) -> float:
    if gbm is None:
        gbm = _mc_gbm()
    if spec is None:
        spec = _barrier_spec(**spec_kw)
    return OptionValuation(gbm, spec, PricingMethod.MONTE_CARLO, params=params).present_value()


class TestBarrierMCInceptionHit:
    """MC paths where the barrier is already triggered at time zero."""

    # -- Continuous monitoring --

    def test_continuous_ko_inception_pv_zero(self):
        """Continuous UOC with S >= H → MC weight = 0 → PV = 0."""
        pv = _mc_price(
            _mc_gbm(spot=120.0),
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
        )
        assert np.isclose(pv, 0.0, atol=1e-10)

    def test_continuous_ko_inception_rebate_at_hit(self):
        """Continuous UOC at inception with AT_HIT rebate → PV = rebate."""
        pv = _mc_price(
            _mc_gbm(spot=120.0),
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
            rebate=5.0,
            rebate_timing=RebateTiming.AT_HIT,
        )
        assert np.isclose(pv, 5.0, atol=1e-10)

    def test_continuous_ko_inception_rebate_at_expiry(self):
        """Continuous UOC at inception with AT_EXPIRY rebate → PV = rebate * df."""
        from derivatives_pricing.utils import calculate_year_fraction

        T = calculate_year_fraction(PRICING_DATE, MATURITY)
        df = float(flat_curve(PRICING_DATE, MATURITY, RATE).df(T))

        pv = _mc_price(
            _mc_gbm(spot=120.0),
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=120.0,
            rebate=5.0,
            rebate_timing=RebateTiming.AT_EXPIRY,
        )
        assert np.isclose(pv, 5.0 * df, rtol=1e-6)

    def test_continuous_ki_inception_equals_vanilla(self):
        """Continuous UIC with S >= H → knocked in at inception → vanilla."""
        gbm = _mc_gbm(spot=120.0)
        pv_ki = _mc_price(
            gbm,
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
        pv_vanilla = OptionValuation(
            gbm, vanilla_spec, PricingMethod.MONTE_CARLO, params=_MC_SEED
        ).present_value()
        assert np.isclose(pv_ki, pv_vanilla, rtol=1e-6)

    # -- Discrete monitoring --

    def test_discrete_ko_inception_pv_zero(self):
        """Discrete DOP with S <= H → MC weight = 0 → PV = 0."""
        pv = _mc_price(
            _mc_gbm(spot=80.0),
            option_type=OptionType.PUT,
            direction=BarrierDirection.DOWN,
            action=BarrierAction.OUT,
            barrier=80.0,
            monitoring=BarrierMonitoring.DISCRETE,
            num_observations=12,
        )
        assert np.isclose(pv, 0.0, atol=1e-10)

    def test_discrete_ki_inception_equals_vanilla_aligned(self):
        """Discrete DIC at inception with grid-aligned observations → exact match.

        Setting num_observations = num_steps + 1 ensures monitoring dates
        land exactly on the simulation grid, so no extra dates are injected
        and the random draws are identical to vanilla.
        """
        gbm = _mc_gbm(spot=80.0)
        pv_ki = _mc_price(
            gbm,
            direction=BarrierDirection.DOWN,
            action=BarrierAction.IN,
            barrier=80.0,
            monitoring=BarrierMonitoring.DISCRETE,
            num_observations=_MC_STEPS + 1,
        )
        vanilla_spec = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=STRIKE,
            maturity=MATURITY,
        )
        pv_vanilla = OptionValuation(
            gbm, vanilla_spec, PricingMethod.MONTE_CARLO, params=_MC_SEED
        ).present_value()
        assert np.isclose(pv_ki, pv_vanilla, rtol=1e-10)

    def test_discrete_ki_inception_equals_vanilla_unaligned(self):
        """Discrete DIC at inception with non-aligned observations → MC noise.

        With num_observations=12, monitoring dates are injected into the
        grid, changing its size and thus the random draws. Both prices
        converge to the same expectation; we compare within MC noise.
        """
        gbm = _mc_gbm(spot=80.0)
        pv_ki = _mc_price(
            gbm,
            direction=BarrierDirection.DOWN,
            action=BarrierAction.IN,
            barrier=80.0,
            monitoring=BarrierMonitoring.DISCRETE,
            num_observations=12,
        )
        vanilla_spec = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=STRIKE,
            maturity=MATURITY,
        )
        pv_vanilla = OptionValuation(
            gbm, vanilla_spec, PricingMethod.MONTE_CARLO, params=_MC_SEED
        ).present_value()
        assert np.isclose(pv_ki, pv_vanilla, rtol=0.01)


class TestBarrierMCDiscreteRebate:
    """Discrete monitoring rebate paths in MC."""

    def test_discrete_ko_rebate_at_hit_positive(self):
        """Discrete KO with rebate AT_HIT: PV should include rebate component."""
        pv_no_rebate = _mc_price(
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=110.0,
            rebate=0.0,
            monitoring=BarrierMonitoring.DISCRETE,
            num_observations=12,
        )
        pv_with_rebate = _mc_price(
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=110.0,
            rebate=5.0,
            rebate_timing=RebateTiming.AT_HIT,
            monitoring=BarrierMonitoring.DISCRETE,
            num_observations=12,
        )
        assert pv_with_rebate > pv_no_rebate

    def test_discrete_ko_rebate_at_expiry_positive(self):
        """Discrete KO with rebate AT_EXPIRY: PV should include rebate component."""
        pv_no_rebate = _mc_price(
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=110.0,
            rebate=0.0,
            monitoring=BarrierMonitoring.DISCRETE,
            num_observations=12,
        )
        pv_with_rebate = _mc_price(
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=110.0,
            rebate=5.0,
            rebate_timing=RebateTiming.AT_EXPIRY,
            monitoring=BarrierMonitoring.DISCRETE,
            num_observations=12,
        )
        assert pv_with_rebate > pv_no_rebate

    def test_discrete_ki_rebate_at_expiry_positive(self):
        """Discrete KI with rebate AT_EXPIRY: never-knocked-in paths receive rebate."""
        pv_no_rebate = _mc_price(
            direction=BarrierDirection.DOWN,
            action=BarrierAction.IN,
            barrier=85.0,
            rebate=0.0,
            rebate_timing=RebateTiming.AT_EXPIRY,
            monitoring=BarrierMonitoring.DISCRETE,
            num_observations=12,
        )
        pv_with_rebate = _mc_price(
            direction=BarrierDirection.DOWN,
            action=BarrierAction.IN,
            barrier=85.0,
            rebate=5.0,
            rebate_timing=RebateTiming.AT_EXPIRY,
            monitoring=BarrierMonitoring.DISCRETE,
            num_observations=12,
        )
        assert pv_with_rebate > pv_no_rebate


class TestBarrierMCContinuousRebateAtExpiry:
    """Continuous monitoring KO rebate AT_EXPIRY path in MC."""

    def test_continuous_ko_rebate_at_expiry_positive(self):
        """Continuous KO with AT_EXPIRY rebate: PV includes discounted rebate."""
        pv_no_rebate = _mc_price(
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=110.0,
            rebate=0.0,
        )
        pv_with_rebate = _mc_price(
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            barrier=110.0,
            rebate=5.0,
            rebate_timing=RebateTiming.AT_EXPIRY,
        )
        assert pv_with_rebate > pv_no_rebate

    def test_continuous_ki_rebate_at_expiry_positive(self):
        """Continuous KI with AT_EXPIRY rebate: never-in paths receive rebate."""
        pv_no_rebate = _mc_price(
            direction=BarrierDirection.DOWN,
            action=BarrierAction.IN,
            barrier=85.0,
            rebate=0.0,
            rebate_timing=RebateTiming.AT_EXPIRY,
        )
        pv_with_rebate = _mc_price(
            direction=BarrierDirection.DOWN,
            action=BarrierAction.IN,
            barrier=85.0,
            rebate=5.0,
            rebate_timing=RebateTiming.AT_EXPIRY,
        )
        assert pv_with_rebate > pv_no_rebate


class TestBarrierMCNonBarrierAwareBasis:
    """Test the non-barrier-aware LSM regression path."""

    def test_american_ko_without_barrier_aware_basis(self):
        """American KO with barrier_aware_basis=False should still price reasonably."""
        params_aware = MonteCarloParams(random_seed=42, barrier_aware_basis=True)
        params_naive = MonteCarloParams(random_seed=42, barrier_aware_basis=False)

        spec = _barrier_spec(
            exercise_type=ExerciseType.AMERICAN,
            direction=BarrierDirection.DOWN,
            action=BarrierAction.OUT,
            barrier=85.0,
        )
        gbm = _mc_gbm()

        pv_aware = OptionValuation(
            gbm, spec, PricingMethod.MONTE_CARLO, params=params_aware
        ).present_value()
        pv_naive = OptionValuation(
            gbm, spec, PricingMethod.MONTE_CARLO, params=params_naive
        ).present_value()

        # Both should be positive and in the same ballpark
        assert pv_aware > 0
        assert pv_naive > 0
        assert np.isclose(pv_naive, pv_aware, rtol=0.05), (
            f"barrier_aware_basis=False ({pv_naive:.4f}) vs True ({pv_aware:.4f})"
        )
