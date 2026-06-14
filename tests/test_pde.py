"""PDE FD grid/method/solver equivalence tests.

Verifies that different PDE finite-difference schemes (Explicit, Implicit,
Crank-Nicolson, Explicit-Hull) and space grids (SPOT, LOG_SPOT) produce
consistent prices.  These are internal engine tests — cross-method and
QuantLib comparisons live in test_quantlib_comparison.py.
"""

from dataclasses import replace as dc_replace
import datetime as dt
import logging

import numpy as np
import pytest

from derivatives_pricing.exceptions import StabilityError, UnsupportedFeatureError
from derivatives_pricing.enums import (
    BarrierAction,
    BarrierDirection,
    BarrierMonitoring,
    DayCountConvention,
    ExerciseType,
    GreekCalculationMethod,
    OptionType,
    PDEEarlyExercise,
    PDEMethod,
    PDESpaceGrid,
    PricingMethod,
    RebateTiming,
)
from derivatives_pricing.market_environment import MarketData
from derivatives_pricing.rates import DiscountCurve
from derivatives_pricing.utils import calculate_year_fraction
from derivatives_pricing.valuation import OptionValuation, UnderlyingData
from derivatives_pricing.valuation.contracts import (
    BarrierSpec,
    DoubleBarrierSpec,
    PayoffSpec,
    VanillaSpec,
)
from derivatives_pricing.valuation.pde import (
    _FDBarrierValuation,
    _FDDoubleBarrierValuation,
    _fd_barrier_ki_core,
    _fd_double_barrier_ki_core,
    _resolve_pde_spot_steps,
)
from helpers import (
    flat_curve,
    market_data,
    underlying,
    spec,
    PRICING_DATE,
    MATURITY,
)
from derivatives_pricing.valuation.params import PDEParams

logger = logging.getLogger(__name__)


def test_pde_fd_grid_method_equivalence_european():
    """PDE FD variants should be in the same neighborhood for European options."""
    q_curve = flat_curve(PRICING_DATE, MATURITY, 0.01)
    ud = underlying(initial_value=100.0, dividend_curve=q_curve)
    sp = spec(strike=100.0, option_type=OptionType.CALL, exercise=ExerciseType.EUROPEAN)

    base_params = PDEParams(spot_steps=160, time_steps=240)
    baseline = OptionValuation(ud, sp, PricingMethod.PDE_FD, params=base_params).present_value()

    for method in (
        PDEMethod.IMPLICIT,
        PDEMethod.EXPLICIT,
        PDEMethod.EXPLICIT_HULL,
        PDEMethod.CRANK_NICOLSON,
    ):
        for grid in (PDESpaceGrid.SPOT, PDESpaceGrid.LOG_SPOT):
            params = PDEParams(
                spot_steps=160,
                time_steps=240,
                method=method,
                space_grid=grid,
                american_solver=PDEEarlyExercise.INTRINSIC,
            )

            if (
                method in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL)
                and grid is PDESpaceGrid.SPOT
            ):
                with pytest.raises(
                    StabilityError, match="Explicit spot-grid scheme likely unstable"
                ):
                    OptionValuation(ud, sp, PricingMethod.PDE_FD, params=params).present_value()
                continue

            pv = OptionValuation(ud, sp, PricingMethod.PDE_FD, params=params).present_value()
            assert np.isclose(pv, baseline, rtol=0.005)


def test_pde_fd_grid_method_equivalence_american():
    """PDE FD American variants should be in the same neighborhood."""
    q_curve = flat_curve(PRICING_DATE, MATURITY, 0.0)
    ud = underlying(initial_value=95.0, dividend_curve=q_curve)
    sp = spec(strike=100.0, option_type=OptionType.PUT, exercise=ExerciseType.AMERICAN)

    base_params = PDEParams(spot_steps=160, time_steps=240)
    baseline = OptionValuation(ud, sp, PricingMethod.PDE_FD, params=base_params).present_value()

    for method in (
        PDEMethod.IMPLICIT,
        PDEMethod.EXPLICIT,
        PDEMethod.EXPLICIT_HULL,
        PDEMethod.CRANK_NICOLSON,
    ):
        for grid in (PDESpaceGrid.SPOT, PDESpaceGrid.LOG_SPOT):
            for solver in (PDEEarlyExercise.INTRINSIC, PDEEarlyExercise.GAUSS_SEIDEL):
                params = PDEParams(
                    spot_steps=160,
                    time_steps=240,
                    method=method,
                    space_grid=grid,
                    american_solver=solver,
                    max_iter=20_000,
                )

                if (
                    method in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL)
                    and solver is PDEEarlyExercise.GAUSS_SEIDEL
                ):
                    with pytest.raises(
                        UnsupportedFeatureError, match="GAUSS_SEIDEL is not supported"
                    ):
                        OptionValuation(ud, sp, PricingMethod.PDE_FD, params=params).present_value()
                    continue

                if (
                    method in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL)
                    and grid is PDESpaceGrid.SPOT
                ):
                    with pytest.raises(
                        StabilityError, match="Explicit spot-grid scheme likely unstable"
                    ):
                        OptionValuation(ud, sp, PricingMethod.PDE_FD, params=params).present_value()
                    continue

                pv = OptionValuation(ud, sp, PricingMethod.PDE_FD, params=params).present_value()
                assert np.isclose(pv, baseline, rtol=0.005)


class TestPDEGridTheta:
    """Verify PDE grid theta sign and magnitude against BSM analytical theta."""

    @pytest.fixture(autouse=True)
    def setup(self):
        q_curve = flat_curve(PRICING_DATE, MATURITY, 0.0)
        self.ud = underlying(initial_value=100.0, volatility=0.20, dividend_curve=q_curve)
        self.pde_params = PDEParams(spot_steps=400, time_steps=400)

    def test_call_grid_theta_negative(self):
        spec_call = spec(strike=100.0, option_type=OptionType.CALL, exercise=ExerciseType.EUROPEAN)
        ov = OptionValuation(self.ud, spec_call, PricingMethod.PDE_FD, params=self.pde_params)
        theta = ov.theta(greek_calc_method=GreekCalculationMethod.GRID)
        assert theta < 0

    def test_put_grid_theta_negative(self):
        spec_put = spec(strike=100.0, option_type=OptionType.PUT, exercise=ExerciseType.EUROPEAN)
        ov = OptionValuation(self.ud, spec_put, PricingMethod.PDE_FD, params=self.pde_params)
        theta = ov.theta(greek_calc_method=GreekCalculationMethod.GRID)
        assert theta < 0

    @pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
    def test_grid_theta_close_to_bsm(self, option_type: OptionType):
        spec_vanilla = spec(strike=100.0, option_type=option_type, exercise=ExerciseType.EUROPEAN)
        bsm = OptionValuation(self.ud, spec_vanilla, PricingMethod.BSM)
        pde = OptionValuation(self.ud, spec_vanilla, PricingMethod.PDE_FD, params=self.pde_params)

        theta_bsm = bsm.theta()
        theta_pde = pde.theta(greek_calc_method=GreekCalculationMethod.GRID)

        assert np.isclose(theta_pde, theta_bsm, rtol=0.02)


# ═══════════════════════════════════════════════════════════════════════════
# Custom payoff PDE equivalence
# ═══════════════════════════════════════════════════════════════════════════


def _bull_call_spread(S):
    return np.maximum(S - 95.0, 0) - np.maximum(S - 115.0, 0)


def _capped_strangle(S):
    return np.minimum(40.0, np.maximum(90.0 - S, 0) + np.maximum(S - 110.0, 0))


_CUSTOM_PAYOFFS = [
    pytest.param(_bull_call_spread, id="bull_call_spread"),
    pytest.param(_capped_strangle, id="capped_strangle"),
]


class TestPDECustomPayoffMethodEquivalence:
    """PDE FD variants should agree for custom PayoffSpec payoffs."""

    @pytest.fixture(autouse=True)
    def setup(self):
        q_curve = flat_curve(PRICING_DATE, MATURITY, 0.01)
        self.ud = underlying(initial_value=100.0, dividend_curve=q_curve)

    @pytest.mark.parametrize("payoff_fn", _CUSTOM_PAYOFFS)
    def test_european(self, payoff_fn):
        sp = PayoffSpec(
            exercise_type=ExerciseType.EUROPEAN,
            maturity=MATURITY,
            payoff_fn=payoff_fn,
        )
        base_params = PDEParams(spot_steps=400, time_steps=400)
        baseline = OptionValuation(
            self.ud, sp, PricingMethod.PDE_FD, params=base_params
        ).present_value()

        for method in (
            PDEMethod.IMPLICIT,
            PDEMethod.EXPLICIT,
            PDEMethod.EXPLICIT_HULL,
            PDEMethod.CRANK_NICOLSON,
        ):
            for grid in (PDESpaceGrid.SPOT, PDESpaceGrid.LOG_SPOT):
                params = PDEParams(
                    spot_steps=400,
                    time_steps=400,
                    method=method,
                    space_grid=grid,
                )

                if (
                    method in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL)
                    and grid is PDESpaceGrid.SPOT
                ):
                    with pytest.raises(
                        StabilityError, match="Explicit spot-grid scheme likely unstable"
                    ):
                        OptionValuation(
                            self.ud, sp, PricingMethod.PDE_FD, params=params
                        ).present_value()
                    continue

                pv = OptionValuation(
                    self.ud, sp, PricingMethod.PDE_FD, params=params
                ).present_value()
                assert np.isclose(pv, baseline, rtol=0.005)

    @pytest.mark.parametrize("payoff_fn", _CUSTOM_PAYOFFS)
    def test_american(self, payoff_fn):
        sp = PayoffSpec(
            exercise_type=ExerciseType.AMERICAN,
            maturity=MATURITY,
            payoff_fn=payoff_fn,
        )
        base_params = PDEParams(spot_steps=400, time_steps=400)
        baseline = OptionValuation(
            self.ud, sp, PricingMethod.PDE_FD, params=base_params
        ).present_value()

        for method in (
            PDEMethod.IMPLICIT,
            PDEMethod.EXPLICIT,
            PDEMethod.EXPLICIT_HULL,
            PDEMethod.CRANK_NICOLSON,
        ):
            for grid in (PDESpaceGrid.SPOT, PDESpaceGrid.LOG_SPOT):
                is_explicit = method in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL)
                params = PDEParams(
                    spot_steps=400,
                    time_steps=400,
                    method=method,
                    space_grid=grid,
                    american_solver=(
                        PDEEarlyExercise.INTRINSIC if is_explicit else PDEEarlyExercise.GAUSS_SEIDEL
                    ),
                )

                if is_explicit and grid is PDESpaceGrid.SPOT:
                    with pytest.raises(
                        StabilityError, match="Explicit spot-grid scheme likely unstable"
                    ):
                        OptionValuation(
                            self.ud, sp, PricingMethod.PDE_FD, params=params
                        ).present_value()
                    continue

                pv = OptionValuation(
                    self.ud, sp, PricingMethod.PDE_FD, params=params
                ).present_value()
                assert np.isclose(pv, baseline, rtol=0.02)


# ═══════════════════════════════════════════════════════════════════════════
# Barrier PDE equivalence
# ═══════════════════════════════════════════════════════════════════════════


def _forward_curve(*, times: tuple[float, ...], forwards: tuple[float, ...]) -> DiscountCurve:
    return DiscountCurve.from_forwards(
        times=np.array(times, dtype=float),
        forwards=np.array(forwards, dtype=float),
    )


def _barrier_pde_value(scenario: dict, params: PDEParams) -> float:
    r_curve = scenario.get("r_curve") or flat_curve(PRICING_DATE, MATURITY, scenario["rate"])
    q_curve = scenario.get("q_curve")
    md = market_data(pricing_date=PRICING_DATE, discount_curve=r_curve)
    ud = underlying(
        initial_value=scenario["spot"],
        volatility=scenario["volatility"],
        market_data=md,
        dividend_curve=q_curve,
    )
    barrier_spec = BarrierSpec(
        option_type=scenario["option_type"],
        exercise_type=scenario["exercise_type"],
        strike=scenario["strike"],
        maturity=MATURITY,
        barrier=scenario["barrier"],
        direction=scenario["direction"],
        action=scenario["action"],
        monitoring=scenario["monitoring"],
        rebate=scenario.get("rebate", 0.0),
        rebate_timing=scenario.get("rebate_timing", RebateTiming.AT_HIT),
        num_observations=scenario.get("num_observations"),
        monitoring_dates=scenario.get("monitoring_dates"),
    )
    return OptionValuation(ud, barrier_spec, PricingMethod.PDE_FD, params=params).present_value()


_EUROPEAN_BARRIER_SCENARIOS = [
    pytest.param(
        {
            "spot": 104.0,
            "strike": 100.0,
            "volatility": 0.24,
            "rate": 0.045,
            "option_type": OptionType.CALL,
            "exercise_type": ExerciseType.EUROPEAN,
            "direction": BarrierDirection.DOWN,
            "action": BarrierAction.OUT,
            "barrier": 86.0,
            "monitoring": BarrierMonitoring.CONTINUOUS,
            "q_curve": flat_curve(PRICING_DATE, MATURITY, 0.012),
        },
        id="eu_down_out_call_continuous",
    ),
    pytest.param(
        {
            "spot": 96.0,
            "strike": 101.0,
            "volatility": 0.28,
            "rate": 0.035,
            "option_type": OptionType.PUT,
            "exercise_type": ExerciseType.EUROPEAN,
            "direction": BarrierDirection.UP,
            "action": BarrierAction.IN,
            "barrier": 114.0,
            "monitoring": BarrierMonitoring.CONTINUOUS,
            "rebate": 1.75,
            "rebate_timing": RebateTiming.AT_EXPIRY,
        },
        id="eu_up_in_put_rebate_continuous",
    ),
    pytest.param(
        {
            "spot": 101.0,
            "strike": 97.0,
            "volatility": 0.22,
            "rate": 0.04,
            "option_type": OptionType.CALL,
            "exercise_type": ExerciseType.EUROPEAN,
            "direction": BarrierDirection.DOWN,
            "action": BarrierAction.IN,
            "barrier": 89.0,
            "monitoring": BarrierMonitoring.DISCRETE,
            "rebate": 1.25,
            "rebate_timing": RebateTiming.AT_EXPIRY,
            "num_observations": 12,
            "q_curve": flat_curve(PRICING_DATE, MATURITY, 0.02),
        },
        id="eu_down_in_call_discrete",
    ),
    pytest.param(
        {
            "spot": 93.0,
            "strike": 99.0,
            "volatility": 0.26,
            "rate": 0.03,
            "option_type": OptionType.PUT,
            "exercise_type": ExerciseType.EUROPEAN,
            "direction": BarrierDirection.UP,
            "action": BarrierAction.OUT,
            "barrier": 117.0,
            "monitoring": BarrierMonitoring.DISCRETE,
            "rebate": 2.0,
            "num_observations": 10,
            "q_curve": flat_curve(PRICING_DATE, MATURITY, 0.005),
        },
        id="eu_up_out_put_discrete_rebate",
    ),
    pytest.param(
        {
            "spot": 108.0,
            "strike": 102.0,
            "volatility": 0.21,
            "rate": 0.05,
            "option_type": OptionType.CALL,
            "exercise_type": ExerciseType.EUROPEAN,
            "direction": BarrierDirection.UP,
            "action": BarrierAction.OUT,
            "barrier": 124.0,
            "monitoring": BarrierMonitoring.CONTINUOUS,
            "r_curve": _forward_curve(times=(0.0, 0.2, 0.6, 1.0), forwards=(0.035, 0.055, 0.04)),
            "q_curve": _forward_curve(times=(0.0, 0.4, 0.8, 1.0), forwards=(0.01, 0.018, 0.008)),
        },
        id="eu_up_out_call_nonflat_curves",
    ),
]


_AMERICAN_BARRIER_SCENARIOS = [
    pytest.param(
        {
            "spot": 94.0,
            "strike": 100.0,
            "volatility": 0.23,
            "rate": 0.042,
            "option_type": OptionType.PUT,
            "exercise_type": ExerciseType.AMERICAN,
            "direction": BarrierDirection.DOWN,
            "action": BarrierAction.OUT,
            "barrier": 82.0,
            "monitoring": BarrierMonitoring.CONTINUOUS,
        },
        id="am_down_out_put_continuous",
    ),
    pytest.param(
        {
            "spot": 101.0,
            "strike": 99.0,
            "volatility": 0.27,
            "rate": 0.038,
            "option_type": OptionType.CALL,
            "exercise_type": ExerciseType.AMERICAN,
            "direction": BarrierDirection.UP,
            "action": BarrierAction.OUT,
            "barrier": 119.0,
            "monitoring": BarrierMonitoring.CONTINUOUS,
            "rebate": 3.0,
            "q_curve": flat_curve(PRICING_DATE, MATURITY, 0.015),
        },
        id="am_up_out_call_rebate_continuous",
    ),
    pytest.param(
        {
            "spot": 103.0,
            "strike": 98.0,
            "volatility": 0.25,
            "rate": 0.04,
            "option_type": OptionType.CALL,
            "exercise_type": ExerciseType.AMERICAN,
            "direction": BarrierDirection.DOWN,
            "action": BarrierAction.IN,
            "barrier": 88.0,
            "monitoring": BarrierMonitoring.CONTINUOUS,
            "rebate": 1.5,
            "rebate_timing": RebateTiming.AT_EXPIRY,
            "q_curve": flat_curve(PRICING_DATE, MATURITY, 0.01),
        },
        id="am_down_in_call_continuous",
    ),
    pytest.param(
        {
            "spot": 97.0,
            "strike": 100.0,
            "volatility": 0.29,
            "rate": 0.033,
            "option_type": OptionType.PUT,
            "exercise_type": ExerciseType.AMERICAN,
            "direction": BarrierDirection.UP,
            "action": BarrierAction.IN,
            "barrier": 113.0,
            "monitoring": BarrierMonitoring.CONTINUOUS,
        },
        id="am_up_in_put_continuous",
    ),
    pytest.param(
        {
            "spot": 98.0,
            "strike": 102.0,
            "volatility": 0.24,
            "rate": 0.047,
            "option_type": OptionType.PUT,
            "exercise_type": ExerciseType.AMERICAN,
            "direction": BarrierDirection.DOWN,
            "action": BarrierAction.IN,
            "barrier": 87.0,
            "monitoring": BarrierMonitoring.DISCRETE,
            "rebate": 1.25,
            "rebate_timing": RebateTiming.AT_EXPIRY,
            "num_observations": 9,
            "r_curve": _forward_curve(times=(0.0, 0.3, 0.7, 1.0), forwards=(0.03, 0.05, 0.042)),
            "q_curve": _forward_curve(times=(0.0, 0.25, 0.75, 1.0), forwards=(0.008, 0.015, 0.005)),
        },
        id="am_down_in_put_discrete_nonflat",
    ),
]


@pytest.mark.parametrize("scenario", _EUROPEAN_BARRIER_SCENARIOS)
def test_pde_fd_barrier_equivalence_european(scenario):
    """Barrier PDE European variants should be in the same neighborhood."""
    base_params = PDEParams(spot_steps=800, time_steps=800, space_grid=PDESpaceGrid.LOG_SPOT)
    baseline = _barrier_pde_value(scenario, base_params)

    for method in (
        PDEMethod.IMPLICIT,
        PDEMethod.CRANK_NICOLSON,
    ):
        for grid in (PDESpaceGrid.SPOT, PDESpaceGrid.LOG_SPOT):
            params = PDEParams(
                spot_steps=800,
                time_steps=800,
                method=method,
                space_grid=grid,
            )

            pv = _barrier_pde_value(scenario, params)
            assert np.isclose(pv, baseline, rtol=0.015)


_PARITY_MONITORING_CONFIGS = [
    pytest.param(
        BarrierMonitoring.CONTINUOUS,
        None,
        None,
        id="continuous",
    ),
    # Continuous monitoring + discrete divs: parity-safe baseline for
    # the discrete-dividend jump path before any monitoring reset.
    pytest.param(
        BarrierMonitoring.CONTINUOUS,
        None,
        [
            (dt.datetime(2025, 3, 15), 1.0),
            (dt.datetime(2025, 6, 15), 1.0),
            (dt.datetime(2025, 9, 15), 1.0),
            (dt.datetime(2025, 12, 15), 1.0),
        ],
        id="continuous_discrete_divs",
    ),
    # Discrete monitoring + discrete divs on non-coincident dates:
    # isolates the discrete monitoring codepath from the
    # reset-vs-divjump ordering interaction.
    pytest.param(
        BarrierMonitoring.DISCRETE,
        [dt.datetime(2025, m, 28) for m in range(1, 13)],
        [
            (dt.datetime(2025, 3, 15), 1.0),
            (dt.datetime(2025, 6, 15), 1.0),
            (dt.datetime(2025, 9, 15), 1.0),
            (dt.datetime(2025, 12, 15), 1.0),
        ],
        id="discrete_noncoincident_divs",
    ),
    # Discrete monitoring + discrete divs coincident on monitoring
    # dates: regression case for the KO ordering fix (reset on ex-div
    # surface before divjump).  Pre-fix this disagreed with direct KI
    # by ~1–10%; post-fix it agrees to grid-resolution precision.
    pytest.param(
        BarrierMonitoring.DISCRETE,
        [dt.datetime(2025, m, 28) for m in range(1, 13)],
        [
            (dt.datetime(2025, 3, 28), 1.0),
            (dt.datetime(2025, 6, 28), 1.0),
            (dt.datetime(2025, 9, 28), 1.0),
            (dt.datetime(2025, 12, 28), 1.0),
        ],
        id="discrete_coincident_divs",
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize(
    "monitoring,monitoring_dates,discrete_dividends",
    _PARITY_MONITORING_CONFIGS,
)
@pytest.mark.parametrize(
    "option_type,strike,direction,barrier,rebate",
    [
        pytest.param(OptionType.CALL, 100.0, BarrierDirection.DOWN, 90.0, 0.0, id="down_in_call"),
        pytest.param(OptionType.PUT, 100.0, BarrierDirection.DOWN, 90.0, 0.0, id="down_in_put"),
        pytest.param(
            OptionType.CALL, 95.0, BarrierDirection.UP, 105.0, 2.0, id="up_in_call_rebate"
        ),
        pytest.param(OptionType.PUT, 100.0, BarrierDirection.UP, 105.0, 2.0, id="up_in_put_rebate"),
    ],
)
def test_pde_fd_barrier_european_ki_parity_matches_direct(
    option_type: OptionType,
    strike: float,
    direction: BarrierDirection,
    barrier: float,
    rebate: float,
    monitoring: BarrierMonitoring,
    monitoring_dates: list[dt.datetime] | None,
    discrete_dividends: list[tuple[dt.datetime, float]] | None,
):
    """European KI parity pricing should closely track the direct KI PDE solve.

    Covers continuous and discrete monitoring crossed with continuous
    or discrete dividends, including the coincident-date regression
    case for the KO core ordering fix.  Dividend mechanism is mutually
    exclusive: ``dividend_curve`` for the continuous case,
    ``discrete_dividends`` (with no continuous curve) for the rest.
    """
    curve_r = DiscountCurve.flat(0.05, 2)
    curve_q = DiscountCurve.flat(0.03, 2) if discrete_dividends is None else None
    pricing_date = dt.datetime(2025, 1, 1)
    maturity = dt.datetime(2025, 12, 31)

    md = MarketData(pricing_date, curve_r, currency="USD")
    ud = UnderlyingData(
        initial_value=96.0,
        volatility=0.25,
        market_data=md,
        dividend_curve=curve_q,
        discrete_dividends=discrete_dividends,
    )
    barrier_spec = BarrierSpec(
        option_type=option_type,
        exercise_type=ExerciseType.EUROPEAN,
        strike=strike,
        maturity=maturity,
        barrier=barrier,
        direction=direction,
        action=BarrierAction.IN,
        monitoring=monitoring,
        monitoring_dates=monitoring_dates,
        rebate=rebate,
        rebate_timing=RebateTiming.AT_EXPIRY,
    )
    params = PDEParams(
        spot_steps=600,
        time_steps=600,
        method=PDEMethod.CRANK_NICOLSON,
        space_grid=PDESpaceGrid.LOG_SPOT,
        rannacher_steps=2,
    )

    valuation = OptionValuation(ud, barrier_spec, PricingMethod.PDE_FD, params=params)
    impl = _FDBarrierValuation(valuation)

    parity_price = impl.present_value()
    direct_price = float(_fd_barrier_ki_core(**impl._base_solve_args())[0])

    abs_diff = abs(parity_price - direct_price)
    rel_diff = abs_diff / max(abs(direct_price), 1e-12) * 100
    div_label = (
        "continuous_q" if discrete_dividends is None else f"{len(discrete_dividends)}_discrete_divs"
    )
    logger.info(
        "KI parity vs direct [%s/%s K=%g H=%g rebate=%g monitoring=%s divs=%s]: "
        "parity=%.6f direct=%.6f abs_diff=%.4f rel_diff=%.4f%%",
        option_type.value,
        direction.value,
        strike,
        barrier,
        rebate,
        monitoring.value,
        div_label,
        parity_price,
        direct_price,
        abs_diff,
        rel_diff,
    )

    assert np.isclose(parity_price, direct_price, rtol=0.003)


# Double-barrier continuous-monitoring configs: a continuous dividend yield,
# and discrete dividends (non-coincident — continuous monitoring has no
# observation dates to coincide with).  Mirrors _PARITY_MONITORING_CONFIGS
# but with CONTINUOUS monitoring (double-barrier PDE is continuous-only).
_DOUBLE_PARITY_DIVIDEND_CONFIGS = [
    pytest.param(None, id="continuous_yield"),
    pytest.param(
        [
            (dt.datetime(2025, 3, 15), 1.0),
            (dt.datetime(2025, 6, 15), 1.0),
            (dt.datetime(2025, 9, 15), 1.0),
        ],
        id="discrete_divs",
    ),
]


@pytest.mark.parametrize("discrete_dividends", _DOUBLE_PARITY_DIVIDEND_CONFIGS)
@pytest.mark.parametrize(
    "option_type,strike,lower_barrier,upper_barrier,rebate",
    [
        pytest.param(OptionType.CALL, 100.0, 85.0, 120.0, 0.0, id="dki_call"),
        pytest.param(OptionType.PUT, 100.0, 85.0, 120.0, 0.0, id="dki_put"),
        pytest.param(OptionType.CALL, 95.0, 82.0, 125.0, 2.0, id="dki_call_rebate"),
        pytest.param(OptionType.PUT, 105.0, 82.0, 125.0, 2.0, id="dki_put_rebate"),
    ],
)
def test_pde_fd_double_barrier_european_ki_parity_matches_direct(
    option_type: OptionType,
    strike: float,
    lower_barrier: float,
    upper_barrier: float,
    rebate: float,
    discrete_dividends,
):
    """European double-KI parity pricing should track the direct two-surface solve.

    The engine routes European double knock-in through in-out parity
    (``V_DKI = V_vanilla + R·df_T − V_DKO``).  The direct
    ``_fd_double_barrier_ki_core`` (with ``early_exercise=False``) solves the
    inactive surface on the corridor coupled to the active vanilla at both
    barriers.  Both must agree to grid-resolution precision.
    """
    curve_r = DiscountCurve.flat(0.05, 2)
    curve_q = DiscountCurve.flat(0.03, 2) if discrete_dividends is None else None
    pricing_date = dt.datetime(2025, 1, 1)
    maturity = dt.datetime(2025, 12, 31)

    md = MarketData(pricing_date, curve_r, currency="USD")
    ud = UnderlyingData(
        initial_value=100.0,
        volatility=0.25,
        market_data=md,
        dividend_curve=curve_q,
        discrete_dividends=discrete_dividends,
    )
    spec = DoubleBarrierSpec(
        option_type=option_type,
        exercise_type=ExerciseType.EUROPEAN,
        strike=strike,
        maturity=maturity,
        lower_barrier=lower_barrier,
        upper_barrier=upper_barrier,
        action=BarrierAction.IN,
        monitoring=BarrierMonitoring.CONTINUOUS,
        rebate=rebate,
        rebate_timing=RebateTiming.AT_EXPIRY,
    )
    params = PDEParams(
        spot_steps=600,
        time_steps=600,
        method=PDEMethod.CRANK_NICOLSON,
        space_grid=PDESpaceGrid.LOG_SPOT,
        rannacher_steps=2,
    )

    valuation = OptionValuation(ud, spec, PricingMethod.PDE_FD, params=params)
    impl = _FDDoubleBarrierValuation(valuation)

    parity_price = float(valuation.present_value())
    direct_price = float(_fd_double_barrier_ki_core(**impl._base_solve_args())[0])

    abs_diff = abs(parity_price - direct_price)
    rel_diff = abs_diff / max(abs(direct_price), 1e-12) * 100
    logger.info(
        "DoubleKI parity vs direct [%s K=%g L=%g U=%g rebate=%g divs=%s]: "
        "parity=%.6f direct=%.6f abs_diff=%.4f rel_diff=%.4f%%",
        option_type.value,
        strike,
        lower_barrier,
        upper_barrier,
        rebate,
        "discrete" if discrete_dividends is not None else "continuous_q",
        parity_price,
        direct_price,
        abs_diff,
        rel_diff,
    )

    assert np.isclose(parity_price, direct_price, rtol=5e-3, atol=5e-3), (
        f"parity {parity_price:.6f} vs direct {direct_price:.6f}"
    )


@pytest.mark.parametrize("discrete_dividends", _DOUBLE_PARITY_DIVIDEND_CONFIGS)
@pytest.mark.parametrize(
    "option_type,strike,lower_barrier,upper_barrier,rebate",
    [
        pytest.param(OptionType.CALL, 100.0, 85.0, 120.0, 0.0, id="dki_call"),
        pytest.param(OptionType.PUT, 100.0, 85.0, 120.0, 0.0, id="dki_put"),
        pytest.param(OptionType.CALL, 95.0, 82.0, 125.0, 2.0, id="dki_call_rebate"),
        pytest.param(OptionType.PUT, 105.0, 82.0, 125.0, 2.0, id="dki_put_rebate"),
    ],
)
def test_pde_fd_double_barrier_discrete_european_ki_parity_matches_direct(
    option_type: OptionType,
    strike: float,
    lower_barrier: float,
    upper_barrier: float,
    rebate: float,
    discrete_dividends,
):
    """Discrete European double-KI parity should track the direct two-surface solve.

    Discrete-monitoring analogue of
    ``test_pde_fd_double_barrier_european_ki_parity_matches_direct``.  The
    engine prices European double knock-in via in-out parity
    (``V_DKI = V_vanilla + R·df_T − V_DKO``, with the now discrete-aware KO
    core); the direct ``_fd_double_barrier_ki_core`` (``early_exercise=False``)
    solves both surfaces on the Boyle-Tian half-step grid and couples the
    inactive surface to the active one at every monitoring date.  Both must
    agree to grid-resolution precision.
    """
    curve_r = DiscountCurve.flat(0.05, 2)
    curve_q = DiscountCurve.flat(0.03, 2) if discrete_dividends is None else None
    pricing_date = dt.datetime(2025, 1, 1)
    maturity = dt.datetime(2025, 12, 31)

    md = MarketData(pricing_date, curve_r, currency="USD")
    ud = UnderlyingData(
        initial_value=100.0,
        volatility=0.25,
        market_data=md,
        dividend_curve=curve_q,
        discrete_dividends=discrete_dividends,
    )
    spec = DoubleBarrierSpec(
        option_type=option_type,
        exercise_type=ExerciseType.EUROPEAN,
        strike=strike,
        maturity=maturity,
        lower_barrier=lower_barrier,
        upper_barrier=upper_barrier,
        action=BarrierAction.IN,
        monitoring=BarrierMonitoring.DISCRETE,
        num_observations=50,
        rebate=rebate,
        rebate_timing=RebateTiming.AT_EXPIRY,
    )
    # CN on the half-step discrete grid keeps the KO (parity) and KI (direct)
    # solves on the same grid family for a clean comparison.
    params = PDEParams(
        spot_steps=600,
        time_steps=1200,
        method=PDEMethod.CRANK_NICOLSON,
        space_grid=PDESpaceGrid.LOG_SPOT,
        rannacher_steps=2,
    )

    valuation = OptionValuation(ud, spec, PricingMethod.PDE_FD, params=params)
    impl = _FDDoubleBarrierValuation(valuation)

    parity_price = float(valuation.present_value())
    direct_price = float(_fd_double_barrier_ki_core(**impl._base_solve_args())[0])

    abs_diff = abs(parity_price - direct_price)
    rel_diff = abs_diff / max(abs(direct_price), 1e-12) * 100
    logger.info(
        "DoubleKI discrete parity vs direct [%s K=%g L=%g U=%g rebate=%g divs=%s]: "
        "parity=%.6f direct=%.6f abs_diff=%.4f rel_diff=%.4f%%",
        option_type.value,
        strike,
        lower_barrier,
        upper_barrier,
        rebate,
        "discrete" if discrete_dividends is not None else "continuous_q",
        parity_price,
        direct_price,
        abs_diff,
        rel_diff,
    )

    assert np.isclose(parity_price, direct_price, rtol=5e-3, atol=5e-3), (
        f"parity {parity_price:.6f} vs direct {direct_price:.6f}"
    )


def test_pde_fd_barrier_european_ki_rebate_grid_matches_direct_near_spot():
    """Returned KI grids should include the rebate term near the spot node."""
    curve_r = DiscountCurve.flat(0.05, 2)
    curve_q = DiscountCurve.flat(0.03, 2)
    pricing_date = dt.datetime(2025, 1, 1)
    maturity = dt.datetime(2025, 12, 31)

    md = MarketData(pricing_date, curve_r, currency="USD")
    ud = UnderlyingData(
        initial_value=96.0,
        volatility=0.25,
        market_data=md,
        dividend_curve=curve_q,
    )
    barrier_spec = BarrierSpec(
        option_type=OptionType.PUT,
        exercise_type=ExerciseType.EUROPEAN,
        strike=100.0,
        maturity=maturity,
        barrier=105.0,
        direction=BarrierDirection.UP,
        action=BarrierAction.IN,
        monitoring=BarrierMonitoring.CONTINUOUS,
        rebate=50.0,
        rebate_timing=RebateTiming.AT_EXPIRY,
    )
    params = PDEParams(
        spot_steps=400,
        time_steps=400,
        method=PDEMethod.CRANK_NICOLSON,
        space_grid=PDESpaceGrid.LOG_SPOT,
        rannacher_steps=2,
    )

    valuation = OptionValuation(ud, barrier_spec, PricingMethod.PDE_FD, params=params)
    impl = _FDBarrierValuation(valuation)

    _, S_parity, V_parity, V_parity_prev, _ = impl._solve()
    _, S_direct, V_direct, V_direct_prev, _ = _fd_barrier_ki_core(**impl._base_solve_args())

    V_direct_on_parity = np.interp(S_parity, S_direct, V_direct)
    V_direct_prev_on_parity = np.interp(S_parity, S_direct, V_direct_prev)

    j = int(np.searchsorted(S_parity, float(ud.initial_value)))
    j = max(1, min(j, len(S_parity) - 2))
    window = slice(max(0, j - 3), min(len(S_parity), j + 4))

    assert np.allclose(V_parity[window], V_direct_on_parity[window], atol=0.002)
    assert np.allclose(V_parity_prev[window], V_direct_prev_on_parity[window], atol=0.002)


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param(
            {
                "spot": 100.0,
                "strike": 100.0,
                "volatility": 0.25,
                "rate": 0.05,
                "option_type": OptionType.CALL,
                "direction": BarrierDirection.DOWN,
                "barrier": 90.0,
                "rebate": 4.0,
                "num_observations": 12,
            },
            id="down_out_call_num_observations",
        ),
        pytest.param(
            {
                "spot": 100.0,
                "strike": 100.0,
                "volatility": 0.25,
                "rate": 0.05,
                "option_type": OptionType.PUT,
                "direction": BarrierDirection.UP,
                "barrier": 110.0,
                "rebate": 3.0,
                "monitoring_dates": [dt.datetime(2025, m, 28) for m in range(1, 13)],
            },
            id="up_out_put_monitoring_dates",
        ),
    ],
)
def test_pde_fd_discrete_ko_at_expiry_rebate_matches_direct_ki_parity(scenario: dict):
    """Discrete KO expiry rebates should satisfy in/out parity with a direct KI solve.

    This specifically exercises the maturity monitoring slice where the KO core
    resets the knocked-out region. With a non-zero rebate paid at expiry, the
    contract identity is:

        V_KO + V_KI = V_vanilla + rebate * df(T)

    We price KO through the public PDE path, KI through the direct two-surface
    core (to avoid the European-KI parity facade being circular), and vanilla
    with the same PDE engine. Before the maturity-reset fix, the KO leg missed
    the rebate on the terminal monitored slice and this assertion failed.
    """
    curve_r = flat_curve(PRICING_DATE, MATURITY, scenario["rate"])
    curve_q = flat_curve(PRICING_DATE, MATURITY, 0.01)
    md = market_data(pricing_date=PRICING_DATE, discount_curve=curve_r)
    ud = underlying(
        initial_value=scenario["spot"],
        volatility=scenario["volatility"],
        market_data=md,
        dividend_curve=curve_q,
    )
    ko_spec = BarrierSpec(
        option_type=scenario["option_type"],
        exercise_type=ExerciseType.EUROPEAN,
        strike=scenario["strike"],
        maturity=MATURITY,
        barrier=scenario["barrier"],
        direction=scenario["direction"],
        action=BarrierAction.OUT,
        monitoring=BarrierMonitoring.DISCRETE,
        rebate=scenario["rebate"],
        rebate_timing=RebateTiming.AT_EXPIRY,
        num_observations=scenario.get("num_observations"),
        monitoring_dates=scenario.get("monitoring_dates"),
    )
    ki_spec = dc_replace(ko_spec, action=BarrierAction.IN)
    vanilla_spec = VanillaSpec(
        option_type=scenario["option_type"],
        exercise_type=ExerciseType.EUROPEAN,
        strike=scenario["strike"],
        maturity=MATURITY,
    )
    params = PDEParams.for_barriers(monitoring=BarrierMonitoring.DISCRETE)

    ko_price = OptionValuation(ud, ko_spec, PricingMethod.PDE_FD, params=params).present_value()
    ki_valuation = OptionValuation(ud, ki_spec, PricingMethod.PDE_FD, params=params)
    ki_direct_price = float(
        _fd_barrier_ki_core(**_FDBarrierValuation(ki_valuation)._base_solve_args())[0]
    )
    vanilla_price = OptionValuation(
        ud, vanilla_spec, PricingMethod.PDE_FD, params=params
    ).present_value()

    ttm = calculate_year_fraction(
        PRICING_DATE,
        MATURITY,
        day_count_convention=md.day_count_convention,
    )
    rebate_leg = scenario["rebate"] * float(curve_r.df(ttm))
    ki_ko_sum = ko_price + ki_direct_price
    vanilla_rebate_sum = vanilla_price + rebate_leg
    abs_diff = ki_ko_sum - vanilla_rebate_sum
    rel_diff_pct = 100.0 * abs(abs_diff) / max(abs(vanilla_rebate_sum), 1.0e-12)

    monitoring_label = (
        f"num_obs={scenario['num_observations']}"
        if scenario.get("num_observations") is not None
        else f"monitoring_dates={len(scenario['monitoring_dates'])}"
    )
    logger.info(
        "Discrete KO expiry rebate parity %s/%s K=%g H=%g R=%g %s | "
        "ko=%.4f ki_direct=%.4f vanilla=%.4f rebate_leg=%.4f | "
        "ki_ko_sum=%.4f vanilla_rebate_sum=%.4f diff=%.4f rel=%.4f%%",
        scenario["option_type"].value,
        scenario["direction"].value,
        scenario["strike"],
        scenario["barrier"],
        scenario["rebate"],
        monitoring_label,
        ko_price,
        ki_direct_price,
        vanilla_price,
        rebate_leg,
        ki_ko_sum,
        vanilla_rebate_sum,
        abs_diff,
        rel_diff_pct,
    )

    assert np.isclose(
        ki_ko_sum,
        vanilla_rebate_sum,
        rtol=0.003,
        atol=1e-4,
    )


@pytest.mark.parametrize(
    "option_type,spot,strike,direction,barrier,rebate",
    [
        pytest.param(
            OptionType.CALL,
            100.0,
            100.0,
            BarrierDirection.DOWN,
            90.0,
            0.0,
            id="down_in_call_no_rebate",
        ),
        pytest.param(
            OptionType.PUT,
            100.0,
            100.0,
            BarrierDirection.UP,
            110.0,
            2.0,
            id="up_in_put_rebate",
        ),
        pytest.param(
            OptionType.PUT,
            100.0,
            100.0,
            BarrierDirection.DOWN,
            90.0,
            1.5,
            id="down_in_put_rebate",
        ),
    ],
)
def test_pde_fd_barrier_european_ki_facade_vs_direct_core_greeks(
    option_type: OptionType,
    spot: float,
    strike: float,
    direction: BarrierDirection,
    barrier: float,
    rebate: float,
):
    """European KI facade (parity) and direct two-surface core should agree on PV + greeks.

    The facade path for European KI goes through ``_compute_ki_components``
    (V_KI = V_vanilla − V_KO + rebate PV leg). The direct path calls
    ``_fd_barrier_ki_core(early_exercise=False)`` — the same two-surface coupled
    solver used for American KI but with exercise disabled. Both solve the same
    continuous barrier pricing problem via different numerics, so PV and grid
    greeks should match to within grid-refinement error.
    """
    curve_r = DiscountCurve.flat(0.05, 2)
    curve_q = DiscountCurve.flat(0.02, 2)
    pricing_date = dt.datetime(2025, 1, 1)
    maturity = dt.datetime(2025, 12, 31)

    md = MarketData(pricing_date, curve_r, currency="USD")
    ud = UnderlyingData(
        initial_value=spot,
        volatility=0.25,
        market_data=md,
        dividend_curve=curve_q,
    )
    barrier_spec = BarrierSpec(
        option_type=option_type,
        exercise_type=ExerciseType.EUROPEAN,
        strike=strike,
        maturity=maturity,
        barrier=barrier,
        direction=direction,
        action=BarrierAction.IN,
        monitoring=BarrierMonitoring.CONTINUOUS,
        rebate=rebate,
        rebate_timing=RebateTiming.AT_EXPIRY,
    )
    params = PDEParams.for_barriers(
        monitoring=BarrierMonitoring.CONTINUOUS,
        spot_steps=800,
        time_steps=800,
    )

    # Facade (parity) path
    valuation = OptionValuation(ud, barrier_spec, PricingMethod.PDE_FD, params=params)
    facade_pv = valuation.present_value()
    facade_delta = valuation.delta()
    facade_gamma = valuation.gamma()
    facade_theta = valuation.theta()

    # Direct two-surface core path (early_exercise=False)
    impl = _FDBarrierValuation(valuation)
    direct_result = _fd_barrier_ki_core(**impl._base_solve_args())
    direct_pv = float(direct_result[0])
    direct_delta = impl._grid_delta_from_result(direct_result, spot)
    direct_gamma = impl._grid_gamma_from_result(direct_result, spot)
    direct_theta = impl._grid_theta_from_result(direct_result, spot)

    logger.info(
        "KI facade-vs-direct [%s %s H=%g S=%g rebate=%g]: "
        "PV parity=%.6f direct=%.6f | "
        "Δ parity=%.6f direct=%.6f | "
        "Γ parity=%.6f direct=%.6f | "
        "Θ parity=%.6f direct=%.6f",
        option_type.value,
        direction.value,
        barrier,
        spot,
        rebate,
        facade_pv,
        direct_pv,
        facade_delta,
        direct_delta,
        facade_gamma,
        direct_gamma,
        facade_theta,
        direct_theta,
    )

    assert np.isclose(facade_pv, direct_pv, rtol=0.005, atol=1e-3)
    assert np.isclose(facade_delta, direct_delta, rtol=0.01, atol=1e-3)
    assert np.isclose(facade_gamma, direct_gamma, rtol=0.01, atol=1e-3)
    assert np.isclose(facade_theta, direct_theta, rtol=0.01, atol=1e-3)


@pytest.mark.parametrize(
    "option_type,spot,strike,lower_barrier,upper_barrier,rebate",
    [
        pytest.param(OptionType.CALL, 100.0, 100.0, 85.0, 120.0, 0.0, id="dki_call_no_rebate"),
        pytest.param(OptionType.PUT, 100.0, 100.0, 85.0, 120.0, 0.0, id="dki_put_no_rebate"),
        pytest.param(OptionType.CALL, 100.0, 95.0, 82.0, 125.0, 2.0, id="dki_call_rebate"),
        pytest.param(OptionType.PUT, 100.0, 105.0, 82.0, 125.0, 2.0, id="dki_put_rebate"),
    ],
)
def test_pde_fd_double_barrier_european_ki_facade_vs_direct_core_greeks(
    option_type: OptionType,
    spot: float,
    strike: float,
    lower_barrier: float,
    upper_barrier: float,
    rebate: float,
):
    """European double-KI facade (parity) and direct two-surface core agree on PV + greeks.

    The double-barrier analogue of
    ``test_pde_fd_barrier_european_ki_facade_vs_direct_core_greeks``.  The
    facade path prices European double knock-in via in-out parity
    (``V_DKI = V_vanilla + R·df_T − V_DKO``) and reads grid greeks off the
    reconstructed KI surface on the KO corridor grid.  The direct path calls
    ``_fd_double_barrier_ki_core(early_exercise=False)`` — the same two-surface
    coupled solver used for American double-KI, with exercise disabled — and
    reads grid greeks off its (full-domain) grid.  Both solve the same problem
    via different numerics, so PV and grid greeks should agree to within
    grid-refinement error.
    """
    curve_r = DiscountCurve.flat(0.05, 2)
    curve_q = DiscountCurve.flat(0.02, 2)
    pricing_date = dt.datetime(2025, 1, 1)
    maturity = dt.datetime(2025, 12, 31)

    md = MarketData(pricing_date, curve_r, currency="USD")
    ud = UnderlyingData(
        initial_value=spot,
        volatility=0.25,
        market_data=md,
        dividend_curve=curve_q,
    )
    spec = DoubleBarrierSpec(
        option_type=option_type,
        exercise_type=ExerciseType.EUROPEAN,
        strike=strike,
        maturity=maturity,
        lower_barrier=lower_barrier,
        upper_barrier=upper_barrier,
        action=BarrierAction.IN,
        monitoring=BarrierMonitoring.CONTINUOUS,
        rebate=rebate,
        rebate_timing=RebateTiming.AT_EXPIRY,
    )
    params = PDEParams(
        spot_steps=800,
        time_steps=800,
        method=PDEMethod.CRANK_NICOLSON,
        space_grid=PDESpaceGrid.LOG_SPOT,
        rannacher_steps=2,
    )

    # Facade (parity) path.
    valuation = OptionValuation(ud, spec, PricingMethod.PDE_FD, params=params)
    facade_pv = valuation.present_value()
    facade_delta = valuation.delta()
    facade_gamma = valuation.gamma()
    facade_theta = valuation.theta()

    # Direct two-surface core path (early_exercise=False).
    impl = _FDDoubleBarrierValuation(valuation)
    direct_result = _fd_double_barrier_ki_core(**impl._base_solve_args())
    _, S, V, _, last_dtau = direct_result
    direct_pv = float(direct_result[0])
    j = impl._spot_grid_index(S, spot)
    direct_delta = impl._grid_delta_at_spot(S, V, j, spot)
    direct_gamma = impl._grid_gamma_safe(S, V, j, spot)
    direct_theta = impl._grid_theta_bs_identity(S, V, j, spot, last_dtau)

    logger.info(
        "DoubleKI facade-vs-direct [%s K=%g L=%g U=%g rebate=%g]: "
        "PV parity=%.6f direct=%.6f | Δ parity=%.6f direct=%.6f | "
        "Γ parity=%.6f direct=%.6f | Θ parity=%.6f direct=%.6f",
        option_type.value,
        strike,
        lower_barrier,
        upper_barrier,
        rebate,
        facade_pv,
        direct_pv,
        facade_delta,
        direct_delta,
        facade_gamma,
        direct_gamma,
        facade_theta,
        direct_theta,
    )

    assert np.isclose(facade_pv, direct_pv, rtol=0.005, atol=1e-3)
    assert np.isclose(facade_delta, direct_delta, rtol=0.01, atol=1e-3)
    assert np.isclose(facade_gamma, direct_gamma, rtol=0.02, atol=1e-3)
    assert np.isclose(facade_theta, direct_theta, rtol=0.02, atol=1e-3)


@pytest.mark.slow
@pytest.mark.parametrize("scenario", _AMERICAN_BARRIER_SCENARIOS)
def test_pde_fd_barrier_equivalence_american(scenario):
    """Barrier PDE American robust variants should be in the same neighborhood.

    This test deliberately excludes the explicit schemes. American barrier
    problems combine a free boundary with a barrier boundary, and the explicit
    variants are materially less robust here than implicit/CN formulations.
    We therefore treat IMPLICIT / CRANK_NICOLSON across spot/log grids and
    intrinsic/PSOR exercise handling as the equivalence set.
    """
    baseline_by_solver = {
        solver: _barrier_pde_value(
            scenario,
            PDEParams(
                spot_steps=600,
                time_steps=600,
                method=PDEMethod.CRANK_NICOLSON,
                space_grid=PDESpaceGrid.LOG_SPOT,
                american_solver=solver,
                max_iter=20_000,
            ),
        )
        for solver in (PDEEarlyExercise.INTRINSIC, PDEEarlyExercise.GAUSS_SEIDEL)
    }

    for method in (
        PDEMethod.IMPLICIT,
        PDEMethod.CRANK_NICOLSON,
    ):
        for grid in (PDESpaceGrid.SPOT, PDESpaceGrid.LOG_SPOT):
            for solver in (PDEEarlyExercise.INTRINSIC, PDEEarlyExercise.GAUSS_SEIDEL):
                params = PDEParams(
                    spot_steps=600,
                    time_steps=600,
                    method=method,
                    space_grid=grid,
                    american_solver=solver,
                    max_iter=20_000,
                )

                pv = _barrier_pde_value(scenario, params)
                # American barrier PDEs are more sensitive than vanilla PDEs
                # even across robust formulations, so use a wider neighborhood
                # tolerance than the plain vanilla equivalence tests.
                assert np.isclose(pv, baseline_by_solver[solver], rtol=0.05)


@pytest.mark.slow
@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT], ids=["call", "put"])
def test_pde_fd_double_barrier_equivalence_american(option_type):
    """American double knock-out: cross-scheme convergence (no external benchmark).

    Unlike European double barriers (Boyle-Tian / Zvan-Vetzal-Forsyth) and
    American double knock-in (QuantLib binomial), American double *knock-out*
    has no usable external reference — QL's binomial is biased low and slow on
    KO greeks (both barriers truncate the payoff).  So we cross-validate our own
    FD schemes instead: the CN production default, an IMPLICIT solve, and
    EXPLICIT_HULL should agree on PV **and** greeks.

    CN and IMPLICIT share the same fine 1200-node corridor grid (only the time
    scheme differs) so they agree tightly.  EXPLICIT_HULL is stability-capped on
    the corridor-truncated continuous grid (``dz = corridor/spot_steps`` needs
    ``λ = dz/(σ√Δt) ≥ 1``), throttling its spatial resolution to
    ``≈ corridor·√(time_steps)/(σ√T)`` (~105 nodes even at 6000 time steps), so
    it tracks the CN reference a little less tightly (especially gamma) — but
    still closely.
    """
    spot, strike, sigma, rate, div = 100.0, 100.0, 0.25, 0.05, 0.02
    lower, upper = 85.0, 120.0
    ttm = calculate_year_fraction(PRICING_DATE, MATURITY)

    md = MarketData(PRICING_DATE, DiscountCurve.flat(rate), currency="USD")
    ud = UnderlyingData(
        initial_value=spot,
        volatility=sigma,
        market_data=md,
        dividend_curve=DiscountCurve.flat(div),
    )
    spec = DoubleBarrierSpec(
        option_type=option_type,
        exercise_type=ExerciseType.AMERICAN,
        strike=strike,
        maturity=MATURITY,
        lower_barrier=lower,
        upper_barrier=upper,
        action=BarrierAction.OUT,
        monitoring=BarrierMonitoring.CONTINUOUS,
    )

    def greeks(params: PDEParams | None = None) -> dict[str, float]:
        v = OptionValuation(ud, spec, PricingMethod.PDE_FD, params)
        return {
            "pv": float(v.present_value()),
            "delta": float(v.delta()),
            "gamma": float(v.gamma()),
            "theta": float(v.theta()),
        }

    cn = greeks()  # uses default CN params

    implicit = greeks(
        PDEParams.for_double_barriers(
            monitoring=BarrierMonitoring.CONTINUOUS,
            method=PDEMethod.IMPLICIT,
            # IMPLICIT is O(Δt) in time vs CN's O(Δt²); a few more steps brings
            # it closer to CN (GAUSS_SEIDEL PSOR, so kept modest for runtime).
            time_steps=1600,
        )
    )
    # EXPLICIT_HULL: keep spot_steps one under the corridor stability cap.
    ex_time_steps = 6000
    corridor = np.log(upper / lower)
    ex_spot_steps = int(corridor / (sigma * np.sqrt(ttm / ex_time_steps))) - 1
    explicit = greeks(
        PDEParams.for_double_barriers(
            monitoring=BarrierMonitoring.DISCRETE,  # → EXPLICIT_HULL + INTRINSIC
            spot_steps=ex_spot_steps,
            time_steps=ex_time_steps,
        )
    )

    for name, g in (("CN", cn), ("IMPLICIT", implicit), ("EXPLICIT_HULL", explicit)):
        logger.info(
            "AM DKO %s [%-13s]: pv=%.6f delta=%.6f gamma=%.6f theta=%.6f",
            option_type.value,
            name,
            g["pv"],
            g["delta"],
            g["gamma"],
            g["theta"],
        )

    # IMPLICIT shares CN's fine spatial grid (only the time scheme differs), so
    # they agree to ~1e-3 on PV and ~1e-5 on greeks — a tight band.
    assert np.isclose(implicit["pv"], cn["pv"], rtol=2e-3, atol=5e-4)
    assert np.isclose(implicit["delta"], cn["delta"], rtol=5e-3, atol=5e-4)
    assert np.isclose(implicit["gamma"], cn["gamma"], rtol=2e-2, atol=5e-4)
    assert np.isclose(implicit["theta"], cn["theta"], rtol=2e-2, atol=5e-4)
    # EXPLICIT_HULL is spatially throttled by stability (~105 corridor nodes vs
    # CN's 1200), so it tracks a touch less tightly (~0.07% PV) but still close.
    assert np.isclose(explicit["pv"], cn["pv"], rtol=2e-3, atol=1e-3)
    assert np.isclose(explicit["delta"], cn["delta"], rtol=1e-2, atol=1e-3)
    assert np.isclose(explicit["gamma"], cn["gamma"], rtol=5e-2, atol=1e-3)
    assert np.isclose(explicit["theta"], cn["theta"], rtol=3e-2, atol=1e-3)


def test_european_knock_in_grid_gamma_uses_native_surface_parity():
    """European KI grid gamma should follow vanilla-minus-KO parity."""
    r_curve = DiscountCurve.flat(0.05)
    q_curve = DiscountCurve.flat(0.02)
    md = MarketData(PRICING_DATE, r_curve, currency="USD")
    ud = UnderlyingData(
        initial_value=100.0,
        volatility=0.25,
        market_data=md,
        dividend_curve=q_curve,
    )
    params = PDEParams(spot_steps=400, time_steps=400)

    ki_spec = BarrierSpec(
        option_type=OptionType.CALL,
        exercise_type=ExerciseType.EUROPEAN,
        strike=105.0,
        maturity=MATURITY,
        barrier=115.0,
        direction=BarrierDirection.UP,
        action=BarrierAction.IN,
        monitoring=BarrierMonitoring.CONTINUOUS,
        rebate=0.0,
        rebate_timing=RebateTiming.AT_HIT,
    )
    ko_spec = BarrierSpec(
        option_type=OptionType.CALL,
        exercise_type=ExerciseType.EUROPEAN,
        strike=105.0,
        maturity=MATURITY,
        barrier=115.0,
        direction=BarrierDirection.UP,
        action=BarrierAction.OUT,
        monitoring=BarrierMonitoring.CONTINUOUS,
        rebate=0.0,
        rebate_timing=RebateTiming.AT_HIT,
    )
    vanilla_spec = spec(
        strike=105.0,
        option_type=OptionType.CALL,
        exercise=ExerciseType.EUROPEAN,
    )

    ki = OptionValuation(ud, ki_spec, PricingMethod.PDE_FD, params=params)
    ko = OptionValuation(ud, ko_spec, PricingMethod.PDE_FD, params=params)
    vanilla = OptionValuation(ud, vanilla_spec, PricingMethod.PDE_FD, params=params)

    gamma_grid = ki.gamma(greek_calc_method=GreekCalculationMethod.GRID)
    gamma_parity = vanilla.gamma(greek_calc_method=GreekCalculationMethod.GRID) - ko.gamma(
        greek_calc_method=GreekCalculationMethod.GRID
    )
    gamma_numerical = ki.gamma(greek_calc_method=GreekCalculationMethod.NUMERICAL)

    assert np.isclose(gamma_grid, gamma_parity, rtol=1.0e-4, atol=1.0e-4)
    assert np.isclose(gamma_grid, gamma_numerical, rtol=0.01, atol=1.0e-4)


def test_european_knock_in_grid_delta_uses_native_surface_parity():
    """European KI grid delta should follow vanilla-minus-KO parity."""
    r_curve = DiscountCurve.flat(0.05)
    q_curve = DiscountCurve.flat(0.02)
    md = MarketData(PRICING_DATE, r_curve, currency="USD")
    ud = UnderlyingData(
        initial_value=100.0,
        volatility=0.25,
        market_data=md,
        dividend_curve=q_curve,
    )
    params = PDEParams(spot_steps=2400, time_steps=800, space_grid=PDESpaceGrid.LOG_SPOT)

    ki_spec = BarrierSpec(
        option_type=OptionType.PUT,
        exercise_type=ExerciseType.EUROPEAN,
        strike=100.0,
        maturity=MATURITY,
        barrier=85.0,
        direction=BarrierDirection.DOWN,
        action=BarrierAction.IN,
        monitoring=BarrierMonitoring.CONTINUOUS,
        rebate=0.0,
        rebate_timing=RebateTiming.AT_HIT,
    )
    ko_spec = dc_replace(ki_spec, action=BarrierAction.OUT)

    vanilla_spec = spec(
        strike=100.0,
        option_type=OptionType.PUT,
        exercise=ExerciseType.EUROPEAN,
    )

    ki = OptionValuation(ud, ki_spec, PricingMethod.PDE_FD, params=params)
    ko = OptionValuation(ud, ko_spec, PricingMethod.PDE_FD, params=params)
    vanilla = OptionValuation(ud, vanilla_spec, PricingMethod.PDE_FD, params=params)

    delta_grid = ki.delta(greek_calc_method=GreekCalculationMethod.GRID)
    delta_parity = vanilla.delta(greek_calc_method=GreekCalculationMethod.GRID) - ko.delta(
        greek_calc_method=GreekCalculationMethod.GRID
    )
    delta_numerical = ki.delta(greek_calc_method=GreekCalculationMethod.NUMERICAL, epsilon=2.5)

    assert np.isclose(delta_grid, delta_parity, rtol=1.0e-4, atol=1.0e-4)
    assert np.isclose(delta_grid, delta_numerical, rtol=0.01, atol=1.0e-4)


def test_european_knock_in_grid_theta_uses_native_surface_parity():
    """European KI grid theta should follow vanilla-plus-rebate-minus-KO parity."""
    r_curve = DiscountCurve.flat(0.05)
    q_curve = DiscountCurve.flat(0.02)
    md = MarketData(PRICING_DATE, r_curve, currency="USD")
    ud = UnderlyingData(
        initial_value=100.0,
        volatility=0.25,
        market_data=md,
        dividend_curve=q_curve,
    )
    params = PDEParams(spot_steps=400, time_steps=400)

    ki_spec = BarrierSpec(
        option_type=OptionType.PUT,
        exercise_type=ExerciseType.EUROPEAN,
        strike=100.0,
        maturity=MATURITY,
        barrier=85.0,
        direction=BarrierDirection.DOWN,
        action=BarrierAction.IN,
        monitoring=BarrierMonitoring.CONTINUOUS,
        rebate=3.0,
        rebate_timing=RebateTiming.AT_EXPIRY,
    )

    ko_spec = dc_replace(ki_spec, action=BarrierAction.OUT)

    vanilla_spec = spec(
        strike=100.0,
        option_type=OptionType.PUT,
        exercise=ExerciseType.EUROPEAN,
    )

    ki = OptionValuation(ud, ki_spec, PricingMethod.PDE_FD, params=params)
    ko = OptionValuation(ud, ko_spec, PricingMethod.PDE_FD, params=params)
    vanilla = OptionValuation(ud, vanilla_spec, PricingMethod.PDE_FD, params=params)

    theta_grid = ki.theta(greek_calc_method=GreekCalculationMethod.GRID)
    theta_numerical = ki.theta(
        greek_calc_method=GreekCalculationMethod.NUMERICAL, time_bump_days=21.0
    )

    ttm = calculate_year_fraction(PRICING_DATE, MATURITY)
    rebate_pv = ki_spec.rebate * float(r_curve.df(ttm))
    rebate_theta = rebate_pv * 0.05 / 365.0
    theta_parity = (
        vanilla.theta(greek_calc_method=GreekCalculationMethod.GRID)
        + rebate_theta
        - ko.theta(greek_calc_method=GreekCalculationMethod.GRID)
    )

    assert np.isclose(theta_grid, theta_parity, rtol=1.0e-4, atol=1.0e-4)
    assert np.isclose(theta_grid, theta_numerical, rtol=0.02, atol=1.0e-4)


@pytest.mark.parametrize(
    ("monitoring", "monitoring_dates"),
    [
        (BarrierMonitoring.CONTINUOUS, None),
        (BarrierMonitoring.DISCRETE, (PRICING_DATE, MATURITY)),
    ],
)
def test_knock_out_triggered_at_inception_grid_greeks_zero_without_rebate(
    monitoring, monitoring_dates
):
    """PDE KO greeks should collapse to zero when the contract is already dead."""
    ud = underlying(initial_value=120.0)
    params = PDEParams(spot_steps=300, time_steps=300)
    ko_spec = BarrierSpec(
        option_type=OptionType.CALL,
        exercise_type=ExerciseType.EUROPEAN,
        strike=100.0,
        maturity=MATURITY,
        barrier=120.0,
        direction=BarrierDirection.UP,
        action=BarrierAction.OUT,
        monitoring=monitoring,
        monitoring_dates=monitoring_dates,
        rebate=0.0,
        rebate_timing=RebateTiming.AT_HIT,
    )

    valuation = OptionValuation(ud, ko_spec, PricingMethod.PDE_FD, params=params)

    assert np.isclose(valuation.present_value(), 0.0, atol=1.0e-12)
    assert np.isclose(valuation.delta(), 0.0, atol=1.0e-12)
    assert np.isclose(valuation.gamma(), 0.0, atol=1.0e-12)
    assert np.isclose(valuation.theta(), 0.0, atol=1.0e-12)
    assert np.isclose(valuation.rho(), 0.0, atol=1.0e-12)


@pytest.mark.parametrize(
    ("monitoring", "monitoring_dates"),
    [
        (BarrierMonitoring.CONTINUOUS, None),
        (BarrierMonitoring.DISCRETE, (PRICING_DATE, MATURITY)),
    ],
)
def test_knock_out_triggered_at_inception_grid_greeks_match_fixed_expiry_rebate(
    monitoring, monitoring_dates
):
    """PDE KO greeks should match the resolved expiry rebate when already triggered."""
    rate = 0.05
    md = MarketData(PRICING_DATE, flat_curve(PRICING_DATE, MATURITY, rate), currency="USD")
    ud = underlying(initial_value=120.0, market_data=md)
    params = PDEParams(spot_steps=300, time_steps=300)
    rebate = 5.0
    ko_spec = BarrierSpec(
        option_type=OptionType.CALL,
        exercise_type=ExerciseType.EUROPEAN,
        strike=100.0,
        maturity=MATURITY,
        barrier=120.0,
        direction=BarrierDirection.UP,
        action=BarrierAction.OUT,
        monitoring=monitoring,
        monitoring_dates=monitoring_dates,
        rebate=rebate,
        rebate_timing=RebateTiming.AT_EXPIRY,
    )

    valuation = OptionValuation(ud, ko_spec, PricingMethod.PDE_FD, params=params)
    ttm = calculate_year_fraction(PRICING_DATE, MATURITY)
    discount_factor = float(md.discount_curve.df(ttm))
    expected_pv = rebate * discount_factor
    expected_theta = expected_pv * rate / 365.0
    expected_rho = -expected_pv * ttm * 0.01

    assert np.isclose(valuation.present_value(), expected_pv, rtol=1.0e-12, atol=1.0e-12)
    assert np.isclose(valuation.delta(), 0.0, atol=1.0e-12)
    assert np.isclose(valuation.gamma(), 0.0, atol=1.0e-12)
    assert np.isclose(valuation.theta(), expected_theta, rtol=0.01, atol=1.0e-6)
    assert np.isclose(valuation.rho(), expected_rho, rtol=0.02, atol=1.0e-6)


def test_knock_in_triggered_at_inception_grid_greeks_match_vanilla():
    """Triggered-at-inception KI should reduce to vanilla for PDE grid greeks."""
    q_curve = flat_curve(PRICING_DATE, MATURITY, 0.02)
    ud = underlying(initial_value=120.0, dividend_curve=q_curve)
    params = PDEParams(spot_steps=400, time_steps=400)
    ki_spec = BarrierSpec(
        option_type=OptionType.CALL,
        exercise_type=ExerciseType.EUROPEAN,
        strike=100.0,
        maturity=MATURITY,
        barrier=115.0,
        direction=BarrierDirection.UP,
        action=BarrierAction.IN,
        monitoring=BarrierMonitoring.CONTINUOUS,
        rebate=4.0,
        rebate_timing=RebateTiming.AT_EXPIRY,
    )
    vanilla_spec = spec(
        strike=100.0,
        option_type=OptionType.CALL,
        exercise=ExerciseType.EUROPEAN,
    )

    ki = OptionValuation(ud, ki_spec, PricingMethod.PDE_FD, params=params)
    vanilla = OptionValuation(ud, vanilla_spec, PricingMethod.PDE_FD, params=params)

    assert np.isclose(ki.present_value(), vanilla.present_value(), rtol=1.0e-6, atol=1.0e-6)
    assert np.isclose(ki.delta(), vanilla.delta(), rtol=1.0e-6, atol=1.0e-6)
    assert np.isclose(ki.gamma(), vanilla.gamma(), rtol=1.0e-6, atol=1.0e-6)
    assert np.isclose(ki.theta(), vanilla.theta(), rtol=1.0e-6, atol=1.0e-6)
    assert np.isclose(ki.rho(), vanilla.rho(), rtol=1.0e-6, atol=1.0e-6)


# ═══════════════════════════════════════════════════════════════════════════
# American flat barrier vega — PDE FD scheme/grid self-consistency
# ═══════════════════════════════════════════════════════════════════════════
# American barrier vega has no cross-engine reference (binomial blocked,
# BSM is European-only, QL barrier engines don't expose vega).  We
# self-consistency-check PDE FD across scheme (CN vs IMPLICIT) and grid
# topology (LOG_SPOT vs SPOT) at a high resolution (2400×800).  All three
# configs should agree to a few percent.

_AM_BARRIER_VEGA_RHO_SCENARIOS = [
    # (direction, action, option_type, strike, barrier, rebate, rebate_timing, id)
    (
        BarrierDirection.DOWN,
        BarrierAction.OUT,
        OptionType.CALL,
        100.0,
        85.0,
        5.0,
        RebateTiming.AT_HIT,
        "am_down_out_call_rebate",
    ),
    (
        BarrierDirection.UP,
        BarrierAction.OUT,
        OptionType.PUT,
        100.0,
        120.0,
        5.0,
        RebateTiming.AT_HIT,
        "am_up_out_put_rebate",
    ),
    (
        BarrierDirection.DOWN,
        BarrierAction.IN,
        OptionType.CALL,
        100.0,
        85.0,
        5.0,
        RebateTiming.AT_EXPIRY,
        "am_down_in_call_rebate",
    ),
    (
        BarrierDirection.UP,
        BarrierAction.IN,
        OptionType.PUT,
        100.0,
        120.0,
        5.0,
        RebateTiming.AT_EXPIRY,
        "am_up_in_put_rebate",
    ),
]

_AM_VEGA_RHO_SPOT = 100.0
_AM_VEGA_RHO_VOL = 0.25
_AM_VEGA_RHO_RATE = 0.05
_AM_VEGA_RHO_DIV = 0.02


def _build_am_barrier_valuation(
    *, direction, action, option_type, strike, barrier, rebate, rebate_timing, params
) -> OptionValuation:
    r_curve = flat_curve(PRICING_DATE, MATURITY, _AM_VEGA_RHO_RATE)
    q_curve = flat_curve(PRICING_DATE, MATURITY, _AM_VEGA_RHO_DIV)
    md = market_data(pricing_date=PRICING_DATE, discount_curve=r_curve)
    ud = underlying(
        initial_value=_AM_VEGA_RHO_SPOT,
        volatility=_AM_VEGA_RHO_VOL,
        market_data=md,
        dividend_curve=q_curve,
    )
    barrier_spec = BarrierSpec(
        option_type=option_type,
        exercise_type=ExerciseType.AMERICAN,
        strike=strike,
        maturity=MATURITY,
        barrier=barrier,
        direction=direction,
        action=action,
        monitoring=BarrierMonitoring.CONTINUOUS,
        rebate=rebate,
        rebate_timing=rebate_timing,
    )
    return OptionValuation(ud, barrier_spec, PricingMethod.PDE_FD, params=params)


@pytest.mark.slow
@pytest.mark.parametrize(
    "direction,action,option_type,strike,barrier,rebate,rebate_timing",
    [pytest.param(*p[:-1], id=p[-1]) for p in _AM_BARRIER_VEGA_RHO_SCENARIOS],
)
def test_american_barrier_vega_pde_scheme_grid_consistency(
    direction, action, option_type, strike, barrier, rebate, rebate_timing
):
    """American flat barrier vega should be consistent across PDE FD
    scheme (CN vs IMPLICIT) and grid topology (LOG_SPOT vs SPOT) at high
    resolution (2400×800).  This is the only available cross-validation
    for American barrier vega — no other engine in the library supports
    it (binomial barrier vega is blocked by the Boyle-Lau guard, BSM is
    European-only, QL barrier engines don't expose vega).
    """
    spot_steps = 2400
    time_steps = 800

    cn_log = PDEParams(
        spot_steps=spot_steps,
        time_steps=time_steps,
        method=PDEMethod.CRANK_NICOLSON,
        space_grid=PDESpaceGrid.LOG_SPOT,
    )
    implicit_log = PDEParams(
        spot_steps=spot_steps,
        time_steps=time_steps,
        method=PDEMethod.IMPLICIT,
        space_grid=PDESpaceGrid.LOG_SPOT,
    )
    cn_spot = PDEParams(
        spot_steps=spot_steps,
        time_steps=time_steps,
        method=PDEMethod.CRANK_NICOLSON,
        space_grid=PDESpaceGrid.SPOT,
    )

    common = dict(
        direction=direction,
        action=action,
        option_type=option_type,
        strike=strike,
        barrier=barrier,
        rebate=rebate,
        rebate_timing=rebate_timing,
    )

    ov_cn_log = _build_am_barrier_valuation(params=cn_log, **common)
    ov_implicit_log = _build_am_barrier_valuation(params=implicit_log, **common)
    ov_cn_spot = _build_am_barrier_valuation(params=cn_spot, **common)

    vega_cn_log = ov_cn_log.vega()
    vega_implicit_log = ov_implicit_log.vega()
    vega_cn_spot = ov_cn_spot.vega()

    # Reference: CN + LOG_SPOT (the default high-quality config). The other
    # two configurations should agree closely.
    vega_rtol, vega_atol = 0.01, 1e-3

    assert np.isclose(vega_implicit_log, vega_cn_log, rtol=vega_rtol, atol=vega_atol), (
        f"vega CN_LOG={vega_cn_log:.6f} vs IMPLICIT_LOG={vega_implicit_log:.6f}"
    )
    assert np.isclose(vega_cn_spot, vega_cn_log, rtol=vega_rtol, atol=vega_atol), (
        f"vega CN_LOG={vega_cn_log:.6f} vs CN_SPOT={vega_cn_spot:.6f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# spot_steps=None ("auto") resolution
# ─────────────────────────────────────────────────────────────────────────────
class TestResolvePdeSpotSteps:
    """Unit tests for ``_resolve_pde_spot_steps``.

    Auto log-step is ``dz = lambda * sigma * sqrt(dt)`` — lambda = sqrt(3)
    (Hull, stability-pinned) for the explicit family, lambda = 1/2 (accuracy
    choice, floored at 200) for CN/IMPLICIT.  Spot (non-log) grids fall back
    to the fixed default 200.  Reference numbers below use sigma=0.2, T=0.5,
    spot=strike=100, smax_mult=4 => vanilla log-span = ln(16) ~= 2.772589.
    """

    SPOT = 100.0
    STRIKE = 100.0
    VOL = 0.20
    TTM = 0.5

    def _resolve(self, params, spec_obj=None):
        if spec_obj is None:
            spec_obj = VanillaSpec(
                option_type=OptionType.CALL,
                exercise_type=ExerciseType.EUROPEAN,
                strike=self.STRIKE,
                maturity=MATURITY,
            )
        return _resolve_pde_spot_steps(
            spec=spec_obj,
            spot=self.SPOT,
            strike=self.STRIKE,
            volatility=self.VOL,
            time_to_maturity=self.TTM,
            params=params,
        )

    def _dko_spec(self):
        return DoubleBarrierSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.STRIKE,
            maturity=MATURITY,
            lower_barrier=95.0,
            upper_barrier=125.0,
            action=BarrierAction.OUT,
            monitoring=BarrierMonitoring.CONTINUOUS,
        )

    def test_int_spot_steps_honored_verbatim(self):
        p = PDEParams(spot_steps=500, space_grid=PDESpaceGrid.LOG_SPOT)
        assert self._resolve(p) == 500

    def test_bare_defaults_resolve_to_200(self):
        """Bare PDEParams() (CN + SPOT grid) keeps the historical 200 grid."""
        assert self._resolve(PDEParams()) == 200

    def test_cn_spot_grid_falls_back_to_200(self):
        p = PDEParams(spot_steps=None, time_steps=800, space_grid=PDESpaceGrid.SPOT)
        assert self._resolve(p) == 200

    def test_cn_log_auto_uses_half_lambda(self):
        """dz = 0.5*0.2*sqrt(0.5/800) = 0.0025 -> ceil(ln(16)/dz) = 1110."""
        p = PDEParams(spot_steps=None, time_steps=800, space_grid=PDESpaceGrid.LOG_SPOT)
        assert self._resolve(p) == 1110

    def test_cn_log_auto_floored_at_200(self):
        """Absurdly coarse time grid (4 steps) backs out ~79 -> floored to 200."""
        p = PDEParams(spot_steps=None, time_steps=4, space_grid=PDESpaceGrid.LOG_SPOT)
        assert self._resolve(p) == 200

    def test_cn_log_auto_refines_with_time_steps(self):
        """Auto-CN grids scale with sqrt(time_steps): 4x steps -> 2x finer space."""
        p_1x = PDEParams(spot_steps=None, time_steps=800, space_grid=PDESpaceGrid.LOG_SPOT)
        p_4x = PDEParams(spot_steps=None, time_steps=3200, space_grid=PDESpaceGrid.LOG_SPOT)
        n_1x, n_4x = self._resolve(p_1x), self._resolve(p_4x)
        assert np.isclose(n_4x / n_1x, 2.0, rtol=2.0e-3)

    def test_explicit_hull_log_auto_unchanged(self):
        """dz_hull = 0.2*sqrt(3*0.5/3000) ~= 0.0044721 -> ceil(ln(16)/dz) = 620."""
        p = PDEParams(
            spot_steps=None,
            time_steps=3000,
            method=PDEMethod.EXPLICIT_HULL,
            space_grid=PDESpaceGrid.LOG_SPOT,
        )
        assert self._resolve(p) == 620

    def test_cn_log_barrier_level_widens_span(self):
        """An UP barrier at 150 stretches the covered span: ln(600/25) -> 1272 steps."""
        barrier_spec = BarrierSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.STRIKE,
            maturity=MATURITY,
            barrier=150.0,
            direction=BarrierDirection.UP,
            action=BarrierAction.OUT,
            monitoring=BarrierMonitoring.CONTINUOUS,
        )
        p = PDEParams(spot_steps=None, time_steps=800, space_grid=PDESpaceGrid.LOG_SPOT)
        assert self._resolve(p, spec_obj=barrier_spec) == 1272

    def test_cn_continuous_double_barrier_corridor_floored(self):
        """Corridor ln(125/95)/0.0025 ~= 110 -> floored to 200 (CN is stable; finer is safe)."""
        p = PDEParams(spot_steps=None, time_steps=800, space_grid=PDESpaceGrid.LOG_SPOT)
        assert self._resolve(p, spec_obj=self._dko_spec()) == 200

    def test_cn_continuous_double_barrier_corridor_unfloored_when_fine(self):
        """time_steps=80000 -> dz=2.5e-4 -> round(corridor/dz) = 1098 (> floor)."""
        p = PDEParams(spot_steps=None, time_steps=80_000, space_grid=PDESpaceGrid.LOG_SPOT)
        assert self._resolve(p, spec_obj=self._dko_spec()) == 1098

    def test_explicit_continuous_double_barrier_corridor_not_floored(self):
        """Explicit corridor counts stay stability-pinned (~61), NOT floored to 200:
        flooring the count up would shrink dz below Hull spacing and break lambda >= 1."""
        p = PDEParams(
            spot_steps=None,
            time_steps=3000,
            method=PDEMethod.EXPLICIT_HULL,
            space_grid=PDESpaceGrid.LOG_SPOT,
        )
        assert self._resolve(p, spec_obj=self._dko_spec()) == 61


# ─────────────────────────────────────────────────────────────────────────────
# Auto-grid knock-in: separate (frozen) vanilla-leg node count
# ─────────────────────────────────────────────────────────────────────────────
class TestKnockInParityVanillaSpotSteps:
    """Auto-grid European KI prices the vanilla parity leg on its own grid.

    European KI is priced by in-out parity (V_KI = V_vanilla - V_KO).  The
    vanilla leg lives on the full free-far-field domain, not the corridor-
    truncated KO grid, so under ``spot_steps=None`` ``OptionValuation`` freezes a
    separate ``parity_vanilla_spot_steps`` for it (see ``_freeze_pde_grid``).
    Without it, a continuous double-KI's vanilla leg inherits the corridor-
    pinned / 200-floored count and is badly under-resolved.
    """

    PD = dt.datetime(2025, 1, 1)
    MAT = PD + dt.timedelta(days=365)
    SPOT = STRIKE = 100.0
    VOL = 0.20
    LOWER, UPPER = 85.0, 125.0

    def _ud(self):
        mkt = MarketData(
            self.PD,
            DiscountCurve.flat(0.05),
            currency="USD",
            day_count_convention=DayCountConvention.ACT_365F,
        )
        return UnderlyingData(
            initial_value=self.SPOT,
            volatility=self.VOL,
            market_data=mkt,
            dividend_curve=DiscountCurve.flat(0.02),
        )

    def _dspec(self, action):
        return DoubleBarrierSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.STRIKE,
            maturity=self.MAT,
            lower_barrier=self.LOWER,
            upper_barrier=self.UPPER,
            action=action,
            monitoring=BarrierMonitoring.CONTINUOUS,
        )

    @staticmethod
    def _auto():
        return PDEParams(spot_steps=None, time_steps=400, space_grid=PDESpaceGrid.LOG_SPOT)

    def test_freeze_resolves_separate_vanilla_count_for_ki(self):
        """KI freezes a larger free-far-field vanilla count than the corridor pin."""
        ov = OptionValuation(
            self._ud(), self._dspec(BarrierAction.IN), PricingMethod.PDE_FD, params=self._auto()
        )
        assert ov._params.spot_steps == 200  # corridor count floored
        assert ov._params.parity_vanilla_spot_steps == 555  # free-far-field over ln(16)

    def test_freeze_no_vanilla_count_for_ko(self):
        """KO is a single solve — no parity vanilla leg, so the field stays None."""
        ov = OptionValuation(
            self._ud(), self._dspec(BarrierAction.OUT), PricingMethod.PDE_FD, params=self._auto()
        )
        assert ov._params.parity_vanilla_spot_steps is None

    def test_explicit_spot_steps_leaves_vanilla_count_none(self):
        """Explicit spot_steps is honored verbatim on every leg (field stays None)."""
        expl = PDEParams(spot_steps=400, time_steps=400, space_grid=PDESpaceGrid.LOG_SPOT)
        ov = OptionValuation(
            self._ud(), self._dspec(BarrierAction.IN), PricingMethod.PDE_FD, params=expl
        )
        assert ov._params.spot_steps == 400
        assert ov._params.parity_vanilla_spot_steps is None

    def test_user_set_vanilla_count_honored_with_auto_spot_steps(self):
        """An explicit parity_vanilla_spot_steps survives the auto-grid freeze."""
        params = PDEParams(
            spot_steps=None,
            parity_vanilla_spot_steps=640,
            time_steps=400,
            space_grid=PDESpaceGrid.LOG_SPOT,
        )
        ov = OptionValuation(
            self._ud(), self._dspec(BarrierAction.IN), PricingMethod.PDE_FD, params=params
        )
        assert ov._params.spot_steps == 200  # corridor still auto-resolved
        assert ov._params.parity_vanilla_spot_steps == 640  # user value not clobbered

    def test_user_set_vanilla_count_honored_with_explicit_spot_steps(self):
        """parity_vanilla_spot_steps overrides the vanilla leg independently of spot_steps."""
        params = PDEParams(
            spot_steps=300,
            parity_vanilla_spot_steps=900,
            time_steps=400,
            space_grid=PDESpaceGrid.LOG_SPOT,
        )
        ov = OptionValuation(
            self._ud(), self._dspec(BarrierAction.IN), PricingMethod.PDE_FD, params=params
        )
        assert ov._params.spot_steps == 300
        assert ov._params.parity_vanilla_spot_steps == 900

    def test_auto_ki_accurate_against_bsm(self):
        """Auto-grid double-KI tracks the Kunitomo-Ikeda closed form to <2e-4 rel.

        Pre-fix the vanilla leg ran at the 200-floored corridor count and the
        relative error was ~5.8e-3; resolving it at the free-far-field count
        brings it to ~9e-5.
        """
        ud = self._ud()
        truth = OptionValuation(
            ud, self._dspec(BarrierAction.IN), PricingMethod.BSM
        ).present_value()
        pv = OptionValuation(
            ud, self._dspec(BarrierAction.IN), PricingMethod.PDE_FD, params=self._auto()
        ).present_value()
        assert np.isclose(pv, truth, rtol=2e-4, atol=0.0), (
            f"auto double-KI {pv:.6f} vs BSM {truth:.6f} (rel {abs(pv - truth) / truth:.2e})"
        )
