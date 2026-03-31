"""PDE FD grid/method/solver equivalence tests.

Verifies that different PDE finite-difference schemes (Explicit, Implicit,
Crank-Nicolson, Explicit-Hull) and space grids (SPOT, LOG_SPOT) produce
consistent prices.  These are internal engine tests — cross-method and
QuantLib comparisons live in test_quantlib_comparison.py.
"""

import numpy as np
import pytest

from derivatives_pricing.exceptions import StabilityError, UnsupportedFeatureError
from derivatives_pricing.enums import (
    BarrierAction,
    BarrierDirection,
    BarrierMonitoring,
    ExerciseType,
    GreekCalculationMethod,
    OptionType,
    PDEEarlyExercise,
    PDEMethod,
    PDESpaceGrid,
    PricingMethod,
    RebateTiming,
)
from derivatives_pricing.rates import DiscountCurve
from derivatives_pricing.valuation.contracts import BarrierSpec, PayoffSpec
from helpers import (
    flat_curve,
    market_data,
    underlying,
    spec,
    PRICING_DATE,
    MATURITY,
)
from derivatives_pricing.valuation import OptionValuation
from derivatives_pricing.valuation.params import PDEParams


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
    base_params = PDEParams(spot_steps=180, time_steps=260)
    baseline = _barrier_pde_value(scenario, base_params)

    for method in (
        PDEMethod.IMPLICIT,
        PDEMethod.EXPLICIT,
        PDEMethod.EXPLICIT_HULL,
        PDEMethod.CRANK_NICOLSON,
    ):
        for grid in (PDESpaceGrid.SPOT, PDESpaceGrid.LOG_SPOT):
            params = PDEParams(
                spot_steps=180,
                time_steps=260,
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
                    _barrier_pde_value(scenario, params)
                continue

            pv = _barrier_pde_value(scenario, params)
            assert np.isclose(pv, baseline, rtol=0.015)


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
