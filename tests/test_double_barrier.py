"""Double-barrier PDE_FD pricing tests against Boyle-Tian (1998) reference values.

Boyle & Tian (1998) "An explicit finite difference approach to the pricing of
barrier options", Applied Mathematical Finance, 5:1, 17-43.

``TestDoubleBarrierAgainstBoyleTian`` covers continuous-monitoring double
knock-out CALLS with the paper's standard corridor:

    K = 100, sigma = 25%, r = 10% (continuously compounded), q = 0,
    lower barrier L = 90, upper barrier U = 140.

``TestDoubleBarrierMonitoringFrequencyAgainstTian`` covers *discretely*
monitored double knock-out calls across six monitoring frequencies — the
double-barrier analogue of the paper's Table 8 single-barrier DOC sweep.

Reference values are the paper's converged finite-difference figures (4dp).
Our PDE_FD engine (Crank-Nicolson on a log-spot grid truncated at both
barriers for continuous monitoring; Boyle-Tian half-step placement of both
barriers on a full grid for discrete monitoring) is itself validated to
~1e-6 against QuantLib's ``AnalyticDoubleBarrierEngine`` (see
``scripts/double_barrier_example.py``), so the tolerances here absorb the
paper's 4dp rounding plus the small difference between its modified-explicit
scheme and ours.
"""

import datetime as dt
import logging

import numpy as np
import pytest

from derivatives_pricing.enums import (
    BarrierAction,
    BarrierMonitoring,
    DayCountConvention,
    ExerciseType,
    OptionType,
    PricingMethod,
)
from derivatives_pricing.market_environment import MarketData
from derivatives_pricing.rates import DiscountCurve
from derivatives_pricing.valuation import OptionValuation, UnderlyingData
from derivatives_pricing.valuation.contracts import DoubleBarrierSpec
from derivatives_pricing.utils import calculate_year_fraction

logger = logging.getLogger(__name__)


class TestDoubleBarrierAgainstBoyleTian:
    """Continuous double knock-out call PVs vs Boyle-Tian (1998)."""

    PRICING_DATE = dt.datetime(2025, 1, 1)
    STRIKE = 100.0
    SIGMA = 0.25
    RATE = 0.10
    LOWER_BARRIER = 90.0
    UPPER_BARRIER = 140.0

    @classmethod
    def _market_data(cls) -> MarketData:
        return MarketData(
            cls.PRICING_DATE,
            DiscountCurve.flat(cls.RATE, 2.0),
            currency="USD",
            day_count_convention=DayCountConvention.ACT_365F,
        )

    @classmethod
    def _underlying(cls, spot: float) -> UnderlyingData:
        return UnderlyingData(
            initial_value=spot,
            volatility=cls.SIGMA,
            market_data=cls._market_data(),
            dividend_curve=DiscountCurve.flat(0.0, 2.0),
        )

    @classmethod
    def _maturity(cls, t_years: float) -> dt.datetime:
        """Maturity datetime whose ACT/365F year fraction is exactly ``t_years``."""
        maturity = cls.PRICING_DATE + dt.timedelta(days=365.0 * t_years)
        assert np.isclose(
            calculate_year_fraction(cls.PRICING_DATE, maturity, DayCountConvention.ACT_365F),
            t_years,
        ), "maturity helper must reproduce the requested year fraction"
        return maturity

    @classmethod
    def _double_ko_call_pv(cls, spot: float, t_years: float) -> float:
        spec = DoubleBarrierSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=cls.STRIKE,
            maturity=cls._maturity(t_years),
            lower_barrier=cls.LOWER_BARRIER,
            upper_barrier=cls.UPPER_BARRIER,
            action=BarrierAction.OUT,
            monitoring=BarrierMonitoring.CONTINUOUS,
        )
        valuation = OptionValuation(cls._underlying(spot), spec, PricingMethod.PDE_FD)
        return float(valuation.present_value())

    # ── Tables 2 & 3: S0 = 95, double-KO call across maturities ──────────
    @pytest.mark.parametrize(
        "t_years,paper_pv",
        [
            pytest.param(1.0, 1.4581, id="T_1y"),
            pytest.param(1.0 / 12.0, 1.1515, id="T_1mo"),
            pytest.param(1.0 / 24.0, 0.4759, id="T_half_mo"),
        ],
    )
    def test_double_ko_call_maturity_sweep(
        self, t_years: float, paper_pv: float, request: pytest.FixtureRequest
    ):
        """DKO call at S0=95 matches the paper across 1y / 1mo / half-month."""
        pv = self._double_ko_call_pv(95.0, t_years)
        logger.info(
            "BT98 DoubleKO maturity %s | T=%.4fy paper=%.4f dp_fd=%.4f diff=%.4f",
            request.node.callspec.id,
            t_years,
            paper_pv,
            pv,
            abs(pv - paper_pv),
        )
        assert np.isclose(pv, paper_pv, rtol=0.0, atol=2.0e-3), (
            f"DKO call T={t_years:.4f}y: got {pv:.6f}, expected {paper_pv:.4f}"
        )

    # ── Table 5: T = 1y, double-KO call as spot approaches the lower barrier ─
    @pytest.mark.parametrize(
        "spot,paper_pv",
        [
            pytest.param(92.0, 0.6262, id="S_92_0"),
            pytest.param(91.0, 0.3196, id="S_91_0"),
            pytest.param(90.5, 0.1613, id="S_90_5"),
            pytest.param(90.4, 0.1293, id="S_90_4"),
            pytest.param(90.3, 0.0972, id="S_90_3"),
            pytest.param(90.2, 0.0649, id="S_90_2"),
        ],
    )
    def test_double_ko_call_near_lower_barrier(
        self, spot: float, paper_pv: float, request: pytest.FixtureRequest
    ):
        """DKO call (T=1y) matches Table 5 as spot nears the lower barrier (90)."""
        pv = self._double_ko_call_pv(spot, 1.0)
        logger.info(
            "BT98 DoubleKO Table5 %s | spot=%.1f paper=%.4f dp_fd=%.4f diff=%.4f",
            request.node.callspec.id,
            spot,
            paper_pv,
            pv,
            abs(pv - paper_pv),
        )
        assert np.isclose(pv, paper_pv, rtol=0.0, atol=3.0e-3), (
            f"DKO call S0={spot}: got {pv:.6f}, expected {paper_pv:.4f}"
        )


class TestDoubleBarrierMonitoringFrequencyAgainstTian:
    """Discretely-monitored double-KO call PVs across monitoring frequencies.

    The double-barrier analogue of Boyle-Tian's Table 8 single-barrier DOC
    sweep: same setup (S0 = K = 100, sigma = 20%, T = 0.5 yr, r = 10%, q = 0,
    lower barrier H = 95) but with an added upper barrier U = 140, making the
    contract a double knock-out call.  Tian's reference PVs sweep six
    monitoring frequencies under the Cheuk-Vorst (1994) convention
    (1 yr = 4 q = 12 m = 52 w = 250 trading days = 1000 trading hours), here
    scaled to the half-year maturity.

    Our half-step double-barrier scheme (``PDEMethod.EXPLICIT_HULL`` by default
    for discrete monitoring; CONTINUOUS uses Crank-Nicolson) matches every
    frequency to <=0.15% **except** the hourly row, where the paper's reference
    (4.7536) sits ~1.9% below our cross-scheme consensus (~4.84 under both
    EXPLICIT_HULL refinement and an independent Crank-Nicolson solve).  This
    mirrors the documented hourly artefact in the single-barrier Table 8 (see
    ``TestBarrierPresentValueAgainstBoyleTianTable8`` in ``test_barrier.py``),
    so the hourly row carries a wider tolerance.
    """

    PRICING_DATE = dt.datetime(2025, 1, 1)
    SPOT = 100.0
    STRIKE = 100.0
    SIGMA = 0.20
    RATE = 0.10
    LOWER_BARRIER = 95.0
    UPPER_BARRIER = 140.0
    T_YEARS = 0.5
    MATURITY = PRICING_DATE + dt.timedelta(days=T_YEARS * 365)

    assert calculate_year_fraction(PRICING_DATE, MATURITY, DayCountConvention.ACT_365F) == 0.5, (
        "Paper maturity should be exactly 0.5 years under ACT/365F"
    )

    @classmethod
    def _market_data(cls) -> MarketData:
        return MarketData(
            cls.PRICING_DATE,
            DiscountCurve.flat(cls.RATE, 2.0),
            currency="USD",
            day_count_convention=DayCountConvention.ACT_365F,
        )

    @classmethod
    def _underlying(cls) -> UnderlyingData:
        return UnderlyingData(
            initial_value=cls.SPOT,
            volatility=cls.SIGMA,
            market_data=cls._market_data(),
            dividend_curve=DiscountCurve.flat(0.0, 2.0),
        )

    @classmethod
    def _double_ko_call_pv(
        cls, monitoring: BarrierMonitoring, num_observations: int | None = None
    ) -> float:
        spec = DoubleBarrierSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=cls.STRIKE,
            maturity=cls.MATURITY,
            lower_barrier=cls.LOWER_BARRIER,
            upper_barrier=cls.UPPER_BARRIER,
            action=BarrierAction.OUT,
            monitoring=monitoring,
            num_observations=num_observations,
        )
        valuation = OptionValuation(cls._underlying(), spec, PricingMethod.PDE_FD)
        return float(valuation.present_value())

    # (frequency, num_observations | None for continuous, paper_pv, rtol).
    # Observation counts follow Cheuk-Vorst (1994) per-year frequencies scaled
    # to T = 0.5 yr: hourly 1000/yr -> 500, daily 250/yr -> 125, weekly
    # 52/yr -> 26, monthly 12/yr -> 6, quarterly 4/yr -> 2.  The hourly row's
    # wider rtol absorbs the paper's known ~1.6% low bias at that density.
    _FREQUENCIES = [
        pytest.param("continuous", None, 4.5580, 2.0e-3, id="continuous"),
        pytest.param("hourly", 500, 4.7536, 2.5e-2, id="hourly"),
        pytest.param("daily", 125, 5.0784, 2.0e-3, id="daily"),
        pytest.param("weekly", 26, 5.6032, 2.0e-3, id="weekly"),
        pytest.param("monthly", 6, 6.4117, 2.0e-3, id="monthly"),
        pytest.param("quarterly", 2, 7.1541, 2.0e-3, id="quarterly"),
    ]

    @pytest.mark.parametrize("frequency,num_observations,paper_pv,rtol", _FREQUENCIES)
    def test_double_ko_call_monitoring_frequency_sweep(
        self,
        frequency: str,
        num_observations: int | None,
        paper_pv: float,
        rtol: float,
    ):
        """DKO call matches Tian across continuous -> quarterly monitoring."""
        monitoring = (
            BarrierMonitoring.CONTINUOUS if num_observations is None else BarrierMonitoring.DISCRETE
        )
        pv = self._double_ko_call_pv(monitoring, num_observations)
        logger.info(
            "BT98 Table8-DKO freq=%-10s N=%-4s | paper=%.4f dp_fd=%.4f diff=%.4f rel=%.4f%%",
            frequency,
            "—" if num_observations is None else num_observations,
            paper_pv,
            pv,
            abs(pv - paper_pv),
            abs(pv - paper_pv) / paper_pv * 100,
        )
        assert np.isclose(pv, paper_pv, rtol=rtol, atol=1.0e-4), (
            f"DKO call {frequency} (N={num_observations}): got {pv:.6f}, expected {paper_pv:.4f}"
        )
