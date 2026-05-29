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
    """Continuous double knock-out call PVs vs Boyle-Tian (1998).

    Both engines that price double barriers are compared side-by-side:

    * ``BSM`` — Kunitomo-Ikeda (1992) analytical infinite-series closed form;
    * ``PDE_FD`` — Crank-Nicolson on a log-spot grid truncated at both barriers.

    The paper reports "N/A" for closed-form double-knock-out values, so the
    reference column here is **Boyle-Tian's own explicit FD at 5000 steps**.
    Both our engines are therefore compared against a third
    numerical reference (BT98's 5000-step modified-explicit scheme), so each
    carries its own discretisation residual *plus* the paper's: K-I's
    deviation reflects how closely BT98's FD itself approximated the K-I
    limit, and our PDE_FD's deviation reflects two FD schemes' agreement at
    different resolutions.  Tolerances absorb the 4dp paper rounding plus
    this two-method gap.
    """

    PRICING_DATE = dt.datetime(2025, 1, 1)
    STRIKE = 100.0
    SIGMA = 0.25
    RATE = 0.10
    LOWER_BARRIER = 90.0
    UPPER_BARRIER = 140.0

    # Per-engine tolerances against Boyle-Tian's *FD* reference.
    # Both engines come in within ±3e-4 of paper across the sweep, so a 1e-3
    # atol comfortably absorbs paper rounding plus the BT98-FD-vs-our-engine
    # gap on either side.
    _TOLS: dict[PricingMethod, dict[str, float]] = {
        PricingMethod.BSM: dict(rtol=0.0, atol=1.0e-3),
        PricingMethod.PDE_FD: dict(rtol=0.0, atol=1.0e-3),
    }

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
    def _double_ko_call_pv(cls, spot: float, t_years: float, method: PricingMethod) -> float:
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
        valuation = OptionValuation(cls._underlying(spot), spec, method)
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
        engine_pvs: dict[PricingMethod, float] = {
            method: self._double_ko_call_pv(95.0, t_years, method)
            for method in (PricingMethod.BSM, PricingMethod.PDE_FD)
        }
        logger.info(
            "BT98 DoubleKO maturity %s | T=%.4fy paper=%.4f dp_an=%.4f dp_fd=%.4f",
            request.node.callspec.id,
            t_years,
            paper_pv,
            engine_pvs[PricingMethod.BSM],
            engine_pvs[PricingMethod.PDE_FD],
        )
        for method, pv in engine_pvs.items():
            assert np.isclose(pv, paper_pv, **self._TOLS[method]), (
                f"{method.name} DKO call T={t_years:.4f}y: got {pv:.6f}, expected {paper_pv:.4f}"
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
        engine_pvs: dict[PricingMethod, float] = {
            method: self._double_ko_call_pv(spot, 1.0, method)
            for method in (PricingMethod.BSM, PricingMethod.PDE_FD)
        }
        logger.info(
            "BT98 DoubleKO Table5 %s | spot=%.1f paper=%.4f dp_an=%.4f dp_fd=%.4f",
            request.node.callspec.id,
            spot,
            paper_pv,
            engine_pvs[PricingMethod.BSM],
            engine_pvs[PricingMethod.PDE_FD],
        )
        for method, pv in engine_pvs.items():
            assert np.isclose(pv, paper_pv, **self._TOLS[method]), (
                f"{method.name} DKO call S0={spot}: got {pv:.6f}, expected {paper_pv:.4f}"
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


class TestDoubleBarrierGreeksAgainstBoyleTian:
    """Continuous double knock-out call Greeks vs Boyle-Tian (1998).

    The double-barrier analogue of the paper's Table 6 single-barrier DOC
    Greeks, using the same corridor and spot sweep as the Table 5 PV test
    (``TestDoubleBarrierAgainstBoyleTian.test_double_ko_call_near_lower_barrier``):

        K = 100, sigma = 25%, r = 10%, q = 0, T = 1 yr,
        lower barrier L = 90, upper barrier U = 140,

    as spot approaches the lower barrier.  Both double-barrier engines are
    compared:

    * ``BSM`` — Kunitomo-Ikeda (1992) closed-form PV; delta and gamma come
      from central-difference bump-and-revalue around the K-I price, theta
      from the Black-Scholes PDE identity
      ``Θ = rV − (r−q)SΔ − ½σ²S²Γ``.
    * ``PDE_FD`` — grid greeks read straight off the backward solve; theta
      also via the same BS-PDE identity (with parabolic-Lagrange at-spot
      interpolation for ``V``).

    The paper reports "N/A" for closed-form double-knock-out greeks, so the
    reference column here is **Boyle-Tian's own explicit FD at 5000 steps**.
    Both our engines therefore compare against a third numerical reference,
    yet each lands within ~2e-3 of paper across the sweep — essentially the
    4dp paper rounding floor.

    The paper reports annualized theta; the library theta is per-day, so the
    comparison scales by 365.
    """

    PRICING_DATE = dt.datetime(2025, 1, 1)
    STRIKE = 100.0
    SIGMA = 0.25
    RATE = 0.10
    LOWER_BARRIER = 90.0
    UPPER_BARRIER = 140.0
    MATURITY = PRICING_DATE + dt.timedelta(days=365)

    # spot → (delta, gamma, theta_annualized) starred figures from the paper.
    _PAPER_GREEKS: dict[float, tuple[float, float, float]] = {
        95.0: (0.2535, -0.0165, 2.3985),
        92.0: (0.2998, -0.0141, 1.0271),
        91.0: (0.3133, -0.0129, 0.5238),
        90.5: (0.3196, -0.0123, 0.2644),
        90.4: (0.3208, -0.0121, 0.2119),
        90.3: (0.3220, -0.0120, 0.1592),
        90.2: (0.3232, -0.0119, 0.1064),
    }

    # Per-greek tolerances, shared across both engines.  All three greeks
    # sit essentially at the paper's 4dp rounding floor — delta/gamma via
    # GRID (FD) or central-difference bump (BSM), theta via the Black-
    # Scholes PDE identity ``Θ = rV − (r−q)SΔ − ½σ²S²Γ`` on both engines.
    _TOLS: dict[str, dict[str, float]] = {
        "delta": dict(rtol=2.0e-3, atol=5.0e-4),
        "gamma": dict(rtol=5.0e-3, atol=2.0e-4),
        "theta": dict(rtol=5.0e-3, atol=2.0e-3),
    }

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
    def _spec(cls) -> DoubleBarrierSpec:
        return DoubleBarrierSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=cls.STRIKE,
            maturity=cls.MATURITY,
            lower_barrier=cls.LOWER_BARRIER,
            upper_barrier=cls.UPPER_BARRIER,
            action=BarrierAction.OUT,
            monitoring=BarrierMonitoring.CONTINUOUS,
        )

    @classmethod
    def _engine_greek(cls, spot: float, method: PricingMethod, greek: str) -> float:
        """Return a single greek; theta annualized (×365) like the paper."""
        valuation = OptionValuation(cls._underlying(spot), cls._spec(), method)
        value = float(getattr(valuation, greek)())
        if greek == "theta":
            value *= 365.0
        return value

    @pytest.mark.parametrize(
        "spot",
        list(_PAPER_GREEKS.keys()),
        ids=[f"spot_{s:.1f}".replace(".", "_") for s in _PAPER_GREEKS],
    )
    @pytest.mark.parametrize("greek", ["delta", "gamma", "theta"])
    def test_double_ko_call_greek_matches_paper(self, spot: float, greek: str):
        """DKO call (delta/gamma/theta) matches Boyle-Tian as spot nears L=90."""
        paper_value = self._PAPER_GREEKS[spot][["delta", "gamma", "theta"].index(greek)]
        engine_values: dict[PricingMethod, float] = {
            method: self._engine_greek(spot, method, greek)
            for method in (PricingMethod.BSM, PricingMethod.PDE_FD)
        }
        logger.info(
            "BT98 DoubleKO Greeks spot=%.2f %-5s | paper=%.4f dp_an=%.4f dp_fd=%.4f",
            spot,
            greek,
            paper_value,
            engine_values[PricingMethod.BSM],
            engine_values[PricingMethod.PDE_FD],
        )
        tol = self._TOLS[greek]
        for method, value in engine_values.items():
            assert np.isclose(value, paper_value, **tol), (
                f"{method.name} {greek} mismatch at spot={spot}: "
                f"got {value:.4f}, expected {paper_value:.4f}"
            )


@pytest.mark.slow
class TestDoubleBarrierGreeksFrequencyAgainstTian:
    """Discretely-monitored double-KO call Greeks across monitoring frequencies.

    The double-barrier analogue of Boyle-Tian's Table 9 single-barrier DOC
    Greeks sweep — same scenario as the Table-8-style PV sweep
    (``TestDoubleBarrierMonitoringFrequencyAgainstTian``):

        S0 = K = 100, sigma = 20%, T = 0.5 yr, r = 10%, q = 0,
        lower barrier H = 95, upper barrier U = 140,

    swept across continuous / daily / weekly / monthly / quarterly monitoring
    (Cheuk-Vorst 1994 frequencies scaled to the half-year maturity).  Only
    PDE_FD prices double barriers; its grid Greeks come from the same backward
    solve.  The paper reports annualized theta, so the comparison scales by 365.

    No params are passed: the test relies on the engine's **default** params
    for double barriers (``PDEParams.for_double_barriers``), which lifts the
    discrete grid to ``spot_steps=2000, time_steps=6000``.  This doubles as a
    check that the default delivers paper-grade greeks without the caller
    tuning the grid — discrete-monitoring theta is the most resolution-
    sensitive figure (the knock-out resets inject a fresh discontinuity at
    every observation date), and on the default every Greek tracks the paper to
    within ~2e-4 (delta/gamma) / ~5e-3 (annualized theta).
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

    # frequency → (monitoring_kind, (delta, gamma, theta_annualized)).
    # monitoring_kind: "continuous" or N obs scaled by T = 0.5 yr.
    _PAPER_GREEKS: dict[str, tuple[str | int, tuple[float, float, float]]] = {
        "continuous": ("continuous", (0.7853, -0.0491, 2.4127)),
        "daily": (125, (0.7585, -0.0440, 1.7219)),  # 250 / yr
        "weekly": (26, (0.7271, -0.0372, 0.7274)),  # 52  / yr
        "monthly": (6, (0.6373, -0.0086, -4.0225)),  # 12  / yr
        "quarterly": (2, (0.5484, 0.0036, -5.4920)),  # 4   / yr
    }

    _TOLS: dict[str, dict[str, float]] = {
        "delta": dict(rtol=1.0e-2, atol=1.0e-3),
        "gamma": dict(rtol=2.0e-2, atol=1.0e-3),
        "theta": dict(rtol=2.0e-2, atol=1.0e-2),
    }

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
        )

    @classmethod
    def _engine_greeks(cls, monitoring_kind: str | int) -> dict[str, float]:
        """Return {delta, gamma, theta} from one PDE_FD solve; theta annualized."""
        monitoring = (
            BarrierMonitoring.CONTINUOUS
            if monitoring_kind == "continuous"
            else BarrierMonitoring.DISCRETE
        )
        num_observations = None if monitoring_kind == "continuous" else int(monitoring_kind)
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
        # No explicit params → engine resolves PDEParams.for_double_barriers,
        # i.e. the production default for double-barrier discrete monitoring.
        valuation = OptionValuation(cls._underlying(), spec, PricingMethod.PDE_FD)
        # delta/gamma/theta all read from the one cached backward solve.
        return {
            "delta": float(valuation.delta()),
            "gamma": float(valuation.gamma()),
            "theta": float(valuation.theta()) * 365.0,
        }

    @pytest.mark.parametrize("frequency", list(_PAPER_GREEKS.keys()))
    def test_double_ko_call_greek_frequency_sweep(self, frequency: str):
        """DKO call Greeks match Tian across continuous → quarterly monitoring."""
        monitoring_kind, paper = self._PAPER_GREEKS[frequency]
        paper_greeks = dict(zip(["delta", "gamma", "theta"], paper))
        engine_greeks = self._engine_greeks(monitoring_kind)

        for greek in ("delta", "gamma", "theta"):
            paper_value = paper_greeks[greek]
            value = engine_greeks[greek]
            logger.info(
                "BT98 Table9-DKO Greeks freq=%-10s %-5s | paper=%.4f dp_fd=%.4f diff=%.4f",
                frequency,
                greek,
                paper_value,
                value,
                abs(value - paper_value),
            )
            assert np.isclose(value, paper_value, **self._TOLS[greek]), (
                f"{greek} mismatch for {frequency}: got {value:.4f}, expected {paper_value:.4f}"
            )


class TestDoubleBarrierDiscreteDividendAgainstZvan:
    """European double-KO call with a discrete dividend vs Zvan-Vetzal-Forsyth.

    Setup: S0 = K = 100, sigma = 20%, T = 0.5 yr, r = 10%, q = 0, L = 95,
    U = 125, plus a single $2 cash dividend at T-t = 0.25 (calendar t = 0.25y).
    The paper tabulates the European double-KO across monitoring frequencies;
    weekly uses 25 observations (a week = 5 days of a 250-day year, so
    T = 0.5y gives 25 weeks).

    Reference PVs (2dp): continuous 1.92, daily 2.32, weekly 2.80.
    """

    PRICING_DATE = dt.datetime(2025, 1, 1)
    SPOT = 100.0
    STRIKE = 100.0
    SIGMA = 0.20
    RATE = 0.10
    LOWER_BARRIER = 95.0
    UPPER_BARRIER = 125.0
    T_YEARS = 0.5
    MATURITY = PRICING_DATE + dt.timedelta(days=T_YEARS * 365)
    DIV_AMOUNT = 2.0
    DIV_DATE = PRICING_DATE + dt.timedelta(days=0.25 * 365)  # T-t = 0.25

    assert calculate_year_fraction(PRICING_DATE, MATURITY, DayCountConvention.ACT_365F) == 0.5, (
        "Paper maturity should be exactly 0.5 years under ACT/365F"
    )
    assert calculate_year_fraction(PRICING_DATE, DIV_DATE, DayCountConvention.ACT_365F) == 0.25, (
        "Dividend should fall at exactly t = 0.25 years (T-t = 0.25)"
    )

    @classmethod
    def _underlying(cls) -> UnderlyingData:
        market = MarketData(
            cls.PRICING_DATE,
            DiscountCurve.flat(cls.RATE, 2.0),
            currency="USD",
            day_count_convention=DayCountConvention.ACT_365F,
        )
        return UnderlyingData(
            initial_value=cls.SPOT,
            volatility=cls.SIGMA,
            market_data=market,
            dividend_curve=None,  # the $2 cash dividend is the only payout
            discrete_dividends=[(cls.DIV_DATE, cls.DIV_AMOUNT)],
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

    @pytest.mark.parametrize(
        "frequency,num_observations,paper_pv",
        [
            pytest.param("continuous", None, 1.92, id="continuous"),
            pytest.param("daily", 125, 2.32, id="daily"),
            pytest.param("weekly", 25, 2.80, id="weekly"),
        ],
    )
    def test_discrete_dividend_double_ko_call_matches_paper(
        self, frequency: str, num_observations: int | None, paper_pv: float
    ):
        """European DKO call with a $2 discrete dividend matches Zvan-Vetzal-Forsyth."""
        monitoring = (
            BarrierMonitoring.CONTINUOUS if num_observations is None else BarrierMonitoring.DISCRETE
        )
        pv = self._double_ko_call_pv(monitoring, num_observations)
        logger.info(
            "ZVF discrete-div DKO freq=%-10s N=%-4s | paper=%.4f dp_fd=%.4f diff=%.4f rel=%.4f%%",
            frequency,
            "—" if num_observations is None else num_observations,
            paper_pv,
            pv,
            abs(pv - paper_pv),
            abs(pv - paper_pv) / paper_pv * 100,
        )
        # atol floor (~6e-3) absorbs the paper's 2dp rounding (±0.005); rtol
        # (0.3%) scales the model residual with the option value.
        assert np.isclose(pv, paper_pv, rtol=3e-3, atol=6e-3), (
            f"discrete-div DKO call {frequency}: got {pv:.6f}, expected {paper_pv:.4f}"
        )
