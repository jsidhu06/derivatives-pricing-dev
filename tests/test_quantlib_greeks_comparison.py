"""Compare Greeks between derivatives_pricing and QuantLib for reference.

Sections
--------
1. Vanilla European Greeks: DP engines vs QuantLib (broad scenarios)
2. Vanilla American Greeks: DP engines vs QuantLib FD (broad scenarios)
3. European Asian MC numerical Greeks vs QuantLib
"""

from __future__ import annotations

from collections.abc import Sequence
import datetime as dt
import logging
from typing import TYPE_CHECKING

import numpy as np
import pytest

from derivatives_pricing.enums import (
    AsianAveraging,
    BarrierAction,
    BarrierDirection,
    DayCountConvention,
    ExerciseType,
    OptionType,
    PricingMethod,
)
from derivatives_pricing.market_environment import MarketData
from derivatives_pricing.rates import DiscountCurve
from helpers import (
    assert_greeks_close,
    market_data,
    make_vanilla_spec,
    underlying,
)
from derivatives_pricing.valuation import (
    AsianSpec,
    BarrierSpec,
    VanillaSpec,
    OptionValuation,
    UnderlyingData,
)
from derivatives_pricing.stochastic_processes import (
    GBMParams,
    GBMProcess,
    SimulationConfig,
)
from derivatives_pricing.valuation.params import (
    BinomialParams,
    MonteCarloParams,
    PDEParams,
)
from derivatives_pricing.utils import calculate_year_fraction

if TYPE_CHECKING:
    import QuantLib as ql_typing

pytestmark = pytest.mark.slow

ql = pytest.importorskip("QuantLib")

logger = logging.getLogger(__name__)


def _ql_dcc(dcc: DayCountConvention):
    """Map a DP DayCountConvention to the corresponding QuantLib DayCounter."""
    if dcc is DayCountConvention.ACT_360:
        return ql.Actual360()
    return ql.Actual365Fixed()


# Convention mapping: QL → DP
# delta, gamma: same
# vega: QL per 100% vol → /100 to match DP per 1 vol-pt
# theta: QL per year → /365 to match DP per day
# rho: QL per 100% rate → /100 to match DP per 1%

_QL_SCALE = {"delta": 1.0, "gamma": 1.0, "vega": 1 / 100, "theta": 1 / 365, "rho": 1 / 100}

# ── Shared constants ────────────────────────────────────────────────────

PRICING_DATE = dt.datetime(2025, 1, 1)
MATURITY = PRICING_DATE + dt.timedelta(days=365)
RISK_FREE = 0.05
VOL = 0.20
CURRENCY = "USD"

BINOM_CFG = BinomialParams(num_steps=1500)
PDE_CFG = PDEParams(spot_steps=200, time_steps=200, max_iter=20_000)
MC_CFG = MonteCarloParams(random_seed=42)

# ── derivatives_pricing helpers ─────────────────────────────────────────


def _market_data(
    discount_curve: DiscountCurve | None = None,
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> MarketData:
    if discount_curve is None:
        ttm = calculate_year_fraction(PRICING_DATE, MATURITY, dcc)
        discount_curve = DiscountCurve.flat(RISK_FREE, end_time=ttm)
    return market_data(
        pricing_date=PRICING_DATE,
        discount_curve=discount_curve,
        currency=CURRENCY,
        day_count_convention=dcc,
    )


def _spec(
    *,
    strike: float,
    option_type: OptionType,
    exercise_type: ExerciseType = ExerciseType.EUROPEAN,
) -> VanillaSpec:
    return make_vanilla_spec(
        strike=strike,
        maturity=MATURITY,
        option_type=option_type,
        exercise_type=exercise_type,
        currency=CURRENCY,
    )


def _underlying(
    *,
    spot: float,
    risk_free_curve: DiscountCurve | None = None,
    dividend_curve: DiscountCurve | None = None,
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None = None,
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> UnderlyingData:
    return underlying(
        initial_value=spot,
        volatility=VOL,
        market_data=_market_data(discount_curve=risk_free_curve, dcc=dcc),
        dividend_curve=dividend_curve,
        discrete_dividends=discrete_dividends,
    )


def _gbm(
    *,
    spot: float,
    risk_free_curve: DiscountCurve | None = None,
    dividend_curve: DiscountCurve | None = None,
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None = None,
    paths: int = 500_000,
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> GBMProcess:
    return GBMProcess(
        _market_data(discount_curve=risk_free_curve, dcc=dcc),
        GBMParams(
            initial_value=spot,
            volatility=VOL,
            dividend_curve=dividend_curve,
            discrete_dividends=discrete_dividends,
        ),
        SimulationConfig(
            paths=paths,
            num_steps=52,
            end_date=MATURITY,
        ),
    )


# ── QuantLib helpers ────────────────────────────────────────────────────


def _ql_setup() -> "ql_typing.Date":
    eval_date = ql.Date(PRICING_DATE.day, PRICING_DATE.month, PRICING_DATE.year)
    ql.Settings.instance().evaluationDate = eval_date
    return eval_date


def _ql_curve_handle_from_discount_curve(
    curve: DiscountCurve,
    *,
    eval_date: "ql_typing.Date",
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> "ql_typing.YieldTermStructureHandle":
    ql_dc = _ql_dcc(dcc)
    denom = 360.0 if dcc is DayCountConvention.ACT_360 else 365.0
    dates = [eval_date]
    for t in curve.times[1:]:
        dates.append(eval_date + int(round(float(t) * denom)))
    return ql.YieldTermStructureHandle(ql.DiscountCurve(dates, list(curve.dfs), ql_dc))


def _ql_dividend_vector(
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None,
) -> "ql_typing.DividendVector":
    if not discrete_dividends:
        return ql.DividendVector([], [])
    dates: list[ql_typing.Date] = []
    amounts: list[float] = []
    for ex_date, amount in discrete_dividends:
        if PRICING_DATE <= ex_date <= MATURITY:
            dates.append(ql.Date(ex_date.day, ex_date.month, ex_date.year))
            amounts.append(float(amount))
    return ql.DividendVector(dates, amounts)


def _ql_process(
    eval_date: "ql_typing.Date",
    *,
    spot: float,
    dividend_yield: float = 0.0,
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> "ql_typing.BlackScholesMertonProcess":
    """BSM process with flat rate, dividend yield, and vol."""
    ql_dc = _ql_dcc(dcc)
    return ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(spot)),
        ql.YieldTermStructureHandle(ql.FlatForward(eval_date, dividend_yield, ql_dc)),
        ql.YieldTermStructureHandle(ql.FlatForward(eval_date, RISK_FREE, ql_dc)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(eval_date, ql.TARGET(), VOL, ql_dc)),
    )


def _ql_european_option(
    *,
    spot: float,
    strike: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
    risk_free_curve: DiscountCurve | None = None,
    dividend_curve: DiscountCurve | None = None,
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> "ql_typing.VanillaOption":
    """QuantLib European with AnalyticEuropeanEngine (flat or term-structured curves)."""
    eval_date = _ql_setup()
    ql_maturity = ql.Date(MATURITY.day, MATURITY.month, MATURITY.year)
    ql_type = ql.Option.Put if option_type is OptionType.PUT else ql.Option.Call
    ql_dc = _ql_dcc(dcc)
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql_type, strike),
        ql.EuropeanExercise(ql_maturity),
    )

    if risk_free_curve is None and dividend_curve is None:
        process = _ql_process(eval_date, spot=spot, dividend_yield=dividend_yield, dcc=dcc)
    else:
        rf_handle = (
            _ql_curve_handle_from_discount_curve(risk_free_curve, eval_date=eval_date, dcc=dcc)
            if risk_free_curve is not None
            else ql.YieldTermStructureHandle(ql.FlatForward(eval_date, RISK_FREE, ql_dc))
        )
        div_handle = (
            _ql_curve_handle_from_discount_curve(dividend_curve, eval_date=eval_date, dcc=dcc)
            if dividend_curve is not None
            else ql.YieldTermStructureHandle(ql.FlatForward(eval_date, dividend_yield, ql_dc))
        )
        process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(spot)),
            div_handle,
            rf_handle,
            ql.BlackVolTermStructureHandle(ql.BlackConstantVol(eval_date, ql.TARGET(), VOL, ql_dc)),
        )

    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return option


def _ql_american_fd_option(
    *,
    spot: float,
    strike: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
    risk_free_curve: DiscountCurve | None = None,
    dividend_curve: DiscountCurve | None = None,
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None = None,
    grid_points: int = 200,
    time_steps: int = 400,
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> "ql_typing.VanillaOption":
    """QuantLib American with FdBlackScholesVanillaEngine."""
    eval_date = _ql_setup()
    ql_maturity = ql.Date(MATURITY.day, MATURITY.month, MATURITY.year)
    ql_type = ql.Option.Put if option_type is OptionType.PUT else ql.Option.Call
    ql_dc = _ql_dcc(dcc)
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql_type, strike),
        ql.AmericanExercise(eval_date, ql_maturity),
    )
    if risk_free_curve is None and dividend_curve is None:
        process = _ql_process(eval_date, spot=spot, dividend_yield=dividend_yield, dcc=dcc)
    else:
        rf_handle = (
            _ql_curve_handle_from_discount_curve(risk_free_curve, eval_date=eval_date, dcc=dcc)
            if risk_free_curve is not None
            else ql.YieldTermStructureHandle(ql.FlatForward(eval_date, RISK_FREE, ql_dc))
        )
        div_handle = (
            _ql_curve_handle_from_discount_curve(dividend_curve, eval_date=eval_date, dcc=dcc)
            if dividend_curve is not None
            else ql.YieldTermStructureHandle(ql.FlatForward(eval_date, dividend_yield, ql_dc))
        )
        process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(spot)),
            div_handle,
            rf_handle,
            ql.BlackVolTermStructureHandle(ql.BlackConstantVol(eval_date, ql.TARGET(), VOL, ql_dc)),
        )

    engine = ql.FdBlackScholesVanillaEngine(
        process,
        _ql_dividend_vector(discrete_dividends),
        grid_points,
        time_steps,
    )
    option.setPricingEngine(engine)
    return option


def _ql_greek(option: "ql_typing.VanillaOption", greek: str) -> float | None:
    """Return a QuantLib greek when available, else ``None``."""
    try:
        return float(getattr(option, greek)())
    except RuntimeError:
        return None


def _ql_scaled_greeks(
    option: "ql_typing.VanillaOption",
    *,
    allow_missing: bool,
) -> dict[str, float | None]:
    """Return QuantLib Greeks scaled to derivatives_pricing conventions."""
    values: dict[str, float | None] = {}
    for greek, scale in _QL_SCALE.items():
        val = _ql_greek(option, greek)
        if val is None and allow_missing:
            values[greek] = None
        else:
            values[greek] = None if val is None else val * scale
    return values


# Convention conversions (QuantLib → derivatives_pricing):
#   vega:  QL returns d(V)/d(σ);  DP returns d(V)/d(σ)/100  (per 1 vol-pt)
#   theta: QL returns d(V)/d(t) per year; DP returns per calendar day (/365)
#   rho:   QL returns d(V)/d(r);  DP returns d(V)/d(r)/100 (per 1% rate)


# ═══════════════════════════════════════════════════════════════════════
# 1. Vanilla European Greeks: DP engines vs QuantLib (broad scenarios)
# ═══════════════════════════════════════════════════════════════════════

_EU_VANILLA_SCENARIOS = [
    pytest.param(
        100.0,
        100.0,
        OptionType.CALL,
        "flat",
        "none",
        DayCountConvention.ACT_365F,
        id="atm_call_flat_ACT365F",
    ),
    pytest.param(
        100.0,
        95.0,
        OptionType.PUT,
        "flat",
        "flat",
        DayCountConvention.ACT_360,
        id="itm_put_flat_div_ACT360",
    ),
    pytest.param(
        90.0,
        100.0,
        OptionType.CALL,
        "nonflat",
        "nonflat",
        DayCountConvention.ACT_360,
        id="otm_call_nonflat_ACT360",
    ),
    pytest.param(
        110.0,
        100.0,
        OptionType.PUT,
        "nonflat",
        "none",
        DayCountConvention.ACT_365F,
        id="otm_put_nonflat_ACT365F",
    ),
]


def _resolve_curve(
    kind: str,
    *,
    is_dividend: bool,
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> DiscountCurve | None:
    if kind == "none":
        return None
    ttm = calculate_year_fraction(PRICING_DATE, MATURITY, dcc)
    if kind == "flat":
        rate = 0.03 if is_dividend else RISK_FREE
        return DiscountCurve.flat(rate, end_time=ttm)
    if kind == "nonflat":
        forwards = np.array([0.01, 0.02, 0.015]) if is_dividend else np.array([0.03, 0.05, 0.06])
        return DiscountCurve.from_forwards(times=np.array([0.0, 0.25, 0.5, ttm]), forwards=forwards)
    raise ValueError(f"unsupported curve kind: {kind}")


@pytest.mark.parametrize("spot,strike,option_type,rate_kind,div_kind,dcc", _EU_VANILLA_SCENARIOS)
@pytest.mark.parametrize(
    "engine,tols",
    [
        (
            PricingMethod.BSM,
            {"delta": 1e-4, "gamma": 1e-4, "vega": 1e-4, "theta": 1e-4, "rho": 1e-4},
        ),
        (
            PricingMethod.BINOMIAL,
            {"delta": 0.01, "gamma": 0.03, "vega": 0.08, "theta": 0.30, "rho": 0.08},
        ),
        (
            PricingMethod.PDE_FD,
            {"delta": 0.02, "gamma": 0.05, "vega": 0.10, "theta": 0.30, "rho": 0.08},
        ),
        (
            PricingMethod.MONTE_CARLO,
            {"delta": 0.03, "gamma": 0.10, "vega": 0.05, "theta": 0.10, "rho": 0.10},
        ),
    ],
)
def test_vanilla_european_greeks_vs_quantlib(
    spot, strike, option_type, rate_kind, div_kind, dcc, engine, tols
):
    """DP vanilla European Greeks align with QuantLib across flat/non-flat curve scenarios."""
    r_curve = _resolve_curve(rate_kind, is_dividend=False, dcc=dcc)
    q_curve = _resolve_curve(div_kind, is_dividend=True, dcc=dcc)

    underlying = (
        _gbm(spot=spot, risk_free_curve=r_curve, dividend_curve=q_curve, dcc=dcc)
        if engine is PricingMethod.MONTE_CARLO
        else _underlying(spot=spot, risk_free_curve=r_curve, dividend_curve=q_curve, dcc=dcc)
    )
    params = (
        MC_CFG
        if engine is PricingMethod.MONTE_CARLO
        else BINOM_CFG
        if engine is PricingMethod.BINOMIAL
        else PDE_CFG
        if engine is PricingMethod.PDE_FD
        else None
    )

    ov = OptionValuation(
        underlying,
        _spec(strike=strike, option_type=option_type),
        engine,
        params=params,
    )
    ql_opt = _ql_european_option(
        spot=spot,
        strike=strike,
        option_type=option_type,
        risk_free_curve=r_curve,
        dividend_curve=q_curve,
        dcc=dcc,
    )

    ql_values = _ql_scaled_greeks(ql_opt, allow_missing=False)
    dp_values = {
        "delta": ov.delta(),
        "gamma": ov.gamma(),
        "vega": ov.vega(),
        "theta": ov.theta(),
        "rho": ov.rho(),
    }

    assert_greeks_close(
        lhs=dp_values,
        rhs=ql_values,
        tols=tols,
        log_prefix=f"{engine.name} EU {option_type.value} S={spot:.0f} K={strike:.0f} {dcc.value}",
        lhs_name="DP",
        rhs_name="QL",
        skip_missing_rhs=False,
        logger=logger,
    )


# ═══════════════════════════════════════════════════════════════════════
# 2. Vanilla American Greeks: DP engines vs QuantLib FD (broad scenarios)
# ═══════════════════════════════════════════════════════════════════════

_AM_VANILLA_SCENARIOS = [
    pytest.param(
        90.0,
        100.0,
        OptionType.PUT,
        "flat",
        "none",
        None,
        DayCountConvention.ACT_365F,
        id="itm_put_flat_ACT365F",
    ),
    pytest.param(
        110.0,
        100.0,
        OptionType.CALL,
        "flat",
        "flat",
        None,
        DayCountConvention.ACT_360,
        id="itm_call_flat_div_ACT360",
    ),
    pytest.param(
        100.0,
        100.0,
        OptionType.PUT,
        "nonflat",
        "nonflat",
        None,
        DayCountConvention.ACT_360,
        id="atm_put_nonflat_ACT360",
    ),
    pytest.param(
        100.0,
        100.0,
        OptionType.PUT,
        "nonflat",
        "none",
        [
            (PRICING_DATE + dt.timedelta(days=90), 0.50),
            (PRICING_DATE + dt.timedelta(days=270), 0.50),
        ],
        DayCountConvention.ACT_365F,
        id="atm_put_nonflat_discrete_ACT365F",
    ),
]


@pytest.mark.parametrize(
    "spot,strike,option_type,rate_kind,div_kind,discrete_dividends,dcc", _AM_VANILLA_SCENARIOS
)
@pytest.mark.parametrize(
    "engine,tols",
    [
        (
            PricingMethod.BINOMIAL,
            {"delta": 0.03, "gamma": 0.08, "vega": 0.12, "theta": 0.35, "rho": 0.12},
        ),
        (
            PricingMethod.PDE_FD,
            {"delta": 0.03, "gamma": 0.08, "vega": 0.15, "theta": 0.35, "rho": 0.15},
        ),
    ],
)
def test_vanilla_american_greeks_vs_quantlib(
    spot,
    strike,
    option_type,
    rate_kind,
    div_kind,
    discrete_dividends,
    dcc,
    engine,
    tols,
):
    """DP vanilla American Greeks align with QuantLib FD for broad curve/dividend scenarios."""
    r_curve = _resolve_curve(rate_kind, is_dividend=False, dcc=dcc)
    q_curve = _resolve_curve(div_kind, is_dividend=True, dcc=dcc)

    ov = OptionValuation(
        _underlying(
            spot=spot,
            risk_free_curve=r_curve,
            dividend_curve=q_curve,
            discrete_dividends=discrete_dividends,
            dcc=dcc,
        ),
        _spec(strike=strike, option_type=option_type, exercise_type=ExerciseType.AMERICAN),
        engine,
        params=BINOM_CFG if engine is PricingMethod.BINOMIAL else PDE_CFG,
    )
    ql_opt = _ql_american_fd_option(
        spot=spot,
        strike=strike,
        option_type=option_type,
        risk_free_curve=r_curve,
        dividend_curve=q_curve,
        discrete_dividends=discrete_dividends,
        dcc=dcc,
    )

    dp_values = {
        "delta": ov.delta(),
        "gamma": ov.gamma(),
        "vega": ov.vega(),
        "theta": ov.theta(),
        "rho": ov.rho(),
    }
    ql_values = _ql_scaled_greeks(ql_opt, allow_missing=True)
    assert_greeks_close(
        lhs=dp_values,
        rhs=ql_values,
        tols=tols,
        log_prefix=f"{engine.name} AM {option_type.value} S={spot:.0f} K={strike:.0f} {dcc.value}",
        lhs_name="DP",
        rhs_name="QL",
        skip_missing_rhs=True,
        atol=1e-4,
        logger=logger,
    )


# ═══════════════════════════════════════════════════════════════════════
# 5. European Asian MC numerical Greeks vs QuantLib
# ═══════════════════════════════════════════════════════════════════════

_ASIAN_FIXINGS = tuple(
    dt.datetime(2025, m, 1) if m <= 12 else dt.datetime(2026, m - 12, 1) for m in range(2, 14)
)


def _dt_to_ql(d: dt.datetime) -> "ql_typing.Date":
    return ql.Date(d.day, d.month, d.year)


def _ql_asian_greeks(
    *,
    option_type: OptionType,
    averaging: AsianAveraging,
    strike: float,
    spot: float,
    vol: float,
    risk_free_curve: DiscountCurve,
    dividend_curve: DiscountCurve | None,
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> dict[str, float | None]:
    """Price a European Asian via QuantLib and return available Greeks.

    Geometric analytic engine provides all 5 Greeks.
    TW arithmetic engine provides delta and gamma only.
    """
    eval_date = ql.Date(PRICING_DATE.day, PRICING_DATE.month, PRICING_DATE.year)
    ql.Settings.instance().evaluationDate = eval_date
    ql_dc = _ql_dcc(dcc)

    spot_h = ql.QuoteHandle(ql.SimpleQuote(spot))
    rf_h = _ql_curve_handle_from_discount_curve(risk_free_curve, eval_date=eval_date, dcc=dcc)
    div_h = (
        _ql_curve_handle_from_discount_curve(dividend_curve, eval_date=eval_date, dcc=dcc)
        if dividend_curve is not None
        else ql.YieldTermStructureHandle(ql.FlatForward(eval_date, 0.0, ql_dc))
    )
    vol_h = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(eval_date, ql.TARGET(), vol, ql_dc))
    proc = ql.BlackScholesMertonProcess(spot_h, div_h, rf_h, vol_h)

    ql_type = ql.Option.Put if option_type is OptionType.PUT else ql.Option.Call
    payoff = ql.PlainVanillaPayoff(ql_type, strike)
    exercise = ql.EuropeanExercise(_dt_to_ql(MATURITY))
    ql_fixings = [_dt_to_ql(d) for d in _ASIAN_FIXINGS]

    if averaging is AsianAveraging.GEOMETRIC:
        ql_avg = ql.Average.Geometric
        engine = ql.AnalyticDiscreteGeometricAveragePriceAsianEngine(proc)
    else:
        ql_avg = ql.Average.Arithmetic
        engine = ql.TurnbullWakemanAsianEngine(proc)

    opt = ql.DiscreteAveragingAsianOption(ql_avg, ql_fixings, payoff, exercise)
    opt.setPricingEngine(engine)

    result: dict[str, float | None] = {
        "npv": opt.NPV(),
        "delta": opt.delta(),
        "gamma": opt.gamma(),
    }
    # TW engine does not provide vega/theta/rho
    for greek in ("vega", "theta", "rho"):
        try:
            result[greek] = getattr(opt, greek)()
        except RuntimeError:
            result[greek] = None
    return result


def _dp_asian_mc_greeks(
    *,
    option_type: OptionType,
    averaging: AsianAveraging,
    strike: float,
    spot: float,
    vol: float,
    risk_free_curve: DiscountCurve,
    dividend_curve: DiscountCurve | None,
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> dict[str, float]:
    """Build our MC Asian and compute numerical Greeks."""
    md = MarketData(PRICING_DATE, risk_free_curve, currency=CURRENCY, day_count_convention=dcc)
    params = GBMParams(initial_value=spot, volatility=vol, dividend_curve=dividend_curve)
    sim_cfg = SimulationConfig(
        paths=150_000,
        end_date=MATURITY,
        num_steps=30,
    )
    process = GBMProcess(md, params, sim_cfg)
    spec = AsianSpec(
        averaging=averaging,
        option_type=option_type,
        strike=strike,
        maturity=MATURITY,
        currency=CURRENCY,
        fixing_dates=_ASIAN_FIXINGS,
        exercise_type=ExerciseType.EUROPEAN,
    )
    ov = OptionValuation(
        process,
        spec,
        PricingMethod.MONTE_CARLO,
        params=MonteCarloParams(random_seed=42),
    )
    return {
        "npv": ov.present_value(),
        "delta": ov.delta(),
        "gamma": ov.gamma(),
        "vega": ov.vega(),
        "theta": ov.theta(),
        "rho": ov.rho(),
    }


def _dp_asian_analytical_greeks(
    *,
    option_type: OptionType,
    averaging: AsianAveraging,
    strike: float,
    spot: float,
    vol: float,
    risk_free_curve: DiscountCurve,
    dividend_curve: DiscountCurve | None,
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> dict[str, float]:
    """Build our analytical (BSM) Asian and compute numerical Greeks."""
    md = MarketData(PRICING_DATE, risk_free_curve, currency=CURRENCY, day_count_convention=dcc)
    underlying = UnderlyingData(
        initial_value=spot,
        volatility=vol,
        market_data=md,
        dividend_curve=dividend_curve,
    )
    spec = AsianSpec(
        averaging=averaging,
        option_type=option_type,
        strike=strike,
        maturity=MATURITY,
        currency=CURRENCY,
        fixing_dates=_ASIAN_FIXINGS,
        exercise_type=ExerciseType.EUROPEAN,
    )
    ov = OptionValuation(underlying, spec, PricingMethod.BSM)
    return {
        "npv": ov.present_value(),
        "delta": ov.delta(),
        "gamma": ov.gamma(),
        "vega": ov.vega(),
        "theta": ov.theta(),
        "rho": ov.rho(),
    }


_ASIAN_GREEK_SCENARIOS = [
    # Geometric (all 5 Greeks available from QL analytic engine)
    pytest.param(
        100,
        100,
        0.20,
        OptionType.CALL,
        AsianAveraging.GEOMETRIC,
        "flat",
        "none",
        DayCountConvention.ACT_365F,
        id="geom_call_atm_flat_ACT365F",
    ),
    pytest.param(
        100,
        90,
        0.25,
        OptionType.PUT,
        AsianAveraging.GEOMETRIC,
        "nonflat",
        "flat",
        DayCountConvention.ACT_360,
        id="geom_put_itm_nonflat_ACT360",
    ),
    # Arithmetic (TW engine: delta/gamma only)
    pytest.param(
        100,
        110,
        0.30,
        OptionType.CALL,
        AsianAveraging.ARITHMETIC,
        "flat",
        "flat",
        DayCountConvention.ACT_360,
        id="arith_call_otm_flat_div_ACT360",
    ),
    pytest.param(
        100,
        100,
        0.20,
        OptionType.PUT,
        AsianAveraging.ARITHMETIC,
        "nonflat",
        "none",
        DayCountConvention.ACT_365F,
        id="arith_put_atm_nonflat_ACT365F",
    ),
]


@pytest.mark.parametrize(
    "spot,strike,vol,option_type,averaging,rate_kind,div_kind,dcc",
    _ASIAN_GREEK_SCENARIOS,
)
def test_asian_mc_greeks_vs_quantlib(
    spot,
    strike,
    vol,
    option_type,
    averaging,
    rate_kind,
    div_kind,
    dcc,
):
    """European Asian MC numerical Greeks vs QuantLib analytic/TW Greeks."""
    r_curve = _resolve_curve(rate_kind, is_dividend=False, dcc=dcc)
    q_curve = _resolve_curve(div_kind, is_dividend=True, dcc=dcc)
    assert r_curve is not None

    ql_greeks = _ql_asian_greeks(
        option_type=option_type,
        averaging=averaging,
        strike=strike,
        spot=spot,
        vol=vol,
        risk_free_curve=r_curve,
        dividend_curve=q_curve,
        dcc=dcc,
    )
    dp_greeks = _dp_asian_mc_greeks(
        option_type=option_type,
        averaging=averaging,
        strike=strike,
        spot=spot,
        vol=vol,
        risk_free_curve=r_curve,
        dividend_curve=q_curve,
        dcc=dcc,
    )
    dp_an_greeks = _dp_asian_analytical_greeks(
        option_type=option_type,
        averaging=averaging,
        strike=strike,
        spot=spot,
        vol=vol,
        risk_free_curve=r_curve,
        dividend_curve=q_curve,
        dcc=dcc,
    )

    # Tolerances: MC numerical bump-and-revalue vs analytic/TW
    mc_tols = {"delta": 0.03, "gamma": 0.10, "vega": 0.05, "theta": 0.10, "rho": 0.10}
    # Analytical bump-and-revalue vs QL analytic/TW.  Match MC tolerances —
    # QL's own geometric analytic engine has limitations with non-flat curves
    an_tols = {"delta": 0.03, "gamma": 0.10, "vega": 0.05, "theta": 0.10, "rho": 0.10}

    for greek in ("delta", "gamma", "vega", "theta", "rho"):
        ql_val = ql_greeks[greek]
        if ql_val is None:
            continue  # TW engine doesn't provide this greek
        dp_val = dp_greeks[greek]
        dp_an_val = dp_an_greeks[greek]
        ql_scaled = ql_val * _QL_SCALE[greek]
        logger.info(
            "Asian %s %s %s S=%.0f K=%.0f | DP_MC=%.6f DP_AN=%.6f QL=%.6f",
            averaging.value,
            option_type.value,
            greek,
            spot,
            strike,
            dp_val,
            dp_an_val,
            ql_scaled,
        )
        assert np.isclose(dp_val, ql_scaled, rtol=mc_tols[greek], atol=1e-4), (
            f"{greek}: DP_MC {dp_val:.6f} vs QL {ql_scaled:.6f}"
        )
        assert np.isclose(dp_an_val, ql_scaled, rtol=an_tols[greek], atol=1e-4), (
            f"{greek}: DP_AN {dp_an_val:.6f} vs QL {ql_scaled:.6f}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 4. European Barrier Greeks: DP numerical vs QuantLib bump-and-revalue
# ═══════════════════════════════════════════════════════════════════════════

_BARRIER_SPOT = 100.0
_BARRIER_VOL = 0.25
_BARRIER_RATE = 0.05
_BARRIER_DIV = 0.02

_QL_BARRIER_TYPE = {
    (BarrierDirection.DOWN, BarrierAction.IN): ql.Barrier.DownIn,
    (BarrierDirection.DOWN, BarrierAction.OUT): ql.Barrier.DownOut,
    (BarrierDirection.UP, BarrierAction.IN): ql.Barrier.UpIn,
    (BarrierDirection.UP, BarrierAction.OUT): ql.Barrier.UpOut,
}


def _barrier_nonflat_curves() -> tuple[DiscountCurve, DiscountCurve]:
    """Non-flat rate and dividend curves for barrier tests."""
    ttm = calculate_year_fraction(PRICING_DATE, MATURITY)
    r_times = np.array([0.0, 0.25, 0.5, ttm])
    r_forwards = np.array([0.03, 0.06, 0.04])
    q_times = np.array([0.0, 0.25, 0.5, ttm])
    q_forwards = np.array([0.01, 0.03, 0.005])
    return (
        DiscountCurve.from_forwards(times=r_times, forwards=r_forwards),
        DiscountCurve.from_forwards(times=q_times, forwards=q_forwards),
    )


def _ql_barrier_bump_greeks(
    *,
    direction: BarrierDirection,
    action: BarrierAction,
    barrier: float,
    option_type: OptionType,
    strike: float,
    spot: float,
    vol: float,
    r_curve: DiscountCurve | None = None,
    q_curve: DiscountCurve | None = None,
) -> dict[str, float]:
    """Compute barrier Greeks via QL bump-and-revalue (analytic engine)."""
    eval_date = ql.Date(PRICING_DATE.day, PRICING_DATE.month, PRICING_DATE.year)
    ql_dc = ql.Actual365Fixed()
    maturity_ql = ql.Date(MATURITY.day, MATURITY.month, MATURITY.year)
    ql_type = ql.Option.Put if option_type is OptionType.PUT else ql.Option.Call
    bt = _QL_BARRIER_TYPE[(direction, action)]

    def _price(s, v, r_rate, q_rate, eval_dt=eval_date, rc=r_curve, qc=q_curve):
        ql.Settings.instance().evaluationDate = eval_dt
        sp = ql.QuoteHandle(ql.SimpleQuote(s))
        if rc is not None:
            rf = _ql_curve_handle_from_discount_curve(rc, eval_date=eval_dt)
        else:
            rf = ql.YieldTermStructureHandle(ql.FlatForward(eval_dt, r_rate, ql_dc))
        if qc is not None:
            dv = _ql_curve_handle_from_discount_curve(qc, eval_date=eval_dt)
        else:
            dv = ql.YieldTermStructureHandle(ql.FlatForward(eval_dt, q_rate, ql_dc))
        vl = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(eval_dt, ql.TARGET(), v, ql_dc))
        pr = ql.BlackScholesMertonProcess(sp, dv, rf, vl)
        payoff = ql.PlainVanillaPayoff(ql_type, strike)
        exercise = ql.EuropeanExercise(maturity_ql)
        opt = ql.BarrierOption(bt, barrier, 0.0, payoff, exercise)
        opt.setPricingEngine(ql.AnalyticBarrierEngine(pr))
        return opt.NPV()

    r0 = _BARRIER_RATE
    q0 = _BARRIER_DIV

    pv0 = _price(spot, vol, r0, q0)

    # Delta & gamma
    ds = spot * 0.01
    pv_up = _price(spot + ds, vol, r0, q0)
    pv_dn = _price(spot - ds, vol, r0, q0)
    delta = (pv_up - pv_dn) / (2 * ds)
    gamma = (pv_up - 2 * pv0 + pv_dn) / (ds**2)

    # Vega (per 1 vol point = /100)
    dv = 0.01
    vega = (_price(spot, vol + dv, r0, q0) - _price(spot, vol - dv, r0, q0)) / (2 * dv) / 100

    # Theta (per calendar day = /365)
    eval_fwd = eval_date + 1
    theta = (_price(spot, vol, r0, q0, eval_fwd) - pv0) / (1.0 / 365.0) / 365.0

    # Rho (per 1% = /100) — match DP convention: half-bumps ±dr/2, divide by dr
    dr = 0.01
    if r_curve is not None:
        # Parallel shift: multiply DFs by exp(∓ dr/2 * t)
        times = r_curve.times
        rc_up = DiscountCurve(times=times, dfs=r_curve.dfs * np.exp(-dr / 2 * times))
        rc_dn = DiscountCurve(times=times, dfs=r_curve.dfs * np.exp(+dr / 2 * times))
        rho = (_price(spot, vol, r0, q0, rc=rc_up) - _price(spot, vol, r0, q0, rc=rc_dn)) / dr / 100
    else:
        rho = (_price(spot, vol, r0 + dr / 2, q0) - _price(spot, vol, r0 - dr / 2, q0)) / dr / 100

    ql.Settings.instance().evaluationDate = eval_date
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def _dp_barrier_greeks(
    *,
    direction: BarrierDirection,
    action: BarrierAction,
    barrier: float,
    option_type: OptionType,
    strike: float,
    r_curve: DiscountCurve | None = None,
    q_curve: DiscountCurve | None = None,
) -> dict[str, float]:
    """Compute barrier Greeks via DP (numerical bump-and-revalue)."""
    ttm = calculate_year_fraction(PRICING_DATE, MATURITY)
    rc = r_curve if r_curve is not None else DiscountCurve.flat(_BARRIER_RATE, end_time=ttm)
    qc = q_curve if q_curve is not None else DiscountCurve.flat(_BARRIER_DIV, end_time=ttm)
    md = MarketData(PRICING_DATE, rc, currency=CURRENCY)
    ud = UnderlyingData(
        initial_value=_BARRIER_SPOT,
        volatility=_BARRIER_VOL,
        market_data=md,
        dividend_curve=qc,
    )
    spec = BarrierSpec(
        option_type=option_type,
        exercise_type=ExerciseType.EUROPEAN,
        strike=strike,
        maturity=MATURITY,
        barrier=barrier,
        direction=direction,
        action=action,
    )
    ov = OptionValuation(ud, spec, PricingMethod.BSM)
    return {
        "delta": ov.delta(),
        "gamma": ov.gamma(),
        "vega": ov.vega(),
        "theta": ov.theta(),
        "rho": ov.rho(),
    }


_BARRIER_GREEK_SCENARIOS = [
    # Flat curves — all 4 barrier types
    pytest.param(
        BarrierDirection.DOWN,
        BarrierAction.OUT,
        OptionType.CALL,
        100.0,
        85.0,
        None,
        None,
        id="down_out_call_flat",
    ),
    pytest.param(
        BarrierDirection.UP,
        BarrierAction.IN,
        OptionType.CALL,
        105.0,
        115.0,
        None,
        None,
        id="up_in_call_flat",
    ),
    pytest.param(
        BarrierDirection.DOWN,
        BarrierAction.IN,
        OptionType.PUT,
        100.0,
        85.0,
        None,
        None,
        id="down_in_put_flat",
    ),
    pytest.param(
        BarrierDirection.UP,
        BarrierAction.OUT,
        OptionType.PUT,
        95.0,
        125.0,
        None,
        None,
        id="up_out_put_flat",
    ),
    # Non-flat curves
    pytest.param(
        BarrierDirection.DOWN,
        BarrierAction.OUT,
        OptionType.PUT,
        105.0,
        90.0,
        *_barrier_nonflat_curves(),
        id="down_out_put_nonflat",
    ),
    pytest.param(
        BarrierDirection.UP,
        BarrierAction.OUT,
        OptionType.CALL,
        100.0,
        120.0,
        *_barrier_nonflat_curves(),
        id="up_out_call_nonflat",
    ),
    pytest.param(
        BarrierDirection.DOWN,
        BarrierAction.IN,
        OptionType.CALL,
        95.0,
        80.0,
        *_barrier_nonflat_curves(),
        id="down_in_call_nonflat",
    ),
    pytest.param(
        BarrierDirection.UP,
        BarrierAction.IN,
        OptionType.PUT,
        100.0,
        110.0,
        *_barrier_nonflat_curves(),
        id="up_in_put_nonflat",
    ),
]


@pytest.mark.parametrize(
    "direction,action,option_type,strike,barrier,r_curve,q_curve",
    _BARRIER_GREEK_SCENARIOS,
)
def test_barrier_greeks_vs_quantlib(
    direction,
    action,
    option_type,
    strike,
    barrier,
    r_curve,
    q_curve,
):
    """European barrier numerical Greeks: DP vs QL bump-and-revalue."""
    dp_g = _dp_barrier_greeks(
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
        r_curve=r_curve,
        q_curve=q_curve,
    )
    ql_g = _ql_barrier_bump_greeks(
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
        spot=_BARRIER_SPOT,
        vol=_BARRIER_VOL,
        r_curve=r_curve,
        q_curve=q_curve,
    )

    # Both sides do bump-and-revalue on analytical engines, so tight tolerances.
    tols = {"delta": 1e-4, "gamma": 1e-3, "vega": 1e-4, "theta": 0.02, "rho": 1e-3}

    for greek in ("delta", "gamma", "vega", "theta", "rho"):
        ql_val = ql_g[greek]
        dp_val = dp_g[greek]
        logger.info(
            "Barrier %s-%s %s %s K=%.0f H=%.0f | DP=%.6f QL=%.6f",
            direction.value,
            action.value,
            option_type.value,
            greek,
            strike,
            barrier,
            dp_val,
            ql_val,
        )
        assert np.isclose(dp_val, ql_val, rtol=tols[greek], atol=1e-6), (
            f"{greek}: DP {dp_val:.6f} vs QL {ql_val:.6f}"
        )
