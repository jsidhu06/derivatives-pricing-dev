"""Kunitomo-Ikeda (1992) analytical pricing for European double-barrier options.

Implements the closed-form infinite-series solution for European double
knock-out and (by in-out parity) knock-in options under GBM with flat barriers.

Scope
-----
- European exercise only (no closed form exists for American double barriers).
- Continuous monitoring only (the Broadie-Glasserman-Kou continuity
  correction does not extend cleanly to two barriers).
- Constant (flat-equivalent) ``r`` and ``q`` — the K-I series assumes a
  constant cost of carry ``b = r - q``.  For curved term structures the flat-
  equivalent rates at ``T`` are used; this is an approximation, exact only
  for genuinely flat curves.
- No discrete dividends.
- No rebate.

Anything outside this envelope raises ``UnsupportedFeatureError``; route those
cases through ``PricingMethod.PDE_FD`` instead.

See also
--------
- ``barrier_analytical.py`` — Reiner-Rubinstein analytical pricing for
  *single*-barrier options.

References
----------
Kunitomo, N. and Ikeda, M. (1992). "Pricing Options with Curved Boundaries."
    *Mathematical Finance*, 2(4), 275-298.
Haug, E. G. (2007). *The Complete Guide to Option Pricing Formulas*, 2nd ed.,
    Section 4.17.  Provides the flat-barrier specialisation used here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import norm

from ..enums import (
    BarrierAction,
    BarrierMonitoring,
    ExerciseType,
    OptionType,
    PricingMethod,
)
from ..exceptions import UnsupportedFeatureError
from .contracts import VanillaSpec

if TYPE_CHECKING:
    from .contracts import DoubleBarrierSpec
    from .core import OptionValuation


# Series truncation: terms for n in [-N, N].  K-I converges geometrically for
# typical parameters — 20 terms gives ~1e-12 precision at U/L < 2.
_DEFAULT_N_TERMS = 20


def _kunitomo_ikeda_dko_no_rebate(
    *,
    spot: float,
    strike: float,
    lower: float,
    upper: float,
    df_r: float,
    df_q: float,
    sigma: float,
    time_to_maturity: float,
    option_type: OptionType,
    n_terms: int = _DEFAULT_N_TERMS,
) -> float:
    """Closed-form European double-KO price (no rebate, flat barriers).

    Implements the K-I series in the symmetric ``(lo, hi)`` form so the call
    and put share the same series body — only the integration limits and the
    outer sign flip.  For a call with strike ``K`` clipped to ``[L, U]``:

      * ``(lo, hi) = (max(K, L), U)`` for the CALL,
      * ``(lo, hi) = (L,  min(K, U))`` for the PUT.

    Caller must ensure ``L < spot < U`` (corridor not breached at inception).

    Discount factors are the primary inputs (consistent with how the
    valuation context exposes them via ``DiscountCurve.df``).  The flat-
    equivalent rates ``r`` and ``q`` are derived internally for ``μ`` and the
    drift; the discount factors themselves feed the outer
    ``S·df_q·Σ_S − K·df_r·Σ_K`` scaling.
    """
    # Early-out: payoff is identically zero given the surviving corridor.
    if option_type is OptionType.CALL and strike >= upper:
        return 0.0
    if option_type is OptionType.PUT and strike <= lower:
        return 0.0

    if option_type is OptionType.CALL:
        lo = max(lower, strike)
        hi = upper
    else:
        lo = lower
        hi = min(upper, strike)

    # Flat-equivalent rates back-derived from the discount factors.
    r = -np.log(df_r) / time_to_maturity
    q = -np.log(df_q) / time_to_maturity
    b = r - q
    sigma_sq = sigma * sigma
    sigma_sqrt_T = sigma * np.sqrt(time_to_maturity)
    drift_T = (b + 0.5 * sigma_sq) * time_to_maturity
    mu = 2.0 * b / sigma_sq + 1.0  # flat-barrier μ₁ = μ₃; μ₂ = 0 drops out

    log_L = np.log(lower)
    log_U = np.log(upper)
    log_S = np.log(spot)
    log_lo = np.log(lo)
    log_hi = np.log(hi)
    log_U_over_L = log_U - log_L

    sum_S = 0.0
    sum_K = 0.0
    for n in range(-n_terms, n_terms + 1):
        # log(S · U^{2n} / L^{2n}) — common to the d_1, d_2 numerators.
        log_SU2n_L2n = log_S + 2.0 * n * log_U_over_L
        log_arg_1 = log_SU2n_L2n - log_lo
        log_arg_2 = log_SU2n_L2n - log_hi

        # log(L^{2n+2} / (S · U^{2n})) — common to the d_3, d_4 numerators.
        log_L2n2_over_SU2n = (2.0 * n + 2.0) * log_L - log_S - 2.0 * n * log_U
        log_arg_3 = log_L2n2_over_SU2n - log_lo
        log_arg_4 = log_L2n2_over_SU2n - log_hi

        d_1 = (log_arg_1 + drift_T) / sigma_sqrt_T
        d_2 = (log_arg_2 + drift_T) / sigma_sqrt_T
        d_3 = (log_arg_3 + drift_T) / sigma_sqrt_T
        d_4 = (log_arg_4 + drift_T) / sigma_sqrt_T

        # Geometric prefactors:
        #   R_1 = (U^n / L^n)^μ          R_3 = (U^n / L^n)^{μ-2}
        #   R_2 = (L^{n+1} / (U^n S))^μ  R_4 = (L^{n+1} / (U^n S))^{μ-2}
        log_UL_n = n * log_U_over_L
        log_LUS_n = (n + 1.0) * log_L - n * log_U - log_S

        R_1 = np.exp(mu * log_UL_n)
        R_2 = np.exp(mu * log_LUS_n)
        R_3 = np.exp((mu - 2.0) * log_UL_n)
        R_4 = np.exp((mu - 2.0) * log_LUS_n)

        sum_S += R_1 * (norm.cdf(d_1) - norm.cdf(d_2)) - R_2 * (norm.cdf(d_3) - norm.cdf(d_4))
        sum_K += R_3 * (norm.cdf(d_1 - sigma_sqrt_T) - norm.cdf(d_2 - sigma_sqrt_T)) - R_4 * (
            norm.cdf(d_3 - sigma_sqrt_T) - norm.cdf(d_4 - sigma_sqrt_T)
        )

    if option_type is OptionType.CALL:
        return float(max(spot * df_q * sum_S - strike * df_r * sum_K, 0.0))
    return float(max(strike * df_r * sum_K - spot * df_q * sum_S, 0.0))


class _AnalyticalDoubleBarrierValuation:
    """Kunitomo-Ikeda analytical valuation for European double-barrier options.

    Dispatched by ``OptionValuation`` when ``spec`` is :class:`DoubleBarrierSpec`
    and ``pricing_method`` is ``BSM``.  Supports European exercise, continuous
    monitoring, flat curves, no discrete dividends, no rebate; anything else
    raises ``UnsupportedFeatureError`` — route to ``PricingMethod.PDE_FD``.
    """

    def __init__(self, valuation_ctx: OptionValuation) -> None:
        self.valuation_ctx = valuation_ctx
        self.underlying = valuation_ctx.underlying
        self.spec: DoubleBarrierSpec = valuation_ctx.spec  # type: ignore[assignment]

        spec = self.spec

        if spec.exercise_type is not ExerciseType.EUROPEAN:
            raise UnsupportedFeatureError(
                "Kunitomo-Ikeda analytical pricing is European only "
                f"(got {spec.exercise_type.name}). Use PricingMethod.PDE_FD."
            )
        if spec.monitoring is not BarrierMonitoring.CONTINUOUS:
            raise UnsupportedFeatureError(
                "Kunitomo-Ikeda analytical pricing requires continuous monitoring "
                f"(got {spec.monitoring.name}). The Broadie-Glasserman-Kou "
                "continuity correction does not extend cleanly to two barriers; "
                "use PricingMethod.PDE_FD for discrete monitoring."
            )
        if spec.rebate > 0.0:
            raise UnsupportedFeatureError(
                "Kunitomo-Ikeda analytical pricing does not support rebates. "
                "Use PricingMethod.PDE_FD."
            )
        if self.underlying.discrete_dividends:
            raise UnsupportedFeatureError(
                "Kunitomo-Ikeda analytical pricing does not support discrete "
                "dividends. Use PricingMethod.PDE_FD."
            )

    def solve(self) -> float:
        return self.present_value()

    def present_value(self) -> float:
        spec = self.spec
        ctx = self.valuation_ctx
        underlying = self.underlying

        S = float(underlying.initial_value)
        sigma = float(underlying.volatility)
        T = ctx._maturity_year_fraction()

        df_r = float(ctx.discount_curve.df(T))
        dividend_curve = underlying.dividend_curve
        df_q = float(dividend_curve.df(T)) if dividend_curve is not None else 1.0

        L = float(spec.lower_barrier)
        U = float(spec.upper_barrier)

        if ctx._barrier_triggered_at_inception():
            if spec.action is BarrierAction.OUT:
                return 0.0
            return self._vanilla_pv()

        ko_value = _kunitomo_ikeda_dko_no_rebate(
            spot=S,
            strike=float(spec.strike),
            lower=L,
            upper=U,
            df_r=df_r,
            df_q=df_q,
            sigma=sigma,
            time_to_maturity=T,
            option_type=spec.option_type,
        )
        if spec.action is BarrierAction.OUT:
            return ko_value

        # In-out parity (no rebate): V_DKI = V_vanilla - V_DKO.
        return float(max(self._vanilla_pv() - ko_value, 0.0))

    def theta(self) -> float:
        """Theta via the Black-Scholes PDE identity (same pattern as
        ``_AnalyticalBarrierValuation``).

        Returns per-calendar-day theta.
        """
        ctx = self.valuation_ctx
        underlying = self.underlying

        S = float(underlying.initial_value)
        sigma = float(underlying.volatility)
        T = ctx._maturity_year_fraction()

        df_r = float(ctx.discount_curve.df(T))
        r = -np.log(df_r) / T
        dividend_curve = underlying.dividend_curve
        df_q = float(dividend_curve.df(T)) if dividend_curve is not None else 1.0
        q = -np.log(df_q) / T

        V = ctx.present_value()
        delta = ctx.delta()
        gamma = ctx.gamma()

        theta_annual = r * V - (r - q) * S * delta - 0.5 * sigma * sigma * S * S * gamma
        return float(theta_annual / 365.0)

    def _vanilla_pv(self) -> float:
        """Vanilla European price via the BSM engine — used for DKI parity
        and for the triggered-at-inception KI fallback.
        """
        from .core import OptionValuation

        spec = self.spec
        vanilla_spec = VanillaSpec(
            option_type=spec.option_type,
            exercise_type=spec.exercise_type,
            strike=spec.strike,
            maturity=spec.maturity,
            currency=spec.currency,
            contract_size=spec.contract_size,
        )
        return float(
            OptionValuation(
                underlying=self.underlying,
                spec=vanilla_spec,
                pricing_method=PricingMethod.BSM,
            ).present_value()
        )
