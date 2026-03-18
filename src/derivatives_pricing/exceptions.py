"""Custom exception hierarchy for the derivatives_pricing library.

All library-specific exceptions inherit from :class:`DerivativesAnalyticsError`,
enabling callers to catch *any* library error with a single ``except`` clause::

    try:
        val = OptionValuation(...)
        pv = val.present_value()
    except DerivativesAnalyticsError as exc:
        log.error("Library error: %s", exc)
"""

from __future__ import annotations


class DerivativesAnalyticsError(Exception):
    """Base exception for all library errors."""


# ── Input validation ────────────────────────────────────────────────


class ValidationError(DerivativesAnalyticsError):
    """Invalid input values (out-of-range, non-finite, mutually exclusive inputs, etc.)."""


class ConfigurationError(DerivativesAnalyticsError):
    """Wrong types passed to a public API (e.g. raw int instead of enum)."""


# ── Feature support ─────────────────────────────────────────────────


class UnsupportedFeatureError(DerivativesAnalyticsError):
    """Requested feature combination is not (yet) supported."""


# ── Numerical issues ────────────────────────────────────────────────


class NumericalError(DerivativesAnalyticsError):
    """Base for errors arising from numerical computation."""


class ArbitrageViolationError(NumericalError):
    """Model parameters imply an arbitrage (e.g. risk-neutral probability outside [0, 1])."""


class ConvergenceError(NumericalError):
    """An iterative solver failed to converge within the allowed tolerance / iterations."""


class StabilityError(NumericalError):
    """A numerical scheme's stability conditions are violated."""
