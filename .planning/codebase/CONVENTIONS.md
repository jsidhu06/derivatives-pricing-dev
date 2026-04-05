# Coding Conventions

**Analysis Date:** 2026-04-05

## Naming Patterns

**Files:**
- Module files use `snake_case`: `discount_curve.py`, `bsm.py`, `monte_carlo.py`
- Private implementation classes prefixed with underscore: `_BSMEuropeanValuation`, `_BinomialAmericanValuation`
- Enum definitions in dedicated `enums.py`
- Exception classes in dedicated `exceptions.py`

**Functions:**
- Public functions use `snake_case`: `calculate_year_fraction()`, `pv_discrete_dividends()`, `log_timing()`
- Private helper functions prefixed with underscore: `_day_count_30_360_us()`, `_adjusted_spot()`
- Static methods used for pure utility functions without side effects: `_implied_rate_from_df()`

**Variables:**
- Local variables use `snake_case`: `time_to_maturity`, `discount_factor`, `spot_price`
- Constants in UPPERCASE: `SECONDS_IN_DAY = 86400`
- Parameters maintain clarity: `df_r` (risk-free discount factor), `df_q` (dividend discount factor)
- Loop variables follow convention: `for ex_date, amount in self.discrete_dividends:`

**Types:**
- Enum values use UPPERCASE: `OptionType.CALL`, `ExerciseType.EUROPEAN`, `PricingMethod.BSM`
- NamedTuples document input structures: `_BSMInputs`, `_MCSimulationResult`
- Type hints use modern Python 3.10+ union syntax: `float | None` not `Optional[float]`
- Imported types guarded with `TYPE_CHECKING` block to prevent circular imports

## Code Style

**Formatting:**
- Formatter: **Ruff** (enforced via pre-commit)
- Line length: 100 characters
- Quote style: Double quotes (`"string"`)
- Target version: Python 3.10+

**Linting:**
- Linter: **Ruff** (enforced via pre-commit)
- Additional tool: **Semgrep** with custom enum identity-comparison rule (`.semgrep/`)
- Config: `pyproject.toml` sections `[tool.ruff]` and `[tool.ruff.format]`

**Type Checking:**
- Type checker: **mypy** (very relaxed/developer-friendly)
- Config: `[tool.mypy]` in `pyproject.toml` with `check_untyped_defs = false`
- Guidelines: Type hints encouraged for clarity but not strictly enforced; focus on important APIs

## Import Organization

**Order:**
1. `from __future__ import annotations` (always first in files with forward references)
2. Standard library imports (e.g., `import datetime as dt`, `from abc import ABC`)
3. Third-party imports (e.g., `import numpy as np`, `from scipy.stats import norm`)
4. Relative local imports (e.g., `from ..enums import OptionType`, `from .core import OptionValuation`)

**Path Aliases:**
- Within package: Use relative imports (`from ..enums import OptionType`, `from .core import UnderlyingData`)
- In tests: Use absolute imports from `derivatives_pricing` root (`from derivatives_pricing.enums import OptionType`)

**Circular Import Prevention:**
- Use `from __future__ import annotations` to defer annotation evaluation
- Guard imports behind `if TYPE_CHECKING:` block for type-only imports
- Example in `src/derivatives_pricing/utils.py`:
  ```python
  if TYPE_CHECKING:
      from .rates import DiscountCurve
  ```

## Error Handling

**Patterns:**
- All validation happens in `__post_init__()` of frozen dataclasses
- Specific exception types thrown based on error category:
  - `ValidationError`: Invalid input values (out-of-range, non-finite, etc.)
  - `ConfigurationError`: Wrong types passed to API (e.g., int instead of enum)
  - `UnsupportedFeatureError`: Valid inputs but unsupported combination
  - `NumericalError` (and subtypes): Computational failures (`ArbitrageViolationError`, `ConvergenceError`, `StabilityError`)
- Exception chaining with `from exc` when converting or rethrowing
- Example from `src/derivatives_pricing/valuation/contracts.py`:
  ```python
  try:
      strike = float(self.strike)
  except (TypeError, ValueError) as exc:
      raise ConfigurationError("VanillaSpec.strike must be numeric") from exc
  ```

**Validation in Frozen Dataclasses:**
- Use `object.__setattr__(self, "field", value)` to set fields during `__post_init__` on frozen classes
- Normalize types and check bounds before assignment
- Example from `src/derivatives_pricing/valuation/core.py`:
  ```python
  object.__setattr__(
      self,
      "discrete_dividends",
      tuple(sorted(cleaned, key=lambda x: x[0])),
  )
  ```

## Logging

**Framework:** Python standard `logging` module

**Patterns:**
- Module-level logger: `logger = logging.getLogger(__name__)`
- Used sparingly for debug timing and warnings
- Context manager `log_timing()` in `src/derivatives_pricing/utils.py` wraps performance-critical sections:
  ```python
  with log_timing(logger, "PDE solve", enabled=debug):
      # expensive computation
  ```
- Warnings module for non-critical issues: `warnings.warn("message", stacklevel=2)`

## Comments

**When to Comment:**
- Explain the "why", not the "what"
- Document non-obvious numerical limits: `if denominator < 1e-300:  # Zero (or near-zero) vol`
- Mark boundary cases and special handling
- Use section markers for organization: `# ─ Input validation ─────` (decorative)

**JSDoc/TSDoc:**
- Use numpy-style docstrings for public APIs
- Parameters documented with types and descriptions
- Returns section specifies type and meaning
- Example from `src/derivatives_pricing/rates.py`:
  ```python
  def calculate_year_fraction(
      start_date,
      end_date,
      day_count_convention: DayCountConvention = DayCountConvention.ACT_365F,
  ) -> float:
      """Calculate year fraction between two dates.
      
      Parameters
      ----------
      start_date
          Start date.
      end_date
          End date.
      day_count_convention
          Day-count basis. Supported values are ``ACT_365F``, ``ACT_360``,
          ``ACT_365_25``, and ``THIRTY_360_US``.
      
      Returns
      -------
      float
          Year fraction between ``start_date`` and ``end_date``.
      """
  ```

## Function Design

**Size:** Prefer focused functions under 50 lines; larger functions broken into private helpers

**Parameters:**
- Keyword-only parameters for clarity when many arguments: `def __init__(self, ..., *, kw_only_arg)`
- Use positional-or-keyword for simple cases
- Dataclasses preferred over many individual parameters (see `UnderlyingData`, `SimulationConfig`)

**Return Values:**
- Return concrete types, not `Any` (use type hints for clarity)
- For multiple returns, use NamedTuple or dataclass: `_BSMInputs`, `ImpliedVolResult`
- Prefer single responsibility: Greeks methods return float, not dict

**Bump-and-Revalue Pattern:**
- Greeks computed via numerical differentiation create fresh `UnderlyingData` instances:
  ```python
  def _bump_underlying(self, param_name: str, bump: float) -> UnderlyingData:
      return self.underlying.replace(  # Uses dataclasses.replace()
          **{param_name: getattr(self.underlying, param_name) + bump}
      )
  ```
- Never mutate existing `UnderlyingData` objects (frozen by design)

## Module Design

**Exports:**
- Public API controlled via `__all__` in module `__init__.py` files
- Top-level package exports from `src/derivatives_pricing/__init__.py`
- Valuation subpackage exports from `src/derivatives_pricing/valuation/__init__.py`
- Example from `src/derivatives_pricing/__init__.py`:
  ```python
  __all__ = [
      "MarketData",
      "DiscountCurve",
      "OptionValuation",
      "VanillaSpec",
      # ...
  ]
  ```

**Barrel Files:**
- Subpackage `__init__.py` files re-export internal classes for public API
- Simplifies consumer imports: `from derivatives_pricing import OptionValuation`
- Hides implementation details like `_BSMEuropeanValuation`

**Private Modules:**
- Engine implementations in `valuation/` are private (prefixed underscore): `_bsm.py`, `_binomial.py`
- Accessed via dispatcher `OptionValuation` only, never directly imported

---

*Convention analysis: 2026-04-05*
