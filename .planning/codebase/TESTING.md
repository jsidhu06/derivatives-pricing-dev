# Testing Patterns

**Analysis Date:** 2026-04-05

## Test Framework

**Runner:**
- pytest 9.x (config: `[tool.pytest.ini_options]` in `pyproject.toml`)
- Config file: `pyproject.toml` (no separate `pytest.ini`)

**Assertion Library:**
- pytest built-in assertions with numpy helpers
- `numpy.isclose()` for numerical comparisons

**Run Commands:**
```bash
pytest src/derivatives_pricing/tests/ -q              # Run all tests
pytest tests/ -v                                      # Run tests with verbose output
pytest tests/ --runslow                               # Include slow tests (marked with @pytest.mark.slow)
pytest tests/ -q --cov=derivatives_pricing --cov-report=html  # Run with coverage report
```

## Test File Organization

**Location:**
- Primary test directory: `/home/jsidhu/repos/derivatives-pricing/tests/` (at repo root)
- Shared test fixtures: `tests/conftest.py`
- Test helpers: `tests/helpers.py`
- Test files co-located in single `tests/` directory (not distributed with source)

**Naming:**
- Test files: `test_*.py` (e.g., `test_bsm_valuation.py`, `test_greeks.py`)
- Test classes: `Test<FeatureName>` (e.g., `TestBSMValuation`, `TestGreeksSetup`)
- Test methods: `test_<specific_case>` (e.g., `test_bsm_call_option_atm`, `test_zero_vol_pricing`)

**Structure:**
```
tests/
├── conftest.py              # pytest fixtures and setup
├── helpers.py               # Shared factory functions and assertions
├── test_bsm_valuation.py    # Black-Scholes-Merton tests
├── test_greeks.py           # Greek calculation tests
├── test_discount_curve.py   # DiscountCurve tests
├── test_barrier.py          # Barrier option tests
├── test_asian.py            # Asian option tests
├── test_edge_cases.py       # Boundary/edge case tests
└── test_quantlib_comparison.py  # QuantLib cross-validation
```

## Test Structure

**Suite Organization:**
- Test classes group related tests (one feature per class)
- Each class can have autouse fixtures for shared setup
- Example from `tests/test_greeks.py`:
  ```python
  class TestGreeksSetup:
      """Base setup for greek tests with common parameters + factory helpers."""
      
      @pytest.fixture(autouse=True)
      def setup(self):
          self.pricing_date = dt.datetime(2025, 1, 1)
          self.maturity = dt.datetime(2026, 1, 1)
          self.spot = 100.0
          # ... more setup
  
  class TestDeltaBasicProperties(TestGreeksSetup):
      # Inherits setup, adds delta-specific tests
  ```

**Patterns:**

*Setup Pattern:*
- Fixtures are function-scoped (default in conftest.py)
- Per-test class setup via `@pytest.fixture(autouse=True)` method
- Factory methods preferred over shared fixtures for mutable state
- Example from `tests/test_greeks.py`:
  ```python
  def _make_ud(self, *, spot=None, vol=None, dividend_curve=None) -> UnderlyingData:
      """Factory: create fresh UnderlyingData without mutation."""
      md = self.market_data if pricing_date is None else MarketData(...)
      return UnderlyingData(
          initial_value=self.spot if spot is None else spot,
          volatility=self.volatility if vol is None else vol,
          market_data=md,
          dividend_curve=dividend_curve,
      )
  ```

*Teardown Pattern:*
- Rarely needed (frozen dataclasses prevent state mutation)
- Fixtures clean up resources via yield pattern if needed
- Example from `tests/conftest.py` (simple fixtures, no cleanup):
  ```python
  @pytest.fixture()
  def discount_curve(pricing_date, maturity, risk_free_rate):
      return flat_curve(pricing_date, maturity, risk_free_rate)
  ```

*Assertion Pattern:*
- Numerical assertions use `np.isclose()` with explicit tolerances:
  ```python
  assert np.isclose(result, expected, rtol=1e-4)
  ```
- Error assertions use `pytest.raises()` with message matching:
  ```python
  with pytest.raises(ValidationError, match=r"must be non-negative"):
      OptionValuation(bad_underlying, spec, method)
  ```
- Shared assertion helper `assert_greeks_close()` in `tests/helpers.py` for greek comparisons:
  ```python
  assert_greeks_close(
      lhs=my_greeks,
      rhs=reference_greeks,
      tols={"delta": 0.01, "gamma": 0.001},
      log_prefix="BSM vs QL",
      lhs_name="derivatives_pricing",
      rhs_name="QuantLib",
  )
  ```

## Fixture Chain

Fixtures in `tests/conftest.py` chain to build test data (function-scoped):

```
pricing_date (dt.datetime)
  ↓
maturity (dt.datetime)
  ↓
discount_curve (DiscountCurve)
  ↓
market_data (MarketData)
  ↓
underlying_data (UnderlyingData)
```

Example fixture definitions from `tests/conftest.py`:
```python
@pytest.fixture()
def pricing_date() -> dt.datetime:
    return PRICING_DATE  # dt.datetime(2025, 1, 1)

@pytest.fixture()
def discount_curve(pricing_date, maturity, risk_free_rate) -> DiscountCurve:
    return flat_curve(pricing_date, maturity, risk_free_rate)

@pytest.fixture()
def market_data(pricing_date, discount_curve, currency) -> MarketData:
    return MarketData(pricing_date, discount_curve, currency=currency)

@pytest.fixture()
def underlying_data(market_data) -> UnderlyingData:
    return UnderlyingData(
        initial_value=SPOT,
        volatility=VOL,
        market_data=market_data,
    )
```

## Mocking

**Framework:** pytest built-in monkeypatch (minimal mocking philosophy)

**Patterns:**
- Rarely use mocking; prefer real objects with test data
- When needed, use monkeypatch for method override:
  ```python
  def test_with_mock(monkeypatch):
      mock_result = 42.0
      monkeypatch.setattr(SomeClass, "method", lambda self: mock_result)
      # test code
  ```
- QuantLib tests use `pytest.importorskip("QuantLib")` to conditionally skip

**What to Mock:**
- External dependencies (QuantLib for conditional testing)
- Heavy I/O operations (not applicable in this library)

**What NOT to Mock:**
- Internal option valuation engines (test with real implementations)
- Discount curves and market data (use test factories)
- Stochastic processes (use SimulationConfig with seeds)

## Fixtures and Factories

**Test Data:**
- Constants defined in `tests/helpers.py`: `PRICING_DATE`, `MATURITY`, `SPOT`, `STRIKE`, `VOL`, `RATE`
- Factory functions in `tests/helpers.py`:
  ```python
  def flat_curve(pricing_date: dt.datetime, maturity: dt.datetime, rate: float) -> DiscountCurve:
      """Build flat discount curve for test."""
      ttm = calculate_year_fraction(pricing_date, maturity)
      return DiscountCurve.flat(rate, end_time=ttm)
  
  def underlying(...) -> UnderlyingData:
      """Build UnderlyingData; signature mirrors production dataclass."""
      return UnderlyingData(...)
  
  def spec(...) -> VanillaSpec:
      """Build VanillaSpec with defaults."""
      return VanillaSpec(...)
  ```

**Location:**
- Shared factories in `tests/helpers.py`
- Test-class-local factories in method: `def _make_ud(self, ...)`
- Prefer factories over shared fixtures for stateful objects (allows independent test mutations)

## Coverage

**Requirements:** No explicit target (not enforced)

**View Coverage:**
```bash
pytest tests/ --cov=derivatives_pricing --cov-report=html
# Open htmlcov/index.html
```

**Config:** `[tool.coverage.run]` and `[tool.coverage.report]` in `pyproject.toml`:
```ini
[tool.coverage.run]
branch = true
source = ["derivatives_pricing"]

[tool.coverage.report]
show_missing = true
skip_covered = true
```

## Test Types

**Unit Tests:**
- Scope: Individual functions and classes (e.g., `DiscountCurve.df()`, `calculate_year_fraction()`)
- Approach: Test with valid inputs, boundary values, and error cases
- Files: `test_discount_curve.py`, `test_utils_parity_forward.py`
- Example from `tests/test_discount_curve.py`:
  ```python
  class TestDiscountCurveConstruction:
      def test_flat_curve_basic(self):
          curve = DiscountCurve.flat(rate=0.05, end_time=1.0)
          assert np.isclose(curve.flat_rate, 0.05, rtol=1e-4)
  ```

**Integration Tests:**
- Scope: Option valuation across methods (BSM, binomial, PDE, Monte Carlo)
- Approach: Compare results between methods, validate economic properties (American ≥ European)
- Files: `test_bsm_valuation.py`, `test_greeks.py`, `test_barrier.py`, `test_asian.py`
- Example from `tests/test_bsm_valuation.py`:
  ```python
  def test_bsm_call_put_parity(self):
      """Test BSM call-put parity: C - P = S*exp(-q*T) - K*exp(-r*T)."""
      call_price = _bsm(underlying(), spec(OptionType.CALL))
      put_price = _bsm(underlying(), spec(OptionType.PUT))
      parity_rhs = SPOT * np.exp(-0.0 * 1.0) - STRIKE * np.exp(-RATE * 1.0)
      assert np.isclose(call_price - put_price, parity_rhs, rtol=1e-10)
  ```

**Cross-Validation Tests:**
- Scope: Compare derivatives_pricing results against QuantLib
- Approach: Optional comparison tests (skip if QuantLib not installed)
- Files: `test_quantlib_comparison.py`, `test_quantlib_greeks_comparison.py`
- Example from `tests/test_quantlib_comparison.py`:
  ```python
  ql = pytest.importorskip("QuantLib")  # Skip if not installed
  
  def test_european_call_vs_ql(spot, strike, vol, rate, days, option_type):
      # Compute with both libraries
      dp_price = OptionValuation(underlying, spec, method).present_value()
      ql_price = _ql_european_price(...)
      assert np.isclose(dp_price, ql_price, rtol=0.01)
  ```

**Edge Case Tests:**
- Scope: Boundary conditions (zero vol, near-zero expiry, deep ITM/OTM, extreme rates)
- Approach: Verify correct behavior or graceful failure
- Files: `test_edge_cases.py`
- Example from `tests/test_edge_cases.py`:
  ```python
  @pytest.mark.parametrize(
      "option_type,strike,expected_positive",
      [
          (OptionType.CALL, 90.0, True),   # ITM call
          (OptionType.CALL, 110.0, False), # OTM call
      ],
  )
  def test_bsm_zero_vol_pricing(self, option_type, strike, expected_positive):
      ud = _underlying(vol=0.0)
      spec = _spec(strike=strike, option_type=option_type)
      pv = _pv(ud, spec, PricingMethod.BSM)
      assert (pv > 0) == expected_positive
  ```

**Slow Tests:**
- Marked with `@pytest.mark.slow`
- Skipped by default; use `pytest tests/ --runslow` to include
- Typically: high-dimensional Monte Carlo, fine PDE grids, QuantLib comparisons
- Example from `tests/test_asian.py`:
  ```python
  @pytest.mark.slow
  def test_asian_three_method_convergence(...):
      # Expensive comparison across BSM, binomial, Monte Carlo
  ```

## Common Patterns

**Async Testing:**
- Not applicable (no async code in derivatives_pricing)

**Error Testing:**
- Use `pytest.raises()` context manager with error type and message match
- Example from `tests/test_edge_cases.py`:
  ```python
  def test_negative_volatility_raises(self):
      with pytest.raises(ValidationError, match="volatility must be >= 0"):
          UnderlyingData(initial_value=100.0, volatility=-0.1, market_data=market_data)
  ```

**Parametrized Tests:**
- Use `@pytest.mark.parametrize()` for multiple input combinations
- Example from `tests/test_edge_cases.py`:
  ```python
  @pytest.mark.parametrize(
      "spot,strike,vol,short_rate,days,option_type",
      [
          (100, 100, 0.2, 0.05, 365, OptionType.CALL),
          (100, 100, 0.2, 0.05, 365, OptionType.PUT),
          # ... more cases
      ]
  )
  def test_american_geq_european(spot, strike, vol, short_rate, days, option_type):
      # Test with each parameter combination
  ```

**Test Organization with Inheritance:**
- Base class with `@pytest.fixture(autouse=True) setup()` provides common fixtures
- Child test classes inherit setup and add specific tests
- Example from `tests/test_greeks.py`:
  ```python
  class TestGreeksSetup:
      @pytest.fixture(autouse=True)
      def setup(self):
          # Shared setup
  
  class TestDeltaBasicProperties(TestGreeksSetup):
      def test_delta_call_positive(self):
          # Uses inherited setup
  ```

---

*Testing analysis: 2026-04-05*
