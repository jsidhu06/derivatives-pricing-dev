# External Integrations

**Analysis Date:** 2026-04-05

## APIs & External Services

**None** - This is a pure computational library with no external API calls.

The library does not integrate with any remote APIs, cloud services, or external data feeds. All pricing and Greek calculations are performed locally using provided market data.

## Data Storage

**Databases:** None - No database integrations

**File Storage:** None - No file I/O integrations

The library operates entirely in-memory. Market data (`DiscountCurve`, `MarketData`, `StochasticProcess`) are passed in as Python objects. No persistence layer exists.

**Caching:** None - No caching framework

## Authentication & Identity

**Auth Provider:** None - No authentication required

The library is a computational library with no user authentication, API tokens, or credential management. All inputs are passed directly as function arguments.

## Monitoring & Observability

**Error Tracking:** None - No error tracking service

**Logging:** Standard Python logging module

- Framework: `logging` (stdlib)
- Configuration: Uses module-level loggers via `logging.getLogger(__name__)` in each module
- Locations where logging is configured:
  - `src/derivatives_pricing/valuation/core.py` - logger for dispatcher
  - `src/derivatives_pricing/valuation/pde.py` - logger for PDE solver
  - `src/derivatives_pricing/valuation/monte_carlo.py` - logger for MC engine with std error warnings
  - `src/derivatives_pricing/valuation/binomial.py` - logger for binomial tree
  - `src/derivatives_pricing/valuation/asian_analytical.py` - logger for analytical Asian
  - `src/derivatives_pricing/valuation/implied_volatility.py` - logger for IV solver

Logging is used for:
- Timing information (via `log_timing()` context manager in `utils.py`)
- Monte Carlo standard error warnings
- Debug-level diagnostics in numerical solvers

No external logging service (Datadog, CloudWatch, Stackdriver) is integrated.

## CI/CD & Deployment

**Hosting:** GitHub (source code repository)

- Repository: https://github.com/jsidhu06/derivatives-pricing/

**CI Pipeline:** GitHub Actions

- Config file: `.github/workflows/ci.yml`
- Workflow: `ci` (callable from pull_request, push to main, tags v*, manual dispatch)
- Matrix testing: Python 3.10-3.13 × Ubuntu/Windows/macOS
- Key checks:
  - Dependency lockfile verification (`pip-compile`)
  - Linting (ruff check/format)
  - Type checking (mypy)
  - Unit tests (pytest with --cov)
  - Notebook execution (jupyter nbconvert)
  - Coverage reporting (codecov/codecov-action@v5)

**Package Publishing:** PyPI

- Distribution: setuptools-based wheel/source distribution
- Published as: `derivatives-pricing`
- Installation: `pip install derivatives-pricing`

## Environment Configuration

**Required env vars:** None

The library requires no environment variables. All configuration is via:
1. Function arguments (pricing parameters, market data)
2. Python dataclass instances (`VanillaSpec`, `OptionValuation`, `MarketData`, etc.)
3. `pyproject.toml` build/test metadata

**Secrets location:** Not applicable

No credentials, API keys, or secrets are required or used by the library.

## Webhooks & Callbacks

**Incoming:** None

**Outgoing:** None

The library does not make any outbound HTTP calls or handle webhooks. It is purely a computational library that operates on in-memory data structures.

## Optional Test-Only Integration

**QuantLib (optional, test-only):**

- Package: QuantLib >=1.36,<2
- Installation: `pip install derivatives-pricing[quantlib]`
- Usage: Validation/comparison tests only
- Files: `tests/test_quantlib_comparison.py`, `tests/test_quantlib_greeks_comparison.py`
- Test marker: `@pytest.mark.slow` (requires `--runslow` flag to execute)
- Purpose: Verify pricing results match QuantLib implementation as benchmark
- Not imported in core library code - only in test files via `pytest.importorskip("QuantLib")`

## Optional Acceleration Integration

**Numba (optional, graceful fallback):**

- Package: numba >=0.60,<0.65
- Installation: `pip install derivatives-pricing[numba]`
- Usage: JIT compilation for PDE solvers
- File: `src/derivatives_pricing/valuation/pde.py`
- Pattern: Try-except with identity decorator fallback
  ```python
  try:
      from numba import njit as _njit
  except ModuleNotFoundError:
      def _njit(*args, **kwargs):
          if args and callable(args[0]):
              return args[0]
          return lambda fn: fn
  ```
- Purpose: Optional acceleration (library works without it)

---

*Integration audit: 2026-04-05*
