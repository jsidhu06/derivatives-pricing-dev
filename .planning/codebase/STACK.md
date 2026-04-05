# Technology Stack

**Analysis Date:** 2026-04-05

## Languages

**Primary:**
- Python 3.10+ - All production code, library implementation
- Supports Python 3.10, 3.11, 3.12, 3.13 (tested in CI)

## Runtime

**Environment:**
- Python 3.10 or higher (specified in `pyproject.toml`)

**Package Manager:**
- pip with setuptools backend
- Lockfile: `.pre-commit-config.yaml` and `requirements/` directory with `dev.txt`
- Build system: setuptools 61.0+ with PEP 517/518 compliance

## Frameworks

**Core Framework:**
- None - Pure Python library (no web framework or application framework)

**Mathematical/Scientific:**
- numpy >=1.24,<3.0 - Array operations, linear algebra, numerical computations
  - Used throughout: `rates.py`, `stochastic_processes.py`, `valuation/pde.py`, `valuation/monte_carlo.py`, `valuation/binomial.py`
  - Critical for: discount curve interpolation, binomial tree lattices, PDE grid solvers, Monte Carlo path generation
- scipy >=1.10,<3.0 - Scientific computing, optimization, statistics
  - Used in: `valuation/bsm.py`, `valuation/asian_analytical.py`, `valuation/barrier_analytical.py`, `valuation/implied_volatility.py`
  - Critical for: normal distribution CDF (scipy.stats.norm), root finding (scipy.optimize)
- pandas >=1.5,<4.0 - Data manipulation and time series
  - Used in: `stochastic_processes.py`, `valuation/core.py`, `valuation/monte_carlo.py`, `valuation/binomial.py`
  - Purpose: Date-range generation, time grid creation via pd.date_range()

**Testing:**
- pytest >=9,<10 - Test runner and framework
  - Config: `pyproject.toml` with testpaths=["tests"], marked slow tests with `@pytest.mark.slow`
  - Extras: pytest-cov for coverage, pytest-xdist for parallel test execution
- pytest-cov >=6,<7 - Coverage reporting
- pytest-xdist >=3.5,<4 - Parallel test execution

**Build/Dev:**
- ruff >=0.15 - Formatter and linter (replaces black/flake8)
  - Config: `pyproject.toml` [tool.ruff] section - line-length=100, target-version=py310
  - Formatting: double-quote strings
- mypy >=1.19 - Static type checker
  - Config: `pyproject.toml` [tool.mypy] - lenient settings, ignore_missing_imports=true
- pylint >=3.3,<4 - Code quality linting
  - Config: `pyproject.toml` with disabled rules for docstrings, naming, complexity
- pre-commit >=4.5,<5 - Git hook framework (`.pre-commit-config.yaml` present)
- pip-tools >=7.5,<8 - Lockfile and dependency management

## Optional Dependencies

**For QuantLib Integration Testing:**
- QuantLib >=1.36,<2 - Optional integration via `[quantlib]` extra
  - Used in: `tests/test_quantlib_comparison.py`, `tests/test_quantlib_greeks_comparison.py`
  - Purpose: Validation and comparison of pricing results (not used in core library)
  - Installation: `pip install derivatives-pricing[quantlib]`

**For Numerical Acceleration:**
- numba >=0.60,<0.65 - Optional JIT compilation via `[numba]` extra
  - Used in: `valuation/pde.py` with conditional import and identity decorator fallback
  - Purpose: Acceleration of PDE solvers (gracefully degrades if not installed)
  - Installation: `pip install derivatives-pricing[numba]`

**For Jupyter/Notebook Support:**
- jupyterlab >=4.3,<5
- nbconvert >=7.16,<8
- ipykernel >=6.29,<7
- matplotlib >=3.9,<4
- plotly >=6,<7
- Installation: `pip install derivatives-pricing[jupyter]`

## Configuration

**Environment:**
- No `.env` file required - pure library with no environment configuration
- No external secrets, API keys, or credentials needed
- Python path configuration: `pythonpath = ["src"]` in `pyproject.toml` [tool.pytest.ini_options]

**Build:**
- `pyproject.toml` - PEP 518 project metadata and configuration
  - Build backend: setuptools.build_meta
  - Package discovery: src layout (`where = ["src"]`)
  - Marker file: `src/derivatives_pricing/py.typed` (PEP 561 type hints marker)
- No `setup.py` (uses modern pyproject.toml only)

**CI/CD:**
- GitHub Actions in `.github/workflows/ci.yml`
  - Test matrix: Python 3.10, 3.11, 3.12, 3.13
  - Platforms: Ubuntu, Windows, macOS
  - Lint/format checks: Ruff, Mypy
  - Test commands: pytest with --cov, --runslow flags
  - Coverage upload: codecov/codecov-action@v5
  - Notebook execution: jupyter nbconvert

## Platform Requirements

**Development:**
- Python 3.10+ with pip
- Unix-like shell for scripts (bash)
- For pre-commit hooks: `.pre-commit-config.yaml` defines hooks
- Git for version control and CI/CD

**Production:**
- Python 3.10+
- numpy, scipy, pandas (as specified in dependencies)
- No system libraries or external binaries required
- Cross-platform compatible (tested on Ubuntu, Windows, macOS)

**Optional/Enhancement:**
- For JIT acceleration: numba installation
- For validation/testing: QuantLib installation
- For visualization: matplotlib, plotly (via jupyter extra)

---

*Stack analysis: 2026-04-05*
