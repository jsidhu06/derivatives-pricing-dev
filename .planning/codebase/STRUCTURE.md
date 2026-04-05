# Codebase Structure

**Analysis Date:** 2026-04-05

## Directory Layout

```
src/derivatives_pricing/
├── __init__.py                    # Public API exports
├── enums.py                       # Enum definitions (OptionType, ExerciseType, PricingMethod, etc.)
├── exceptions.py                  # Exception hierarchy
├── rates.py                       # DiscountCurve with log-linear interpolation
├── market_environment.py          # MarketData, CorrelationContext containers
├── stochastic_processes.py        # GBMProcess, JDProcess, SRDProcess, PathSimulation, SimulationConfig
├── utils.py                       # Pure utility functions (day-count, forwards, dividends)
│
└── valuation/
    ├── __init__.py                # Valuation public API exports
    ├── core.py                    # OptionValuation facade, UnderlyingData, spec contracts
    ├── contracts.py               # VanillaSpec, PayoffSpec, AsianSpec, BarrierSpec
    ├── params.py                  # MonteCarloParams, BinomialParams, PDEParams
    │
    ├── bsm.py                     # Black-Scholes-Merton pricing engine
    ├── binomial.py                # Binomial tree pricing engine
    ├── monte_carlo.py             # Monte Carlo pricing engine
    ├── pde.py                     # Finite-difference PDE pricing engine
    ├── asian_analytical.py        # Analytical Asian (geometric averaging)
    ├── barrier_analytical.py      # Analytical barrier options
    └── implied_volatility.py      # Implied volatility solver

tests/                            # (Not in /src/derivatives_pricing/ - separate test suite)
```


## Directory Purposes

**`src/derivatives_pricing/`:**
- Purpose: Top-level package namespace; public API consolidation
- Contains: Core domain objects, enums, exceptions
- Key files: `__init__.py` (re-exports public API), `enums.py`, `exceptions.py`

**`src/derivatives_pricing/valuation/`:**
- Purpose: Valuation-specific logic; all pricing engines and contracts
- Contains: Four pricing engines (BSM, binomial, MC, PDE), contract specs, dispatcher, parameters
- Key files: `core.py` (OptionValuation), `contracts.py` (specs), `params.py` (engine-specific tuning)


## Key File Locations

**Entry Points:**
- `src/derivatives_pricing/__init__.py`: Imports and re-exports all public classes/functions (MarketData, OptionValuation, DiscountCurve, GBMProcess, Enums, exceptions)
- `src/derivatives_pricing/valuation/__init__.py`: Exports spec classes and OptionValuation

**Configuration:**
- `src/derivatives_pricing/enums.py`: All enum definitions (immutable, string values)
- `src/derivatives_pricing/valuation/params.py`: Frozen dataclasses for method-specific params

**Core Logic:**
- `src/derivatives_pricing/valuation/core.py`: `OptionValuation` facade and `UnderlyingData` container (1217 lines)
- `src/derivatives_pricing/valuation/contracts.py`: Contract specs (VanillaSpec, PayoffSpec, AsianSpec, BarrierSpec) (537 lines)
- `src/derivatives_pricing/stochastic_processes.py`: Process definitions and PathSimulation (923 lines)

**Pricing Engines:**
- `src/derivatives_pricing/valuation/bsm.py`: Black-Scholes-Merton analytical pricing (301 lines)
- `src/derivatives_pricing/valuation/binomial.py`: Binomial tree pricing for vanilla, Asian, barrier (1148 lines)
- `src/derivatives_pricing/valuation/monte_carlo.py`: Monte Carlo with Longstaff-Schwartz (1816 lines)
- `src/derivatives_pricing/valuation/pde.py`: Finite-difference PDE solver (2530 lines) — **largest file**
- `src/derivatives_pricing/valuation/asian_analytical.py`: Analytical Asian (geometric only) (467 lines)
- `src/derivatives_pricing/valuation/barrier_analytical.py`: Analytical barrier options (563 lines)

**Specialized:**
- `src/derivatives_pricing/rates.py`: DiscountCurve with log-linear interpolation, curve building (228 lines)
- `src/derivatives_pricing/market_environment.py`: MarketData and CorrelationContext (129 lines)
- `src/derivatives_pricing/utils.py`: Pure utility functions (449 lines)
- `src/derivatives_pricing/valuation/implied_volatility.py`: IV solver via root-finding (429 lines)


## Naming Conventions

**Files:**
- Snake_case: `bsm.py`, `monte_carlo.py`, `stochastic_processes.py`
- Single engine per file (e.g., `bsm.py` contains BSM-specific classes)
- Test files: `test_*.py` (in separate tests/ directory, not co-located)

**Directories:**
- Snake_case: `valuation/`, `src/`, `tests/`
- Feature grouping: All pricing in `valuation/`

**Classes:**
- PascalCase public: `OptionValuation`, `UnderlyingData`, `VanillaSpec`, `GBMProcess`
- Underscore-prefix private: `_BSMValuationBase`, `_BinomialEuropeanValuation`, `_MCAmericanValuation` (private engine implementations)
- Frozen dataclasses: All data containers use `@dataclass(frozen=True, slots=True)`

**Functions:**
- snake_case: `present_value()`, `delta()`, `implied_volatility()`, `calculate_year_fraction()`
- Helper methods prefixed with underscore: `_bump_underlying()`, `_build_impl()`, `_asian_fixing_dates()`

**Enums:**
- PascalCase class, UPPER_SNAKE_CASE values: `OptionType.CALL`, `ExerciseType.EUROPEAN`, `PricingMethod.BSM`
- String values: `OptionType.CALL.value == "call"`

**Exception Classes:**
- PascalCase, `Error` suffix: `ValidationError`, `ConfigurationError`, `UnsupportedFeatureError`, `NumericalError`


## Where to Add New Code

**New Pricing Method (e.g., Monte Carlo with Quasi-Random Numbers):**
- Create new file: `src/derivatives_pricing/valuation/qmc.py`
- Define private engine classes following pattern: `_QMCEuropeanValuation`, `_QMCAmericanValuation`, etc.
- Add entries to registries in `src/derivatives_pricing/valuation/core.py` (create new registries or add to existing if method supports those specs)
- Export from `src/derivatives_pricing/valuation/__init__.py` if needed for tests

**New Contract Type (e.g., Spread Options):**
- Create new spec class in `src/derivatives_pricing/valuation/contracts.py` (or new file if large)
- Add new registry in `src/derivatives_pricing/valuation/core.py`: `_SPREAD_REGISTRY`
- Implement engine methods in each pricing file (bsm.py, binomial.py, monte_carlo.py, pde.py)
- Add validation for spec in `OptionValuation.__init__`
- Export new spec from `src/derivatives_pricing/valuation/__init__.py` and top-level `__init__.py`

**New Enum (e.g., CalendarConvention for future use):**
- Add to `src/derivatives_pricing/enums.py` (all enums in single file)
- Use string values: `class CalendarConvention(Enum): ACTUAL = "actual", BUSINESS = "business"`
- Export from `src/derivatives_pricing/__init__.py`

**New Utility Function (e.g., Adjusted Option Pricing):**
- Add to `src/derivatives_pricing/utils.py` as pure function
- Document parameters and return type
- Export from `src/derivatives_pricing/__init__.py` if part of public API

**New Stochastic Process (e.g., CEV Model):**
- Create process parameters class in `src/derivatives_pricing/stochastic_processes.py`: `@dataclass(frozen=True, slots=True, kw_only=True) class CEVParams`
- Create process class inheriting from `PathSimulation`: `class CEVProcess(PathSimulation)`
- Implement `simulate()` method following `GBMProcess` pattern
- Export from `src/derivatives_pricing/__init__.py`

## Special Directories

**`.planning/codebase/`:**
- Purpose: Generated architecture/structure analysis documents
- Generated: Yes (by `/gsd-map-codebase`)
- Committed: Yes (for reference across CLI sessions)

**`src/derivatives_pricing/__pycache__/`:**
- Purpose: Python bytecode cache
- Generated: Yes (automatic by Python interpreter)
- Committed: No (git-ignored)

**`tests/` (Outside src/):**
- Purpose: Test suite for derivatives_pricing package
- Key files: `conftest.py` (pytest fixtures), `helpers.py` (test factories), `test_*.py` (test modules)
- Pattern: Mirrors structure of src/ but with test_ prefix
- Not in analysis scope but follows `pytest src/derivatives_pricing/tests/ -q` convention


## Import Conventions

**Within Package (relative imports):**
```python
# From valuation/bsm.py importing core
from ..valuation.core import OptionValuation, UnderlyingData
from ..enums import OptionType
from ..exceptions import ValidationError
```

**In Tests (absolute imports):**
```python
# In tests/test_*.py
from derivatives_pricing import OptionValuation, UnderlyingData, VanillaSpec
from derivatives_pricing.enums import OptionType, ExerciseType
```

**Path Aliases:**
- None configured; relative imports used throughout
- All relative imports use `from __future__ import annotations` at file top for forward references

**Barrel Files:**
- `src/derivatives_pricing/__init__.py`: Re-exports public API (OptionValuation, specs, enums, exceptions)
- `src/derivatives_pricing/valuation/__init__.py`: Re-exports valuation-specific public API

**Circular Import Prevention:**
- `from __future__ import annotations` used in files with potential TYPE_CHECKING guards
- Example: `bsm.py` uses `if TYPE_CHECKING: from .core import OptionValuation`


## Module Dependency Graph

```
exceptions.py  
↓
enums.py → exceptions.py
↓
utils.py → enums.py, exceptions.py
↓
rates.py → enums.py, exceptions.py, utils.py
↓
market_environment.py → enums.py, exceptions.py, rates.py, utils.py
↓
stochastic_processes.py → enums.py, exceptions.py, rates.py, market_environment.py, utils.py
↓
valuation/contracts.py → enums.py, exceptions.py, utils.py
↓
valuation/params.py → enums.py, exceptions.py
↓
valuation/bsm.py → utils.py, enums.py, exceptions.py, (TYPE_CHECKING: core.py)
↓
valuation/binomial.py, monte_carlo.py, pde.py, asian_analytical.py, barrier_analytical.py
    ↓ (all depend on)
valuation/core.py → all above + contracts.py + params.py + stochastic_processes.py
    ↓
__init__.py → re-exports all public symbols
```

**Acyclic:** No circular dependencies; all import arrows point inward toward core.py, which is imported by tests and user code.


## File Size Reference (for refactoring decisions)

| File | Lines | Role |
|------|-------|------|
| `pde.py` | 2530 | Largest; finite-difference solver complexity |
| `monte_carlo.py` | 1816 | Longstaff-Schwartz, pathwise Greeks |
| `core.py` | 1217 | Dispatcher, Greeks orchestration |
| `binomial.py` | 1148 | Tree construction, Hull Asian logic |
| `stochastic_processes.py` | 923 | GBM/JD/SRD process definitions |
| `utils.py` | 449 | Pure utilities |
| `implied_volatility.py` | 429 | IV solver |
| `asian_analytical.py` | 467 | Geometric Asian formula |
| `barrier_analytical.py` | 563 | Analytical barrier formulas |
| `contracts.py` | 537 | Spec dataclasses |
| `rates.py` | 228 | Discount curve |
| `params.py` | 246 | Parameter classes |
| `market_environment.py` | 129 | Market data containers |
| `enums.py` | 134 | Enums (all in one file) |
| `exceptions.py` | 54 | Exception hierarchy |

---

*Structure analysis: 2026-04-05*
