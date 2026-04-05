# Architecture

**Analysis Date:** 2026-04-05

## Pattern Overview

**Overall:** Registry-based dispatcher with frozen dataclass entities

The codebase uses a **facade pattern** where `OptionValuation` (in `src/derivatives_pricing/valuation/core.py`) is the central orchestration layer. It accepts a contract specification and underlying market data, looks up the appropriate pricing engine in a registry keyed by `(PricingMethod, ExerciseType)`, and delegates computation to a private engine class. All data containers are **immutable frozen dataclasses** with `__post_init__` validation.

**Key Characteristics:**
- **Single public interface**: `OptionValuation` handles all option types (vanilla, Asian, barrier, custom payoffs) through polymorphic specs
- **Private implementation engines**: Four pricing engines (`bsm.py`, `binomial.py`, `monte_carlo.py`, `pde.py`) each define private `_*Valuation` classes
- **Immutable data model**: All data (market data, underlying, specs) uses frozen dataclasses; mutations via `dataclasses.replace()`
- **Explicit method selection**: Greeks can use analytical, tree, PDE grid, pathwise, or numerical (bump-and-revalue) methods
- **Three registries**: `_VANILLA_REGISTRY`, `_ASIAN_REGISTRY`, `_BARRIER_REGISTRY` map method/exercise pairs to implementations


## Layers

**Data Container Layer:**
- Purpose: Immutable representations of market state, contract terms, and simulation parameters
- Location: `src/derivatives_pricing/market_environment.py`, `src/derivatives_pricing/rates.py`, `src/derivatives_pricing/valuation/contracts.py`, `src/derivatives_pricing/stochastic_processes.py`
- Contains: `MarketData`, `DiscountCurve`, `CorrelationContext`, `VanillaSpec`, `PayoffSpec`, `AsianSpec`, `BarrierSpec`, `UnderlyingData`, `GBMParams`, `JDParams`, `SRDParams`, `SimulationConfig`
- Depends on: Enums (`enums.py`), exceptions
- Used by: `OptionValuation`, all pricing engines

**Valuation Facade Layer:**
- Purpose: Public entry point that dispatches to appropriate engine and orchestrates Greeks calculation
- Location: `src/derivatives_pricing/valuation/core.py` (class `OptionValuation`)
- Contains: Dispatcher logic, registry lookup, Greeks methods (delta, gamma, vega, theta, rho), bump-and-revalue framework, control variate logic
- Depends on: All contract specs, all engine implementations, params classes
- Used by: End-user code

**Engine Implementation Layer:**
- Purpose: Method-specific pricing implementations (each private `_*Valuation` class)
- Location: `src/derivatives_pricing/valuation/bsm.py`, `binomial.py`, `monte_carlo.py`, `pde.py`, `asian_analytical.py`, `barrier_analytical.py`
- Contains: Private engine classes (`_BSMEuropeanValuation`, `_BinomialAmericanValuation`, etc.), Greeks calculations (analytical/tree/pathwise where available), payoff evaluation
- Depends on: Underlying data, specs, market data, stochastic processes (for MC)
- Used by: `OptionValuation` (via registry instantiation)

**Market Data & Calibration Layer:**
- Purpose: Discount curves, interest-rate models, stochastic process definitions
- Location: `src/derivatives_pricing/rates.py`, `stochastic_processes.py`, `market_environment.py`
- Contains: `DiscountCurve` with log-linear interpolation, GBM/jump-diffusion/mean-reverting process parameters, market context (pricing date, currency)
- Depends on: Enums, exceptions
- Used by: All layers above

**Utilities & Core Logic:**
- Purpose: Pure functions for day-count conventions, forward calculations, dividend treatment, put-call parity
- Location: `src/derivatives_pricing/utils.py`
- Contains: `calculate_year_fraction`, `pv_discrete_dividends`, `put_call_parity`, validation helpers
- Depends on: Enums
- Used by: All layers


## Data Flow

**Pricing Flow (Deterministic Methods - BSM, Binomial, PDE):**

1. User creates `UnderlyingData` (spot, vol, market data including pricing date and discount curve)
2. User creates contract spec: `VanillaSpec`, `AsianSpec`, `BarrierSpec`, or `PayoffSpec`
3. User instantiates `OptionValuation(underlying=UnderlyingData, spec=..., pricing_method=...)`
4. `OptionValuation.__init__` validates inputs, resolves contract parameters (maturity, strike, exercise type)
5. Registry lookup: `_VANILLA_REGISTRY[(pricing_method, exercise_type)]` → engine class
6. Engine instantiated and stored in `self._impl`
7. User calls `val.present_value()` → delegates to `self._impl.present_value()`
8. Engine extracts spot, strike, time-to-maturity from inputs and performs method-specific calculation

**Pricing Flow (Monte Carlo):**

1. User creates `GBMProcess` (or `JDProcess`/`SRDProcess`) with simulation config
2. User creates contract spec
3. User instantiates `OptionValuation(underlying=GBMProcess, spec=..., pricing_method=MONTE_CARLO)`
4. `OptionValuation.__init__` validates `underlying` is a `PathSimulation`, injects observation dates into sim config
5. Registry lookup for MC engine class
6. Engine stored in `self._impl`
7. User calls `val.present_value()` → engine internally calls `underlying.simulate()` to generate paths, evaluates payoff on paths, discounts back

**Greeks Calculation Flow (Bump-and-Revalue):**

1. User calls `val.delta()` (or gamma, vega, theta, rho)
2. Method resolves greek calculation method (analytical → tree → grid → pathwise → numerical)
3. If engine provides native analytical/tree/pathwise method, call `self._impl.delta()` etc.
4. If bumping required (numerical), extract bump size (or use default)
5. Call `self._bump_underlying(initial_value=s0 + epsilon)` → returns **new** `UnderlyingData` with bumped field
6. Call `self._build_valuation(underlying=bumped)` → creates **new** `OptionValuation` with bumped underlying
7. Call `present_value()` on new valuation, compute finite difference

**State Management:**

- `OptionValuation` stores constructor arguments as private attributes (`self._underlying`, `self._spec`, `self._pricing_method`, `self._impl`)
- All properties are **read-only** (no setters)
- Greeks calculations are **stateless** — each bump creates a new temporary valuation instance
- Thread-safe: frozen dataclasses prevent accidental mutation, bump-and-revalue creates fresh instances
- `PathSimulation` instances are **defensively copied** in `OptionValuation.__init__` because simulation modifies internal state (time grid, random normals)


## Key Abstractions

**Spec Hierarchy:**
- Purpose: Represent different contract types in a type-discriminated way
- Examples: `VanillaSpec` (plain calls/puts), `PayoffSpec` (custom payoffs), `AsianSpec` (path-dependent averages), `BarrierSpec` (knock-in/out options)
- Pattern: Frozen dataclasses with strict type/value validation in `__post_init__`; `OptionValuation.__init__` uses `isinstance` checks to dispatch behavior
- Contract-specific details: maturity, strike, exercise type, optional attributes (averaging method for Asian, barrier level for barrier options)

**Underlying Variants:**
- Purpose: Distinguish deterministic (analytical/tree/PDE) pricing from Monte Carlo simulation
- Examples: `UnderlyingData` (minimal spot + vol + market context), `GBMProcess` (+ simulation config, dividend curve), `JDProcess` (+ jump parameters), `SRDProcess` (+ mean-reversion)
- Pattern: `UnderlyingData` is a simple container; `GBMProcess` et al inherit from `PathSimulation` abstract base
- Usage: Deterministic engines extract spot/vol from `UnderlyingData`; MC engines extract `GBMProcess` and call `.simulate()`

**Market Context:**
- Purpose: Bundle pricing date, discount curve, day-count convention, currency
- Examples: `MarketData`, embedded in `UnderlyingData`
- Pattern: Single immutable container; forward/discount calculations route through embedded `DiscountCurve`
- Consistency: All pricing, Greeks, and simulation respect the same pricing date and curve

**Valuation Parameters:**
- Purpose: Method-specific knobs (tree steps, MC paths, regularization, solver tolerances)
- Examples: `MonteCarloParams` (deg, ridge_lambda, seed), `BinomialParams` (num_steps, asian_tree_averages), `PDEParams` (smax_mult, spot_steps, method, omega)
- Pattern: Frozen dataclasses with validation in `__post_init__`; optional in `OptionValuation.__init__` (defaults chosen by engine)
- Contract-level vs method-level: Contract is input to `OptionValuation`; params are selected/resolved in `_resolve_params(...)`


## Entry Points

**Primary Entry Point - OptionValuation:**
- Location: `src/derivatives_pricing/valuation/core.py`
- Triggers: Direct instantiation by user code
- Responsibilities: 
  - Registry dispatch to appropriate engine
  - Input validation (currency match, maturity > pricing date)
  - Greeks orchestration (method selection, bump-and-revalue)
  - Control variate adjustments
  - Contract-specific date resolution (Asian fixings, barrier monitoring)

**Secondary Entry Points - Utility Functions:**
- `as_underlying_data(process)` in `core.py`: Convert `GBMProcess` to `UnderlyingData` (for running deterministic method on MC-designed input)
- `implied_volatility(...)` in `implied_volatility.py`: Invert option price to find vol (root-finding on `OptionValuation` PV)

**Configuration Entry Points:**
- `DiscountCurve.from_forwards(...)`, `.from_zero_rates(...)`: Construct curves from market data
- `GBMProcess(...)`, `JDProcess(...)`, `SRDProcess(...)`: Construct stochastic processes with simulation config


## Error Handling

**Strategy:** Exception hierarchy with domain-specific subclasses; validation in `__post_init__` of all dataclasses

**Patterns:**

- **Validation errors** (`ValidationError`): Invalid values (negative vol, strike < 0, maturity <= pricing date, conflicting inputs like both fixing_dates and num_observations)
  - Raised in `__post_init__` on specs, underlying, market data
  - Example: `VanillaSpec.__post_init__` raises if strike is not finite

- **Configuration errors** (`ConfigurationError`): Wrong types (raw int instead of enum, non-callable payoff function)
  - Raised in `__post_init__` and in `OptionValuation.__init__`
  - Example: `OptionValuation.__init__` raises if `pricing_method` is not a `PricingMethod` enum

- **Feature not supported** (`UnsupportedFeatureError`): Feature combination not implemented (cross-currency valuation, custom payoff with BSM)
  - Raised in `OptionValuation.__init__` and engine methods
  - Example: BSM requires CALL/PUT, not PayoffSpec

- **Numerical errors** (`NumericalError` subtree):
  - `ArbitrageViolationError`: Model parameters imply arbitrage (risk-neutral prob outside [0,1])
  - `ConvergenceError`: Iterative solver failed
  - `StabilityError`: Stability conditions violated (CFL for explicit PDE schemes)


## Cross-Cutting Concerns

**Logging:** 
- Module-level logger in each engine: `logger = logging.getLogger(__name__)`
- Debug-level timings when `params.log_timings=True`
- Warning-level for numerical stability issues, large memory allocations

**Validation:**
- All inputs validated in `__post_init__` before object construction completes
- Day-count/year-fraction calculations centralized in `utils.calculate_year_fraction`
- Discrete dividend chronological ordering enforced in `UnderlyingData.__post_init__`

**Authentication & Currency:**
- Single currency per valuation (no cross-currency derivatives)
- Currency validated: spec currency must match underlying currency (if both provided)
- Currency exposed as read-only property `OptionValuation.currency`

**Dividend Handling:**
- Two approaches: discrete dividends (ex-dates with cash amounts) or continuous dividend yield curve
- Warning issued if both provided (continuous enters drift, discrete subtracted at ex-dates)
- Forwarding calculations in engines adjust spot and discount factors accordingly

---

*Architecture analysis: 2026-04-05*
