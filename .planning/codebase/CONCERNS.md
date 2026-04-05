# Codebase Concerns

## Tech Debt

- **Assert statements in production code** — Several modules use `assert` for validation (e.g., `valuation/pde.py`, `valuation/monte_carlo.py`). These are silently skipped with `python -O`, which could mask bugs in optimized deployments.
- **Implicit numba dependency** — Numba is optional but some code paths assume JIT availability. Fallback behavior may not be fully tested.
- **Complex LSM implementation** — The Longstaff-Schwartz Monte Carlo implementation in `valuation/monte_carlo.py` is a large, tightly coupled function that is difficult to unit-test in isolation.

## Numerical Stability

- **PDE CFL conditions** — Crank-Nicolson scheme in `valuation/pde.py` does not enforce CFL checks. Extreme parameter combinations (high vol, fine spatial grid, coarse time grid) could produce oscillations.
- **Barrier monitoring edge cases** — Floating-point comparisons at barrier levels (spot vs barrier) can produce inconsistent knock-in/knock-out decisions near the boundary.
- **Regression basis conditioning** — LSM regression in Monte Carlo uses polynomial basis functions that can become ill-conditioned for high-degree fits or extreme moneyness.
- **Discrete barrier PDE convergence** — PDE solver exhibits first-order convergence for discrete barrier monitoring due to value surface discontinuities at reset dates. Known limitation; Binomial and MC converge faster for these cases.

## Fragile Areas

- **Asian observed-average state machine** — `valuation/asian_analytical.py` and Monte Carlo Asian paths have complex observation-date logic that is sensitive to date ordering and deduplication.
- **Barrier inception comparisons** — Barrier-at-inception logic (spot == barrier at t=0) uses floating-point equality with tolerance. Edge cases around this boundary require careful handling.
- **PDE barrier coupling** — Two-surface coupled PDE for American knock-in barriers (`_fd_barrier_ki_core` in `valuation/pde.py`) is complex and sensitive to grid resolution.
- **Control variate dispatch** — Monte Carlo control variate selection depends on exercise type and option type combinations; not all paths are exercised in tests.

## Performance Bottlenecks

- **PDE grid construction** — Large grids (800x800+) allocate significant memory for tridiagonal systems. Memory scales linearly with spot_steps but computation scales as O(spot_steps * time_steps).
- **MC pathwise Greeks** — Bump-and-revalue Greeks require full re-simulation for each Greek, multiplying MC cost by the number of Greeks requested.
- **Binomial tree time-step scaling** — Binomial pricing is O(n^2) in time steps; high step counts (1000+) are slow for American barriers.

## Test Coverage Gaps

- **IV edge cases** — Implied volatility solver edge cases (deep OTM/ITM, near-expiry) may not be fully covered.
- **Barrier with non-flat curves** — American barrier Greeks with non-flat term structures have limited QuantLib comparison coverage (QL binomial engine approximates with single flat rate).
- **Payoff discontinuities** — Digital/binary-like payoffs in custom payoff functions are not tested with PDE solver.
- **Asian weekend/holiday gaps** — Asian option observation dates falling on weekends/holidays not explicitly tested.

## Known Limitations

- **Binomial day-count restriction** — Binomial engine rejects certain day-count conventions that don't produce clean time-step fractions.
- **Discrete barrier monitoring requirements** — Discrete monitoring requires explicit `num_observations` or `monitoring_dates`; no automatic schedule generation.
- **QuantLib comparison scope** — QL does not support discrete barrier monitoring or American exercise with FD barrier engine, limiting external validation for these features.

## Security

- **Custom payoff callable injection** — `PayoffSpec` accepts arbitrary callables. No sandboxing or validation of user-supplied payoff functions.
- **Numba JIT risks** — If numba is installed, JIT compilation of user-influenced code paths could theoretically be exploited, though risk is low in typical usage.

## Scaling Limits

- **PDE spatial grid** — Memory and time scale with grid size. Grids beyond ~2000x2000 become impractical on typical hardware.
- **MC path memory** — Large path counts (500k+) with many time steps consume significant memory for path storage, especially with antithetic variates doubling effective paths.
- **Barrier monitoring frequency** — Very high observation counts for discrete monitoring (daily for multi-year options) create many time-step injection points, slowing PDE and tree methods.

## Dependencies at Risk

- **SciPy special functions** — BSM analytics depend on `scipy.stats.norm`. Any breaking changes in SciPy's distribution API would affect core pricing.
- **Optional numba** — Numba version compatibility can break silently; tested versions should be pinned.
- **QuantLib comparison tests** — Tests marked `@pytest.mark.slow` depend on QuantLib-Python SWIG bindings, which can be fragile across Python/QL version combinations.
