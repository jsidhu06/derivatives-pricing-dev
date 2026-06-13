"""Finite difference (PDE) valuation implementations.

This package follows the same structure as other valuation modules:
private implementation classes that plug into OptionValuation.

Current scope
-------------
PDE via finite differences for European and American options:
- vanilla call/put and custom payoffs (PayoffSpec)
- barrier options (BarrierSpec): continuous KO via truncated-grid
  Dirichlet BC, continuous KI via in-out parity (European) and
  two-surface coupled PDE solver (American), discrete monitoring
  via full-grid resets at observation dates
- double-barrier options (DoubleBarrierSpec): same coverage with
  Boyle-Tian half-step placement generalised to two barriers
- time stepping: implicit, explicit, or Crank-Nicolson
- optional Rannacher smoothing for Crank-Nicolson
- spatial grids: spot or log-spot
- American handling: intrinsic projection or Gauss-Seidel/PSOR

Layout
------
- ``_kernels``: tridiagonal solver, tau/dividend schedules, vanilla grid
  builders, operator coefficients, time-step kernels, validation
- ``_core``: ``_fd_core`` backward solver, grid-greek mixin, vanilla
  European/American valuation classes
- ``_barrier_grids``: barrier/double-barrier grid builders and
  ``spot_steps`` auto-resolution
- ``_barrier_cores``: KO and two-surface KI core solvers
- ``_barrier_valuation``: barrier valuation classes
"""

# ── Cross-module API ─────────────────────────────────────────────────────────
# The package's actual contract: imported by ``valuation.core`` for the
# (PricingMethod, ExerciseType) registry and the construction-time grid freeze.
from ._core import (
    _FDEuropeanValuation,
    _FDAmericanValuation,
)
from ._barrier_valuation import (
    _FDBarrierValuation,
    _FDDoubleBarrierValuation,
)
from ._barrier_grids import (
    _resolve_pde_spot_steps,
)

# ── Internals re-exported for white-box tests ─────────
# NOT part of the cross-module contract — implementation details that the test
# suite deliberately reach into.
# New white-box access should import straight from the defining submodule instead
# (e.g. ``from ...pde._barrier_cores import _fd_barrier_ko_core``).
from ._kernels import (
    _build_log_grid,
    _build_spot_grid,
)
from ._barrier_grids import (
    _build_double_barrier_discrete_grid,
)
from ._barrier_cores import (
    _fd_barrier_ki_core,
    _fd_double_barrier_ki_core,
)

__all__ = [
    # cross-module API
    "_FDEuropeanValuation",
    "_FDAmericanValuation",
    "_FDBarrierValuation",
    "_FDDoubleBarrierValuation",
    "_resolve_pde_spot_steps",
    # internals exposed for tests
    "_build_log_grid",
    "_build_spot_grid",
    "_build_double_barrier_discrete_grid",
    "_fd_barrier_ki_core",
    "_fd_double_barrier_ki_core",
]
