"""Parameter classes for method-specific valuation configuration.

Each pricing method (Monte Carlo, Binomial, etc.) has its own parameter class
that explicitly documents the configuration options available for that method.
"""

from __future__ import annotations

from dataclasses import dataclass, replace as dc_replace
from typing import Any
import warnings

from ..enums import BarrierMonitoring, PDEEarlyExercise, PDEMethod, PDESpaceGrid
from ..exceptions import ValidationError


@dataclass(frozen=True, slots=True)
class MonteCarloParams:
    """Parameters for Monte Carlo option valuation.

    Attributes
    ----------
    random_seed:
        Random seed for reproducibility. The default is fixed (not None) so
        that bump-and-revalue greeks share common random numbers — without CRN,
        the sampling noise dominates the central-difference signal at typical path
        counts (especially for gamma and tight bumps).  Pass ``None``
        explicitly to use fresh entropy each call (e.g. for portfolio-level
        risk where you want noise to diversify across positions).
        Default: ``42``.
    deg:
        Laguerre polynomial degree for Longstaff-Schwartz regression
        (American only). Typical range: 2-3. Default: ``3``.
    ridge_lambda:
        Ridge (Tikhonov) regularisation parameter for the LSM regression.
        A small positive value stabilises the solve when ITM points are
        few or collinear.  Default: ``1e-8``.
    min_itm:
        Minimum number of in-the-money paths required to run the regression
        at each exercise date.  If fewer paths are ITM, the continuation
        value falls back to the discounted next-step value (path-wise).
        Default: ``25``.
    log_timings:
        If ``True``, log debug timing for solver execution.
        Default: ``False``.
    std_error_warn_ratio:
        If set, emit a warning log when MC standard error exceeds
        std_error_warn_ratio * |PV|. Use None to disable.
        Default: ``0.1``.
    control_variate_european:
        Apply control variate adjustment for American options using the
        analytical European price and the MC European price from the same
        simulation.  Only applicable to American exercise pricing.
        Default: ``False``.
    barrier_aware_basis:
        When ``True``, American barrier Monte Carlo augments the LSM
        regression basis with barrier-distance features and a local near/far
        split for American knock-out options. This is mainly useful where
        the continuation surface changes sharply near the absorbing barrier.
        Default: ``True``. Only applicable to KO barrier options; ignored otherwise.
    """

    random_seed: int | None = 42
    deg: int = 3
    ridge_lambda: float = 1e-8
    min_itm: int = 25
    log_timings: bool = False
    std_error_warn_ratio: float | None = 0.1
    control_variate_european: bool = False
    barrier_aware_basis: bool = True

    def __post_init__(self) -> None:
        for name in ("deg", "min_itm"):
            if type(getattr(self, name)) is not int:
                raise ValidationError(
                    f"{name} must be an int, got {type(getattr(self, name)).__name__}"
                )
        if self.deg < 1:
            raise ValidationError(f"deg must be >= 1, got {self.deg}")
        if self.ridge_lambda < 0:
            raise ValidationError(f"ridge_lambda must be >= 0, got {self.ridge_lambda}")
        if self.min_itm < 1:
            raise ValidationError(f"min_itm must be >= 1, got {self.min_itm}")
        if self.std_error_warn_ratio is not None and self.std_error_warn_ratio <= 0:
            raise ValidationError(
                f"std_error_warn_ratio must be > 0 when set, got {self.std_error_warn_ratio}"
            )
        if not isinstance(self.barrier_aware_basis, bool):
            raise ValidationError(
                f"barrier_aware_basis must be a bool, got {type(self.barrier_aware_basis).__name__}"
            )


@dataclass(frozen=True, slots=True)
class BinomialParams:
    """Parameters for binomial tree option valuation.

    Attributes
    ----------
    num_steps:
        Number of time steps in the binomial tree.
        More steps increase accuracy but also computation time.
        Default: ``500``.
    mc_paths:
        Number of Monte Carlo paths when sampling the binomial tree
        (used for path-dependent payoffs like Asian options). If None,
        Monte Carlo sampling is disabled and Hull-style averages are used.
        Default: ``None``.
    random_seed:
        Random seed for binomial-tree Monte Carlo sampling. This param is ignored
        if mc_paths is None.
        Default: ``None``.
    asian_tree_averages:
        Number of representative averages per node for Hull-style Asian
        binomial tree valuation. Used when mc_paths is None.
        Practical guidance: a ratio of ``asian_tree_averages / num_steps``
        around **1.5–2.0** offers the best accuracy-per-compute trade-off.
        Below 1.0 the tree exhibits significant upward interpolation bias;
        above 2.0 accuracy gains are marginal while memory grows as
        O(asian_tree_averages * num_steps^2).
        Default: ``None``.
    control_variate_european:
        Apply Hull-style control variate adjustment for American options using
        BSM European price and the numerical European price from the same method.
        Only applicable to vanilla call/put American pricing.
        Default: ``False``.
    log_timings:
        If ``True``, log debug timing for solver execution.
        Default: ``False``.
    """

    num_steps: int = 500
    mc_paths: int | None = None
    random_seed: int | None = None
    asian_tree_averages: int | None = None
    control_variate_european: bool = False
    log_timings: bool = False

    def __post_init__(self) -> None:
        for name in ("num_steps",):
            if type(getattr(self, name)) is not int:
                raise ValidationError(
                    f"{name} must be an int, got {type(getattr(self, name)).__name__}"
                )
        if self.num_steps < 1:
            raise ValidationError(f"num_steps must be >= 1, got {self.num_steps}")
        if self.mc_paths is not None and self.asian_tree_averages is not None:
            raise ValidationError(
                "Only one of mc_paths and asian_tree_averages can be set, got both"
            )
        if self.mc_paths is not None and type(self.mc_paths) is not int:
            raise ValidationError(f"mc_paths must be an int, got {type(self.mc_paths).__name__}")
        if self.asian_tree_averages is not None and type(self.asian_tree_averages) is not int:
            raise ValidationError(
                f"asian_tree_averages must be an int, got {type(self.asian_tree_averages).__name__}"
            )
        if self.mc_paths is not None and self.mc_paths < 1:
            raise ValidationError(f"mc_paths must be >= 1, got {self.mc_paths}")
        if self.asian_tree_averages is not None and self.asian_tree_averages < 1:
            raise ValidationError(
                f"asian_tree_averages must be >= 1, got {self.asian_tree_averages}"
            )
        if self.asian_tree_averages is not None:
            ratio = self.asian_tree_averages / self.num_steps
            if ratio < 1.5:
                warnings.warn(
                    "asian_tree_averages / num_steps < 1.5; "
                    "Hull-style Asian valuation may exhibit upward interpolation bias. "
                    "A ratio of 1.5–2.0 is recommended.",
                    RuntimeWarning,
                )
            if ratio > 2.1:
                warnings.warn(
                    "asian_tree_averages is large relative to num_steps; "
                    "memory usage may be high with limited accuracy gains.",
                    RuntimeWarning,
                )
            est_bytes = self.asian_tree_averages * (self.num_steps + 1) ** 2 * 8
            if est_bytes > 1_000_000_000:
                est_gib = est_bytes / (1024**3)
                warnings.warn(
                    f"Estimated memory for Hull Asian grid is ~{est_gib:.2f} GiB; "
                    "consider reducing num_steps or asian_tree_averages.",
                    RuntimeWarning,
                )


@dataclass(frozen=True, slots=True)
class PDEParams:
    """Parameters for PDE finite difference option valuation.

    Parameters
    ----------
    smax_mult
        Multiplier for the maximum spot in the grid domain, where
        ``S_max = smax_mult * max(spot, strike)``. Default: ``4.0``.
    spot_steps
        Number of spatial grid steps. Higher values improve resolution.
        Default: ``None`` ("auto").  On a **log spatial grid** auto sizes the
        step to ``dz = λ·σ·√Δt`` with one ``λ`` per scheme family: the
        **explicit family** is stability-pinned at Hull's trinomial
        ``λ = √3`` (``dz_hull = σ·√(3·Δt)``; covering the target domain, or —
        for a continuously-monitored double barrier — pinning
        ``round(corridor / dz_hull)`` so ``λ ≥ 1`` is guaranteed by
        construction), while **CN/IMPLICIT** use the accuracy choice
        ``λ = 1/2`` (twice as fine as the explicit stability bound at equal
        ``time_steps``), floored at ``200``.  A **spot (non-log) grid**
        falls back to the fixed default ``200``.  The auto value is
        frozen once at ``OptionValuation`` construction so it is identical
        across every bump-and-revalue greek solve.  Pass an int to take full
        manual control.
    parity_vanilla_spot_steps
        Spatial node count for the **vanilla leg** of European knock-in parity
        pricing (``V_KI = V_vanilla - V_KO``).  That leg lives on the *full*
        free-far-field domain rather than the barrier-truncated KO grid, so it
        warrants its own resolution.  Default: ``None`` (auto / library-managed)
        — most users leave it there:

        - With ``spot_steps=None`` and a knock-in spec, ``OptionValuation``
          resolves a separate vanilla-appropriate free-far-field count here
          (frozen once, like ``spot_steps``) so the vanilla leg is not starved
          by the corridor-pinned KO count.
        - With an explicit ``spot_steps`` and ``None`` here, the vanilla leg
          reuses ``spot_steps``.

        Set an explicit int to override the vanilla-leg resolution directly
        (finer manual control); your value is then honored verbatim and frozen.
        Ignored for non-knock-in specs (knock-outs are a single solve with no
        vanilla leg).
    time_steps
        Number of time steps. Higher values generally improve stability/accuracy.
        Default: ``200``.
    omega
        SOR relaxation parameter for PSOR iterations (American options), in ``(1, 2)``.
        Default: ``1.5``.
    tol
        Convergence tolerance for PSOR iterations. Default: ``1e-6``.
    max_iter
        Maximum PSOR iterations per time step. Default: ``20_000``.
    method
        Time-stepping scheme (IMPLICIT, EXPLICIT, EXPLICIT_HULL, CRANK_NICOLSON).
        Default: ``PDEMethod.CRANK_NICOLSON``.
    rannacher_steps
        Number of initial Crank-Nicolson intervals replaced by two implicit
        half-steps each (Rannacher smoothing). Set ``0`` to disable.
        Default: ``2``.
    space_grid
        Spatial discretization in spot space or log-spot space.
        Default: ``PDESpaceGrid.SPOT``.
    american_solver
        Early-exercise handling for American options.
        Default: ``PDEEarlyExercise.GAUSS_SEIDEL``.
    control_variate_european
        Apply Hull-style control-variate adjustment for American vanilla call/put pricing.
        Default: ``False``.
    log_timings
        If ``True``, emit debug timing logs for solver execution.
        Default: ``False``.
    """

    smax_mult: float = 4.0
    spot_steps: int | None = None
    parity_vanilla_spot_steps: int | None = None
    time_steps: int = 200
    omega: float = 1.5
    tol: float = 1e-6
    max_iter: int = 20_000
    method: PDEMethod = PDEMethod.CRANK_NICOLSON
    rannacher_steps: int = 2
    space_grid: PDESpaceGrid = PDESpaceGrid.SPOT
    american_solver: PDEEarlyExercise = PDEEarlyExercise.GAUSS_SEIDEL
    control_variate_european: bool = False
    log_timings: bool = False

    @classmethod
    def for_barriers(
        cls,
        *,
        monitoring: BarrierMonitoring,
        **overrides: Any,
    ) -> PDEParams:
        """Create PDE params tuned for barrier pricing.

        Returns a ``PDEParams`` instance with a finer grid and log-spot
        spatial discretization suitable for barrier pricing.  The time-
        marching ``method`` is chosen from ``monitoring``:

        - ``BarrierMonitoring.CONTINUOUS`` → ``PDEMethod.CRANK_NICOLSON``.
          Continuous monitoring has a single payoff discontinuity at
          maturity; the default ``rannacher_steps=2`` startup dampens it
          and CN's higher-order time accuracy gives the best PV/greek
          quality on the rest of the time march.
        - ``BarrierMonitoring.DISCRETE`` → ``PDEMethod.EXPLICIT_HULL`` with
          ``american_solver=PDEEarlyExercise.INTRINSIC``.  Discrete
          monitoring projects ``V(S, t_i) = (discounted) rebate`` past the
          barrier at every observation date, introducing a fresh step discontinuity
          each time.  ``EXPLICIT_HULL``'s per-step cost is a single vectorised
          matrix-vector multiply, so we can crank ``time_steps`` up cheaply.

        The discrete (explicit) default sets ``spot_steps=None`` ("auto")
        with a log spatial grid: an explicit scheme's spacing ``dz`` is
        pinned by stability to Hull's
        trinomial step ``dz_hull = σ·√(3·Δt)`` and is therefore wholly
        ``time_steps``-driven, so ``spot_steps`` is not an independent
        accuracy dial — the engine sizes the grid to cover the domain at
        ``dz_hull`` (see ``_resolve_pde_spot_steps``). Continuous (CN) passes
        an int spot_steps`` since CN's ``dz`` is a free accuracy/speed
        choice with no physical scale.

        ``monitoring`` is required (no default) so the dependency is
        explicit at the call site.  Any other keyword argument accepted
        by the constructor can be passed to override individual fields,
        including ``method``.
        """
        if monitoring is BarrierMonitoring.DISCRETE:
            method = PDEMethod.EXPLICIT_HULL
            spot_steps, time_steps = None, 3000
            american_solver = PDEEarlyExercise.INTRINSIC
        else:
            method = PDEMethod.CRANK_NICOLSON
            spot_steps, time_steps = 1200, 800
            american_solver = PDEEarlyExercise.GAUSS_SEIDEL

        defaults = cls(
            spot_steps=spot_steps,
            time_steps=time_steps,
            space_grid=PDESpaceGrid.LOG_SPOT,
            method=method,
            american_solver=american_solver,
        )

        return dc_replace(defaults, **overrides) if overrides else defaults

    @classmethod
    def for_double_barriers(
        cls,
        *,
        monitoring: BarrierMonitoring,
        **overrides: Any,
    ) -> PDEParams:
        """Create PDE params tuned for *double*-barrier pricing.

        Same scheme selection as :meth:`for_barriers` (CRANK_NICOLSON for
        continuous, EXPLICIT_HULL for discrete), but with a finer default
        ``time_steps`` for **discrete** monitoring.  A discretely-monitored
        double barrier injects a knock-out reset at *both* barriers on every
        observation date, so the value surface carries fresh discontinuities at
        each end of the corridor; empirically theta in particular needed more temporal
        resolution than the single-barrier discrete default to settle.
        ``time_steps`` is therefore lifted (3000 -> 6000).  ``spot_steps``
        inherits the discrete default ``None`` ("auto"): being explicit, the
        log spatial grid is sized to Hull's stability-pinned ``dz = σ·√(3·Δt)``.
        ``EXPLICIT_HULL`` makes the higher ``time_steps`` cheap
        (single vectorised matrix-vector multiply per step, no PSOR iteration).

        For **continuous** monitoring the corridor-truncated Crank-Nicolson grid
        already resolves PVs and greeks to ~1e-4 vs Boyle-Tian (1998) using the
        ``for_barriers`` defaults (all ``spot_steps`` sit inside the corridor),
        so this method deliberately falls through to identical params there.

        Any keyword accepted by the constructor overrides the chosen defaults.
        """
        richer = (
            dict(spot_steps=None, time_steps=6000)
            if monitoring is BarrierMonitoring.DISCRETE
            else {}
        )
        merged = {**richer, **overrides}  # explicit overrides win over the bump
        return cls.for_barriers(monitoring=monitoring, **merged)

    def __post_init__(self) -> None:
        for name in ("time_steps", "max_iter", "rannacher_steps"):
            if type(getattr(self, name)) is not int:
                raise ValidationError(
                    f"{name} must be an int, got {type(getattr(self, name)).__name__}"
                )
        # spot_steps may be None ("auto" — resolved at OptionValuation
        # construction); validate the int range only when explicitly given.
        if self.spot_steps is not None:
            if type(self.spot_steps) is not int:
                raise ValidationError(
                    f"spot_steps must be an int or None, got {type(self.spot_steps).__name__}"
                )
            if self.spot_steps < 3:
                raise ValidationError(f"spot_steps must be >= 3, got {self.spot_steps}")
        # parity_vanilla_spot_steps is auto-managed (set by OptionValuation's
        # grid freeze for auto-grid knock-ins); validate like spot_steps if set.
        if self.parity_vanilla_spot_steps is not None:
            if type(self.parity_vanilla_spot_steps) is not int:
                raise ValidationError(
                    "parity_vanilla_spot_steps must be an int or None, got "
                    f"{type(self.parity_vanilla_spot_steps).__name__}"
                )
            if self.parity_vanilla_spot_steps < 3:
                raise ValidationError(
                    f"parity_vanilla_spot_steps must be >= 3, got {self.parity_vanilla_spot_steps}"
                )
        if self.smax_mult <= 0:
            raise ValidationError(f"smax_mult must be positive, got {self.smax_mult}")
        if self.time_steps < 1:
            raise ValidationError(f"time_steps must be >= 1, got {self.time_steps}")
        if not (1.0 < self.omega < 2.0):
            raise ValidationError(f"omega must be in (1.0, 2.0), got {self.omega}")
        if self.tol <= 0:
            raise ValidationError(f"tol must be positive, got {self.tol}")
        if self.max_iter < 1:
            raise ValidationError(f"max_iter must be >= 1, got {self.max_iter}")
        if self.rannacher_steps < 0:
            raise ValidationError(f"rannacher_steps must be >= 0, got {self.rannacher_steps}")
        if not isinstance(self.method, PDEMethod):
            raise ValidationError(f"method must be a PDEMethod, got {self.method}")
        if not isinstance(self.space_grid, PDESpaceGrid):
            raise ValidationError(f"space_grid must be a PDESpaceGrid, got {self.space_grid}")
        if not isinstance(self.american_solver, PDEEarlyExercise):
            raise ValidationError(
                f"american_solver must be a PDEEarlyExercise, got {self.american_solver}"
            )


# Type alias for any valuation parameters
ValuationParams = MonteCarloParams | BinomialParams | PDEParams
