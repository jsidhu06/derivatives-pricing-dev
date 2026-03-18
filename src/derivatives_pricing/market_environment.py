"""Market data containers for valuation and simulation."""

from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass, field
import datetime as dt
import numpy as np
from .enums import DayCountConvention
from .rates import DiscountCurve
from .exceptions import ValidationError
from .utils import validate_naive_datetime


@dataclass(frozen=True, slots=True)
class MarketData:
    """Market data required for valuation/simulation."""

    pricing_date: dt.datetime
    discount_curve: DiscountCurve
    currency: str
    day_count_convention: DayCountConvention = DayCountConvention.ACT_365F

    def __post_init__(self) -> None:
        validate_naive_datetime(self.pricing_date, "pricing_date")
        if not isinstance(self.discount_curve, DiscountCurve):
            raise ValidationError(
                f"discount_curve must be a DiscountCurve, got {type(self.discount_curve).__name__}"
            )
        if not isinstance(self.currency, str) or not self.currency:
            raise ValidationError("currency must be a non-empty string")
        if not isinstance(self.day_count_convention, DayCountConvention):
            raise ValidationError(
                f"day_count_convention must be a DayCountConvention enum, "
                f"got {type(self.day_count_convention).__name__}"
            )


@dataclass(frozen=True, slots=True)
class CorrelationContext:
    """Shared correlation/scenario context for multi-asset simulation.

    Users supply a symmetric correlation matrix (with ones on the diagonal)
    and an ordered sequence of asset names.  The Cholesky factor and the
    name-to-index mapping are derived in ``__post_init__``.

    Parameters
    ----------
    correlation_matrix : np.ndarray, shape (n_assets, n_assets)
        Symmetric, positive-definite correlation matrix with ones on
        the diagonal.  Row/column *i* corresponds to ``asset_names[i]``.
    random_numbers : np.ndarray, shape (n_assets, n_time_intervals, n_paths)
        Pre-generated independent standard-normal draws.  Axis 0 is
        indexed by ``asset_names``.
    asset_names : Sequence[str]
        Ordered identifiers for each asset/process.  Length must equal
        ``correlation_matrix.shape[0]``.
    """

    correlation_matrix: np.ndarray
    random_numbers: np.ndarray
    asset_names: Sequence[str]

    # Derived (not user-facing) -------------------------------------------------
    cholesky_matrix: np.ndarray = field(init=False, repr=False)
    _asset_index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        C = np.asarray(self.correlation_matrix, dtype=float)

        # --- shape checks ---
        if C.ndim != 2 or C.shape[0] != C.shape[1]:
            raise ValidationError(f"correlation_matrix must be square, got shape {C.shape}")
        n = C.shape[0]

        if len(self.asset_names) != n:
            raise ValidationError(
                f"asset_names length ({len(self.asset_names)}) must match "
                f"correlation_matrix dimension ({n})"
            )

        # --- duplicate names ---
        if len(set(self.asset_names)) != len(self.asset_names):
            raise ValidationError("asset_names must contain unique entries")

        # --- symmetry ---
        if not np.allclose(C, C.T, atol=1e-12):
            raise ValidationError("correlation_matrix must be symmetric")

        # --- unit diagonal ---
        if not np.allclose(np.diag(C), 1.0, atol=1e-12):
            raise ValidationError(
                "correlation_matrix diagonal must be 1 (it is a correlation matrix, not covariance)"
            )

        # --- positive-definite (Cholesky will fail otherwise) ---
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError as exc:
            raise ValidationError("correlation_matrix is not positive-definite") from exc

        object.__setattr__(self, "cholesky_matrix", L)

        # --- random_numbers shape ---
        rn = self.random_numbers
        if rn.ndim != 3:
            raise ValidationError(
                f"random_numbers must be 3-D (n_assets, n_steps, n_paths), got ndim={rn.ndim}"
            )
        if rn.shape[0] != n:
            raise ValidationError(
                f"random_numbers axis-0 ({rn.shape[0]}) must match number of assets ({n})"
            )

        # --- build index mapping ---
        object.__setattr__(
            self, "_asset_index", {name: i for i, name in enumerate(self.asset_names)}
        )

    def asset_index(self, name: str) -> int:
        """Return the integer index for *name*.

        Raises ``ValidationError`` if *name* is not in ``asset_names``.
        """
        try:
            return self._asset_index[name]
        except KeyError:
            raise ValidationError(
                f"'{name}' not found in asset_names: {list(self.asset_names)}"
            ) from None
