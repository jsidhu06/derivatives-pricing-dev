"""Tests for day-count conventions."""

import datetime as dt
import numpy as np

from derivatives_pricing.enums import DayCountConvention
from derivatives_pricing.utils import calculate_year_fraction


# ---------------------------------------------------------------------------
# ACT/365F
# ---------------------------------------------------------------------------


class TestACT365F:
    """Tests for ACT/365F day-count convention."""

    def test_one_year(self):
        start = dt.datetime(2025, 1, 1)
        end = dt.datetime(2026, 1, 1)
        assert np.isclose(calculate_year_fraction(start, end, DayCountConvention.ACT_365F), 1.0)

    def test_one_day(self):
        start = dt.datetime(2025, 6, 15)
        end = dt.datetime(2025, 6, 16)
        assert np.isclose(
            calculate_year_fraction(start, end, DayCountConvention.ACT_365F), 1.0 / 365.0
        )

    def test_same_date_returns_zero(self):
        d = dt.datetime(2025, 3, 15)
        assert calculate_year_fraction(d, d, DayCountConvention.ACT_365F) == 0.0

    def test_leap_year(self):
        """2024 is a leap year → 366 actual days."""
        start = dt.datetime(2024, 1, 1)
        end = dt.datetime(2025, 1, 1)
        assert np.isclose(
            calculate_year_fraction(start, end, DayCountConvention.ACT_365F), 366.0 / 365.0
        )

    def test_default_convention(self):
        """ACT/365F is the default when no convention is specified."""
        start = dt.datetime(2025, 1, 1)
        end = dt.datetime(2025, 7, 1)
        assert calculate_year_fraction(start, end) == calculate_year_fraction(
            start, end, DayCountConvention.ACT_365F
        )

    def test_half_year(self):
        """182 days ≈ 0.4986 year fraction under ACT/365F."""
        start = dt.datetime(2025, 1, 1)
        end = dt.datetime(2025, 7, 2)  # 182 days
        expected = 182.0 / 365.0
        assert np.isclose(
            calculate_year_fraction(start, end, DayCountConvention.ACT_365F), expected
        )


# ---------------------------------------------------------------------------
# ACT/360
# ---------------------------------------------------------------------------


class TestACT360:
    """Tests for ACT/360 day-count convention."""

    def test_one_year_actual(self):
        """365 actual days → 365/360 under ACT/360."""
        start = dt.datetime(2025, 1, 1)
        end = dt.datetime(2026, 1, 1)
        expected = 365.0 / 360.0
        assert np.isclose(calculate_year_fraction(start, end, DayCountConvention.ACT_360), expected)

    def test_one_day(self):
        start = dt.datetime(2025, 6, 15)
        end = dt.datetime(2025, 6, 16)
        assert np.isclose(
            calculate_year_fraction(start, end, DayCountConvention.ACT_360), 1.0 / 360.0
        )

    def test_same_date_returns_zero(self):
        d = dt.datetime(2025, 3, 15)
        assert calculate_year_fraction(d, d, DayCountConvention.ACT_360) == 0.0

    def test_90_days(self):
        """90 days → exactly 0.25 under ACT/360."""
        start = dt.datetime(2025, 1, 1)
        end = dt.datetime(2025, 4, 1)  # 90 actual days
        assert np.isclose(
            calculate_year_fraction(start, end, DayCountConvention.ACT_360), 90.0 / 360.0
        )


# ---------------------------------------------------------------------------
# ACT/365.25
# ---------------------------------------------------------------------------


class TestACT36525:
    """Tests for ACT/365.25 day-count convention (accounts for leap years on average)."""

    def test_one_year(self):
        start = dt.datetime(2025, 1, 1)
        end = dt.datetime(2026, 1, 1)
        expected = 365.0 / 365.25
        assert np.isclose(
            calculate_year_fraction(start, end, DayCountConvention.ACT_365_25), expected
        )

    def test_four_years_exactly_four(self):
        """1461 actual days (4 years including 1 leap) → 1461/365.25 = 4.0."""
        start = dt.datetime(2024, 1, 1)
        end = dt.datetime(2028, 1, 1)
        expected = 1461.0 / 365.25
        assert np.isclose(
            calculate_year_fraction(start, end, DayCountConvention.ACT_365_25), expected, rtol=1e-10
        )

    def test_same_date_returns_zero(self):
        d = dt.datetime(2025, 3, 15)
        assert calculate_year_fraction(d, d, DayCountConvention.ACT_365_25) == 0.0


# ---------------------------------------------------------------------------
# 30/360 US
# ---------------------------------------------------------------------------


class TestThirty360US:
    """Tests for 30/360 US day-count convention."""

    def test_jan31_to_feb28(self):
        """Original test: Jan 31 → Feb 28 = 28 days → 28/360."""
        start = dt.datetime(2025, 1, 31)
        end = dt.datetime(2025, 2, 28)
        expected = 28.0 / 360.0
        assert np.isclose(
            calculate_year_fraction(start, end, DayCountConvention.THIRTY_360_US), expected
        )

    def test_one_full_year(self):
        """Jan 1 → Jan 1 next year = 360/360 = 1.0."""
        start = dt.datetime(2025, 1, 1)
        end = dt.datetime(2026, 1, 1)
        expected = 360.0 / 360.0
        assert np.isclose(
            calculate_year_fraction(start, end, DayCountConvention.THIRTY_360_US), expected
        )

    def test_same_month(self):
        """15th to 25th of same month = 10/360."""
        start = dt.datetime(2025, 6, 15)
        end = dt.datetime(2025, 6, 25)
        expected = 10.0 / 360.0
        assert np.isclose(
            calculate_year_fraction(start, end, DayCountConvention.THIRTY_360_US), expected
        )

    def test_same_date_returns_zero(self):
        d = dt.datetime(2025, 3, 15)
        assert calculate_year_fraction(d, d, DayCountConvention.THIRTY_360_US) == 0.0

    def test_end_of_month_31st(self):
        """Month with 31 days: d2=31 stays 31 when d1 is not 30 or 31."""
        start = dt.datetime(2025, 1, 1)
        end = dt.datetime(2025, 1, 31)
        # d1=1 (not 30 or 31), so d2=31 stays → (0*360 + 0*30 + (31-1)) / 360 = 30/360
        expected = 30.0 / 360.0
        assert np.isclose(
            calculate_year_fraction(start, end, DayCountConvention.THIRTY_360_US), expected
        )

    def test_31st_to_31st(self):
        """When both d1 and d2 are 31, both become 30."""
        start = dt.datetime(2025, 1, 31)
        end = dt.datetime(2025, 3, 31)
        # d1=31→30, d2=31→30 (since d1 was 30/31) → (0*360 + 2*30 + 0) / 360 = 60/360
        expected = 60.0 / 360.0
        assert np.isclose(
            calculate_year_fraction(start, end, DayCountConvention.THIRTY_360_US), expected
        )

    def test_quarter(self):
        """3 months: Jan 1 → Apr 1 = 90/360 = 0.25."""
        start = dt.datetime(2025, 1, 1)
        end = dt.datetime(2025, 4, 1)
        expected = 90.0 / 360.0
        assert np.isclose(
            calculate_year_fraction(start, end, DayCountConvention.THIRTY_360_US), expected
        )


# ---------------------------------------------------------------------------
# Cross-convention comparisons
# ---------------------------------------------------------------------------


class TestCrossConventionComparisons:
    """Verify ordering relations between conventions for the same date pair."""

    def test_act_360_greater_than_act_365f(self):
        """ACT/360 fraction > ACT/365F fraction for any positive period."""
        start = dt.datetime(2025, 1, 1)
        end = dt.datetime(2025, 7, 1)
        frac_360 = calculate_year_fraction(start, end, DayCountConvention.ACT_360)
        frac_365f = calculate_year_fraction(start, end, DayCountConvention.ACT_365F)
        assert frac_360 > frac_365f

    def test_act_365f_greater_than_act_365_25(self):
        """ACT/365F > ACT/365.25 because 365 < 365.25 denominator."""
        start = dt.datetime(2025, 1, 1)
        end = dt.datetime(2025, 7, 1)
        frac_365f = calculate_year_fraction(start, end, DayCountConvention.ACT_365F)
        frac_365_25 = calculate_year_fraction(start, end, DayCountConvention.ACT_365_25)
        assert frac_365f > frac_365_25

    def test_ordering_act_360_gt_365f_gt_365_25(self):
        """ACT/360 > ACT/365F > ACT/365.25 for any positive period."""
        start = dt.datetime(2025, 1, 1)
        end = dt.datetime(2025, 7, 1)
        frac_360 = calculate_year_fraction(start, end, DayCountConvention.ACT_360)
        frac_365f = calculate_year_fraction(start, end, DayCountConvention.ACT_365F)
        frac_365_25 = calculate_year_fraction(start, end, DayCountConvention.ACT_365_25)
        assert frac_360 > frac_365f > frac_365_25
