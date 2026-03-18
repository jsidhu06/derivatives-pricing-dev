import datetime as dt
import pytest

from derivatives_pricing.enums import AsianAveraging, ExerciseType, OptionType
from derivatives_pricing.exceptions import ConfigurationError, ValidationError
from derivatives_pricing.valuation import AsianSpec, PayoffSpec, VanillaSpec
from derivatives_pricing.valuation import contracts


def test_contract_types_reexport_from_valuation_module():
    """Public re-exports should point to the canonical contracts module classes."""
    assert VanillaSpec is contracts.VanillaSpec
    assert PayoffSpec is contracts.PayoffSpec
    assert AsianSpec is contracts.AsianSpec


def test_vanilla_spec_instantiation_smoke():
    spec = VanillaSpec(
        option_type=OptionType.CALL,
        exercise_type=ExerciseType.EUROPEAN,
        strike=100.0,
        maturity=dt.datetime(2026, 1, 1),
        currency="USD",
    )
    assert spec.option_type is OptionType.CALL
    assert spec.exercise_type is ExerciseType.EUROPEAN
    assert spec.strike == 100.0


def test_payoff_spec_instantiation_smoke():
    spec = PayoffSpec(
        exercise_type=ExerciseType.AMERICAN,
        maturity=dt.datetime(2026, 1, 1),
        payoff_fn=lambda s: s,
        currency="USD",
    )
    assert spec.exercise_type is ExerciseType.AMERICAN
    assert spec.strike is None


def test_asian_spec_instantiation_smoke():
    spec = AsianSpec(
        averaging=AsianAveraging.ARITHMETIC,
        option_type=OptionType.PUT,
        exercise_type=ExerciseType.EUROPEAN,
        strike=100.0,
        maturity=dt.datetime(2026, 1, 1),
        currency="USD",
        num_observations=12,
    )
    assert spec.averaging is AsianAveraging.ARITHMETIC
    assert spec.option_type is OptionType.PUT
    assert spec.exercise_type is ExerciseType.EUROPEAN


class TestAsianSpecValidation:
    """Test AsianSpec validation rejects invalid configurations."""

    def _base_kw(self, **overrides):
        kw = dict(
            averaging=AsianAveraging.ARITHMETIC,
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=100.0,
            maturity=dt.datetime(2026, 1, 1),
            num_observations=12,
        )
        kw.update(overrides)
        return kw

    def test_num_observations_less_than_2(self):
        with pytest.raises(ValidationError, match="num_observations"):
            AsianSpec(**self._base_kw(num_observations=1))

    def test_num_observations_not_int(self):
        with pytest.raises(ValidationError, match="num_observations"):
            AsianSpec(**self._base_kw(num_observations=5.5))

    def test_observed_average_without_count(self):
        with pytest.raises(ValidationError, match="observed_average and observed_count"):
            AsianSpec(**self._base_kw(observed_average=100.0))

    def test_observed_count_without_average(self):
        with pytest.raises(ValidationError, match="observed_average and observed_count"):
            AsianSpec(**self._base_kw(observed_count=3))

    def test_observed_average_negative(self):
        with pytest.raises(ValidationError, match="observed_average must be > 0"):
            AsianSpec(**self._base_kw(observed_average=-10.0, observed_count=2))

    def test_observed_count_zero(self):
        with pytest.raises(ValidationError, match="observed_count must be a positive"):
            AsianSpec(**self._base_kw(observed_average=100.0, observed_count=0))

    def test_both_fixing_dates_and_num_observations(self):
        with pytest.raises(ValidationError, match="exactly one"):
            AsianSpec(
                **self._base_kw(
                    num_observations=12,
                    fixing_dates=[dt.datetime(2025, 6, 1)],
                )
            )

    def test_neither_fixing_dates_nor_num_observations(self):
        with pytest.raises(ValidationError, match="exactly one"):
            AsianSpec(**self._base_kw(num_observations=None))

    def test_fixing_dates_not_ascending(self):
        dates = [dt.datetime(2025, 6, 1), dt.datetime(2025, 3, 1)]
        with pytest.raises(ValidationError, match="ascending"):
            AsianSpec(**self._base_kw(num_observations=None, fixing_dates=dates))

    def test_fixing_dates_beyond_maturity(self):
        dates = [dt.datetime(2027, 1, 1)]
        with pytest.raises(ValidationError, match="beyond maturity"):
            AsianSpec(**self._base_kw(num_observations=None, fixing_dates=dates))

    def test_fixing_dates_duplicates(self):
        fixing_date = dt.datetime(2025, 6, 1)
        with pytest.raises(ValidationError, match="unique"):
            AsianSpec(
                **self._base_kw(num_observations=None, fixing_dates=[fixing_date, fixing_date])
            )

    def test_strike_negative(self):
        with pytest.raises(ValidationError, match="strike"):
            AsianSpec(**self._base_kw(strike=-10.0))

    def test_invalid_averaging_type(self):
        with pytest.raises(ConfigurationError, match="averaging"):
            AsianSpec(**self._base_kw(averaging="arithmetic"))
