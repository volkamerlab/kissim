"""
Unit and regression test for the kissim.utils.
"""

import pytest
import numpy as np

from kissim.utils import (
    set_n_cores,
    calculate_first_second_third_moments,
    min_max_normalization_scalar,
)


@pytest.mark.parametrize(
    "n_cores",
    [1000000000000],
)
def test_get_n_cores_valueerror(n_cores):
    """
    Test if number of cores are set correctly.
    """

    with pytest.raises(ValueError):
        set_n_cores(n_cores)


@pytest.mark.parametrize(
    "values, moments",
    [
        ([0], [0, 0, 0]),
        ([0, 0], [0, 0, 0]),
        ([1, 0], [0.5, 0.5, 0]),
        ([3, 0, 0], [1, 1.4142135, 1.2599210]),
        ([], [np.nan, np.nan, np.nan]),
        ([np.nan, np.nan], [np.nan, np.nan, np.nan]),
    ],
)
def test_calculate_first_second_third_moment(values, moments):
    """
    Test static method that calculates the first three moments of a distribution.
    """

    moments_calculated = calculate_first_second_third_moments(values)
    if len(values) > 0 and not all(np.isnan(values)):
        assert pytest.approx(moments_calculated, abs=1e-6) == moments
    else:
        assert all(moments_calculated) == all(moments)


@pytest.mark.parametrize(
    "value, minimum, maximum, value_normalized",
    [(15, 10, 20, 0.5), (10, 10, 20, 0.0), (0, 10, 20, 0.0), (np.nan, 10, 20, np.nan)],
)
def test_min_max_normalization_scalar(value, minimum, maximum, value_normalized):
    """
    Test min-max normalization
    """

    value_normalized_calculated = min_max_normalization_scalar(value, minimum, maximum)
    if not np.isnan(value):
        assert pytest.approx(value_normalized_calculated, abs=1e-4) == value_normalized
