"""
Unit and regression test for the kissim.utils.
"""

import pytest
import numpy as np

from kissim.utils import set_n_cores, calculate_first_second_third_moments


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
