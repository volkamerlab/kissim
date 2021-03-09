"""
Unit and regression test for the kissim.utils.
"""

import pytest

from kissim.utils import set_n_cores


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