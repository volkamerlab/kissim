"""
Unit and regression test for the kissim.comparison.weights module.
"""

import pytest
import numpy as np

from kissim.comparison import weights


@pytest.mark.parametrize(
    "feature_weights, feature_weights_formatted",
    [
        (None, np.array([0.0667] * 15)),
        (
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ),
    ],
)
def test_format_weights(feature_weights, feature_weights_formatted):
    """
    Test if feature weights are added correctly to feature distance DataFrame.

    Parameters
    ----------
    feature_weights : None or list of float
        Feature weights.
    feature_weights_formatted : list of float
        Formatted feature weights of length 15.
    """

    feature_weights_formatted_calculated = weights.format_weights(feature_weights)

    assert np.isclose(
        np.std(feature_weights_formatted),
        np.std(feature_weights_formatted_calculated),
        rtol=1e-04,
    )


@pytest.mark.parametrize("feature_weights", [{"a": 0}, "bla"])
def test_format_weights_typeerror(feature_weights):
    """
    Test if wrong data type of input feature weights raises TypeError.
    """

    with pytest.raises(TypeError):
        weights.format_weights(feature_weights)


@pytest.mark.parametrize(
    "feature_weights",
    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0]],
)
def test_format_weights_valueerror(feature_weights):
    """
    Test if wrong data type of input feature weights raises TypeError.
    """

    with pytest.raises(ValueError):
        weights.format_weights(feature_weights)
