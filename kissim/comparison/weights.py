"""
kissim.comparison.weights

Defines the feature weights.
"""

import numpy as np


def format_weights(feature_weights=None):
    """
    Format feature weights.

    Parameters
    ----------
    feature_weights : None or list of float
        Feature weights of the following form:
        (i) None
            Default feature weights: All features equally distributed to 1/15
            (15 features in total).
        (ii) By feature (list of 15 floats):
            Features to be set in the following order: size, hbd, hba, charge, aromatic,
            aliphatic, sco, exposure, distance_to_centroid, distance_to_hinge_region,
            distance_to_dfg_region, distance_to_front_pocket, moment1, moment2, and moment3.
            All floats must sum up to 1.0.

    Returns
    -------
    np.ndarray
        Feature weights.
    """

    if feature_weights is None:
        feature_weights = [1.0 / 15] * 15  # TODO do not hardcode 15

    elif isinstance(feature_weights, (list, np.ndarray)):
        # Check if feature weight keys are correct
        if len(feature_weights) != 15:
            raise ValueError(f"List must have length 15, but has length {len(feature_weights)}.")
        # Sum of weights must be 1.0
        if not np.isclose(sum(feature_weights), 1.0, rtol=1e-04):
            raise ValueError(f"Sum of all weights must be one, but is {sum(feature_weights)}.")

    else:
        raise TypeError(
            f'Data type of "feature_weights" parameter must be list, '
            f"but is {type(feature_weights)}."
        )

    return np.array(feature_weights)
