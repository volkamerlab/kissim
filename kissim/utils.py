"""
kissim.utils

Useful utilities used across the library.
"""

import logging
import os
import shutil
import tempfile
import contextlib
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from scipy.special import cbrt
from scipy.stats.stats import moment


logger = logging.getLogger(__name__)


@contextlib.contextmanager
def enter_temp_directory(remove=True):
    """Create and enter a temporary directory; used as context manager."""
    temp_dir = tempfile.mkdtemp()
    cwd = os.getcwd()
    os.chdir(temp_dir)
    logger.debug("Entered %s", temp_dir)
    yield cwd, temp_dir
    os.chdir(cwd)
    logger.debug("Left %s", temp_dir)
    if remove:
        logger.debug("Deleting %s", temp_dir)
        shutil.rmtree(temp_dir)


def set_n_cores(n_cores):
    """
    Set the number of cores to be used for fingerprint generation.

    Parameters
    ----------
    n_cores : int or None
        Number of cores as defined by the user.
        If no number is given, use 1 core.
        If a number is given, raise error if it exceeds the number of available CPUs - 1.

    Returns
    -------
    int
        Number of cores to be used for fingerprint generation.

    Raises
    ------
    ValueError
        If input number of cores exceeds the number of available CPUs - 1.
    """

    max_n_cores = cpu_count()
    if n_cores is None:
        n_cores = 1
    else:
        if n_cores > max_n_cores:
            raise ValueError(
                f"Maximal number of available cores: {max_n_cores}. You chose: {n_cores}."
            )
    logger.info(f"Number of cores used: {n_cores}.")
    return n_cores


def calculate_first_second_third_moments(
    values,
):
    """
    Get first, second, and third moment (mean, standard deviation, and skewness)
    for a distribution of values.
    Note: Moments are based only on non-NaN values.

    Parameters
    ----------
    values : list or numpy.array or pd.Series of (float or int)
        List of values.
    """

    values = np.array(values)

    if len(values) > 0 and not all(np.isnan(values)):
        moment1 = np.nanmean(values)
        # Second and third moment: delta degrees of freedom = 0 (divisor N)
        moment2 = np.nanstd(values, ddof=0)
        moment3 = cbrt(moment(values, moment=3, nan_policy="omit"))
        return moment1, moment2, moment3
    else:
        return np.nan, np.nan, np.nan


def spatial_min_max_from_fingerprint_generator(
    fingerprint_generator, feature="distances", fine_grained=True
):
    """
    Calculate the minimum and maximum values from fingerprints data.

    Parameters
    ----------
    feature : str
        Choose `"distances"` or `"moments"` features.
    fine_grained : bool
        True (default):
            Distances: Calculate min/max per subpocket for each residue position individually.
            Moments: Calculate min/max per moment for each subpocket individually.
        False:
            Distances: Calculate min/max per subpocket over all residue positions.
            Moments: Calculate min/max per moment over all subpockets.

    Returns
    -------
    pandas.DataFrame
        Distances:
            For each subpocket save min/max (index) calculated over all residue positions or
            per residue position (column(s)).
        Moments:
            For each moment save min/max (index) calculated over all subpockets or
            per subpocket (column(s)).
    """

    if feature == "distances":
        fps_data = fingerprint_generator.distances_exploded()
        index1_name = "subpocket"
        if fine_grained:
            column_names = fps_data.index.get_level_values(1).unique().to_list()
        else:
            column_names = ["any"]
    elif feature == "moments":
        fps_data = fingerprint_generator.moments_exploded().stack().unstack(1)
        index1_name = "moment"
        if fine_grained:
            column_names = fps_data.index.get_level_values(1).unique().to_list()
        else:
            column_names = ["any"]
    else:
        raise KeyError(f"Unknown feature. Choose between distances and moments.")

    min_max = []
    for subpocket, data in fps_data.items():

        data = data.to_frame()
        if fine_grained:
            data = data.unstack()

        data_min, data_max = data.min(), data.max()

        data_min = round(data_min, 2)
        data_max = round(data_max, 2)

        min_max.append([subpocket, "min"] + data_min.tolist())
        min_max.append([subpocket, "max"] + data_max.tolist())

    min_max = pd.DataFrame(min_max, columns=[index1_name, "min_max"] + column_names).set_index(
        [index1_name, "min_max"]
    )

    return min_max


def min_max_normalization_vector(vector, minimum, maximum):
    """
    Normalize vector, either based on a single minimum/maximum or based on a element-wise minimum
    /maximum.

    Parameters
    ----------
    vector : list of float
        Vector to be normalized.
    minimum : int/float or list of int/float
        Minimum value or vector (same length as `vector` and `maximum`)
    maximum : int/float or list of int/float
        Maximum value or vector (same length as `vector` and `minimum`)

    Returns
    -------
    list of int/float
        Normalized vector
    """

    if isinstance(minimum, (int, float)):
        minimum = [minimum]
    if isinstance(maximum, (int, float)):
        maximum = [maximum]

    if len(minimum) == len(maximum) == 1:
        return [min_max_normalization_scalar(v_i, minimum[0], maximum[0]) for v_i in vector]
    elif len(minimum) == len(maximum) == len(vector) > 1:
        return [
            min_max_normalization_scalar(v_i, min_i, max_i)
            for v_i, min_i, max_i in zip(vector, minimum, maximum)
        ]
    else:
        raise ValueError("Inputs do not match; please refer to docstring.")


def min_max_normalization_scalar(scalar, minimum, maximum):
    """
    Normalize a value using minimum-maximum normalization.
    Values equal or lower / greater than the minimum / maximum value are set to 0.0 / 1.0.

    Parameters
    ----------
    scalar : float or int
        Value to be normalized.
    minimum : float or int
        Minimum value for normalization, values equal/greater than this minimum are set to 0.0.
    maximum : float or int
        Maximum value for normalization, values equal/greater than this maximum are set to 1.0.

    Returns
    -------
    float
        Normalized value.
    """

    if np.isnan(scalar):
        return np.nan
    elif minimum < scalar < maximum:
        return (scalar - minimum) / float(maximum - minimum)
    elif scalar <= minimum:
        return 0.0
    else:
        return 1.0
