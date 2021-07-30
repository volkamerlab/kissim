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
