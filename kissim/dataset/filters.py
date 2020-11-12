"""
kissim.dataset.filters

Defines filters for the KLIFS dataset.
"""

from functools import wraps
import datetime
import logging

logger = logging.getLogger(__name__)


def log_step(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.datetime.now()
        time_taken = str(start_time - end_time)
        logger.info(f"{func.__name__:<30}{result.shape[0]:>7} structures ({time_taken}s)")
        return result

    return wrapper


@log_step
def make_copy(dataframe):
    """
    Make a copy of the input DataFrame.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input DataFrame.

    Returns
    -------
    pandas.DataFrame
        Copy of input DataFrame.
    """
    return dataframe.copy()


@log_step
def select_species(structures, species):
    """
    Filter structures based on selected species.

    Parameters
    ----------
    structures : pandas.DataFrame
        Structures DataFrame from opencadd.databases.klifs module.
    species : str or list of str
        Species.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame.
    """
    if not isinstance(species, list):
        species = [species]
    return structures[structures["species.klifs"].isin(species)]


@log_step
def select_dfg(structures, dfg_conformation):
    """
    Filter structures based on selected DFG conformation(s).

    Parameters
    ----------
    structures : pandas.DataFrame
        Structures DataFrame from opencadd.databases.klifs module.
    dfg_conformation : str of list of str
        DFG conformation(s).

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame.
    """
    if not isinstance(dfg_conformation, list):
        dfg_conformation = [dfg_conformation]
    return structures[structures["structure.dfg"].isin(dfg_conformation)]


@log_step
def select_resolution(structures, resolution_max):
    """
    Filter structures for structure with a resolution equal or lower than a given value.

    Parameters
    ----------
    structures : pandas.DataFrame
        Structures DataFrame from opencadd.databases.klifs module.
    resolution_max : int or float
        Maximum resolution value.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame.
    """
    return structures[structures["structure.resolution"] <= resolution_max]


@log_step
def select_qualityscore(structures, qualityscore_min):
    """
    Filter structures for structures with a quality score equal or greater than a given value.

    Parameters
    ----------
    structures : pandas.DataFrame
        Structures DataFrame from opencadd.databases.klifs module.
    qualityscore_min : int or float
        Minimum KLIFS quality score value.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame.
    """
    return structures[structures["structure.qualityscore"] >= qualityscore_min]


@log_step
def select_best_pdb_kinase_pairs(structures):
    """
    Pick only the best structure (based on the quality score) among all structures per kinase-PDB
    pair.

    Parameters
    ----------
    structures : pandas.DataFrame
        Structures DataFrame from opencadd.databases.klifs module.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame.
    """
    # Sort structures by kinase, PDB and descending qualityscore
    structures.sort_values(
        by=["kinase.klifs_name", "structure.pdb_id", "structure.qualityscore"],
        ascending=[True, True, False],
        inplace=True,
    )
    # Drop duplicate kinase-PDB pairs, keep only the pair with best quality score
    structures.drop_duplicates(
        subset=["kinase.klifs_name", "structure.pdb_id"], keep="first", inplace=True
    )
    return structures
