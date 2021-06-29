"""
Calculates structure distance matrices and kinase distance matrices.
"""

import numpy as np
import pandas as pd


def structure_distance_matrix(structure_distances, coverage_min=0.0):
    """
    Get fingerprint distances for all structure pairs in the form of a matrix (DataFrame).

    Parameters
    ----------
    structure_distances : pandas.DataFrame
        Fingerprint distance and bit coverage for each structure pair (kinase pair).
    fill : bool
        Fill or fill not (default) lower triangle of distance matrix.
    coverage_min : float
        Returns only pairs with a user-defined minimum coverage (defaults to 0.0, i.e. no
        coverage restrictions).

    Returns
    -------
    pandas.DataFrame
        Structure distance matrix.
    """

    data = structure_distances

    # Filter by coverage
    data = data[data["bit_coverage"] >= coverage_min]
    # Data for upper half of the matrix
    pairs_upper = data[["structure.1", "structure.2", "distance"]]
    # Data for lower half of the matrix
    pairs_lower = pairs_upper.rename(
        columns={"structure.1": "structure.2", "structure.2": "structure.1"}
    )

    # Concatenate upper and lower matrix data
    pairs = pd.concat([pairs_upper, pairs_lower]).sort_values(["structure.1", "structure.2"])
    # Convert to matrix
    matrix = pairs.pivot(columns="structure.2", index="structure.1", values="distance")
    # Matrix diagonal is NaN > set to 0.0
    np.fill_diagonal(matrix.values, 0)

    return matrix


def kinase_distance_matrix(
    structure_distances, by="minimum", fill_diagonal=True, coverage_min=0.0
):
    """
    Extract per kinase pair one distance value from the set of structure pair distance values
    and return these  fingerprint distances for all kinase pairs in the form of a matrix
    (DataFrame).

    Parameters
    ----------
    structure_distances : pandas.DataFrame
        Fingerprint distance and bit coverage for each structure pair (kinase pair).
    by : str
        Condition on which the distance value per kinase pair is extracted from the set of
        distances values per structure pair. Default: Minimum distance value.
    fill_diagonal : bool
        Fill diagonal with 0 (same kinase has distance of 0) by default. If `False`, diagonal
        will be a experimental values calculated based on the structure pairs per kinase pair.
        Is by default set to False, if `by="size"`.
    coverage_min : float
        Returns only pairs with a user-defined minimum coverage (defaults to 0.0, i.e. no
        coverage restrictions).

    Returns
    -------
    pandas.DataFrame
        Kinase distance matrix.
    """

    if by == "size":
        fill_diagonal = False

    # Data for upper half of the matrix
    pairs_upper = kinase_distances(structure_distances, by, coverage_min).reset_index()[
        ["kinase.1", "kinase.2", "distance"]
    ]
    # Data for lower half of the matrix
    pairs_lower = pairs_upper.rename(columns={"kinase.1": "kinase.2", "kinase.2": "kinase.1"})

    # Concatenate upper and lower matrix data
    pairs = (
        pd.concat([pairs_upper, pairs_lower])
        .sort_values(["kinase.1", "kinase.2"])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Convert to matrix
    matrix = pairs.pivot(columns="kinase.2", index="kinase.1", values="distance")

    if fill_diagonal:
        np.fill_diagonal(matrix.values, 0)

    # If matrix contains number of structure pairs: NaN > 0, cast to int
    if by == "size":
        matrix = matrix.astype("int64")

    return matrix


def kinase_distances(structure_distances, by="minimum", coverage_min=0.0):
    """
    Extract per kinase pair one distance value from the set of structure pair distance values.

    Parameters
    ----------
    structure_distances : pandas.DataFrame
        Fingerprint distance and bit coverage for each structure pair (kinase pair).
    by : str
        Condition on which the distance value per kinase pair is extracted from the set of
        distances values per structure pair. Default: Minimum distance value.
    coverage_min : float
        Returns only pairs with a user-defined minimum coverage (defaults to 0.0, i.e. no
        coverage restrictions).

    Returns
    -------
    pandas.DataFrame
        Fingerprint distance and coverage for kinase pairs.
    """

    data = structure_distances

    # Filter by coverage
    data = data[data["bit_coverage"] >= coverage_min].reset_index()
    # Group by kinase names
    structure_distances_grouped_by_kinases = data.groupby(by=["kinase.1", "kinase.2"], sort=False)

    # Get distance values per kinase pair based on given condition
    # Note: For min/max we'd like to know which structure pairs were selected!
    by_terms = "minimum maximum mean median size std".split()

    if by == "minimum":
        kinase_distances = data.iloc[
            structure_distances_grouped_by_kinases["distance"].idxmin()
        ].set_index(["kinase.1", "kinase.2"])
    elif by == "maximum":
        kinase_distances = data.iloc[
            structure_distances_grouped_by_kinases["distance"].idxmax()
        ].set_index(["kinase.1", "kinase.2"])
    elif by == "mean":
        kinase_distances = structure_distances_grouped_by_kinases.mean()[["distance"]]
    elif by == "median":
        kinase_distances = structure_distances_grouped_by_kinases.median()[["distance"]]
    elif by == "size":
        kinase_distances = structure_distances_grouped_by_kinases.size().to_frame("distance")
    elif by == "std":
        kinase_distances = structure_distances_grouped_by_kinases.std()[["distance"]]
        kinase_distances = round(kinase_distances, 3)
    else:
        raise ValueError(f'Condition "by" unknown. Choose from: {", ".join(by_terms)}')

    return kinase_distances
