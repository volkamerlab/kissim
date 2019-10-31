"""
Unit and regression test for kinsim_structure.similarity functions.
"""

import numpy as np
import pandas as pd
import pytest

from kinsim_structure.similarity import _euclidean_distance, _get_values_without_nan, calculate_similarity, _calc_feature_distance


@pytest.mark.parametrize('fingerprint1, fingerprint2, measure, score, coverage', [
    (
        pd.DataFrame([[1, 2], [3, 0]]),
        pd.DataFrame([[1, 1], [1, 1]]),
        'ballester',
        0.5,
        1.0
    ),
    (
        pd.DataFrame([[1, 2], [3, None]]),
        pd.DataFrame([[1, 1], [1, None]]),
        'ballester',
        0.5,
        0.75
    ),
    (
        pd.DataFrame([[1, 2], [3, np.nan]]),
        pd.DataFrame([[1, 1], [1, np.nan]]),
        'ballester',
        0.5,
        0.75
    )
])
def test_calculate_similarity(fingerprint1, fingerprint2, measure, score, coverage):
    """
    Test pairwise fingerprint similarity (similarity score and bit coverage) calculation given a defined measure.

    Parameters
    ----------
    fingerprint1 : pandas.DataFrame (or 1D array-like)
        Fingerprint for molecule.
    fingerprint2 : pandas.DataFrame (or 1D array-like)
        Fingerprint for molecule.
    measure : str
        Similarity measurement method:
         - ballester (inverse of the translated and scaled Manhattan distance)
    score : float
        Similarity score.
    coverage : float
         Coverage (ratio of bits used for similarity score).
    """

    assert calculate_similarity(fingerprint1, fingerprint2, measure='ballester') == (score, coverage)




@pytest.mark.parametrize('mol2_filenames, pdb_filenames, chain_ids', [
    (
        ['ABL1/2g2i_chainA/pocket.mol2', 'AAK1/4wsq_altA_chainB/pocket.mol2'],
        ['2g2i.cif', '4wsq.cif'],
        ['A', 'B']
    )
])
def test_calc_feature_distances(mol2_filenames, pdb_filenames, chain_ids):
    """
    Test data type and dimensions of feature distances between two fingerprints.

    Parameters
    ----------
    mol2_filenames : list of str
        Paths to two mol2 files.
    pdb_filenames : list of str
        Paths to two cif files.
    chain_ids : list of str
        Two chain IDs.
    """

    # Fingerprint 1
    mol2_path1 = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / mol2_filenames[0]
    pdb_path1 = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / pdb_filenames[0]

    klifs_molecule_loader1 = KlifsMoleculeLoader(mol2_path=mol2_path1)
    pdb_chain_loader1 = PdbChainLoader(pdb_path=pdb_path1, chain_id=chain_ids[0])

    fingerprint1 = Fingerprint()
    fingerprint1.from_molecule(klifs_molecule_loader1.molecule, pdb_chain_loader1.chain)

    # Fingerprint 2
    mol2_path2 = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / mol2_filenames[1]
    pdb_path2 = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / pdb_filenames[1]

    klifs_molecule_loader2 = KlifsMoleculeLoader(mol2_path=mol2_path2)
    pdb_chain_loader2 = PdbChainLoader(pdb_path=pdb_path2, chain_id=chain_ids[1])

    fingerprint2 = Fingerprint()
    fingerprint2.from_molecule(klifs_molecule_loader2.molecule, pdb_chain_loader2.chain)

    # Get feature distances and check if format is correct
    feature_distances = _calc_feature_distances(
        fingerprint1=fingerprint1,
        fingerprint2=fingerprint2,
        distance_measure='euclidean',
        normalized=True
    )

    assert list(feature_distances.keys()) == 'physicochemical distances moments'.split()
    assert [len(value) for key, value in feature_distances.items()] == [8, 4, 3]


@pytest.mark.parametrize('feature_values1, feature_values2, distance_measure, distance', [
    ([0, 0], [4, 3], 'euclidean', 2.5),
    (pd.Series([0, 0]), pd.Series([4, 3]), 'euclidean', 2.5),
    (pd.Series([0, 0, np.nan]), pd.Series([4, 3, 1]), 'euclidean', 2.5)
])
def test_calc_feature_distance(feature_values1, feature_values2, distance_measure, distance):
    """
    Test distance calculation for two value (feature) lists.

    Parameters
    ----------
    feature_values1 : list or pandas.Series
        Value list (same length as values2).
    feature_values2 : list or pandas.Series
        Value list (same length as values1).
    distance_measure : str
        Distance measure.
    distance : float
        Distance between two value lists.
    """

    distance_calculated = _calc_feature_distance(feature_values1, feature_values2, distance_measure)

    assert np.isclose(distance_calculated, distance, rtol=1e-04)


@pytest.mark.parametrize('values1, values2, values_reduced, coverage', [
    ([0, 0, np.nan, 1], [4, 3, 1, np.nan], [[0, 0], [4, 3]], 0.5),
    ([0, 0, np.nan], [4, 3, np.nan], [[0, 0], [4, 3]], 0.6667),
    ([0, 0], [4, 3], [[0, 0], [4, 3]], 1.0)
])
def test_get_values_without_nan(values1, values2, values_reduced, coverage):

    values_reduced_calculated = _get_values_without_nan(values1, values2)

    assert np.isclose(values_reduced_calculated['coverage'], coverage, rtol=1e-04)
    assert np.isclose(values_reduced_calculated['values'], values_reduced, rtol=1e-04).all()


@pytest.mark.parametrize('values1, values2, distance', [
    ([0, 0], [4, 3], 2.5),
    (pd.Series([0, 0]), pd.Series([4, 3]), 2.5)
])
def test_euclidean_distance(values1, values2, distance):
    """
    Test Euclidean distance calculation.

    Parameters
    ----------
    values1 : list or pandas.Series
        Value list (same length as values2).
    values2 : list or pandas.Series
        Value list (same length as values1).
    distance : float
        Euclidean distance between two value lists.
    """

    score_calculated = _euclidean_distance(values1, values2)

    assert np.isclose(score_calculated, distance, rtol=1e-04)