"""
Unit and regression test for kinsim_structure.similarity functions.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kinsim_structure.auxiliary import KlifsMoleculeLoader, PdbChainLoader
from kinsim_structure.encoding import Fingerprint, FEATURE_NAMES
from kinsim_structure.similarity import FeatureDistancesGenerator


@pytest.mark.parametrize('mol2_filenames, pdb_filenames, chain_ids', [
    (
        ['ABL1/2g2i_chainA/pocket.mol2', 'AAK1/4wsq_altA_chainB/pocket.mol2'],
        ['2g2i.cif', '4wsq.cif'],
        ['A', 'B']
    )
])
def test_from_fingerprint_pair(mol2_filenames, pdb_filenames, chain_ids):
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
    feature_distances_generator = FeatureDistancesGenerator()
    feature_distances = feature_distances_generator.from_fingerprint_pair(
        fingerprint1=fingerprint1,
        fingerprint2=fingerprint2,
        distance_measure='euclidean',
        normalized=True
    )

    assert list(feature_distances.keys()) == 'physicochemical distances moments'.split()
    assert [len(value) for key, value in feature_distances.items()] == [8, 4, 3]


@pytest.mark.parametrize('feature_pair, distance_measure, distance', [
    (pd.DataFrame([[4, 0], [0, 3]], columns=['a', 'b']), 'euclidean', 2.5),
    (pd.DataFrame([], columns=['a', 'b']), 'euclidean', np.nan)
])
def test_calc_feature_distance(feature_pair, distance_measure, distance):
    """
    Test distance calculation for two value (feature) lists.

    Parameters
    ----------
    feature_pair : pandas.DataFrame
        Pairwise bits of one feature extracted from two fingerprints (only bit positions without any NaN value).
    distance_measure : str
        Distance measure.
    distance : float
        Distance between two value lists.
    """

    feature_distances_generator = FeatureDistancesGenerator()
    distance_calculated = feature_distances_generator._calc_feature_distance(
        feature_pair,
        distance_measure
    )

    if np.isnan(distance):
        assert np.isnan(distance_calculated)
    else:
        assert np.isclose(distance_calculated, distance, rtol=1e-04)


@pytest.mark.parametrize('feature_pair, distance_measure', [
    (pd.DataFrame([[1, 2], [1, 2]]), None),  # Distance measure is not str
    ('feature_pair', 'euclidean')  # Feature pair is not pandas.DataFrame
])
def test_calc_feature_distance_typeerror(feature_pair, distance_measure):

    with pytest.raises(TypeError):
        feature_distance_generator = FeatureDistancesGenerator()
        feature_distance_generator._calc_feature_distance(feature_pair, distance_measure)


@pytest.mark.parametrize('feature_pair, distance_measure', [
    (pd.DataFrame([[1, 2], [1, 2]]), 'xxx'),  # Distance measure is not implemented
    (pd.DataFrame([[1, 2, 1], [1, 2, 1]]), 'euclidean')  # Feature pair has more than two columns
])
def test_calc_feature_distance_valueerror(feature_pair, distance_measure):

    with pytest.raises(ValueError):
        feature_distance_generator = FeatureDistancesGenerator()
        feature_distance_generator._calc_feature_distance(feature_pair, distance_measure)


@pytest.mark.parametrize('mol2_filenames, pdb_filenames, chain_ids, n_bits_wo_nan_size', [
    (
        ['ABL1/2g2i_chainA/pocket.mol2', 'AAK1/4wsq_altA_chainB/pocket.mol2'],
        ['2g2i.cif', '4wsq.cif'],
        ['A', 'B'],
        82
    )
])
def test_extract_fingerprint_pair(mol2_filenames, pdb_filenames, chain_ids, n_bits_wo_nan_size):

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

    # Fingerprint pair
    feature_distances_generator = FeatureDistancesGenerator()
    pair = feature_distances_generator._extract_fingerprint_pair(fingerprint1, fingerprint2, normalized=True)

    # Correct feature type keys?
    assert pair.keys() == FEATURE_NAMES.keys()

    for feature_type in pair.keys():

        # Correct feature names per feature type?
        assert list(pair[feature_type].keys()) == FEATURE_NAMES[feature_type]

        for feature_name in pair[feature_type].keys():

            # Correct DataFrame column names?
            assert list(pair[feature_type][feature_name].columns) == 'fingerprint1 fingerprint2'.split()

            # Correct number of bits for one example feature?
            if (feature_type == 'physicochemical') and (feature_name) == 'size':
                assert len(pair[feature_type][feature_name]) == n_bits_wo_nan_size


@pytest.mark.parametrize('values1, values2, values_reduced, coverage', [
    ([0, 0, np.nan, 1], [4, 3, 1, np.nan], [[0, 0], [4, 3]], 0.5),
    ([0, 0, np.nan], [4, 3, np.nan], [[0, 0], [4, 3]], 0.6667),
    ([0, 0], [4, 3], [[0, 0], [4, 3]], 1.0)
])
def test_get_values_without_nan(values1, values2, values_reduced, coverage):
    """
    # TODO

    Parameters
    ----------
    values1
    values2
    values_reduced
    coverage
    """

    feature_distances_generator = FeatureDistancesGenerator()
    values_reduced_calculated = feature_distances_generator._get_values_without_nan(values1, values2)

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

    feature_distances_generator = FeatureDistancesGenerator()
    score_calculated = feature_distances_generator._euclidean_distance(values1, values2)

    assert np.isclose(score_calculated, distance, rtol=1e-04)