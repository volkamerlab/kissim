"""
Unit and regression test for kinsim_structure.similarity.FeatureDistances methods.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kinsim_structure.auxiliary import KlifsMoleculeLoader, PdbChainLoader
from kinsim_structure.encoding import Fingerprint, FEATURE_NAMES
from kinsim_structure.similarity import FeatureDistances


def generate_fingerprints_from_files(mol2_filenames, pdb_filenames, chain_ids):
    """
    Helper function: Generate multiple fingerprints from files.

    Parameters
    ----------
    mol2_filenames : list of str
        Paths to multiple mol2 files.
    pdb_filenames : list of str
        Paths to multiple cif files.
    chain_ids : list of str
        Multiple chain IDs.

    Returns
    -------
    list of kinsim_structure.encoding.Fingerprint
        List of fingerprints.
    """

    # Fingerprints
    fingerprints = []

    for mol2_filename, pdb_filename, chain_id in zip(mol2_filenames, pdb_filenames, chain_ids):
        mol2_path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / mol2_filename
        pdb_path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / pdb_filename

        klifs_molecule_loader = KlifsMoleculeLoader(mol2_path=mol2_path)
        pdb_chain_loader = PdbChainLoader(pdb_path=pdb_path, chain_id=chain_id)

        fingerprint = Fingerprint()
        fingerprint.from_molecule(klifs_molecule_loader.molecule, pdb_chain_loader.chain)

        fingerprints.append(fingerprint)

    return fingerprints


@pytest.mark.parametrize('mol2_filenames, pdb_filenames, chain_ids, feature_type_dimension', [
    (
        ['ABL1/2g2i_chainA/pocket.mol2', 'AAK1/4wsq_altA_chainB/pocket.mol2'],
        ['2g2i.cif', '4wsq.cif'],
        ['A', 'B'],
        pd.Series([8, 4, 3], index='physicochemical distances moments'.split())
    )
])
def test_from_fingerprints(mol2_filenames, pdb_filenames, chain_ids, feature_type_dimension):
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

    # Fingerprints
    fingerprints = generate_fingerprints_from_files(mol2_filenames, pdb_filenames, chain_ids)

    # Get feature distances and check if format is correct
    feature_distances = FeatureDistances()
    feature_distances.from_fingerprints(
        fingerprint1=fingerprints[0],
        fingerprint2=fingerprints[1],
        distance_measure='euclidean'
    )

    feature_type_dimension_calculated = feature_distances.data.groupby(by='feature_type', sort=False).size()

    assert all(feature_type_dimension_calculated == feature_type_dimension)


@pytest.mark.parametrize('feature_type, bit_number, bit_coverage', [
    ('moments', 4, 1.0),
    ('moments', 2, 0.5),
    ('moments', 0, 0.0),
    ('physicochemical', 50, 0.59),
    ('distances', 1, 0.01)
])
def test_get_bit_coverage(feature_type, bit_number, bit_coverage):
    """
    Test bit coverage calculation.

    Parameters
    ----------
    feature_type : str
        Feature type: physicochemical, distances or moments.
    bit_number : int
        Number of feature bits used for distance calculation.
    bit_coverage : float
        Bit coverage describing the percentage of bits used for distance calculation.
    """

    feature_distances_generator = FeatureDistances()
    bit_coverage_calculated = feature_distances_generator._get_bit_coverage(feature_type, bit_number)

    assert np.isclose(bit_coverage_calculated, bit_coverage, rtol=1e-02)  # Coverage has two decimals


@pytest.mark.parametrize('feature_type, bit_number', [
    ('xxx', 1),  # Feature type unknown
    ('moments', 5)  # Too many bits
])
def test_get_bit_coverage_valueerror(feature_type, bit_number):
    """
    Test exceptions for bit coverage calculation.

    Parameters
    ----------
    feature_type : str
        Feature type: physicochemical, distances or moments.
    bit_number : int
        Number of feature bits used for distance calculation.
    """

    with pytest.raises(ValueError):
        feature_distances_generator = FeatureDistances()
        feature_distances_generator._get_bit_coverage(feature_type, bit_number)


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
        Type of distance measure, defaults to Euclidean distance.
    distance : float
        Distance between two value lists.
    """

    feature_distances_generator = FeatureDistances()
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
    """
    Test TypeError exceptions in distance calculation for two value (feature) lists.

    Parameters
    ----------
    feature_pair : pandas.DataFrame
        Pairwise bits of one feature extracted from two fingerprints (only bit positions without any NaN value).
    distance_measure : str
        Type of distance measure, defaults to Euclidean distance.
    """

    with pytest.raises(TypeError):
        feature_distance_generator = FeatureDistances()
        feature_distance_generator._calc_feature_distance(feature_pair, distance_measure)


@pytest.mark.parametrize('feature_pair, distance_measure', [
    (pd.DataFrame([[1, 2], [1, 2]]), 'xxx'),  # Distance measure is not implemented
    (pd.DataFrame([[1, 2, 1], [1, 2, 1]]), 'euclidean')  # Feature pair has more than two columns
])
def test_calc_feature_distance_valueerror(feature_pair, distance_measure):
    """
    Test ValueError exceptions in distance calculation for two value (feature) lists.

    Parameters
    ----------
    feature_pair : pandas.DataFrame
        Pairwise bits of one feature extracted from two fingerprints (only bit positions without any NaN value).
    distance_measure : str
        Type of distance measure, defaults to Euclidean distance.
    """

    with pytest.raises(ValueError):
        feature_distance_generator = FeatureDistances()
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
    """
    Test extracting fingerprint pairs for each feature.

    Parameters
    ----------
    mol2_filenames : list of str
        Paths to two mol2 files.
    pdb_filenames : list of str
        Paths to two cif files.
    chain_ids : list of str
        Two chain IDs.
    n_bits_wo_nan_size : int
        Number of bits after removing all positions with any NaN value for size feature.
    """

    # Fingerprints
    fingerprints = generate_fingerprints_from_files(mol2_filenames, pdb_filenames, chain_ids)

    # Fingerprint pair
    feature_distances_generator = FeatureDistances()
    pair = feature_distances_generator._extract_fingerprint_pair(fingerprints[0], fingerprints[1], normalized=True)

    # Correct feature type keys?
    assert pair.keys() == FEATURE_NAMES.keys()

    for feature_type in pair.keys():

        # Correct feature names per feature type?
        assert list(pair[feature_type].keys()) == FEATURE_NAMES[feature_type]

        for feature_name in pair[feature_type].keys():

            # Correct DataFrame column names?
            assert list(pair[feature_type][feature_name].columns) == 'fingerprint1 fingerprint2'.split()

            # Correct number of bits for one example feature?
            if (feature_type == 'physicochemical') and (feature_name == 'size'):
                assert len(pair[feature_type][feature_name]) == n_bits_wo_nan_size


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

    feature_distances_generator = FeatureDistances()
    score_calculated = feature_distances_generator._euclidean_distance(values1, values2)

    assert np.isclose(score_calculated, distance, rtol=1e-04)
