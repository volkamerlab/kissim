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

PATH_TEST_DATA = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data'


def generate_fingerprints_from_files(path_klifs_metadata, paths_mol2, paths_pdb, chain_ids):
    """
    Helper function: Generate multiple fingerprints from files.

    Parameters
    ----------
    path_klifs_metadata : pathlib.Path
        Path to unfiltered KLIFS metadata.
    paths_mol2 : list of pathlib.Path
        Paths to multiple mol2 files.
    paths_pdb : list of pathlib.Path
        Paths to multiple cif files.
    chain_ids : list of str
        Multiple chain IDs.

    Returns
    -------
    list of kinsim_structure.encoding.Fingerprint
        List of fingerprints.
    """

    fingerprints = []

    for path_mol2, path_pdb, chain_id in zip(paths_mol2, paths_pdb, chain_ids):

        klifs_molecule_loader = KlifsMoleculeLoader()
        klifs_molecule_loader.from_file(path_mol2, path_klifs_metadata)
        pdb_chain_loader = PdbChainLoader()
        pdb_chain_loader.from_file(path_pdb, chain_id)

        fingerprint = Fingerprint()
        fingerprint.from_molecule(klifs_molecule_loader.molecule, pdb_chain_loader.chain)

        fingerprints.append(fingerprint)

    return fingerprints


@pytest.mark.parametrize('path_klifs_metadata, paths_mol2, paths_pdb, chain_ids, feature_type_dimension', [
    (
        PATH_TEST_DATA / 'klifs_metadata.csv',
        [
            PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ABL1/2g2i_chainA/pocket.mol2',
            PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/pocket.mol2'
        ],
        [
            PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ABL1/2g2i_chainA/protein_pymol.pdb',
            PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/protein_pymol.pdb'
        ],
        [
            'A',
            'B'
        ],
        pd.Series([8, 4, 3], index='physicochemical distances moments'.split())
    )
])
def test_from_fingerprints(path_klifs_metadata, paths_mol2, paths_pdb, chain_ids, feature_type_dimension):
    """
    Test data type and dimensions of feature distances between two fingerprints.

    Parameters
    ----------
    path_klifs_metadata : pathlib.Path
        Path to unfiltered KLIFS metadata.
    paths_mol2 : list of str
        Paths to two mol2 files.
    paths_pdb : list of str
        Paths to two cif files.
    chain_ids : list of str
        Two chain IDs.
    """

    # Fingerprints
    fingerprints = generate_fingerprints_from_files(path_klifs_metadata, paths_mol2, paths_pdb, chain_ids)

    # Get feature distances and check if format is correct
    feature_distances = FeatureDistances()
    feature_distances.from_fingerprints(
        fingerprint1=fingerprints[0],
        fingerprint2=fingerprints[1],
        distance_measure='scaled_euclidean'
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
    (np.array([[4, 0], [0, 3]]), 'scaled_euclidean', 2.5),
    (np.array([]), 'scaled_euclidean', np.nan)
])
def test_calculate_feature_distance(feature_pair, distance_measure, distance):
    """
    Test distance calculation for two value (feature) lists.

    Parameters
    ----------
    feature_pair : np.ndarray
        Pairwise bits of one feature extracted from two fingerprints (only bit positions without any NaN value).
    distance_measure : str
        Type of distance measure, defaults to Euclidean distance.
    distance : float
        Distance between two value lists.
    """

    feature_distances_generator = FeatureDistances()
    distance_calculated = feature_distances_generator._calculate_feature_distance(
        feature_pair,
        distance_measure
    )

    if np.isnan(distance):
        assert np.isnan(distance_calculated)
    else:
        assert np.isclose(distance_calculated, distance, rtol=1e-04)


@pytest.mark.parametrize('feature_pair, distance_measure', [
    ('feature_pair', 'scaled_euclidean')  # Feature pair is not np.ndarray
])
def test_calculate_feature_distance_typeerror(feature_pair, distance_measure):
    """
    Test TypeError exceptions in distance calculation for two value (feature) lists.

    Parameters
    ----------
    feature_pair : np.ndarray
        Pairwise bits of one feature extracted from two fingerprints (only bit positions without any NaN value).
    distance_measure : str
        Type of distance measure, defaults to Euclidean distance.
    """

    with pytest.raises(TypeError):
        feature_distance_generator = FeatureDistances()
        feature_distance_generator._calculate_feature_distance(feature_pair, distance_measure)


@pytest.mark.parametrize('feature_pair, distance_measure', [
    (np.array([[1, 2], [1, 2]]), 'xxx'),  # Distance measure is not implemented
    (np.array([[1, 2], [1, 2], [1, 2]]), 'scaled_euclidean'),  # Feature pair has more than two rows
    (np.array([[1, 2], [1, 2]]), 11),  # Distance measure is not str
])
def test_calculate_feature_distance_valueerror(feature_pair, distance_measure):
    """
    Test ValueError exceptions in distance calculation for two value (feature) lists.

    Parameters
    ----------
    feature_pair : np.ndarray
        Pairwise bits of one feature extracted from two fingerprints (only bit positions without any NaN value).
    distance_measure : str
        Type of distance measure, defaults to Euclidean distance.
    """

    with pytest.raises(ValueError):
        feature_distance_generator = FeatureDistances()
        feature_distance_generator._calculate_feature_distance(feature_pair, distance_measure)


@pytest.mark.parametrize('path_klifs_metadata, paths_mol2, paths_pdb, chain_ids, n_bits_wo_nan_size', [
    (
        PATH_TEST_DATA / 'klifs_metadata.csv',
        [
            PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ABL1/2g2i_chainA/pocket.mol2',
            PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/pocket.mol2'
        ],
        [
            PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ABL1/2g2i_chainA/protein_pymol.pdb',
            PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/protein_pymol.pdb'
        ],
        [
            'A',
            'B'
        ],
        82
    )
])
def test_extract_fingerprint_pair(path_klifs_metadata, paths_mol2, paths_pdb, chain_ids, n_bits_wo_nan_size):
    """
    Test extracting fingerprint pairs for each feature.

    Parameters
    ----------
    path_klifs_metadata : pathlib.Path
        Path to unfiltered KLIFS metadata.
    paths_mol2 : list of str
        Paths to two mol2 files.
    paths_pdb : list of str
        Paths to two cif files.
    chain_ids : list of str
        Two chain IDs.
    n_bits_wo_nan_size : int
        Number of bits after removing all positions with any NaN value for size feature.
    """

    # Fingerprints
    fingerprints = generate_fingerprints_from_files(path_klifs_metadata, paths_mol2, paths_pdb, chain_ids)

    # Fingerprint pair
    feature_distances_generator = FeatureDistances()
    pair = feature_distances_generator._extract_fingerprint_pair(fingerprints[0], fingerprints[1], normalized=True)

    # Correct feature type keys?
    assert pair.keys() == FEATURE_NAMES.keys()

    for feature_type in pair.keys():

        # Correct feature names per feature type?
        assert list(pair[feature_type].keys()) == FEATURE_NAMES[feature_type]

        for feature_name in pair[feature_type].keys():

            # Correct number of bits for one example feature?
            if (feature_type == 'physicochemical') and (feature_name == 'size'):
                print(pair[feature_type][feature_name])
                assert pair[feature_type][feature_name].shape == (2, n_bits_wo_nan_size)


@pytest.mark.parametrize('values1, values2, distance', [
    (np.array([0, 0]), np.array([4, 3]), 2.5),
    ([0, 0], [4, 3], 2.5),
    (pd.Series([0, 0]), pd.Series([4, 3]), 2.5)
])
def test_scaled_euclidean_distance(values1, values2, distance):
    """
    Test Euclidean distance calculation.

    Parameters
    ----------
    values1 : np.ndarray or list of pd.Series
        Value list (same length as values2).
    values2 : np.ndarray or list of pd.Series
        Value list (same length as values1).
    distance : float
        Euclidean distance between two value lists.
    """

    feature_distances_generator = FeatureDistances()
    score_calculated = feature_distances_generator._scaled_euclidean_distance(values1, values2)

    assert np.isclose(score_calculated, distance, rtol=1e-04)


@pytest.mark.parametrize('values1, values2, distance', [
    ([0, 0], [4, 3], 3.5),
    (np.ndarray([0, 0]), np.ndarray([4, 3]), 3.5),
    (pd.Series([0, 0]), pd.Series([4, 3]), 3.5)
])
def test_scaled_cityblock_distance(values1, values2, distance):
    """
    Test Euclidean distance calculation.

    Parameters
    ----------
    values1 : np.ndarray or list of pd.Series
        Value list (same length as values2).
    values2 : np.ndarray or list of pd.Series
        Value list (same length as values1).
    distance : float
        Euclidean distance between two value lists.
    """

    feature_distances_generator = FeatureDistances()
    score_calculated = feature_distances_generator._scaled_cityblock_distance(values1, values2)

    assert np.isclose(score_calculated, distance, rtol=1e-04)
