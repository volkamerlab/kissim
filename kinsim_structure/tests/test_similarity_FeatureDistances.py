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


@pytest.mark.parametrize('feature1, feature2, distance, bit_coverage', [
    (pd.Series([1, 1, 1, 1]), pd.Series([0, 0, 0, 0]), 0.5, 1.0),
    (pd.Series([1, 1, 1, 1, np.nan]), pd.Series([0, 0, 0, 0, 0]), 0.5, 0.8),
    (pd.Series([1, 1, 1, 1, 1]), pd.Series([0, 0, 0, 0, np.nan]), 0.5, 0.8),
    (pd.Series([1, 1, 1, 1, np.nan]), pd.Series([0, 0, 0, 0, np.nan]), 0.5, 0.8)
])
def test_from_features(feature1, feature2, distance, bit_coverage):
    """
    Test if feature distance and bit coverage is correct for given feature bits.

    Parameters
    ----------
    feature1 : pd.Series
        Feature bits for a given feature in fingerprint 1.
    feature2 : pd.Series
        Feature bits for a given feature in fingerprint 2.
    distance : float
        Distance value for a feature pair.
    bit_coverage : float
        Bit coverage value for a feature pair.
    """

    feature_distances = FeatureDistances()
    distance_calculated, bit_coverage_calculated = feature_distances.from_features(feature1, feature2)

    assert np.isclose(distance_calculated, distance, rtol=1e-04)
    assert np.isclose(bit_coverage_calculated, bit_coverage, rtol=1e-04)


@pytest.mark.parametrize('feature1, feature2', [
    (pd.Series([1, 1, 1, 1]), pd.Series([0, 0, 0]))
])
def test_from_features_valueerror(feature1, feature2):
    """
    Test if feature distance and bit coverage is correct for given feature bits, here if error is raised correctly.

    Parameters
    ----------
    feature1 : pd.Series
        Feature bits for a given feature in fingerprint 1.
    feature2 : pd.Series
        Feature bits for a given feature in fingerprint 2.
    """

    feature_distances = FeatureDistances()

    with pytest.raises(ValueError):
        feature_distances.from_features(feature1, feature2)


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


@pytest.mark.parametrize('values1, values2, distance', [
    ([0, 0], [4, 3], 2.5),
    (np.array([0, 0]), np.array([4, 3]), 2.5),
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
    (np.array([0, 0]), np.array([4, 3]), 3.5),
    (pd.Series([0, 0]), pd.Series([4, 3]), 3.5)
])
def test_scaled_cityblock_distance(values1, values2, distance):
    """
    Test Manhattan distance calculation.

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
