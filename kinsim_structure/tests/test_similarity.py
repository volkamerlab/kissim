"""
Unit and regression tests for kinsim_structure.similarity class methods.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kinsim_structure.auxiliary import KlifsMoleculeLoader, PdbChainLoader
from kinsim_structure.encoding import Fingerprint, FingerprintGenerator, FEATURE_NAMES
from kinsim_structure.similarity import FeatureDistances, FingerprintDistance, \
    FeatureDistancesGenerator, FingerprintDistanceGenerator

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


def generate_feature_distances():
    """
    Get FeatureDistances instance with dummy data, i.e. distances between two fingerprints for each of their features,
    plus details on feature type, feature, feature bit coverage, and feature bit number.

    Returns
    -------
    kinsim_structure.similarity.FeatureDistances
        Distances between two fingerprints for each of their features, plus details on feature type, feature,
        feature bit coverage, and feature bit number.
    """

    molecule_pair_code = ['molecule1', 'molecule2']
    distances = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    bit_coverages = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # FeatureDistances (set class attributes manually)
    feature_distances = FeatureDistances()
    feature_distances.molecule_pair_code = molecule_pair_code
    feature_distances.distances = distances
    feature_distances.bit_coverages = bit_coverages

    return feature_distances


class TestsFeatureDistances:

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
    def test_from_fingerprints(self, path_klifs_metadata, paths_mol2, paths_pdb, chain_ids, feature_type_dimension):
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
    def test_from_features(self, feature1, feature2, distance, bit_coverage):
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
    def test_from_features_valueerror(self, feature1, feature2):
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
    def test_calculate_feature_distance(self, feature_pair, distance_measure, distance):
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
    def test_calculate_feature_distance_typeerror(self, feature_pair, distance_measure):
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
    def test_calculate_feature_distance_valueerror(self, feature_pair, distance_measure):
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
    def test_scaled_euclidean_distance(self, values1, values2, distance):
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
    def test_scaled_cityblock_distance(self, values1, values2, distance):
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


class TestsFingerprintDistance:

    @pytest.mark.parametrize('feature_weights, distance, coverage', [
        (
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25],
                0.5,
                0.75
        )
    ])
    def test_from_feature_distances(self, feature_weights, distance, coverage):
        """
        Test if fingerprint distances are calculated correctly based on feature distances.

        Parameters
        ----------
        feature_weights : dict of float or None
            Feature weights.
        distance : float
            Fingerprint distance.
        coverage : float
            Fingerprint coverage.
        """

        # FeatureDistances (dummy values)
        feature_distances = generate_feature_distances()

        # FingerprintDistance
        fingerprint_distance = FingerprintDistance()
        fingerprint_distance.from_feature_distances(feature_distances, feature_weights)

        # Test class attributes:

        # Molecule codes
        assert fingerprint_distance.molecule_pair_code == feature_distances.molecule_pair_code

        # Fingerprint distance
        assert np.isclose(fingerprint_distance.distance, distance, rtol=1e-04)

        # Fingerprint coverage
        assert np.isclose(fingerprint_distance.bit_coverage, coverage, rtol=1e-04)

    @pytest.mark.parametrize('feature_weights', [
        {'a': 0},
        'bla'
    ])
    def test_format_weights_typeerror(self, feature_weights):
        """
        Test if wrong data type of input feature weights raises TypeError.
        """

        with pytest.raises(TypeError):
            fingerprint_distance = FingerprintDistance()
            fingerprint_distance._format_weights(feature_weights)

    @pytest.mark.parametrize('feature_weights', [
        [0],
    ])
    def test_format_weights_valueerror(self, feature_weights):
        """
        Test if wrong data type of input feature weights raises TypeError.
        """

        with pytest.raises(ValueError):
            fingerprint_distance = FingerprintDistance()
            fingerprint_distance._format_weights(feature_weights)

    @pytest.mark.parametrize('feature_weights, feature_weights_formatted', [
        (
                None,
                np.array([0.0667] * 15)
        ),
        (
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ),
        (
                [1.0, 0.0, 0.0],
                np.array([0.125] * 8 + [0.0] * 7)
        )

    ])
    def test_format_weights(self, feature_weights, feature_weights_formatted):
        """
        Test if feature weights are added correctly to feature distance DataFrame.

        Parameters
        ----------
        feature_weights : None or list of float
            Feature weights.
        feature_weights_formatted : list of float
            Formatted feature weights of length 15.
        """

        # FingerprintDistance
        fingerprint_distance = FingerprintDistance()
        feature_weights_formatted_calculated = fingerprint_distance._format_weights(feature_weights)

        assert np.isclose(
            np.std(feature_weights_formatted),
            np.std(feature_weights_formatted_calculated),
            rtol=1e-04
        )

    @pytest.mark.parametrize('feature_type_weights', [
        ([0.1]),  # Features missing
        ([0.5, 0.5, 0.5]),  # Weights do not sum up to 1.0
    ])
    def test_format_weight_per_feature_type_valueerror(self, feature_type_weights):
        """
        Test if incorrect input feature type weights raise ValueError.
        """

        with pytest.raises(ValueError):
            fingerprint_distance = FingerprintDistance()
            fingerprint_distance._format_weight_per_feature_type(feature_type_weights)

    @pytest.mark.parametrize('feature_type_weights', [
        ({'a': 1.0}),  # Input is no list
    ])
    def test_format_weight_per_feature_type_typeerror(self, feature_type_weights):
        """
        Test if incorrect input feature type weights raise TypeError.
        """

        with pytest.raises(TypeError):
            fingerprint_distance = FingerprintDistance()
            fingerprint_distance._format_weight_per_feature_type(feature_type_weights)

    @pytest.mark.parametrize('feature_type_weights, feature_weights', [
        (
                [0.0, 1.0, 0.0],
                np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0])
        )
    ])
    def test_format_weight_per_feature_type(self, feature_type_weights, feature_weights):
        """
        Test formatting of weights per feature type (weights need to be equally distributed between all features in feature
        type and transformed into a DataFrame).

        Parameters
        ----------
        feature_type_weights : dict of float (3 items) or None
            Weights per feature type which need to sum up to 1.0.
        feature_weights : dict of float (15 items) or None
            Weights per feature which need to sum up to 1.0.
        """

        # FingerprintDistance
        fingerprint_distance = FingerprintDistance()
        print('hallo')
        feature_weights_calculated = fingerprint_distance._format_weight_per_feature_type(feature_type_weights)
        print(feature_weights_calculated)

        # Test weight values
        assert np.isclose(
            np.std(feature_weights_calculated),
            np.std(feature_weights),
            rtol=1e-04
        )

    @pytest.mark.parametrize('feature_weights', [
        (
                [0.1]
        ),  # Features missing
        (
                [0.5, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.125, 0.125, 0.125, 0.125, 0.0, 0.0, 0.0]
        ),  # Weights do not sum up to 1.0
    ])
    def test_format_weight_per_feature_valueerror(self, feature_weights):
        """
        Test if incorrect input feature weights raise ValueError.
        """

        with pytest.raises(ValueError):
            fingerprint_distance = FingerprintDistance()
            fingerprint_distance._format_weight_per_feature(feature_weights)

    @pytest.mark.parametrize('feature_weights', [
        (
                'is_string'
        )  # Input is no list
    ])
    def test_format_weight_per_feature_typeerror(self, feature_weights):
        """
        Test if incorrect input feature weights raise TypeError.
        """

        with pytest.raises(TypeError):
            fingerprint_distance = FingerprintDistance()
            fingerprint_distance._format_weight_per_feature(feature_weights)

    @pytest.mark.parametrize('feature_weights, feature_weights_formatted', [
        (
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ),
        (
                None,
                [0.0667] * 15
        )
    ])
    def test_format_weight_per_feature(self, feature_weights, feature_weights_formatted):
        """
        Test formatting of weights per feature type (weights need to be transformed into a DataFrame).

        Parameters
        ----------
        feature_weights : dict of float or None (15 items)
            Weights per feature which need to sum up to 1.0.
        feature_weights_formatted : xxx
            Formatted feature weights.
        """

        # FingerprintDistance
        fingerprint_distance = FingerprintDistance()
        feature_weights_formatted_calculated = fingerprint_distance._format_weight_per_feature(feature_weights)

        assert np.isclose(
            np.std(feature_weights_formatted_calculated),
            np.std(feature_weights_formatted),
            rtol=1e-04
        )


class TestsFeatureDistancesGenerator:

    @pytest.mark.parametrize('fingerprints, empty_fingerprints', [
        (
                {'a': Fingerprint(), 'b': None},
                {'a': Fingerprint()}
        ),
        (
                {'a': Fingerprint()},
                {'a': Fingerprint()}
        )
    ])
    def test_remove_empty_fingerprints(self, fingerprints, empty_fingerprints):
        """
        Test removal of empty fingerprints (None) from fingerprints dictionary.

        Parameters
        ----------
        fingerprints : dict of kinsim_structure.encoding.Fingerprint
            Dictionary of fingerprints: Keys are molecule codes and values are fingerprint data.
        empty_fingerprints : dict of kinsim_structure.encoding.Fingerprint
            Dictionary of non-empty fingerprints: Keys are molecule codes and values are fingerprint data.
        """

        generator = FeatureDistancesGenerator()
        empty_fingerprints_calculated = generator._remove_empty_fingerprints(fingerprints)

        assert empty_fingerprints_calculated.keys() == empty_fingerprints.keys()

    @pytest.mark.parametrize('fingerprints, pairs', [
        (
                {'a': Fingerprint(), 'b': Fingerprint(), 'c': Fingerprint()},
                [('a', 'b'), ('a', 'c'), ('b', 'c')]
        )
    ])
    def test_get_fingerprint_pairs(self, fingerprints, pairs):
        """
        Test calculation of all fingerprint pair combinations from fingerprints dictionary.

        Parameters
        ----------
        fingerprints : dict of kinsim_structure.encoding.Fingerprint
            Dictionary of fingerprints: Keys are molecule codes and values are fingerprint data.
        pairs : list of list of str
            List of molecule code pairs (list).
        """

        generator = FeatureDistancesGenerator()
        pairs_calculated = generator._get_fingerprint_pairs(fingerprints)

        for pair_calculated, pair in zip(pairs_calculated, pairs):
            assert pair_calculated == pair

    @pytest.mark.parametrize('path_klifs_metadata, paths_mol2, paths_pdb, chain_ids', [
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
                ]
        )

    ])
    def test_get_feature_distances(self, path_klifs_metadata, paths_mol2, paths_pdb, chain_ids):
        """
        Test if return type is instance of FeatureDistance class.

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

        # Fingerprint dictionary and pair names
        pair = [i.molecule_code for i in fingerprints]
        fingerprints = {i.molecule_code: i for i in fingerprints}

        # Test feature distance calculation
        generator = FeatureDistancesGenerator()
        feature_distances_calculated = generator._get_feature_distances(pair, fingerprints)

        assert isinstance(feature_distances_calculated, FeatureDistances)

    @pytest.mark.parametrize('path_klifs_metadata, paths_mol2, paths_pdb, chain_ids', [
        (
                PATH_TEST_DATA / 'klifs_metadata.csv',
                [
                    PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ABL1/2g2i_chainA/pocket.mol2',
                    PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/pocket.mol2',
                    PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/pocket.mol2'
                ],
                [
                    PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ABL1/2g2i_chainA/protein_pymol.pdb',
                    PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/protein_pymol.pdb',
                    PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/protein_pymol.pdb'
                ],
                [
                    'A',
                    'B',
                    'B'
                ]
        )

    ])
    def test_get_feature_distances_from_list(self, path_klifs_metadata, paths_mol2, paths_pdb, chain_ids):
        """
        Test if return type is instance of list of FeatureDistance class.

        Parameters
        ----------
        path_klifs_metadata : pathlib.Path
            Path to unfiltered KLIFS metadata.
        paths_mol2 : list of str
            Paths to multiple mol2 files.
        paths_pdb : list of str
            Paths to multiple cif files.
        chain_ids : list of str
            Multiple chain IDs.
        """

        # Fingerprints
        fingerprints = generate_fingerprints_from_files(path_klifs_metadata, paths_mol2, paths_pdb, chain_ids)

        # Fingerprint dictionary and pair names
        fingerprints = {i.molecule_code: i for i in fingerprints}

        # Test bulk feature distance calculation
        generator = FeatureDistancesGenerator()

        feature_distances_list = generator._get_feature_distances_from_list(
            generator._get_feature_distances, fingerprints
        )

        assert isinstance(feature_distances_list, list)

        for i in feature_distances_list:
            assert isinstance(i, FeatureDistances)

    @pytest.mark.parametrize(
        'path_klifs_metadata, paths_mol2, paths_pdb, chain_ids, distance_measure, feature_weights, molecule_codes, kinase_names',
        [
            (
                    PATH_TEST_DATA / 'klifs_metadata.csv',
                    [
                        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ABL1/2g2i_chainA/pocket.mol2',
                        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/pocket.mol2',
                        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/pocket.mol2'
                    ],
                    [
                        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ABL1/2g2i_chainA/protein_pymol.pdb',
                        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/protein_pymol.pdb',
                        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/protein_pymol.pdb'
                    ],
                    [
                        'A',
                        'B',
                        'B'
                    ],
                    'scaled_euclidean',
                    None,
                    ['HUMAN/ABL1_2g2i_chainA', 'HUMAN/AAK1_4wsq_altA_chainB'],
                    ['AAK1', 'ABL1']
            )
        ]
    )
    def test_from_fingerprints(
            self, path_klifs_metadata, paths_mol2, paths_pdb, chain_ids, distance_measure, feature_weights, molecule_codes,
            kinase_names
    ):
        """
        Test FeatureDistancesGenerator class attributes.

        Parameters
        ----------
        path_klifs_metadata : pathlib.Path
            Path to unfiltered KLIFS metadata.
        paths_mol2 : list of str
            Paths to multiple mol2 files.
        paths_pdb : list of str
            Paths to multiple cif files.
        chain_ids : list of str
            Multiple chain IDs.
        distance_measure : str
            Type of distance measure, defaults to Euclidean distance.
        """

        # Fingerprints
        fingerprints = generate_fingerprints_from_files(path_klifs_metadata, paths_mol2, paths_pdb, chain_ids)

        # Fingerprint dictionary and pair names
        fingerprint_generator = FingerprintGenerator()
        fingerprint_generator.data = {i.molecule_code: i for i in fingerprints}

        # Test FeatureDistancesGenerator class attributes
        feature_distances_generator = FeatureDistancesGenerator()
        feature_distances_generator.from_fingerprint_generator(fingerprint_generator)

        # Test attributes
        assert feature_distances_generator.distance_measure == distance_measure
        assert isinstance(feature_distances_generator.data, dict)

        # Test example value from dictionary
        example_key = list(feature_distances_generator.data.keys())[0]
        assert isinstance(feature_distances_generator.data[example_key], FeatureDistances)


class TestsFingerprintDistanceGenerator:

    @pytest.mark.parametrize('path_klifs_metadata, path_mol2s, path_pdbs, chain_ids', [
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
                ]
        )

    ])
    def test_get_fingerprint_distance(self, path_klifs_metadata, path_mol2s, path_pdbs, chain_ids):
        """
        Test if return type is FingerprintDistance class instance.

        Parameters
        ----------
        path_klifs_metadata : pathlib.Path
            Path to unfiltered KLIFS metadata.
        path_mol2s : list of str
            Paths to two mol2 files.
        path_pdbs : list of str
            Paths to two cif files.
        chain_ids : list of str
            Two chain IDs.
        """

        # Fingerprints
        fingerprints = generate_fingerprints_from_files(path_klifs_metadata, path_mol2s, path_pdbs, chain_ids)

        # FeatureDistances
        feature_distances = FeatureDistances()
        feature_distances.from_fingerprints(fingerprints[0], fingerprints[1])

        # FingerprintDistanceGenerator
        fingerprint_distance_generator = FingerprintDistanceGenerator()
        fingerprint_distance_calculated = fingerprint_distance_generator._get_fingerprint_distance(
            feature_distances
        )

        assert isinstance(fingerprint_distance_calculated, FingerprintDistance)

    @pytest.mark.parametrize('path_klifs_metadata, path_mol2s, path_pdbs, chain_ids', [
        (

                PATH_TEST_DATA / 'klifs_metadata.csv',
                [
                    PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ABL1/2g2i_chainA/pocket.mol2',
                    PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/pocket.mol2',
                    PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/pocket.mol2'
                ],
                [
                    PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ABL1/2g2i_chainA/protein_pymol.pdb',
                    PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/protein_pymol.pdb',
                    PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/protein_pymol.pdb'
                ],
                [
                    'A',
                    'B',
                    'B'
                ]
        )

    ])
    def test_get_fingerprint_distance_from_list(self, path_klifs_metadata, path_mol2s, path_pdbs, chain_ids):
        """
        Test if return type is instance of list of FeatureDistance class instances.

        Parameters
        ----------
        path_klifs_metadata : pathlib.Path
            Path to unfiltered KLIFS metadata.
        path_mol2s : list of str
            Paths to multiple mol2 files.
        path_pdbs : list of str
            Paths to multiple cif files.
        chain_ids : list of str
            Multiple chain IDs.
        """

        # Fingerprints
        fingerprints = generate_fingerprints_from_files(path_klifs_metadata, path_mol2s, path_pdbs, chain_ids)

        # FingerprintGenerator
        fingerprint_generator = FingerprintGenerator()
        fingerprint_generator.data = {i.molecule_code: i for i in fingerprints}

        # FeatureDistancesGenerator
        feature_distances_generator = FeatureDistancesGenerator()
        feature_distances_generator.from_fingerprint_generator(fingerprint_generator)
        feature_distances_list = list(feature_distances_generator.data.values())

        # FingerprintDistanceGenerator
        fingerprint_distance_generator = FingerprintDistanceGenerator()
        fingerprint_distance_list = fingerprint_distance_generator._get_fingerprint_distance_from_list(
            fingerprint_distance_generator._get_fingerprint_distance, feature_distances_list
        )

        assert isinstance(fingerprint_distance_list, list)

        for i in fingerprint_distance_list:
            assert isinstance(i, FingerprintDistance)

    @pytest.mark.parametrize(
        'path_klifs_metadata, path_mol2s, path_pdbs, chain_ids, distance_measure, feature_weights, molecule_codes, kinase_names',
        [
            (
                    PATH_TEST_DATA / 'klifs_metadata.csv',
                    [
                        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ABL1/2g2i_chainA/pocket.mol2',
                        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/pocket.mol2',
                        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/pocket.mol2'
                    ],
                    [
                        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ABL1/2g2i_chainA/protein_pymol.pdb',
                        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/protein_pymol.pdb',
                        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/protein_pymol.pdb'
                    ],
                    [
                        'A',
                        'B',
                        'B'
                    ],
                    'scaled_euclidean',
                    None,
                    ['HUMAN/AAK1_4wsq_altA_chainB', 'HUMAN/ABL1_2g2i_chainA'],
                    ['AAK1', 'ABL1']
            )
        ]
    )
    def test_from_feature_distances_generator(
            self, path_klifs_metadata, path_mol2s, path_pdbs, chain_ids, distance_measure, feature_weights, molecule_codes,
            kinase_names
    ):
        """
        Test FingerprintDistanceGenerator class attributes.

        Parameters
        ----------
        path_klifs_metadata : pathlib.Path
            Path to unfiltered KLIFS metadata.
        path_mol2s : list of str
            Paths to multiple mol2 files.
        path_pdbs : list of str
            Paths to multiple cif files.
        chain_ids : list of str
            Multiple chain IDs.
        distance_measure : str
            Type of distance measure, defaults to Euclidean distance.
        feature_weights : dict of float or None
            Feature weights of the following form:
            (i) None
                Default feature weights: All features equally distributed to 1/15 (15 feature in total).
            (ii) By feature type
                Feature types to be set are: physicochemical, distances, and moments.
            (iii) By feature:
                Features to be set are: size, hbd, hba, charge, aromatic, aliphatic, sco, exposure,
                distance_to_centroid, distance_to_hinge_region, distance_to_dfg_region, distance_to_front_pocket,
                moment1, moment2, and moment3.
            For (ii) and (iii): All floats must sum up to 1.0.
        molecule_codes : list of str
            List of molecule codes associated with input fingerprints.
        kinase_names : list of str
            List of kinase names associated with input fingerprints.
        """

        # Fingerprints
        fingerprints = generate_fingerprints_from_files(path_klifs_metadata, path_mol2s, path_pdbs, chain_ids)

        # FingerprintGenerator
        fingerprint_generator = FingerprintGenerator()
        fingerprint_generator.data = {i.molecule_code: i for i in fingerprints}

        # FeatureDistancesGenerator
        feature_distances_generator = FeatureDistancesGenerator()
        feature_distances_generator.from_fingerprint_generator(fingerprint_generator)

        # FingerprintDistanceGenerator
        fingerprint_distance_generator = FingerprintDistanceGenerator()
        fingerprint_distance_generator.from_feature_distances_generator(feature_distances_generator)

        # Test attributes
        assert fingerprint_distance_generator.distance_measure == distance_measure
        assert fingerprint_distance_generator.feature_weights == feature_weights
        assert fingerprint_distance_generator.molecule_codes == molecule_codes
        assert fingerprint_distance_generator.kinase_names == kinase_names

        assert isinstance(fingerprint_distance_generator.data, pd.DataFrame)
        assert list(
            fingerprint_distance_generator.data.columns) == 'molecule_code_1 molecule_code_2 distance coverage'.split()

    @pytest.mark.parametrize('molecule_codes, data, fill, structure_distance_matrix', [
        (
                'a b c'.split(),
                pd.DataFrame(
                    [['a', 'b', 0.5, 1.0], ['a', 'c', 0.75, 1.0], ['b', 'c', 1.0, 1.0]],
                    columns='molecule_code_1 molecule_code_2 distance coverage'.split()
                ),
                False,
                pd.DataFrame(
                    [[0.0, 0.5, 0.75], [np.nan, 0.0, 1.0], [np.nan, np.nan, 0.0]],
                    columns='a b c'.split(),
                    index='a b c'.split()
                )
        ),
        (
                'a b c'.split(),
                pd.DataFrame(
                    [['a', 'b', 0.5, 1.0], ['a', 'c', 0.75, 1.0], ['b', 'c', 1.0, 1.0]],
                    columns='molecule_code_1 molecule_code_2 distance coverage'.split()
                ),
                True,
                pd.DataFrame(
                    [[0.0, 0.5, 0.75], [0.5, 0.0, 1.0], [0.75, 1.0, 0.0]],
                    columns='a b c'.split(),
                    index='a b c'.split()
                )
        )
    ])
    def test_get_structure_distance_matrix(self, molecule_codes, data, fill, structure_distance_matrix):
        # Set dummy FingerprintDistanceGenerator class attributes
        fingerprint_distance_generator = FingerprintDistanceGenerator()
        fingerprint_distance_generator.molecule_codes = molecule_codes
        fingerprint_distance_generator.data = data

        # Test generation of structure distance matrix
        structure_distance_matrix_calculated = fingerprint_distance_generator.get_structure_distance_matrix(fill)

        assert structure_distance_matrix_calculated.equals(structure_distance_matrix)

    @pytest.mark.parametrize('molecule_codes, kinase_names, data, by, fill, structure_distance_matrix', [
        (
                'HUMAN/kinase1_pdb1 HUMAN/kinase1_pdb2 HUMAN/kinase2_pdb1'.split(),
                'kinase1 kinase2'.split(),
                pd.DataFrame(
                    [
                        ['HUMAN/kinase1_pdb1', 'HUMAN/kinase1_pdb2', 0.5, 1.0],
                        ['HUMAN/kinase1_pdb1', 'HUMAN/kinase2_pdb1', 0.75, 1.0],
                        ['HUMAN/kinase1_pdb2', 'HUMAN/kinase2_pdb1', 1.0, 1.0]
                    ],
                    columns='molecule_code_1 molecule_code_2 distance coverage'.split()
                ),
                'minimum',
                False,
                pd.DataFrame(
                    [[0.5, 0.75], [np.nan, 0.0]],
                    columns='kinase1 kinase2'.split(),
                    index='kinase1 kinase2'.split()
                )
        ),  # Minimum
        (
                'HUMAN/kinase1_pdb1 HUMAN/kinase1_pdb2 HUMAN/kinase2_pdb1'.split(),
                'kinase1 kinase2'.split(),
                pd.DataFrame(
                    [
                        ['HUMAN/kinase1_pdb1', 'HUMAN/kinase1_pdb2', 0.5, 1.0],
                        ['HUMAN/kinase1_pdb1', 'HUMAN/kinase2_pdb1', 0.75, 1.0],
                        ['HUMAN/kinase1_pdb2', 'HUMAN/kinase2_pdb1', 1.0, 1.0]
                    ],
                    columns='molecule_code_1 molecule_code_2 distance coverage'.split()
                ),
                'minimum',
                True,
                pd.DataFrame(
                    [[0.5, 0.75], [0.75, 0.0]],
                    columns='kinase1 kinase2'.split(),
                    index='kinase1 kinase2'.split()
                )
        ),  # Fill=True
        (
                'HUMAN/kinase1_pdb1 HUMAN/kinase1_pdb2 HUMAN/kinase2_pdb1'.split(),
                'kinase1 kinase2'.split(),
                pd.DataFrame(
                    [
                        ['HUMAN/kinase1_pdb1', 'HUMAN/kinase1_pdb2', 0.5, 1.0],
                        ['HUMAN/kinase1_pdb1', 'HUMAN/kinase2_pdb1', 0.75, 1.0],
                        ['HUMAN/kinase1_pdb2', 'HUMAN/kinase2_pdb1', 1.0, 1.0]
                    ],
                    columns='molecule_code_1 molecule_code_2 distance coverage'.split()
                ),
                'maximum',
                False,
                pd.DataFrame(
                    [[0.5, 1.0], [np.nan, 0.0]],
                    columns='kinase1 kinase2'.split(),
                    index='kinase1 kinase2'.split()
                )
        ),  # Maximum
        (
                'HUMAN/kinase1_pdb1 HUMAN/kinase1_pdb2 HUMAN/kinase2_pdb1'.split(),
                'kinase1 kinase2'.split(),
                pd.DataFrame(
                    [
                        ['HUMAN/kinase1_pdb1', 'HUMAN/kinase1_pdb2', 0.5, 1.0],
                        ['HUMAN/kinase1_pdb1', 'HUMAN/kinase2_pdb1', 0.75, 1.0],
                        ['HUMAN/kinase1_pdb2', 'HUMAN/kinase2_pdb1', 1.0, 1.0]
                    ],
                    columns='molecule_code_1 molecule_code_2 distance coverage'.split()
                ),
                'mean',
                False,
                pd.DataFrame(
                    [[0.5, 0.875], [np.nan, 0.0]],
                    columns='kinase1 kinase2'.split(),
                    index='kinase1 kinase2'.split()
                )
        ),  # Minimum
    ])
    def test_get_kinase_distance_matrix(self, molecule_codes, kinase_names, data, by, fill, structure_distance_matrix):
        # Set dummy FingerprintDistanceGenerator class attributes
        fingerprint_distance_generator = FingerprintDistanceGenerator()
        fingerprint_distance_generator.molecule_codes = molecule_codes
        fingerprint_distance_generator.kinase_names = kinase_names
        fingerprint_distance_generator.data = data

        # Test generation of structure distance matrix
        structure_distance_matrix_calculated = fingerprint_distance_generator.get_kinase_distance_matrix(
            by,
            fill
        )

        assert structure_distance_matrix_calculated.equals(structure_distance_matrix)
