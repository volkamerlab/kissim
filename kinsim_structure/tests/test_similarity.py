"""
Unit and regression tests for kinsim_structure.similarity class methods.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kinsim_structure.auxiliary import KlifsMoleculeLoader, PdbChainLoader
from kinsim_structure.encoding import Fingerprint, FingerprintGenerator
from kinsim_structure.similarity import FeatureDistances, FingerprintDistance, \
    FeatureDistancesGenerator, FingerprintDistanceGenerator

PATH_TEST_DATA = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data'


@pytest.fixture(scope="module")
def fingerprint_generator():
    """
    Get FingerprintGenerator instance with dummy data, i.e. multiple fingerprints (encoded pockets).

    Returns
    -------
    kinsim_structure.encoding.FingerprintGenerator
        Fingerprints.
    """

    # Set data paths
    path_klifs_metadata = PATH_TEST_DATA / 'klifs_metadata.csv'
    paths_mol2 = [
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ABL1/2g2i_chainA/pocket.mol2',
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/pocket.mol2',
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/pocket.mol2'
    ]
    paths_pdb = [
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ABL1/2g2i_chainA/protein_pymol.pdb',
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/protein_pymol.pdb',
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/protein_pymol.pdb'
    ]
    chain_ids = [
        'A',
        'B',
        'B'
    ]

    # Generate fingerprints
    fingerprints = []

    for path_mol2, path_pdb, chain_id in zip(paths_mol2, paths_pdb, chain_ids):

        klifs_molecule_loader = KlifsMoleculeLoader()
        klifs_molecule_loader.from_file(path_mol2, path_klifs_metadata)
        pdb_chain_loader = PdbChainLoader()
        pdb_chain_loader.from_file(path_pdb, chain_id)

        fingerprint = Fingerprint()
        fingerprint.from_molecule(klifs_molecule_loader.molecule, pdb_chain_loader.chain)

        fingerprints.append(fingerprint)

    # FingerprintGenerator (set class attribute manually)
    fingerprint_generator = FingerprintGenerator()
    fingerprint_generator.data = {i.molecule_code: i for i in fingerprints}

    return fingerprint_generator


@pytest.fixture(scope="module")
def feature_distances():
    """
    Get FeatureDistances instance with dummy data, i.e. distances and bit coverages between two fingerprints for each
    of their features.

    Returns
    -------
    kinsim_structure.similarity.FeatureDistances
        Distances and bit coverages between two fingerprints for each of their features.
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


@pytest.fixture(scope="module")
def feature_distances_generator():
    """
    Get FeatureDistancesGenerator instance with dummy data.

    Returns
    -------
    kinsim_structure.similarity.FeatureDistancesGenerator
        Feature distances for multiple fingerprint pairs.
    """

    # FeatureDistances
    feature_distances1 = FeatureDistances()
    feature_distances1.molecule_pair_code = ('HUMAN/kinase1_pdb1', 'HUMAN/kinase1_pdb2')
    feature_distances1.distances = [1.0] * 15
    feature_distances1.bit_coverages = [1.0] * 15

    feature_distances2 = FeatureDistances()
    feature_distances2.molecule_pair_code = ('HUMAN/kinase1_pdb1', 'HUMAN/kinase2_pdb1')
    feature_distances2.distances = [0.0] * 15
    feature_distances2.bit_coverages = [1.0] * 15

    feature_distances3 = FeatureDistances()
    feature_distances3.molecule_pair_code = ('HUMAN/kinase1_pdb2', 'HUMAN/kinase2_pdb1')
    feature_distances3.distances = [0.0] * 15
    feature_distances3.bit_coverages = [0.0] * 15

    # FeatureDistancesGenerator
    distance_measure = 'scaled_euclidean'
    data = {
        feature_distances1.molecule_pair_code: feature_distances1,
        feature_distances2.molecule_pair_code: feature_distances2,
        feature_distances3.molecule_pair_code: feature_distances3,
    }

    # FeatureDistancesGenerator
    feature_distances_generator = FeatureDistancesGenerator()
    feature_distances_generator.distance_measure = distance_measure
    feature_distances_generator.data = data

    return feature_distances_generator


@pytest.fixture(scope="module")
def fingerprint_distance_generator():
    """
    Get FingerprintDistanceGenerator instance with dummy data.

    Returns
    -------
    kinsim_structure.similarity.FingerprintDistanceGenerator
        Fingerprint distance for multiple fingerprint pairs.
    """

    molecule_codes = 'HUMAN/kinase1_pdb1 HUMAN/kinase1_pdb2 HUMAN/kinase2_pdb1'.split()
    kinase_names = 'kinase1 kinase2'.split()
    data = pd.DataFrame(
        [
            ['HUMAN/kinase1_pdb1', 'HUMAN/kinase1_pdb2', 0.5, 1.0],
            ['HUMAN/kinase1_pdb1', 'HUMAN/kinase2_pdb1', 0.75, 1.0],
            ['HUMAN/kinase1_pdb2', 'HUMAN/kinase2_pdb1', 1.0, 1.0]
        ],
        columns='molecule_code_1 molecule_code_2 distance coverage'.split()
    )

    # FingerprintDistanceGenerator
    fingerprint_distance_generator = FingerprintDistanceGenerator()
    fingerprint_distance_generator.molecule_codes = molecule_codes
    fingerprint_distance_generator.kinase_names = kinase_names
    fingerprint_distance_generator.data = data

    return fingerprint_distance_generator


class TestsFeatureDistances:
    """
    Test FeatureDistances class methods
    """

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
            Manhattan distance between two value lists.
        """

        feature_distances_generator = FeatureDistances()
        score_calculated = feature_distances_generator._scaled_cityblock_distance(values1, values2)

        assert np.isclose(score_calculated, distance, rtol=1e-04)

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
        Test ValueError exceptions in feature distance calculation.

        Parameters
        ----------
        feature1 : np.ndarray
            Feature bits for a given feature in fingerprint 1.
        feature2 : np.ndarray
            Feature bits for a given feature in fingerprint 2.
        """

        feature_distances = FeatureDistances()

        with pytest.raises(ValueError):
            feature_distances.from_features(feature1, feature2)

    def test_from_fingerprints(self, fingerprint_generator):
        """
        Test data type and dimensions of feature distances between two fingerprints.

        Parameters
        ----------
        fingerprint_generator : FingerprintGenerator
            Multiple fingerprints.
        """

        # Fingerprints
        fingerprints = list(fingerprint_generator.data.values())

        # Get feature distances
        feature_distances = FeatureDistances()
        feature_distances.from_fingerprints(
            fingerprint1=fingerprints[0],
            fingerprint2=fingerprints[1],
            distance_measure='scaled_euclidean'
        )

        # Class attribute types and dimensions correct?
        assert isinstance(feature_distances.molecule_pair_code, tuple)
        assert len(feature_distances.molecule_pair_code) == 2

        assert isinstance(feature_distances.distances, np.ndarray)
        assert len(feature_distances.distances) == 15

        assert isinstance(feature_distances.bit_coverages, np.ndarray)
        assert len(feature_distances.bit_coverages) == 15

        # Class property type and dimension correct?
        assert isinstance(feature_distances.data, pd.DataFrame)

        feature_type_dimension_calculated = feature_distances.data.groupby(by='feature_type', sort=False).size()
        feature_type_dimension = pd.Series([8, 4, 3], index='physicochemical distances moments'.split())
        assert all(feature_type_dimension_calculated == feature_type_dimension)


class TestsFingerprintDistance:
    """
    Test FingerprintDistance class methods.
    """

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
        feature_weights_calculated = fingerprint_distance._format_weight_per_feature_type(feature_type_weights)

        # Test weight values
        assert np.isclose(
            np.std(feature_weights_calculated),
            np.std(feature_weights),
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

    @pytest.mark.parametrize('feature_weights, distance, coverage', [
        (
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25],
            0.5,
            0.75
        )
    ])
    def test_from_feature_distances(self, feature_distances, feature_weights, distance, coverage):
        """
        Test if fingerprint distances are calculated correctly based on feature distances.

        Parameters
        ----------
        feature_distances : kinsim_structure.similarity.FeatureDistances
            Distances and bit coverages between two fingerprints for each of their features.
        feature_weights : dict of float or None
            Feature weights.
        distance : float
            Fingerprint distance.
        coverage : float
            Fingerprint coverage.
        """

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


class TestsFeatureDistancesGenerator:
    """
    Test FeatureDistancesGenerator class methods.
    """

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

    def test_get_feature_distances(self, fingerprint_generator):
        """
        Test if return type is instance of FeatureDistance class.

        Parameters
        ----------
        fingerprint_generator : FingerprintGenerator
            Multiple fingerprints.
        """

        # Get fingerprint pair from FingerprintGenerator
        pair = list(fingerprint_generator.data.keys())[:2]
        fingerprints = fingerprint_generator.data

        # Test feature distance calculation
        feature_distances_generator = FeatureDistancesGenerator()
        feature_distances_calculated = feature_distances_generator._get_feature_distances(pair, fingerprints)

        assert isinstance(feature_distances_calculated, FeatureDistances)

    def test_get_feature_distances_from_list(self, fingerprint_generator):
        """
        Test if return type is instance of list of FeatureDistance class.

        Parameters
        ----------
        fingerprint_generator : FingerprintGenerator
            Multiple fingerprints.
        """

        # Test bulk feature distance calculation
        generator = FeatureDistancesGenerator()

        feature_distances_list = generator._get_feature_distances_from_list(
            generator._get_feature_distances,
            fingerprint_generator.data
        )

        assert isinstance(feature_distances_list, list)

        for i in feature_distances_list:
            assert isinstance(i, FeatureDistances)

    @pytest.mark.parametrize('distance_measure, feature_weights, molecule_codes, kinase_names', [
            (
                'scaled_euclidean',
                None,
                ['HUMAN/ABL1_2g2i_chainA', 'HUMAN/AAK1_4wsq_altA_chainB'],
                ['AAK1', 'ABL1']
            )
        ]
    )
    def test_from_fingerprints(self, fingerprint_generator, distance_measure, feature_weights, molecule_codes, kinase_names):
        """
        Test FeatureDistancesGenerator class attributes.

        Parameters
        ----------
        distance_measure : str
            Type of distance measure, defaults to Euclidean distance.
        """

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
    """
    Test FingerprintDistanceGenerator class methods.
    """

    def test_get_fingerprint_distance(self, feature_distances):
        """
        Test if return type is FingerprintDistance class instance.

        Parameters
        ----------
        feature_distances : kinsim_structure.similarity.FeatureDistances
            Distances and bit coverages between two fingerprints for each of their features.
        """

        fingerprint_distance_generator = FingerprintDistanceGenerator()
        fingerprint_distance_calculated = fingerprint_distance_generator._get_fingerprint_distance(
            feature_distances
        )

        assert isinstance(fingerprint_distance_calculated, FingerprintDistance)

    def test_get_fingerprint_distance_from_list(self, feature_distances_generator):
        """
        Test if return type is instance of list of FingerprintDistance class instances.

        Parameters
        ----------
        feature_distances_generator : FeatureDistancesGenerator
            Feature distances for multiple fingerprints.
        """

        fingerprint_distance_generator = FingerprintDistanceGenerator()
        fingerprint_distance_list = fingerprint_distance_generator._get_fingerprint_distance_from_list(
            fingerprint_distance_generator._get_fingerprint_distance,
            list(feature_distances_generator.data.values())
        )

        assert isinstance(fingerprint_distance_list, list)

        for i in fingerprint_distance_list:
            assert isinstance(i, FingerprintDistance)

    @pytest.mark.parametrize('distance_measure, feature_weights, molecule_codes, kinase_names', [
            (
                'scaled_euclidean',
                None,
                'HUMAN/kinase1_pdb1 HUMAN/kinase1_pdb2 HUMAN/kinase2_pdb1'.split(),
                'kinase1 kinase2'.split()
            )
        ]
    )
    def test_from_feature_distances_generator(
        self,
        feature_distances_generator,
        distance_measure,
        feature_weights,
        molecule_codes,
        kinase_names
    ):
        """
        Test FingerprintDistanceGenerator class attributes.

        Parameters
        ----------
        feature_distances_generator : FeatureDistancesGenerator
            Feature distances for multiple fingerprints.
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

        # FingerprintDistanceGenerator
        fingerprint_distance_generator = FingerprintDistanceGenerator()
        print(feature_distances_generator.data)
        fingerprint_distance_generator.from_feature_distances_generator(feature_distances_generator)

        # Test attributes
        assert fingerprint_distance_generator.distance_measure == distance_measure
        assert fingerprint_distance_generator.feature_weights == feature_weights
        assert fingerprint_distance_generator.molecule_codes == molecule_codes
        assert fingerprint_distance_generator.kinase_names == kinase_names

        assert isinstance(fingerprint_distance_generator.data, pd.DataFrame)

        data_columns = 'molecule_code_1 molecule_code_2 distance coverage'.split()
        assert list(fingerprint_distance_generator.data.columns) == data_columns

    @pytest.mark.parametrize('fill, structure_distance_matrix', [
        (
            False,
            pd.DataFrame(
                [[0.0, 0.5, 0.75], [np.nan, 0.0, 1.0], [np.nan, np.nan, 0.0]],
                columns='HUMAN/kinase1_pdb1 HUMAN/kinase1_pdb2 HUMAN/kinase2_pdb1'.split(),
                index='HUMAN/kinase1_pdb1 HUMAN/kinase1_pdb2 HUMAN/kinase2_pdb1'.split()
            )
        ),
        (
            True,
            pd.DataFrame(
                [[0.0, 0.5, 0.75], [0.5, 0.0, 1.0], [0.75, 1.0, 0.0]],
                columns='HUMAN/kinase1_pdb1 HUMAN/kinase1_pdb2 HUMAN/kinase2_pdb1'.split(),
                index='HUMAN/kinase1_pdb1 HUMAN/kinase1_pdb2 HUMAN/kinase2_pdb1'.split()
            )
        )
    ])
    def test_get_structure_distance_matrix(self, fingerprint_distance_generator, fill, structure_distance_matrix):
        """
        Test if structure distance matrix is correct.

        Parameters
        ----------
        fingerprint_distance_generator : FingerprintDistanceGenerator
            Fingerprint distance for multiple fingerprint pairs.
        fill
        structure_distance_matrix
        """

        # Test generation of structure distance matrix
        structure_distance_matrix_calculated = fingerprint_distance_generator.get_structure_distance_matrix(fill)

        assert structure_distance_matrix_calculated.equals(structure_distance_matrix)

    @pytest.mark.parametrize('by, fill, structure_distance_matrix', [
        (
            'minimum',
            False,
            pd.DataFrame(
                [[0.5, 0.75], [np.nan, 0.0]],
                columns='kinase1 kinase2'.split(),
                index='kinase1 kinase2'.split()
            )
        ),  # Minimum
        (
            'minimum',
            True,
            pd.DataFrame(
                [[0.5, 0.75], [0.75, 0.0]],
                columns='kinase1 kinase2'.split(),
                index='kinase1 kinase2'.split()
            )
        ),  # Fill=True
        (
            'maximum',
            False,
            pd.DataFrame(
                [[0.5, 1.0], [np.nan, 0.0]],
                columns='kinase1 kinase2'.split(),
                index='kinase1 kinase2'.split()
            )
        ),  # Maximum
        (
            'mean',
            False,
            pd.DataFrame(
                [[0.5, 0.875], [np.nan, 0.0]],
                columns='kinase1 kinase2'.split(),
                index='kinase1 kinase2'.split()
            )
        ),  # Minimum
    ])
    def test_get_kinase_distance_matrix(self, fingerprint_distance_generator, by, fill, structure_distance_matrix):
        """
        Test if kinase distance matrix is correct.

        Parameters
        ----------
        fingerprint_distance_generator : FingerprintDistanceGenerator
            Fingerprint distance for multiple fingerprint pairs.
        by : str
            Condition on which the distance value per kinase pair is extracted from the set of distances values per
            structure pair. Default: Minimum distance value.
        fill : bool
            Fill or fill not (default) lower triangle of distance matrix.
        structure_distance_matrix : pandas.DataFrame
            xxx
        """

        # Test generation of structure distance matrix
        structure_distance_matrix_calculated = fingerprint_distance_generator.get_kinase_distance_matrix(
            by,
            fill
        )

        assert structure_distance_matrix_calculated.equals(structure_distance_matrix)
