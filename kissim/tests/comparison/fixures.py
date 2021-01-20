"""
Unit and regression tests for kissim.similarity class methods.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kissim.encoding import Fingerprint, FingerprintGenerator
from kissim.comparison import (
    FeatureDistances,
    FingerprintDistance,
    FeatureDistancesGenerator,
    FingerprintDistanceGenerator,
)

PATH_TEST_DATA = Path(__name__).parent / "kissim" / "tests" / "data"


@pytest.fixture(scope="package")
def fingerprint_generator():
    """
    Get FingerprintGenerator instance with dummy data, i.e. multiple fingerprints
    (encoded pockets).

    Returns
    -------
    kissim.encoding.FingerprintGenerator
        Fingerprints.
    """

    # Set data paths
    path_klifs_metadata = PATH_TEST_DATA / "klifs_metadata.csv"
    paths_mol2 = [
        PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ABL1/2g2i_chainA/pocket.mol2",
        PATH_TEST_DATA / "KLIFS_download" / "HUMAN/AAK1/4wsq_altA_chainB/pocket.mol2",
        PATH_TEST_DATA / "KLIFS_download" / "HUMAN/AAK1/4wsq_altA_chainB/pocket.mol2",
    ]
    paths_pdb = [
        PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ABL1/2g2i_chainA/protein_pymol.pdb",
        PATH_TEST_DATA / "KLIFS_download" / "HUMAN/AAK1/4wsq_altA_chainB/protein_pymol.pdb",
        PATH_TEST_DATA / "KLIFS_download" / "HUMAN/AAK1/4wsq_altA_chainB/protein_pymol.pdb",
    ]
    chain_ids = ["A", "B", "B"]

    # Generate fingerprints
    fingerprints = []

    for path_mol2, path_pdb, chain_id in zip(paths_mol2, paths_pdb, chain_ids):

        fingerprint = Fingerprint.from_structure_klifs_id(paths_mol2)
        fingerprints.append(fingerprint)

    # FingerprintGenerator (set class attribute manually)
    fingerprint_generator = FingerprintGenerator()
    fingerprint_generator.data = {i.molecule_code: i for i in fingerprints}

    return fingerprint_generator


@pytest.fixture(scope="module")
def feature_distances():
    """
    Get FeatureDistances instance with dummy data, i.e. distances and bit coverages between two
    fingerprints for each of their features.

    Returns
    -------
    kissim.similarity.FeatureDistances
        Distances and bit coverages between two fingerprints for each of their features.
    """

    molecule_pair_code = ["molecule1", "molecule2"]
    distances = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    bit_coverages = np.array(
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )

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
    kissim.similarity.FeatureDistancesGenerator
        Feature distances for multiple fingerprint pairs.
    """

    # FeatureDistances
    feature_distances1 = FeatureDistances()
    feature_distances1.molecule_pair_code = ("HUMAN/kinase1_pdb1", "HUMAN/kinase1_pdb2")
    feature_distances1.distances = [1.0] * 15
    feature_distances1.bit_coverages = [1.0] * 15

    feature_distances2 = FeatureDistances()
    feature_distances2.molecule_pair_code = ("HUMAN/kinase1_pdb1", "HUMAN/kinase2_pdb1")
    feature_distances2.distances = [0.0] * 15
    feature_distances2.bit_coverages = [1.0] * 15

    feature_distances3 = FeatureDistances()
    feature_distances3.molecule_pair_code = ("HUMAN/kinase1_pdb2", "HUMAN/kinase2_pdb1")
    feature_distances3.distances = [0.0] * 15
    feature_distances3.bit_coverages = [0.0] * 15

    # FeatureDistancesGenerator
    distance_measure = "scaled_euclidean"
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
    kissim.similarity.FingerprintDistanceGenerator
        Fingerprint distance for multiple fingerprint pairs.
    """

    molecule_codes = "HUMAN/kinase1_pdb1 HUMAN/kinase1_pdb2 HUMAN/kinase2_pdb1".split()
    kinase_names = "kinase1 kinase2".split()
    data = pd.DataFrame(
        [
            ["HUMAN/kinase1_pdb1", "HUMAN/kinase1_pdb2", 0.5, 1.0],
            ["HUMAN/kinase1_pdb1", "HUMAN/kinase2_pdb1", 0.75, 1.0],
            ["HUMAN/kinase1_pdb2", "HUMAN/kinase2_pdb1", 1.0, 1.0],
        ],
        columns="molecule_code_1 molecule_code_2 distance coverage".split(),
    )

    # FingerprintDistanceGenerator
    fingerprint_distance_generator = FingerprintDistanceGenerator()
    fingerprint_distance_generator.molecule_codes = molecule_codes
    fingerprint_distance_generator.kinase_names = kinase_names
    fingerprint_distance_generator.data = data

    return fingerprint_distance_generator
