"""
Fixures to be used in unit testing.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kissim.api import encode
from kissim.comparison import (
    FeatureDistances,
    FingerprintDistance,
    FeatureDistancesGenerator,
    FingerprintDistanceGenerator,
)

PATH_TEST_DATA = Path(__name__).parent / "kissim" / "tests" / "data"


@pytest.fixture(scope="module")
def fingerprint_generator():
    """
    Get FingerprintGenerator instance with dummy data, i.e. multiple fingerprints
    (encoded pockets).

    Returns
    -------
    kissim.encoding.FingerprintGenerator
        Fingerprints.
    """

    # Example structure KLIFS IDs
    structure_klifs_ids = [109, 110, 118]

    # Encode structures
    LOCAL_KLIFS_PATH = PATH_TEST_DATA / "KLIFS_download"
    fingerprint_generator = encode(structure_klifs_ids, local_klifs_session=LOCAL_KLIFS_PATH)

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

    structure_pair_ids = ("molecule1", "molecule2")
    kinase_pair_ids = ("kinase1", "kinase2")
    distances = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    bit_coverages = np.array(
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )

    # FeatureDistances (lf._get_incoming_resp class attributes manually)
    feature_distances = FeatureDistances()
    feature_distances.structure_pair_ids = structure_pair_ids
    feature_distances.kinase_pair_ids = kinase_pair_ids
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
    feature_distances1.structure_pair_ids = ("pdb1", "pdb2")
    feature_distances1.kinase_pair_ids = ("kinase1", "kinase1")
    feature_distances1.distances = np.array([1.0] * 15)
    feature_distances1.bit_coverages = np.array([1.0] * 15)

    feature_distances2 = FeatureDistances()
    feature_distances2.structure_pair_ids = ("pdb1", "pdb3")
    feature_distances2.kinase_pair_ids = ("kinase1", "kinase2")
    feature_distances2.distances = np.array([0.0] * 15)
    feature_distances2.bit_coverages = np.array([1.0] * 15)

    feature_distances3 = FeatureDistances()
    feature_distances3.structure_pair_ids = ("pdb2", "pdb3")
    feature_distances3.kinase_pair_ids = ("kinase1", "kinase2")
    feature_distances3.distances = np.array([0.0] * 15)
    feature_distances3.bit_coverages = np.array([0.0] * 15)

    # FeatureDistancesGenerator
    data = {
        feature_distances1.structure_pair_ids: feature_distances1,
        feature_distances2.structure_pair_ids: feature_distances2,
        feature_distances3.structure_pair_ids: feature_distances3,
    }

    # FeatureDistancesGenerator
    feature_distances_generator = FeatureDistancesGenerator()
    feature_distances_generator.data = data
    feature_distances_generator.structure_kinase_ids = [
        ("pdb1", "kinase1"),
        ("pdb2", "kinase1"),
        ("pdb3", "kinase2"),
    ]

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

    data = pd.DataFrame(
        [
            ["pdb1", "pdb2", "kinase1", "kinase1", 0.75, 1.0],
            ["pdb1", "pdb3", "kinase1", "kinase2", 1.0, 1.0],
            ["pdb2", "pdb3", "kinase1", "kinase2", 0.8, 1.0],
        ],
        columns="structure1 structure2 kinase1 kinase2 distance coverage".split(),
    )

    # FingerprintDistanceGenerator
    fingerprint_distance_generator = FingerprintDistanceGenerator()
    fingerprint_distance_generator.data = data
    fingerprint_distance_generator.structure_kinase_ids = [
        ("pdb1", "kinase1"),
        ("pdb2", "kinase1"),
        ("pdb3", "kinase2"),
    ]

    return fingerprint_distance_generator
