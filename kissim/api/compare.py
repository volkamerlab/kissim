"""
kissim.api.compare

Main API for kissim comparison.
"""

from pathlib import Path
import logging

from kissim.comparison import FeatureDistancesGenerator, FingerprintDistanceGenerator
from kissim.comparison import tree

logger = logging.getLogger(__name__)


def compare(
    fingerprint_generator,
    output_path=None,
    feature_weights=None,
    n_cores=1,
):
    """
    Compare fingerprints (pairwise).

    Parameters
    ----------
    fingerprint_generator : kissim.encoding.FingerprintGenerator
        Fingerprints for KLIFS dataset.
    output_path : str
        Path to output folder.
    feature_weights : None or list of float
        Feature weights of the following form:
        (i) None
            Default feature weights: All features equally distributed to 1/15
            (15 features in total).
        (ii) By feature (list of 15 floats):
            Features to be set in the following order: size, hbd, hba, charge, aromatic,
            aliphatic, sco, exposure, distance_to_centroid, distance_to_hinge_region,
            distance_to_dfg_region, distance_to_front_pocket, moment1, moment2, and moment3.
            All floats must sum up to 1.0.
    n_cores : int
        Number of cores used to generate fingerprint distances.

    Returns
    -------
    feature_distances_generator : kissim.comparison.FeatureDistancesGenerator
        Feature distances for all pairwise fingerprints.
    fingerprint_distance_generator : kissim.comparison.FingerprintDistanceGenerator
        Fingerprint distance for all pairwise fingeprints.
    """

    if output_path is not None:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        feature_distances_filepath = output_path / "feature_distances.csv"
        fingerprint_distance_filepath = output_path / f"fingerprint_distances.csv"
    else:
        feature_distances_filepath = None
        fingerprint_distance_filepath = None

    # Generate feature distances
    feature_distances_generator = compare_fingerprint_features(
        fingerprint_generator, feature_distances_filepath, n_cores
    )

    # Generate fingerprint distance
    fingerprint_distance_generator = weight_feature_distances(
        feature_distances_generator, fingerprint_distance_filepath, feature_weights
    )

    return feature_distances_generator, fingerprint_distance_generator


def compare_fingerprint_features(
    fingerprint_generator,
    output_filepath=None,
    n_cores=1,
):
    """
    Compare fingerprints w.r.t. to their features: Generates per fingerprint pair a distance
    vector, which length equals the number of fingerprint features.

    Parameters
    ----------
    fingerprint_generator : kissim.encoding.FingerprintGenerator
        Fingerprints.
    output_path : str
        Path to output json file containing the feature distances for all fingerprint pairs.
    n_cores : int
        Number of cores used to generate fingerprint distances.
    """

    feature_distances_generator = FeatureDistancesGenerator.from_fingerprint_generator(
        fingerprint_generator, n_cores
    )

    if output_filepath:
        output_filepath = Path(output_filepath)
        feature_distances_generator.to_csv(output_filepath)

    return feature_distances_generator


def weight_feature_distances(
    feature_distances_generator, output_filepath=None, feature_weights=None
):
    """
    Weight feature distances: Generates per fingerprint pair a fingerprint distance.

    Parameters
    ----------
    feature_distances_generator : kissim.encoding.FeatureDistancesGenerator
        Feature distances.
    output_path : str
        Path to output file.
    feature_weights : None or list of float
        Feature weights of the following form:
        (i) None
            Default feature weights: All features equally distributed to 1/15
            (15 features in total).
        (ii) By feature (list of 15 floats):
            Features to be set in the following order: size, hbd, hba, charge, aromatic,
            aliphatic, sco, exposure, distance_to_centroid, distance_to_hinge_region,
            distance_to_dfg_region, distance_to_front_pocket, moment1, moment2, and moment3.
            All floats must sum up to 1.0.
    """

    fingerprint_distance_generator = FingerprintDistanceGenerator.from_feature_distances_generator(
        feature_distances_generator, feature_weights
    )

    if output_filepath:
        output_filepath = Path(output_filepath)

        # Write fingerprint distances to file
        fingerprint_distances_filepath = output_filepath
        fingerprint_distance_generator.to_csv(fingerprint_distances_filepath)

        # Write default kinase distances to file
        kinase_distances_filepath = (
            output_filepath.parent / f"{output_filepath.stem}_to_kinase_matrix.csv"
        )
        kinase_matrix = fingerprint_distance_generator.kinase_distance_matrix()
        kinase_matrix.index.name = None
        kinase_matrix.columns.name = None
        kinase_matrix.to_csv(kinase_distances_filepath, index=True)

        # Write default tree to file
        tree_filepath = output_filepath.parent / f"{output_filepath.stem}_to_kinase_clusters.tree"
        annotation_filepath = output_filepath.parent / "kinase_annotation.csv"
        tree.from_distance_matrix(kinase_matrix, tree_filepath, annotation_filepath)

    return fingerprint_distance_generator
