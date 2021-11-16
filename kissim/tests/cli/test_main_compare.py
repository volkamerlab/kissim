"""
Unit and regression test for kissim's compare CLI.
"""

from pathlib import Path
import pytest
import subprocess

import pandas as pd

from kissim.comparison import FeatureDistancesGenerator, FingerprintDistanceGenerator

PATH_TEST_DATA = Path(__name__).parent / "kissim" / "tests" / "data"


@pytest.mark.parametrize(
    "args",
    [
        f"kissim compare -i {(PATH_TEST_DATA / 'fingerprints.json').absolute()} "
        f"-o {PATH_TEST_DATA.absolute()}",
        f"kissim compare -i {(PATH_TEST_DATA / 'fingerprints.json').absolute()} "
        f"-o {PATH_TEST_DATA.absolute()} -c 2",
        f"kissim compare -i {(PATH_TEST_DATA / 'fingerprints.json').absolute()} "
        f"-o {PATH_TEST_DATA.absolute()} "
        f"-w 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0",
    ],
)
def test_main_compare(fingerprint_generator, args):
    """
    Test CLI for encoding using subprocesses.
    """

    args = args.split()
    input_filepath = Path(args[3])
    output_path = Path(args[5])

    # Hardcode output files
    feature_distances_filepath = output_path / "feature_distances.csv.bz2"
    fingerprint_distance_filepath = output_path / "fingerprint_distances.csv.bz2"
    fingerprint_distances_to_kinase_clusters_filepath = (
        output_path / "fingerprint_distances_to_kinase_clusters.tree"
    )
    fingerprint_distances_to_kinase_matrix_filepath = (
        output_path / "fingerprint_distances_to_kinase_matrix.csv"
    )
    kinase_annotation_filepath = output_path / "kinase_annotation.csv"

    # Generate
    fingerprint_generator.to_json(input_filepath)

    subprocess.run(args, check=True)

    ### Feature distances generator
    # CSV file there?
    assert feature_distances_filepath.exists()
    # CSV file can be loaded as FeatureDistancesGeneration object?
    feature_distances_generator = FeatureDistancesGenerator.from_csv(feature_distances_filepath)
    assert isinstance(feature_distances_generator, FeatureDistancesGenerator)
    assert isinstance(feature_distances_generator.data, pd.DataFrame)

    ### Fingerprint distance generator
    # CSV file there?
    assert fingerprint_distance_filepath.exists()
    # CSV file can be loaded as FingerprintDistanceGeneration object?
    fingerprint_distance_generator = FingerprintDistanceGenerator.from_csv(
        fingerprint_distance_filepath
    )
    assert isinstance(fingerprint_distance_generator, FingerprintDistanceGenerator)
    assert isinstance(fingerprint_distance_generator.data, pd.DataFrame)

    ### Matrix and tree files
    assert fingerprint_distances_to_kinase_clusters_filepath.exists()
    assert fingerprint_distances_to_kinase_matrix_filepath.exists()
    assert kinase_annotation_filepath.exists()

    # Delete file - cannot be done within enter_tmp_directory, since temporary files
    # apparently cannot be read from CLI
    filepaths = [
        input_filepath,
        feature_distances_filepath,
        fingerprint_distance_filepath,
        fingerprint_distances_to_kinase_clusters_filepath,
        fingerprint_distances_to_kinase_matrix_filepath,
        kinase_annotation_filepath,
    ]
    for filepath in filepaths:
        if filepath.exists():
            filepath.unlink()


@pytest.mark.parametrize(
    "args",
    [
        f"kissim compare -i {(PATH_TEST_DATA / 'fingerprints.json').absolute()}",
        f"kissim compare -i {(PATH_TEST_DATA / 'fingerprints.json').absolute()} "
        f"-o {PATH_TEST_DATA.absolute()} "
        f"-w 1.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0",
        f"kissim compare -i {(PATH_TEST_DATA / 'fingerprints.json').absolute()} "
        f"-o {PATH_TEST_DATA.absolute()} -w 1.0",
    ],
)
def test_compare_error(args):
    """
    Test if input arguments cause error.
    """

    args = args.split()

    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(args, check=True)
