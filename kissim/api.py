"""
kissim.api

Main API for kissim encoding and comparison.
"""

import logging

from opencadd.databases.klifs import setup_remote, setup_local

from kissim.encoding import FingerprintGenerator
from kissim.comparison import FeatureDistancesGenerator, FingerprintDistanceGenerator

logger = logging.getLogger(__name__)


def encode(structure_klifs_ids, json_filepath=None, n_cores=1, local_klifs_session=None):
    """
    Encode structures.

    Parameters
    ----------
    structure_klifs_ids : list of int
        Structure KLIFS IDs.
    json_filepath : str or pathlib.Path
        Path to output json file. Default None.
    n_cores : int
        Number of cores used to generate fingerprints.
    local_klifs_session : str or None
        If path to local KLIFS download is given, set up local KLIFS session.
        If None is given, set up remote KLIFS session.

    Returns
    -------
    kissim.encoding.fingerprint_generator
        Fingerprints.
    """

    # Set up KLIFS session
    klifs_session = _setup_klifs_session(local_klifs_session)

    # Generate fingerprints
    fingerprints = FingerprintGenerator.from_structure_klifs_ids(
        structure_klifs_ids, klifs_session, n_cores
    )

    # Optionally: Save fingerprints to json file
    if json_filepath:
        logger.info(f"Write fingerprints to file: {json_filepath}")
        fingerprints.to_json(json_filepath)

    return fingerprints


def compare(
    fingerprint_generator,
    csv_path=None,
    n_cores=1,
    distance_measure="scaled_euclidean",
    feature_weights="101",
):
    """
    Compare fingerprints (pairwise).

    Parameters
    ----------
    fingerprint_generator : kissim.encoding.FingerprintGenerator
        Fingerprints for KLIFS dataset.
    csv_path : str
        TODO
    n_cores : int
        Number of cores used to generate fingerprint distances.
    distance_measures : str
        Distance measures TODO.
    feature_weights : str
        Feature weighting scheme.
    """

    print(csv_path)

    feature_distances_generator = FeatureDistancesGenerator()
    feature_distances_generator.from_fingerprint_generator(fingerprint_generator, distance_measure)
    # TODO save to file

    fingerprint_distance_generator = FingerprintDistanceGenerator()
    fingerprint_distance_generator.from_feature_distances_generator(
        feature_distances_generator, feature_weights
    )
    # TODO save to file


def _setup_klifs_session(local_klifs_session=None):
    """
    Set up KLIFS session.

    Parameters
    ----------
    local_klifs_session : str or None
        If path to local KLIFS download is given, set up local KLIFS session.
        If None is given, set up remote KLIFS session.

    Returns
    -------
    klifs_session : opencadd.databases.klifs.session.Session
        Local or remote KLIFS session.
    """

    if local_klifs_session:
        klifs_session = setup_local(local_klifs_session)
    else:
        klifs_session = setup_remote()
    return klifs_session
