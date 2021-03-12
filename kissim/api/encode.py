"""
kissim.api.encode

Main API for kissim encoding.
"""

import logging

from opencadd.databases.klifs import setup_remote, setup_local

from kissim.encoding import FingerprintGenerator

logger = logging.getLogger(__name__)


def encode(
    structure_klifs_ids, fingerprints_json_filepath=None, n_cores=1, local_klifs_download_path=None
):
    """
    Encode structures.

    Parameters
    ----------
    structure_klifs_ids : list of int
        Structure KLIFS IDs.
    fingerprints_json_filepath : str or pathlib.Path
        Path to output json file. Default None.
    n_cores : int
        Number of cores used to generate fingerprints.
    local_klifs_download_path : str or None
        If path to local KLIFS download is given, set up local KLIFS session.
        If None is given, set up remote KLIFS session.

    Returns
    -------
    kissim.encoding.fingerprint_generator
        Fingerprints.
    """

    # Set up KLIFS session
    klifs_session = _setup_klifs_session(local_klifs_download_path)

    # Generate fingerprints
    fingerprints = FingerprintGenerator.from_structure_klifs_ids(
        structure_klifs_ids, klifs_session, n_cores
    )

    # Optionally: Save fingerprints to json file
    if fingerprints_json_filepath:
        logger.info(f"Write fingerprints to file: {fingerprints_json_filepath}")
        fingerprints.to_json(fingerprints_json_filepath)

    return fingerprints


def _setup_klifs_session(local_klifs_download_path=None):
    """
    Set up KLIFS session.

    Parameters
    ----------
    local_klifs_download_path : str or None
        If path to local KLIFS download is given, set up local KLIFS session.
        If None is given, set up remote KLIFS session.

    Returns
    -------
    klifs_session : opencadd.databases.klifs.session.Session
        Local or remote KLIFS session.
    """

    if local_klifs_download_path:
        klifs_session = setup_local(local_klifs_download_path)
    else:
        klifs_session = setup_remote()
    return klifs_session
