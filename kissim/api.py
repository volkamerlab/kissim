"""
kissim.api

Main API for kissim encoding and comparison.
"""

from opencadd.databases.klifs import setup_remote, setup_local

from kissim.encoding import FingerprintGenerator


def encode(structure_klifs_ids, json_filepath, n_cores=1, local_klifs_session=None):
    """
    Encode structures.

    Parameters
    ----------
    structure_klifs_ids : list of int
        Structure KLIFS IDs.
    json_filepath : str or pathlib.Path
        Path to output json file.
    n_cores : int
        Number of cores used to generate fingerprints.
    local_klifs_session : str or None
        If path to local KLIFS download is given, set up local KLIFS session.
        If None is given, set up remote KLIFS session.
    """

    # Set up KLIFS session
    klifs_session = _setup_klifs_session(local_klifs_session)

    # Generate fingerprints
    fingerprints = FingerprintGenerator.from_structure_klifs_ids(
        structure_klifs_ids, klifs_session, n_cores
    )

    # Save fingerprints to json file
    fingerprints.to_json(json_filepath)


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
