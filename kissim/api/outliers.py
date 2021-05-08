"""
kissim.api.outliers

Main API for removing outlier fingerprints (defined by spatial distances maximum).
"""


import logging
from pathlib import Path

from kissim.encoding import FingerprintGenerator

logger = logging.getLogger(__name__)


def outliers(fingerprints_path, distance_cutoff, fingerprints_wo_outliers_path=None):
    """
    Remove outlier fingerprints (defined by spatial distances maximum).

    Parameters
    ----------
    fingerprints_path : str or pathlib.Path
        Path to fingerprints JSON file.
    distance_cutoff : float
        Tolerated distance maximum; fingerprints with distances greater than this cutoff will be
        removed.
    fingerprints_wo_outliers_path : None or str or pathlib.Path
        Path to fingerprints JSON file with outliers removed.

    Returns
    -------
    kissim.encoding.FingerprintGenerator
        Fingerprints without outliers.
    """

    # Load fingerprints
    logger.info("Read fingerprints...")
    fingerprints_path = Path(fingerprints_path)
    fingerprint_generator = FingerprintGenerator.from_json(fingerprints_path)
    logger.info(f"Number of fingerprints: {len(fingerprint_generator.data)}")

    # Find structures/fingerprints IDs to be removed
    logger.info(
        f"Use the following distance minimum/maximum cutoffs"
        f" to identify outlier structures: {distance_cutoff}"
    )
    remove_structure_ids = []
    for structure_id, fp in fingerprint_generator.data.items():
        if (fp.distances > distance_cutoff).any().any():
            remove_structure_ids.append(structure_id)
    logger.info(f"Structure IDs to be removed: {remove_structure_ids}")

    # Remove fingerprints
    logger.info("Remove fingerprints with distance outliers...")
    for structure_id in remove_structure_ids:
        del fingerprint_generator.data[structure_id]
    logger.info(f"Number of fingerprints: {len(fingerprint_generator.data)}")

    # Optionally: Save to file
    if fingerprints_wo_outliers_path is not None:
        logger.info(f"Save cleaned fingerprints to {fingerprints_wo_outliers_path}...")
        fingerprints_wo_outliers_path = Path(fingerprints_wo_outliers_path)
        fingerprint_generator.to_json(fingerprints_wo_outliers_path)

    return fingerprint_generator
