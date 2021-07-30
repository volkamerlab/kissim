"""
kissim.api.subset

Main API for subsetting fingerprints.
"""


import logging
from pathlib import Path

import numpy as np

from kissim.encoding import Fingerprint, FingerprintGenerator
from kissim.definitions import KLIFS_POCKET_RESIDUE_SUBSET
from kissim.utils import calculate_first_second_third_moments

logger = logging.getLogger(__name__)


def subset(
    fingerprints_path,
    klifs_pocket_residue_subset_type,
    fingerprints_subset_path=None,
    klifs_pocket_residue_subset=KLIFS_POCKET_RESIDUE_SUBSET,
):
    """
    Subset fingerprints based on a list of KLIFS residue IDs.

    Parameters
    ----------
    fingerprints_path : str or pathlib.Path
        Path to fingerprints JSON file.
    klifs_pocket_residue_subset_type : str
        Subset type; corresponding residues will be looked up in `klifs_pocket_residue_subset`.
    fingerprints_subset_path : None or str or pathlib.Path
        Path to subset fingerprints JSON file.
    klifs_pocket_residue_subset : dict
        For different subset types (e.g. DFG conformation), list of KLIFS residue IDs to be
        included in fingerprint subset.

    Returns
    -------
    kissim.encoding.FingerprintGenerator
        Fingerprints with subset of residues only.
    """

    if klifs_pocket_residue_subset_type not in klifs_pocket_residue_subset.keys():
        raise KeyError(
            f"Unknown subset type {klifs_pocket_residue_subset_type}; use one of the following: "
            f"{', '.join(klifs_pocket_residue_subset.keys())}"
        )

    # Load fingerprints
    logger.info("Read fingerprints...")
    fingerprints_path = Path(fingerprints_path)
    fingerprint_generator = FingerprintGenerator.from_json(fingerprints_path)
    logger.info(f"Number of fingerprints: {len(fingerprint_generator.data)}")

    # Find structures/fingerprints IDs to be removed
    klifs_residue_ids = klifs_pocket_residue_subset[klifs_pocket_residue_subset_type]
    logger.info(
        f"Generate subset fingerprint for {klifs_pocket_residue_subset_type} "
        f"({len(klifs_residue_ids)} residues)"
    )
    fingerprint_generator_subset = _subset_fingerprint_generator(
        fingerprint_generator, klifs_residue_ids
    )

    # Log input and output fingerprint lengths
    n_bits = len(fingerprint_generator.data[3].values_array())
    n_bits_subset = len(fingerprint_generator_subset.data[3].values_array())
    n_residues = len(klifs_residue_ids)
    n_residues_theory = 12 * n_residues + 12
    if n_bits_subset == n_residues_theory:
        logger.info(
            f"Number of fingerprint bits in input ({n_bits}) and output ({n_bits_subset})."
        )
    else:
        raise ValueError(
            f"Number of output fingerprint bits must be {n_residues_theory} but is {n_residues}."
        )

    # Optionally: Save to file
    if fingerprints_subset_path is not None:
        logger.info(f"Save subset fingerprints to {fingerprints_subset_path}...")
        fingerprints_subset_path = Path(fingerprints_subset_path)
        fingerprint_generator_subset.to_json(fingerprints_subset_path)

    return fingerprint_generator_subset


def _subset_fingerprint_generator(fingerprint_generator, klifs_residue_ids):
    """
    Subset each fingerprint in the input fingerprint generator.

    Attributes
    ----------
    kissim.encoding.FingerprintGenerator
        Fingerprints.
    klifs_residue_ids : list of int
        List of KLIFS residue IDs to be used for fingerprint subsetting.

    Returns
    -------
    kissim.encoding.FingerprintGenerator
        Fingerprints with subset of residues only.
    """

    # Set up new fingerprint generator
    fingerprint_generator_subset = FingerprintGenerator()
    fingerprint_generator_subset.structure_klifs_ids = fingerprint_generator.structure_klifs_ids
    fingerprint_generator_subset.klifs_session = fingerprint_generator.klifs_session
    fingerprint_generator_subset.data = {}
    fingerprint_generator_subset.data_normalized = {}

    # We will index numpy arrays, so let's start our residue numbering at 0 not 1
    klifs_residue_ixs = [i - 1 for i in klifs_residue_ids]

    for id_, fp in fingerprint_generator.data.items():

        # Initialize new fingerprint
        fp_subset = Fingerprint()
        fp_subset.structure_klifs_id = fp.structure_klifs_id
        fp_subset.kinase_name = fp.kinase_name

        fp_dict = {}

        # Iterate over physiochemical and spatial features
        for feature_name1, features1 in fp.values_dict.items():

            fp_dict[feature_name1] = {}

            # Iterate over all physiochemical features
            if feature_name1 == "physicochemical":
                for feature_name2, features2 in features1.items():
                    fp_dict[feature_name1][feature_name2] = np.array(features2)[
                        klifs_residue_ixs
                    ].tolist()

            # Iterate over all spatial features (includes distances and moments!)
            elif feature_name1 == "spatial":

                for feature_name2, features2 in features1.items():

                    fp_dict[feature_name1][feature_name2] = {}

                    # Iterate over all distances
                    if feature_name2 == "distances":
                        for feature_name3, features3 in features2.items():
                            fp_dict[feature_name1][feature_name2][feature_name3] = np.array(
                                features3
                            )[klifs_residue_ixs].tolist()

                    # Iterate over all moments
                    elif feature_name2 == "moments":
                        # Pass through these iterations to make sure distances have always
                        # finished processing before calculating moments from them.
                        pass

                    elif feature_name2 == "subpocket_centers":
                        # Pass through because subpocket centers stay the same, thus no change
                        # needed
                        pass

                    else:
                        raise KeyError(f"Invalid key in fingerprint: {feature_name2}")

            else:
                raise KeyError(f"Invalid key in fingerprint: {feature_name1}")

        # Calculate moments from subset distances
        for subpocket_name, distances in fp_dict["spatial"]["distances"].items():
            moment1, moment2, moment3 = calculate_first_second_third_moments(distances)
            # Must be cast to list of floats (removing all numpy data types) to allow json dump
            fp_dict["spatial"]["moments"][subpocket_name] = np.array(
                [moment1, moment2, moment3]
            ).tolist()

        fp_subset.values_dict = fp_dict
        fp_subset.residue_ids = np.array(fp.residue_ids)[klifs_residue_ixs].tolist()
        fp_subset.residue_ixs = np.array(fp.residue_ixs)[klifs_residue_ixs].tolist()

        fingerprint_generator_subset.data[id_] = fp_subset

    # If fingerprint generator contains normalized fingerprints
    if fingerprint_generator.data_normalized is not None:
        fingerprint_generator_subset.data_normalized = (
            fingerprint_generator_subset._normalize_fingerprints()
        )

    return fingerprint_generator_subset
