"""
kissim.encoding.fingerprint_generator_normalized

Defines the normalization of a fingerprint generator.
"""

import logging

from kissim.encoding import FingerprintNormalized, FingerprintGeneratorBase

logger = logging.getLogger(__name__)

NORMALIZATION_METHODS = ["min_max"]


class FingerprintGeneratorNormalized(FingerprintGeneratorBase):
    """
    Generate fingerprints for multiple structures.

    Attributes
    ----------
    structure_klifs_id : int
        Structure KLIFS ID.
    klifs_session : opencadd.databases.klifs.session.Session
        Local or remote KLIFS session.
    data : dict of int: kissim.encoding.Fingerprint
        Fingerprints for input structures (by KLIFS ID).
    """

    def __init__(self):

        self.structure_klifs_ids = None
        self.klifs_session = None
        self.data = None

    @classmethod
    def from_fingerprint_generator(
        cls, fingerprint_generator, method="min_max", fine_grained=True
    ):
        """
        Normalize fingerprints.

        Parameters
        ----------
        method : str
            Normalization method.
        fine_grained : bool
            True (default):
                Distances: Calculate min/max per subpocket for each residue position individually.
                Moments: Calculate min/max per moment for each subpocket individually.
            False:
                Distances: Calculate min/max per subpocket over all residue positions.
                Moments: Calculate min/max per moment over all subpockets.

        Returns
        -------
        kissim.encoding.FingerprintGeneratorNormalized
            Normalized fingerprints.
        """

        fingerprint_generator_normalized = cls()
        fingerprint_generator_normalized.structure_klifs_ids = (
            fingerprint_generator.structure_klifs_ids
        )
        fingerprint_generator_normalized.klifs_session = fingerprint_generator.klifs_session
        fingerprint_generator_normalized.data = fingerprint_generator_normalized._normalize(
            fingerprint_generator.data, method=method, fine_grained=fine_grained
        )

        return fingerprint_generator_normalized

    def _normalize(self, fingerprint_generator_data, method, fine_grained):
        """
        Normalize fingerprints.

        Parameters
        ----------
        fingerprint_generator_data : dict
            Fingerprints (values) by fingerprint IDs (keys).
        method : str
            Normalization method.
        fine_grained : bool
            True (default):
                Distances: Calculate min/max per subpocket for each residue position individually.
                Moments: Calculate min/max per moment for each subpocket individually.
            False:
                Distances: Calculate min/max per subpocket over all residue positions.
                Moments: Calculate min/max per moment over all subpockets.

        Returns
        -------
        dict
            Normalized fingerprints (values) by fingerprint IDs (keys).
        """

        if method == NORMALIZATION_METHODS[0]:
            return self._normalize_min_max(fingerprint_generator_data, fine_grained)
        else:
            raise KeyError(
                f"Normalization type unknown. Please choose from {', '.join(NORMALIZATION_METHODS)}"
            )

    def _normalize_min_max(self, fingerprint_generator_data, fine_grained=True):
        """
        Normalize fingerprints in fingerprint generator with min-max normalization.
        Minimum and maximum values are the minimum and maximum values seen per feature in all available fingerprints.

        Parameters
        ----------
        dict
            Fingerprints (values) by fingerprint IDs (keys).

        Returns
        -------
        dict
            Normalized fingerprints (values) by fingerprint ID (keys).
        """
        return {
            key: FingerprintNormalized.from_fingerprint(value, fine_grained)
            for key, value in fingerprint_generator_data.items()
        }
