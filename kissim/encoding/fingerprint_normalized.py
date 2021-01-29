"""
kissim.encoding.fingerprint_normalized

Defines the normalized kissim fingerprint.
"""

import logging

import numpy as np

from kissim.definitions import DISTANCE_CUTOFFS, MOMENT_CUTOFFS
from kissim.encoding import FingerprintBase

logger = logging.getLogger(__name__)


class FingerprintNormalized(FingerprintBase):
    @classmethod
    def from_fingerprint(cls, fingerprint):
        """
        Normalize fingerprint.

        Parameters
        ----------
        fingerprint : kissim.encoding.Fingerprint
            (Unnormalized) fingerprint.

        Returns
        -------
        kissim.encoding.FingerprintNormalized
            Normalized fingerprint.
        """

        fingerprint_normalized = cls()
        fingerprint_normalized.structure_klifs_id = fingerprint.structure_klifs_id
        fingerprint_normalized.residue_ids = fingerprint.residue_ids
        fingerprint_normalized.residue_ixs = fingerprint.residue_ixs

        fingerprint_normalized._normalize(fingerprint)

        return fingerprint_normalized

    def _normalize(self, fingerprint):
        """
        Normalize the fingerprint (set as values_dict attribute in FingerprintNormalized class).

        Parameters
        ----------
        fingerprint : kissim.encoding.Fingerprint
            (Unnormalized) fingerprint.
        """

        values_dict_normalized = {}

        values_dict_normalized["physicochemical"] = self._normalize_physicochemical_bits(
            fingerprint.values_dict["physicochemical"]
        )
        values_dict_normalized["spatial"] = {}
        values_dict_normalized["spatial"]["distances"] = self._normalize_distances_bits(
            fingerprint.values_dict["spatial"]["distances"]
        )
        values_dict_normalized["spatial"]["moments"] = self._normalize_moments_bits(
            fingerprint.values_dict["spatial"]["moments"]
        )

        self.values_dict = values_dict_normalized

    def _normalize_physicochemical_bits(self, values):
        """
        Normalize physicochemical bits.

        Parameters
        ----------
        values : dict of list of float
            Physicochemical bits.

        Returns
        -------
        dict of list of float
            Normalized physicochemical bits.
        """

        values_normalized = {}

        if values is not None:
            values_normalized["size"] = [
                self._min_max_normalization(value, 1.0, 3.0) for value in values["size"]
            ]
            values_normalized["hbd"] = [
                self._min_max_normalization(value, 0.0, 3.0) for value in values["hbd"]
            ]
            values_normalized["hba"] = [
                self._min_max_normalization(value, 0.0, 2.0) for value in values["hba"]
            ]
            values_normalized["charge"] = [
                self._min_max_normalization(value, -1.0, 1.0) for value in values["charge"]
            ]
            values_normalized["aromatic"] = [
                self._min_max_normalization(value, 0.0, 1.0) for value in values["aromatic"]
            ]
            values_normalized["aliphatic"] = [
                self._min_max_normalization(value, 0.0, 1.0) for value in values["aliphatic"]
            ]
            values_normalized["sco"] = [
                self._min_max_normalization(value, 1.0, 3.0) for value in values["sco"]
            ]
            values_normalized["exposure"] = [
                self._min_max_normalization(value, 1.0, 3.0) for value in values["exposure"]
            ]
            return values_normalized

        else:
            return None

    def _normalize_distances_bits(self, values):
        """
        Normalize distances bits (using cutoffs defined for each subpocket).

        Parameters
        ----------
        values : dict of list of float
            Distances bits.

        Returns
        -------
        dict of list of float
            Normalized distances bits.
        """

        values_normalized = {}

        if values is not None:
            for subpocket_name, distances in values.items():
                values_normalized[subpocket_name] = [
                    self._min_max_normalization(
                        distance,
                        DISTANCE_CUTOFFS[subpocket_name][0],
                        DISTANCE_CUTOFFS[subpocket_name][1],
                    )
                    for distance in distances
                ]
            return values_normalized

        else:
            return None

    def _normalize_moments_bits(self, values):
        """
        Normalize moments bits (using cutoffs defined for each moment).

        Parameters
        ----------
        values : dict of list of float
            Moments bits.

        Returns
        -------
        dict of list of float
            Normalized moments bits.
        """

        values_normalized = {}

        if values is not None:
            for subpocket_name, moments in values.items():
                values_normalized[subpocket_name] = [
                    self._min_max_normalization(
                        moment, MOMENT_CUTOFFS[i + 1][0], MOMENT_CUTOFFS[i + 1][1]
                    )
                    for i, moment in enumerate(values[subpocket_name])
                ]
            return values_normalized

        else:
            return None

    @staticmethod
    def _min_max_normalization(value, minimum, maximum):
        """
        Normalize a value using minimum-maximum normalization.
        Values equal or lower / greater than the minimum / maximum value are set to 0.0 / 1.0.

        Parameters
        ----------
        value : float or int
            Value to be normalized.
        minimum : float or int
            Minimum value for normalization, values equal/greater than this minimum are set to 0.0.
        maximum : float or int
            Maximum value for normalization, values equal/greater than this maximum are set to 1.0.

        Returns
        -------
        float
            Normalized value.
        """

        if np.isnan(value):
            return np.nan
        elif minimum < value < maximum:
            return (value - minimum) / float(maximum - minimum)
        elif value <= minimum:
            return 0.0
        else:
            return 1.0
