"""
kissim.encoding.fingerprint_normalized

Defines the normalized kissim fingerprint.
"""

import logging

from kissim.definitions import DISTANCE_CUTOFFS, MOMENT_CUTOFFS, DISCRETE_FEATURE_VALUES
from kissim.utils import min_max_normalization_vector
from kissim.encoding import FingerprintBase

logger = logging.getLogger(__name__)


class FingerprintNormalized(FingerprintBase):
    @classmethod
    def from_fingerprint(cls, fingerprint, fine_grained=True):
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
        fingerprint_normalized.kinase_name = fingerprint.kinase_name
        fingerprint_normalized.residue_ids = fingerprint.residue_ids
        fingerprint_normalized.residue_ixs = fingerprint.residue_ixs
        fingerprint_normalized.values_dict = fingerprint_normalized._normalize(
            fingerprint, fine_grained
        )

        return fingerprint_normalized

    def _normalize(self, fingerprint, fine_grained):
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
            fingerprint.values_dict["spatial"]["distances"], fine_grained
        )
        values_dict_normalized["spatial"]["moments"] = self._normalize_moments_bits(
            fingerprint.values_dict["spatial"]["moments"], fine_grained
        )

        return values_dict_normalized

    def _normalize_physicochemical_bits(self, values_dict):
        """
        Normalize physicochemical bits.

        Parameters
        ----------
        values_dict : dict of list of float
            Physicochemical bits.

        Returns
        -------
        dict of list of float
            Normalized physicochemical bits.
        """

        values_normalized_dict = {}

        if values_dict is not None:
            for feature_name, values in values_dict.items():
                if feature_name in DISCRETE_FEATURE_VALUES.keys():
                    values_normalized_dict[feature_name] = min_max_normalization_vector(
                        values,
                        min(DISCRETE_FEATURE_VALUES[feature_name]),
                        max(DISCRETE_FEATURE_VALUES[feature_name]),
                    )
            return values_normalized_dict

        else:
            return None

    def _normalize_distances_bits(self, values_dict, fine_grained):
        """
        Normalize distances bits (using cutoffs defined for each subpocket).

        Parameters
        ----------
        values_dict : dict of list of float
            Distances bits.

        Returns
        -------
        dict of list of float
            Normalized distances bits.
        """

        if fine_grained:
            cutoffs = DISTANCE_CUTOFFS["fine"]
        else:
            cutoffs = DISTANCE_CUTOFFS["coarse"]

        values_normalized_dict = {}

        if values_dict is not None:
            for subpocket_name, values in values_dict.items():
                values_normalized_dict[subpocket_name] = min_max_normalization_vector(
                    values,
                    cutoffs.loc[(subpocket_name, "min"), :].to_list(),
                    cutoffs.loc[(subpocket_name, "max"), :].to_list(),
                )
            return values_normalized_dict

        else:
            return None

    def _normalize_moments_bits(self, values_dict, fine_grained):
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

        if fine_grained:
            cutoffs = MOMENT_CUTOFFS["fine"]
        else:
            cutoffs = MOMENT_CUTOFFS["coarse"]

        values_normalized_dict = {}

        if values_dict is not None:
            for subpocket_name, values in values_dict.items():
                # This is truly ugly!
                if fine_grained:
                    minimum = cutoffs[cutoffs.index.get_level_values("min_max") == "min"][
                        subpocket_name
                    ]
                    maximum = cutoffs[cutoffs.index.get_level_values("min_max") == "max"][
                        subpocket_name
                    ]
                else:
                    minimum = cutoffs[cutoffs.index.get_level_values("min_max") == "min"]
                    maximum = cutoffs[cutoffs.index.get_level_values("min_max") == "max"]

                values_normalized_dict[subpocket_name] = min_max_normalization_vector(
                    values, minimum.squeeze().to_list(), maximum.squeeze().to_list()
                )
            return values_normalized_dict

        else:
            return None
