"""
kissim.viewer.pair

Visualizes a two structures' fingerprint difference in 3D.
"""

import pandas as pd
from opencadd.databases.klifs import setup_remote

from kissim.encoding import FingerprintGenerator
from kissim.viewer.base import _BaseViewer


class StructurePairViewer(_BaseViewer):
    """
    View a kinase's fingerprint variability.

    Attributes
    ----------
    Inherited from kissim.viewer.base._BaseViewer
    """

    @classmethod
    def from_structure_klifs_ids(
        cls, structure_klifs_id1, structure_klifs_id2, klifs_session=None
    ):
        """
        Initialize viewer from two structure KLIFS IDs: Generate fingerprints and
        fetch structure in PDB format for first structure.
        """

        structure_klifs_ids = [structure_klifs_id1, structure_klifs_id2]
        viewer = cls._from_structure_klifs_id(structure_klifs_ids, klifs_session)

        return viewer

    @property
    def _diff(self):

        features_pair = []
        for _, fingerprint in self._fingerprints.data.items():
            features = pd.concat([fingerprint.physicochemical, fingerprint.distances], axis=1)
            features_pair.append(features)

        return features_pair[0] - features_pair[1]

    def residue_to_color_mapping(self, feature_name, fingerprint, plot_cmap=False):
        """
        Map feature values using color on residues.

        Parameters
        ----------
        feature_name : str
            Name of a continuous feature.
        fingerprint : kissim.encoding.Fingerprint
            Fingerprint.
        plot_cmap : bool
            Plot color map (default: False).

        Returns
        -------
        list of list of [str, str]
            List of color-residue ID pairs. Color given as hex string, residue PDB IDs as string.
        """

        if feature_name not in self._feature_names:
            raise ValueError(f"Feature name {feature_name} unknown.")

        data = self._diff[feature_name]
        data.index = fingerprint.residue_ids
        residue_to_color = self._residue_to_color_mapping(
            feature_name,
            data,
            discrete=False,
            divergent=True,
            label_prefix="difference in ",
            plot_cmap=plot_cmap,
        )

        return residue_to_color
