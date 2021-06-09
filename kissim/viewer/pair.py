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
    _text : str
        PDB text for example structure.
    _fingerprint : str
        Fingerprint for example structure.
    _fingerprints : kissim.encoding.FingerprintGenerator
        Fingerprints.
    """

    @classmethod
    def from_structure_klifs_ids(
        cls, structure_klifs_id1, structure_klifs_id2, klifs_session=None
    ):
        """
        Initialize viewer from two structure KLIFS IDs: Generate fingerprints and
        fetch structure in PDB format for first structure.
        """

        viewer = cls()

        if klifs_session is None:
            klifs_session = setup_remote()
        text = klifs_session.coordinates.to_text(structure_klifs_id1, "complex", "pdb")

        print("Generate fingerprints...")
        fingerprints = FingerprintGenerator.from_structure_klifs_ids(
            [structure_klifs_id1, structure_klifs_id2], klifs_session=klifs_session
        )

        viewer._text = text
        viewer._fingerprint = fingerprints.data[structure_klifs_id1]
        viewer._fingerprints = fingerprints

        return viewer

    @property
    def _diff(self):

        features_pair = []
        for _, fingerprint in self._fingerprints.data.items():
            features = pd.concat([fingerprint.physicochemical, fingerprint.distances], axis=1)
            features_pair.append(features)

        return features_pair[0] - features_pair[1]

    def residue_to_color_mapping(self, feature_name):
        """
        Map feature values using color on residues.

        Parameters
        ----------
        feature_name : str
            Name of a continuous feature.

        Returns
        -------
        list of list of [str, str]
            List of color-residue ID pairs. Color given as hex string, residue PDB IDs as string.
        """

        if feature_name not in self._feature_names:
            raise ValueError(f"Feature name {feature_name} unknown.")

        residue_to_color = self._residue_to_color_mapping(
            feature_name,
            self._diff[feature_name],
            discrete=False,
            divergent=True,
            label_prefix="difference in ",
        )

        return residue_to_color
