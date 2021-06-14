"""
kissim.viewer.kinase

Visualizes a kinase's fingerprint variability in 3D.
"""

from random import sample

from opencadd.databases.klifs import setup_remote

from kissim.encoding import FingerprintGenerator
from kissim.viewer.base import _BaseViewer


class KinaseViewer(_BaseViewer):
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
    def from_kinase_klifs_id(
        cls, kinase_klifs_id, klifs_session=None, example_structure_klifs_id=None, n_sampled=None
    ):
        """
        Initialize viewer from kinase KLIFS ID: Generate fingerprints for all kinase structures and
        fetch example structure in PDB format.
        """

        viewer = cls()

        if klifs_session is None:
            klifs_session = setup_remote()

        # Fetch structures for input kinase
        print("Fetch structures metadata for input kinase...")
        structures = klifs_session.structures.by_kinase_klifs_id(kinase_klifs_id)
        # Sort structures by quality scores
        structures = structures.sort_values(by="structure.qualityscore", ascending=False)
        structure_klifs_ids = structures["structure.klifs_id"].to_list()

        # Fetch example structure PDB
        print("Fetch example PDB...")
        if example_structure_klifs_id is None:
            # Choose structure with best quality score
            example_structure_klifs_id = structure_klifs_ids[0]
        else:
            if example_structure_klifs_id not in structure_klifs_ids:
                raise ValueError(
                    f"Input example structure {example_structure_klifs_id} is not "
                    f"deposited under kinase {kinase_klifs_id} in KLIFS."
                )

        # Optionally: Sample from kinase structures
        if n_sampled is not None:
            if n_sampled < len(structure_klifs_ids):
                samples = list(set(structure_klifs_ids) - set([example_structure_klifs_id]))
                structure_klifs_ids = sample(samples, n_sampled - 1)
                structure_klifs_ids = [example_structure_klifs_id] + structure_klifs_ids

        # Check if we have enough structures for the kinase viewer
        if len(structure_klifs_ids) == 0:
            raise ValueError(f"Kinase viewer cannot be created because kinase no structures.")
        if len(structure_klifs_ids) == 1:
            raise ValueError(
                f"Kinase viewer cannot be created because kinase has only 1 structure.\n"
                f"You can look at the structure with:\n"
                f"from kissim.viewer import StructureViewer\n"
                f"StructureViewer.from_structure_klifs_id({structure_klifs_ids[0]})\n"
            )

        # Get PDB file content
        text = klifs_session.coordinates.to_text(example_structure_klifs_id, "complex", "pdb")

        # Generate fingerprints for all structures
        print("Generate fingerprints...")
        fingerprints = FingerprintGenerator.from_structure_klifs_ids(
            structure_klifs_ids, klifs_session=klifs_session
        )

        viewer._text = text
        viewer._fingerprint = fingerprints.data[example_structure_klifs_id]
        viewer._fingerprints = fingerprints

        return viewer

    @property
    def _std(self):

        return self._fingerprints_features.std(level="residue_ix")

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
            self._std[feature_name],
            discrete=False,
            divergent=False,
            label_prefix="standard deviation of ",
        )

        return residue_to_color
