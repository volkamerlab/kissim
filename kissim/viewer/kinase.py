"""
kissim.viewer.kinase

Visualizes a kinase's fingerprint variability in 3D.
"""

from ipywidgets import interact
import ipywidgets as widgets
from opencadd.databases.klifs import setup_remote

from kissim.encoding import FingerprintGenerator
from kissim.viewer.base import _BaseViewer


class KinaseViewer(_BaseViewer):
    """
    View a kinase's fingerprint TODO.

    Attributes
    ----------
    _text : str
        PDB text for first input structure.
    _fingerprints : kissim.encoding.FingerprintGenerator
        Fingerprints.
    """

    @classmethod
    def from_kinase_klifs_id(
        cls, kinase_klifs_id, klifs_session=None, example_structure_klifs_id=None
    ):
        """
        Initialize viewer from kinase KLIFS ID: Generate fingerprints for all kinase structures and
        fetch example structure in PDB format).
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

    def show(self):
        """
        Show features mapped onto the 3D pocket (select feature interactively).
        """

        interact(
            self._show,
            feature_name=widgets.Dropdown(
                options=self._feature_names,
                value="sco",
                description="Feature: ",
                disabled=False,
            ),
            show_side_chains=widgets.Checkbox(
                value=True, description="Show side chains", disabled=False, indent=False
            ),
        )

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

        residue_to_color = self._continuous_residue_to_color_mapping(
            feature_name, self._std[feature_name], label_prefix="standard deviation of "
        )

        return residue_to_color
