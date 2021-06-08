"""
kissim.viewer.structure

Visualizes a structure's fingerprint in 3D.
"""


from ipywidgets import interact
import ipywidgets as widgets
from opencadd.databases.klifs import setup_remote

from kissim.encoding import Fingerprint
from kissim.viewer.base import _BaseViewer
from kissim.definitions import DISCRETE_FEATURE_VALUES


class StructureViewer(_BaseViewer):
    """
    View a structure's fingerprint in 3D.

    Attributes
    ----------
    _text : str
        PDB text.
    _fingerprint : kissim.encoding.Fingerprint
        Fingerprint.
    """

    @classmethod
    def from_structure_klifs_id(cls, structure_klifs_id, klifs_session=None):
        """
        Initialize viewer from structure KLIFS ID: Generate fingerprint and fetch structure in PDB
        format).
        """

        viewer = cls()

        if klifs_session is None:
            klifs_session = setup_remote()
        text = klifs_session.coordinates.to_text(structure_klifs_id, "complex", "pdb")
        fingerprint = Fingerprint.from_structure_klifs_id(
            structure_klifs_id, klifs_session=klifs_session
        )

        viewer._text = text
        viewer._fingerprint = fingerprint

        return viewer

    def show(self):
        """
        Show features mapped onto the 3D pocket (select feature interactively).
        """

        interact(
            self._show,
            feature_name=widgets.Dropdown(
                options=self._feature_names,
                value="size",
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

        if feature_name in self._fingerprint.physicochemical:
            residue_to_color = self._discrete_residue_to_color_mapping(
                feature_name,
                self._fingerprint.physicochemical[feature_name],
                DISCRETE_FEATURE_VALUES[feature_name],
            )
        elif feature_name in self._fingerprint.distances:
            residue_to_color = self._continuous_residue_to_color_mapping(
                feature_name, self._fingerprint.distances[feature_name]
            )
        else:
            raise ValueError(f"Feature name {feature_name} unknown.")

        return residue_to_color
