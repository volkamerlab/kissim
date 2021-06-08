"""
kissim.viewer.structure

Visualizes a structure's fingerprint in 3D.
"""


import numpy as np
from matplotlib import cm, colors
import nglview
from ipywidgets import interact
import ipywidgets as widgets
from opencadd.databases.klifs import setup_remote

from kissim.encoding import Fingerprint
from kissim.definitions import DISCRETE_FEATURE_VALUES


class StructureViewer:
    """
    View a structure's fingerprint in 3D.

    Attributes
    ----------
    _text : str
        PDB text.
    _fingerprint : kissim.encoding.Fingerprint
        Fingerprint.
    """

    def __init__(self):

        self._text = None
        self._fingerprint = None

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

    @property
    def _feature_names(self):
        """
        All possible feature names.
        """

        feature_names = (
            self._fingerprint.physicochemical.columns.to_list()
            + self._fingerprint.distances.columns.to_list()
        )
        return feature_names

    def _show(self, feature_name, show_side_chains=True):
        """
        Show a feature mapped onto the 3D pocket.
        """

        residue_ids = self._fingerprint.residue_ids
        residue_to_color = self.residue_to_color_mapping(feature_name)
        color_scheme = nglview.color._ColorScheme(residue_to_color, label="scheme_regions")

        view = nglview.NGLWidget()
        view._remote_call("setSize", target="Widget", args=["1000px", "600px"])
        view.camera = "orthographic"
        view.add_component(self._text, ext="pdb")

        if show_side_chains:
            selection = " or ".join([str(i) for i in residue_ids])
            view.clear_representations()
            view.add_representation("cartoon", selection="protein", color="grey")
            view.add_representation("ball+stick", selection=selection, color=color_scheme)
        else:
            view.clear_representations()
            view.add_representation("cartoon", selection="protein", color=color_scheme)

        if feature_name in self._fingerprint.subpocket_centers.columns:
            center = self._fingerprint.subpocket_centers[feature_name].to_list()
            view.shape.add_sphere(center, [0.267004, 0.004874, 0.329415], 1, f"{feature_name}")

        return view.display(gui=True)

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

    def _discrete_residue_to_color_mapping(
        self, feature_name, feature_categories=DISCRETE_FEATURE_VALUES, cmap_name="viridis"
    ):
        """
        Map (discrete) feature values using color on residues.

        Parameters
        ----------
        feature_name : str
            Name of a discrete feature.
        feature_categories : dict
            Dictionary of all possible categories per discrete feature.
        cmap_name : str
            Colormap name (default: viridis).

        Returns
        -------
        list of list of [str, str]
            List of color-residue ID pairs. Color given as hex string, residue PDB IDs as string.
        """

        # Map categories to colors
        feature_categories = feature_categories[feature_name]
        cmap = cm.get_cmap(cmap_name, len(feature_categories))
        # Convert RGB to HEX
        value_to_color = {
            value: colors.rgb2hex(color) for value, color in zip(feature_categories, cmap.colors)
        }

        # Map residues to colors based on category
        features = self._fingerprint.physicochemical[feature_name]
        residue_ids = self._fingerprint.residue_ids

        residue_to_color = []
        for residue_id, value in zip(residue_ids, features):

            try:
                # Look up color for category
                color = value_to_color[value]
            except KeyError:
                # If category unknown (np.nan), choose grey
                color = "#808080"
            residue_to_color.append([color, str(residue_id)])

        return residue_to_color

    def _continuous_residue_to_color_mapping(self, feature_name, cmap_name="viridis"):
        """
        Map (continuous) feature values using color on residues.

        Parameters
        ----------
        feature_name : str
            Name of a continuous feature.
        cmap_name : str
            Colormap name (default: viridis).

        Returns
        -------
        list of list of [str, str]
            List of color-residue ID pairs. Color given as hex string, residue PDB IDs as string.
        """

        # Get fine-grained colormap (1000 colors)
        # Access colors by `value` in [0, 1] using `cmap(value)`
        cmap = cm.get_cmap(cmap_name, 1000)

        # Map residues to colors based on category
        features = self._fingerprint.distances
        # Normalize continuous features by maximum value amongst ALL continuous features
        features = features / features.max().max()
        features = features[feature_name]
        residue_ids = self._fingerprint.residue_ids

        residue_to_color = []
        for residue_id, value in zip(residue_ids, features):
            if np.isnan(value):
                # If no value given, choose grey
                color = "#808080"
            else:
                # Look up color for value
                color = cmap(value)
            # Convert RGB to HEX
            residue_to_color.append([colors.rgb2hex(color), str(residue_id)])

        return residue_to_color

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
            residue_to_color = self._discrete_residue_to_color_mapping(feature_name)
        elif feature_name in self._fingerprint.distances:
            residue_to_color = self._continuous_residue_to_color_mapping(feature_name)
        else:
            raise ValueError(f"Feature name {feature_name} unknown.")

        return residue_to_color
