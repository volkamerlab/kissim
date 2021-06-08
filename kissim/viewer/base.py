"""
kissim.viewer.base

Base class for fingerprint 3D visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import nglview
from ipywidgets import interact
import ipywidgets as widgets

from kissim.definitions import FEATURE_METADATA


class _BaseViewer:
    """
    Base class to view fingerprint data in 3D.

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
        self._fingerprints = None

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

    @property
    def _fingerprints_features(self):

        if self._fingerprints is None:
            raise NotImplementedError(
                f"Property not available; not implemented in the child class of _BaseViewer that "
                f"you are currently using."
            )

        return pd.concat(
            [
                self._fingerprints.physicochemical_exploded(),
                self._fingerprints.distances_exploded(),
            ]
        )[self._feature_names]

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
            view.shape.add_sphere(center, [0.0, 0.0, 0.0], 1, f"{feature_name}")

        return view.display(gui=True)

    def _discrete_residue_to_color_mapping(
        self, feature_name, features, feature_categories, cmap_name="viridis", label_prefix=""
    ):
        """
        Map (discrete) feature values using color on residues.

        Parameters
        ----------
        features : pd.Series
            Values for feature.
        TODO
        feature_categories : dict
            Al possible categories for discrete feature.
        cmap_name : str
            Colormap name (default: viridis).

        Returns
        -------
        list of list of [str, str]
            List of color-residue ID pairs. Color given as hex string, residue PDB IDs as string.
        """

        # Map categories to colors
        cmap = cm.get_cmap(cmap_name, len(feature_categories))
        # Convert RGB to HEX
        value_to_color = {
            value: colors.rgb2hex(color) for value, color in zip(feature_categories, cmap.colors)
        }

        # Map residues to colors based on category
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

        cmap_colorbar(
            cmap,
            feature_name,
            vmin_max=[feature_categories[0], feature_categories[-1]],
            categorial=True,
            label_prefix=label_prefix,
        )

        return residue_to_color

    def _continuous_residue_to_color_mapping(
        self, feature_name, features, cmap_name="viridis", label_prefix=""
    ):
        """
        Map (continuous) feature values using color on residues.

        Parameters
        ----------
        TODO
        features : pd.Series
            Values for feature.
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
        # Normalize continuous features by maximum value
        feature_max = features.max()
        if feature_max != 0:
            features = features / feature_max
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

        cmap_colorbar(
            cmap,
            feature_name,
            vmin_max=[0, feature_max],
            label_prefix=label_prefix,
        )

        return residue_to_color


def cmap_colorbar(cmap, feature_name, vmin_max=None, categorial=False, label_prefix=""):

    label, xticklabels = FEATURE_METADATA[feature_name]

    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)

    if categorial:
        norm = colors.NoNorm(vmin=vmin_max[0], vmax=vmin_max[1])
    else:
        norm = colors.Normalize(vmin=vmin_max[0], vmax=vmin_max[1])

    fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation="horizontal",
        label=f"{label_prefix}{label}",
    )
    if categorial:
        ax.set_xticklabels(xticklabels)
