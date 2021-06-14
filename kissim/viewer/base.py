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

from kissim.definitions import FEATURE_METADATA, DISCRETE_FEATURE_VALUES


class _BaseViewer:
    """
    Base class to view fingerprint data in 3D.

    Attributes
    ----------
    _text : str
        PDB text.
    _fingerprint : kissim.encoding.Fingerprint
        Fingerprint.
    _fingerprints : kissim.encoding.FingerprintGenerator
        Fingerprints.
    _discrete_feature_values : dict of str: list of float
        For all discrete features (keys) list discrete value options (values).
    _feature_metadata : dict of str: tuple of (str, list of str)
        For all features (keys) list a descriptive feature name and a list of descriptive feature
        value names. Useful for plotting!
    """

    def __init__(  # pylint: disable=W0102
        self,
        discrete_feature_values=DISCRETE_FEATURE_VALUES,
        feature_metadata=FEATURE_METADATA,
    ):

        self._text = None
        self._fingerprint = None
        self._fingerprints = None
        self._discrete_feature_values = discrete_feature_values
        self._feature_metadata = feature_metadata

    @property
    def _feature_names(self):
        """
        All possible feature names.

        Returns
        -------
        list of str
            List of feature names.
        """

        feature_names = (
            self._fingerprint.physicochemical.columns.to_list()
            + self._fingerprint.distances.columns.to_list()
        )
        return feature_names

    @property
    def _fingerprints_features(self):
        """
        All fingerprints' feature values.

        Returns
        -------
        pandas.DataFrame
            Features (columns) for each fingerprint and residue (multiindexed row).
        """

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
                value=False, description="Show side chains", disabled=False, indent=False
            ),
        )

    def _show(self, feature_name, show_side_chains=True):
        """
        Show a feature mapped onto the 3D pocket.

        Parameters
        ----------
        feature_name : str
            Feature name.
        show_side_chains : bool
            Show side chains colored by feature defined in `feature_name` (by default) or show
            no side chains and map color onto structure backbone.

        Returns
        -------
        nglview.NGLWidget
            View.
        """

        residue_ids = self._fingerprint.residue_ids
        residue_to_color = self.residue_to_color_mapping(feature_name)
        color_scheme = nglview.color._ColorScheme(residue_to_color, label="scheme_regions")

        view = nglview.NGLWidget()
        view._remote_call("setSize", target="Widget", args=["1000px", "600px"])
        view.camera = "orthographic"
        component = nglview.TextStructure(self._text, ext="pdb")
        view.add_component(component)

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

    def _residue_to_color_mapping(
        self, feature_name, data, discrete=False, divergent=False, label_prefix=""
    ):
        """
        Map (discrete) feature values using color on residues.

        Parameters
        ----------
        feature_name : str
            Feature name.
        data : pd.Series
            Values for feature.
        discrete : None or list
            All possible categories for discrete feature.
        divergent : bool
            Use divergent colormap (PiYG) or sequential colormap (viridis)
        label_prefix

        Returns
        -------
        list of list of [str, str]
            List of color-residue ID pairs. Color given as hex string, residue PDB IDs as string.
        """

        # Define color map
        if divergent:
            cmap_name = "PiYG"
        else:
            cmap_name = "viridis"

        # Define norm
        # Discrete values and seqential colormap
        if discrete and not divergent:
            discrete_options = self._discrete_feature_values[feature_name]
            norm = colors.Normalize(vmin=min(discrete_options), vmax=max(discrete_options))
            cmap = cm.get_cmap(cmap_name, len(discrete_options))
        # Continuous values and seqential colormap
        elif not discrete and not divergent:
            norm = colors.Normalize(vmin=data.min(), vmax=data.max())
            cmap = cm.get_cmap(cmap_name)
        # Continuous values and divergent colormap
        elif not discrete and divergent:
            if data.min() != data.max():
                norm = colors.TwoSlopeNorm(vmin=data.min(), vcenter=0.0, vmax=data.max())
                cmap = cm.get_cmap(cmap_name)
            else:
                norm = colors.NoNorm(vmin=data.min(), vmax=data.min())
                cmap = cm.get_cmap(cmap_name, 1)
        else:
            raise NotImplementedError(
                f"The combination of discrete values and divergent colormap is not implemented."
            )

        # Normalize data
        data_normed = norm(data)
        # Map residues to colors based on category
        residue_ids = self._fingerprint.residue_ids
        residue_to_color = []
        for residue_id, value in zip(residue_ids, data_normed):
            if np.isnan(value):
                # If no value given, choose grey
                color = "#808080"
            else:
                # Look up color for value
                color = cmap(value)
                # Convert RGB to HEX
                color = colors.rgb2hex(color)
            residue_to_color.append([color, str(residue_id)])

        # Add colobar (if discrete, remove normalization)
        if discrete:
            norm = colors.NoNorm(vmin=min(discrete_options), vmax=max(discrete_options))
        label, xticklabels = self._feature_metadata[feature_name]
        self.cmap_colorbar(cmap, norm, f"{label_prefix}{label}", xticklabels)

        return residue_to_color

    @staticmethod
    def cmap_colorbar(cmap, norm, label, xticklabels):
        """
        Plot colormap as colorbar with data-to-color mappings.

        Parameters
        ----------
        cmap : matplotlib.colors.Colormap
            Color map.
        norm : matplotlib.colors.NoNorm or TwoSlopeNorm or Normalize
            Data to color normalizer.
        label : str
            Colorbar label.
        xticklabels : list of str
            Labels for x-axis ticks.
        """

        # Exception: If minimum and maximum the same, colormap only needs one element!
        if norm.vmin == norm.vmax:
            norm = colors.NoNorm(vmin=norm.vmin, vmax=norm.vmax)
            cmap = cm.get_cmap(cmap.name, 1)
            xticklabels = [norm.vmin]

        fig, ax = plt.subplots(figsize=(6, 1))
        fig.subplots_adjust(bottom=0.5)
        fig.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=ax,
            orientation="horizontal",
            label=label,
        )
        # If categorial, exchange tick labels with meaningful text
        if isinstance(norm, colors.NoNorm):
            ax.set_xticklabels(xticklabels)
