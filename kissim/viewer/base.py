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
from opencadd.databases.klifs import setup_remote

from kissim.definitions import FEATURE_METADATA, DISCRETE_FEATURE_VALUES
from kissim.encoding.fingerprint_generator import FingerprintGenerator


class _BaseViewer:
    """
    Base class to view fingerprint data in 3D.

    Attributes
    ----------
    _reference_structure_id : str or int
        ID for reference structure (=primary structure to be shown in the NGLviewer)
    _texts : dict of (str or int): str
        PDB text (values) per structure ID (keys).
    _fingerprints : kissim.encoding.FingerprintGenerator
        Fingerprints.
    _ligands : dict of (str or int): str
        Ligand expo ID (values) per structure ID (keys).
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

        self._reference_structure_id = None
        self._texts = {}
        self._fingerprints = None
        self._ligands = {}
        self._discrete_feature_values = discrete_feature_values
        self._feature_metadata = feature_metadata

    @property
    def _fingerprint(self):
        """
        Fingerprint for reference structure.

        Returns
        -------
        kissim.encoding.Fingerprint
            Fingerprint.
        """

        return self._fingerprints.data[self._reference_structure_id]

    @property
    def _text(self):
        """
        PDB text for reference structure.

        Returns
        -------
        str
            PDB text.
        """

        return self._texts[self._reference_structure_id]

    @property
    def _ligand(self):
        """
        Ligand for reference structure.

        Returns
        -------
        str
            Ligand expo ID.
        """

        return self._ligands[self._reference_structure_id]

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
                self._fingerprints.physicochemical_exploded,
                self._fingerprints.distances_exploded,
            ]
        )[self._feature_names]

    @classmethod
    def _from_structure_klifs_ids(cls, structure_klifs_ids, klifs_session=None):
        """
        Initialize viewer from structure KLIFS ID: Generate fingerprint and fetch structure in PDB
        format.
        """

        viewer = cls()

        if klifs_session is None:
            klifs_session = setup_remote()

        # Get structure text, ligand, and fingerprint
        # With a local KLIFS session PDB files can be missing
        # FingerprintGenerator will omit respective structure KLIFS IDs; the same behaviour must
        # be implemented here for fetching the PDB texts (try-except)
        texts = {}
        for structure_klifs_id in structure_klifs_ids:
            try:
                # Hack! KLIFS IDs may be of type numpy.int32, cast to int!
                structure_klifs_id = int(structure_klifs_id)
                texts[structure_klifs_id] = klifs_session.coordinates.to_text(
                    structure_klifs_id, "complex", "pdb"
                )
            except FileNotFoundError:
                pass
        fingerprints = FingerprintGenerator.from_structure_klifs_ids(
            structure_klifs_ids, klifs_session=klifs_session
        )
        structures = klifs_session.structures.by_structure_klifs_id(structure_klifs_ids)
        ligands = {
            row["structure.klifs_id"]: row["ligand.expo_id"] for _, row in structures.iterrows()
        }

        # Set attributes
        viewer._reference_structure_id = structure_klifs_ids[0]
        viewer._texts = texts
        viewer._fingerprints = fingerprints
        viewer._ligands = ligands

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
                value=False, description="Show side chains", disabled=False, indent=False
            ),
            gui=widgets.Checkbox(
                value=False, description="Show NGL GUI", disabled=False, indent=False
            ),
        )

    def _show(self, feature_name, show_side_chains=True, gui=True):
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

        view = nglview.NGLWidget()
        view._remote_call("setSize", target="Widget", args=["1000px", "600px"])
        view.camera = "orthographic"

        component_counter = 0

        for structure_id, fingerprint in self._fingerprints.data.items():
            text = self._texts[structure_id]
            ligand = self._ligands[structure_id]
            view, component_counter = self._show_structure(
                view,
                structure_id,
                text,
                fingerprint,
                ligand,
                feature_name,
                show_side_chains,
                component_counter,
            )

        if gui:
            return view.display(gui=True)
        else:
            return view

    def _show_structure(
        self,
        view,
        structure_id,
        text,
        fingerprint,
        ligand,
        feature_name,
        show_side_chains,
        component_counter,
    ):
        """
        Show structure with
        - residue coloring based on feature name
        - protein with/without pocket side chains
        - ligand
        - pocket center

        Parameters
        ----------
        view : nglview.widget.NGLWidget
            NGLview widget to draw in.
        structure_id : int
            KLIFS structure ID.
        text : str
            Structure coordinates.
        fingerprint : kissim.encoding.Fingerprint
            Structure fingerprint.
        ligand : str
            Ligand expo ID.
        feature_name : str
            Name of feature whose values shall be mapped onto the structure.
        show_side_chains : bool
            Show side chains.
        component_counter : int
            Latest NGLview component ID before adding new stuff.

        Returns
        -------
        view : nglview.widget.NGLWidget
            NGLview widget with all new components and representations.
        component_counter : int
            Latest NGLview component ID after having added new stuff.
        """

        # Residue coloring based on feature name
        if self._reference_structure_id == structure_id:
            plot_cmap = True
        else:
            plot_cmap = False
        residue_to_color = self.residue_to_color_mapping(feature_name, fingerprint, plot_cmap)
        color_scheme = nglview.color._ColorScheme(residue_to_color, label="scheme_regions")

        component = nglview.TextStructure(text, ext="pdb")
        view.add_component(component, name=f"{structure_id}")

        # Add protein with/without pocket side chains
        if show_side_chains:
            residue_ids = fingerprint.residue_ids
            selection = " or ".join([str(i) for i in residue_ids])
            view.clear_representations(component=component_counter)
            view.add_representation(
                "cartoon",
                selection="protein",
                color="grey",
                name=f"Structure",
                component=component_counter,
            )
            view.add_representation(
                "ball+stick",
                selection=selection,
                color=color_scheme,
                name=f"Pocket",
                component=component_counter,
            )
        else:
            view.clear_representations(component=component_counter)
            view.add_representation(
                "cartoon",
                selection="protein",
                color=color_scheme,
                name=f"Structure",
                component=component_counter,
            )

        # Add reference ligand
        view.add_representation(
            "licorice",
            selection=f"ligand and {ligand}",
            name=f"Ligand",
            component=component_counter,
        )

        component_counter += 1

        # Add pocket center
        if feature_name in fingerprint.subpocket_centers.columns:
            center = fingerprint.subpocket_centers[feature_name].to_list()
            view.shape.add_sphere(center, [0.0, 0.0, 0.0], 1, f"{feature_name}")
            component_counter += 1

        return view, component_counter

    def _residue_to_color_mapping(
        self,
        feature_name,
        data,
        discrete=False,
        cmap_name="viridis",
        label_prefix="",
        plot_cmap=False,
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
            All possible categories for discrete feature. Using the `PiYG` colormap.
        divergent : bool
            Use divergent colormap (PiYG) or sequential colormap (viridis)
        label_prefix : str
            Add prefix to color map label.
        plot_cmap : bool
            Plot color map (default: False).

        Returns
        -------
        list of list of [str, str]
            List of color-residue ID pairs. Color given as hex string, residue PDB IDs as string.
        """

        # Define norm
        # Discrete values and sequential colormap; fingerprint's discrete values
        if discrete and cmap_name == "viridis":
            discrete_options = self._discrete_feature_values[feature_name]
            norm = colors.Normalize(vmin=min(discrete_options), vmax=max(discrete_options))
            cmap = cm.get_cmap(cmap_name, len(discrete_options))
        # Continuous values and sequential colormap; fingerprint's continuous values
        elif not discrete and cmap_name == "viridis":
            norm = colors.Normalize(vmin=data.min(), vmax=data.max())
            cmap = cm.get_cmap(cmap_name)
        # Continuous values and sequential colormap; pair diff
        elif not discrete and cmap_name == "Blues":
            norm = colors.Normalize(vmin=data.min(), vmax=data.max())
            cmap = cm.get_cmap(cmap_name)
        # Continuous values and divergent colormap; pair diff (inactive at the moment)
        elif not discrete and cmap_name == "PiYG":
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
        residue_ids = data.index
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
        if plot_cmap:
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
