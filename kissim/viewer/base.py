"""
kissim.encoding.view

Handles fingerprint visualization in 3D.
"""


import numpy as np
from matplotlib import cm, colors
import nglview
from ipywidgets import interact
import ipywidgets as widgets
from opencadd.databases.klifs import setup_remote

from kissim.encoding import Fingerprint


class Viewer:
    def __init__(self, structure_klifs_id):

        self._text = None
        self._fingerprint = None
        self._feature_names = None

        klifs_session = setup_remote()
        text = klifs_session.coordinates.to_text(structure_klifs_id, "complex", "pdb")
        fingerprint = Fingerprint.from_structure_klifs_id(
            structure_klifs_id, klifs_session=klifs_session
        )
        feature_names = (
            fingerprint.physicochemical.columns.to_list() + fingerprint.distances.columns.to_list()
        )

        self._text = text
        self._fingerprint = fingerprint
        self._feature_names = feature_names

    def _show_discrete(self, feature_name, show_side_chains=True):

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

        func = interact(
            self._show_discrete,
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

    def _discrete_residue_to_color_mapping(self, feature_name):

        feature_values = {
            "size": [1.0, 2.0, 3.0],
            "hbd": [0.0, 1.0, 2.0, 3.0],
            "hba": [0.0, 1.0, 2.0],
            "charge": [-1.0, 0.0, 1.0],
            "aromatic": [0.0, 1.0],
            "aliphatic": [0.0, 1.0],
            "sco": [1.0, 2.0, 3.0],
            "exposure": [1.0, 2.0, 3.0],
        }

        viridis = cm.get_cmap("viridis", len(feature_values[feature_name]))
        value_to_color = {
            value: colors.rgb2hex(color)
            for value, color in zip(feature_values[feature_name], viridis.colors)
        }

        features = self._fingerprint.physicochemical
        features.index = self._fingerprint.residue_ids
        features = features[feature_name]

        residue_to_color = []
        for residue_id, value in zip(features.index, features.values):
            try:
                color = value_to_color[value]
            except KeyError:
                color = "#808080"
            residue_to_color.append([color, str(residue_id)])

        return residue_to_color

    def _continuous_residue_to_color_mapping(self, feature_name):

        features = self._fingerprint.distances
        features = features / features.max().max()
        features.index = self._fingerprint.residue_ids
        features = features[feature_name]

        viridis = cm.get_cmap("viridis", 1000)

        residue_to_color = []
        for residue_id, value in zip(features.index, features.values):
            if np.isnan(value):
                color = "#808080"
            else:
                color = viridis(value)
            residue_to_color.append([colors.rgb2hex(color), str(residue_id)])

        return residue_to_color

    def residue_to_color_mapping(self, feature_name):

        if feature_name in self._fingerprint.physicochemical:
            residue_to_color = self._discrete_residue_to_color_mapping(feature_name)
        elif feature_name in self._fingerprint.distances:
            residue_to_color = self._continuous_residue_to_color_mapping(feature_name)
        else:
            raise ValueError(f"Feature name {feature_name} unknown.")

        return residue_to_color
