"""
kissim.encoding.fingerprint_base

Defines the base kissim fingerprint.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FingerprintBase:
    """
    Fingerprint encoding each of the 85 pre-aligned residues of a KLIFS kinase pocket
    w.r.t. the following features:
    - 8 physicochemical properties (8*85 = 680 bits)
    - Distances to 4 subpocket centers (4*85 = 340 bits)
    - The first 3 moments of respective per-subpocket distance distributions (3*4 = 12 bits)
    The default fingerprint consists of the physicochemical and spatial moment features.

    Attributes
    ----------
    structure_klifs_id : int  # TODO generalize (KLIFS-independent)?
        Structure KLIFS ID.
    kinase_name : str
        Kinase name.
    values_dict : dict of dict (of dict) of list of floats
        Fingerprint values, grouped in a nested dictionary by the following keys
        - "physicochemical"
          - "size", "hbd", "hba", "charge", "aromatic", "aliphatic", "sco", "exposure"
        - "spatial"
          - "distances"
            - "hinge_region", "dfg_region", "front_pocket", "center"
          - "moments"
            - "hinge_region", "dfg_region", "front_pocket", "center"
    residue_ids : list of int
        Pocket residue PDB IDs.
    residue_ixs : list of int
        Pocket residue KLIFS indices (alignment numbering).

    Properties
    ----------
    physicochemical
    distances
    moments

    Notes
    -----
    PHYSICOCHEMICAL features (85 x 8 matrix = 680 bits):

    - Size
    - Pharmacophoric features:
      Hydrogen bond donor, hydrogen bond acceptor, aromatic, aliphatic and charge feature
    - Side chain orientation
    - Solvent exposure

    SPATIAL features:

    - DISTANCE of each residue to 4 reference points (85 x 4 matrix = 340 bits):
      - Pocket center
      - Hinge region (subpocket center)
      - DFG region (subpocket center)
      - Front pocket (subpocket center)
    - MOMENTS for distance distributions for the 4 reference points (4 x 3 matrix = 12 bits):
      - Moment 1: Mean
      - Moment 2: Standard deviation
      - Moment 3: Skewness (cube root)

    The terminology used for the feature hierarchy is the following:
    - Feature category, e.g. spatial or physicochemical
    - Feature type, e.g. distance to centroid or size
    """

    def __init__(self):

        # TODO add more structure metadata? e.g. bound ligand?
        self.structure_klifs_id = None
        self.kinase_name = None
        self.values_dict = None
        self.residue_ids = None
        self.residue_ixs = None

    @property
    def physicochemical(self):
        """
        Physicochemical features.

        Returns
        -------
        pandas.DataFrame
            Feature per physicochemical property (columns) and pocket residue by KLIFS index
            (rows).
        """
        features = self.values_dict["physicochemical"]
        features = pd.DataFrame(features, index=self.residue_ixs)
        features = features[
            ["size", "hbd", "hba", "charge", "aromatic", "aliphatic", "sco", "exposure"]
        ]
        features.index.name = "residue.ix"
        return features

    @property
    def distances(self):
        """
        Spatial distance features.

        Returns
        -------
        pandas.DataFrame
            Distances per subpocket (columns) and pocket residue by KLIFS index (rows).
        """
        features = self.values_dict["spatial"]["distances"]
        features = pd.DataFrame(features, index=self.residue_ixs)
        features.index.name = "residue.ix"
        return features

    @property
    def moments(self):
        """
        Spatial moments features.

        Returns
        -------
        pandas.DataFrame
            First 3 moments (rows) of distance distributions per subpocket (columns).
        """
        features = self.values_dict["spatial"]["moments"]
        features = pd.DataFrame(features, index=[1, 2, 3])
        features.index.name = "moments"
        return features

    @property
    def subpocket_centers(self):
        """
        Subpocket centers' coordinates.

        Returns
        -------
        pandas.DataFrame
            Coordinates x, y, z (rows) for subpocket centers (columns).
        """

        subpocket_centers_dict = self.values_dict["spatial"]["subpocket_centers"]
        subpocket_centers_df = pd.DataFrame(subpocket_centers_dict, index=["x", "y", "z"])

        return subpocket_centers_df

    def values_array(self, physicochemical=True, spatial_distances=True, spatial_moments=True):
        """
        Get the full set or subset of features as 1D array.
        Default set of features includes physicochemical and spatial moments features.

        Parameters
        ----------
        physicochemical : bool
            Include physicochemical features (default: yes).
        spatial_distances : bool
            Include spatial distances features (default: no).
        spatial_moments : bool
            Include spatial moments features (default: yes).

        Returns
        -------
        numpy.ndarray
            1D fingerprint values.
        """

        features = []

        if physicochemical:
            physchem_features = self.physicochemical.to_numpy().flatten()
            features.append(physchem_features)

        if spatial_distances:
            distances_features = self.distances.to_numpy().flatten()
            features.append(distances_features)

        if spatial_moments:
            moments_features = self.moments.to_numpy().flatten()
            features.append(moments_features)

        # Concatenate physicochemical and spatial features
        if len(features) > 0:
            features = np.concatenate(features, axis=0)
        else:
            features = np.array([])

        return features

    @classmethod
    def from_json(cls, filepath):
        """
        Initiate the fingerprint from a json file.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Path to json file.
        """

        filepath = Path(filepath)
        with open(filepath, "r") as f:
            json_string = f.read()
        fingerprint_dict = json.loads(json_string)

        return cls._from_dict(fingerprint_dict)

    @classmethod
    def _from_dict(cls, fingerprint_dict):
        """
        Initiate the fingerprint from a dictionary containing the fingerprint class attributes.

        Parameters
        ----------
        fingerprint_dict : dict
            Fingerprint attributes in the form of a dictionary.
        """

        fingerprint = cls()
        fingerprint.structure_klifs_id = fingerprint_dict["structure_klifs_id"]
        fingerprint.kinase_name = fingerprint_dict["kinase_name"]
        fingerprint.values_dict = fingerprint_dict["values_dict"]
        fingerprint.residue_ids = fingerprint_dict["residue_ids"]
        fingerprint.residue_ixs = fingerprint_dict["residue_ixs"]
        return fingerprint

    def to_json(self, filepath):
        """
        Write Fingerprint class attributes to a json file.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Path to json file.
        """

        json_string = json.dumps(self.__dict__, indent=4)
        filepath = Path(filepath)
        with open(filepath, "w") as f:
            f.write(json_string)
