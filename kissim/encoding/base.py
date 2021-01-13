"""
kissim.encoding.fingerprint

Defines the kissim fingerprint.
"""

import logging

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
    structure_klifs_id : str
        Structure KLIFS ID.
    values_dict : dict of pandas.DataFrame
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

        self.structure_klifs_id = None
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

    def values_array(self, physicochemical=True, spatial_distances=False, spatial_moments=True):
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
            physchem_features = self.values_dict["physicochemical"]
            physchem_features = np.array(list(physchem_features.values())).flatten()
            features.append(physchem_features)

        if spatial_distances:
            distances_features = self.values_dict["spatial"]["distances"]
            distances_features = np.array(list(distances_features.values())).flatten()
            features.append(distances_features)

        if spatial_moments:
            moments_features = self.values_dict["spatial"]["moments"]
            moments_features = np.array(list(moments_features.values())).flatten()
            features.append(moments_features)

        # Concatenate physicochemical and spatial features
        if len(features) > 0:
            features = np.concatenate(features, axis=0)
        else:
            features = np.array([])

        return features
