"""
kissim.encoding.fingerprint

Defines the kissim fingerprint.
"""

import logging

from kissim.io import KlifsToKissimData, PocketBioPython, PocketDataFrame
from kissim.encoding import FingerprintBase
from kissim.encoding.features import (
    SiteAlignFeature,
    SideChainOrientationFeature,
    SolventExposureFeature,
    SubpocketsFeature,
)

logger = logging.getLogger(__name__)


class Fingerprint(FingerprintBase):
    @classmethod
    def from_structure_klifs_id(cls, structure_klifs_id, klifs_session=None):
        """
        Calculate fingerprint for a KLIFS structure (by structure KLIFS ID).

        Parameters
        ----------
        structure_klifs_id : int
            Structure KLIFS ID.
        klifs_session : opencadd.databases.klifs.session.Session or None
            Local or remote KLIFS session.
            If None (default), set up remote KLIFS session.

        Returns
        -------
        kissim.encoding.Fingerprint
            Fingerprint.
        """

        try:
            data = KlifsToKissimData.from_structure_klifs_id(structure_klifs_id, klifs_session)
            if data is None:
                logger.warning(f"{structure_klifs_id}: Empty fingerprint (data unaccessible).")
                fingerprint = None
            else:
                fingerprint = cls.from_text(
                    data.text,
                    data.extension,
                    data.residue_ids,
                    data.residue_ixs,
                    data.structure_klifs_id,
                    data.kinase_name,
                )
            return fingerprint
        except Exception as e:
            logger.error(f"Fingerprint generation throw error for {structure_klifs_id}:  {e}")

    @classmethod
    def from_text(cls, text, extension, residue_ids, residue_ixs, structure_name, kinase_name):
        """
        Calculate fingerprint for a KLIFS structure (by complex data as text and pocket residue
        IDs and indices).

        Parameters
        ----------
        text : str
            Structural complex data as string (file content).
        extension : str
            Structural complex data format (file extension).
        residue_ids : list of int
            Pocket residue IDs.
        residue_ixs : list of int
            Pocket residue indices.
        structure_name : str  # TODO or structure_klifs_id?
            Structure name.
        kinase_name : str
            Kinase name.

        Returns
        -------
        kissim.encoding.Fingerprint
            Fingerprint.
        """

        # BioPython-based and DataFrame-based pocket are both necessary for fingerprint features
        pocket_bp = PocketBioPython.from_text(
            text, extension, residue_ids, residue_ixs, structure_name
        )
        pocket_df = PocketDataFrame.from_text(
            text, extension, residue_ids, residue_ixs, structure_name
        )
        if pocket_bp is None or pocket_df is None:
            logger.warning(f"{structure_name}: Empty fingerprint (pocket unaccessible).")
            fingerprint = None
        else:
            fingerprint = cls()
            fingerprint.structure_klifs_id = structure_name
            fingerprint.kinase_name = kinase_name
            fingerprint.residue_ids = pocket_bp._residue_ids
            fingerprint.residue_ixs = pocket_bp._residue_ixs
            values_dict = {}
            values_dict["physicochemical"] = fingerprint._get_physicochemical_features_dict(
                pocket_bp
            )
            values_dict["spatial"] = fingerprint._get_spatial_features_dict(pocket_df)
            fingerprint.values_dict = values_dict

        return fingerprint

    def _get_physicochemical_features_dict(self, pocket_bp):
        """
        Get physicochemical features.

        Parameters
        ----------
        pocket_bp : kissim.io.PocketBioPython
            Biopython-based pocket object.

        Returns
        -------
        dict of list of float
            Feature values (values) for physicochemical properties (keys).
        """

        # Set up physicochemical features
        features = {}
        # Add SiteAlign features
        for sitealign_feature_name in ["size", "hbd", "hba", "charge", "aromatic", "aliphatic"]:
            feature = SiteAlignFeature.from_pocket(pocket_bp, sitealign_feature_name)
            features[sitealign_feature_name] = feature.values
        # Add side chain orientation feature
        feature = SideChainOrientationFeature.from_pocket(pocket_bp)
        features["sco"] = feature.values
        features["sco.vertex_angle"] = feature._vertex_angles
        # Add solvent exposure feature
        feature = SolventExposureFeature.from_pocket(pocket_bp)
        features["exposure"] = feature.values
        features["exposure.ratio"] = feature._ratio

        return features

    def _get_spatial_features_dict(self, pocket_df):
        """
        Get spatial features (distances and moments).

        Parameters
        ----------
        pocket_df : kissim.io.PocketDataFrame
            DataFrame-based pocket object.

        Returns
        -------
        dict of list of float
            Per-subpocket feature values [and coordinates] (values) for distances and moments
            [and subpocket centers] (keys).
        """

        # Set up spatial features
        features = {}
        # Add subpockets features
        feature = SubpocketsFeature.from_pocket(pocket_df)
        features["distances"] = feature._distances
        features["moments"] = feature._moments
        features["subpocket_centers"] = feature._subpocket_centers

        return features
