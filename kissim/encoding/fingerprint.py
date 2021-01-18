"""
kissim.encoding.fingerprint

Defines the kissim fingerprint.
"""

import logging

from bravado_core.exception import SwaggerMappingError
from opencadd.databases.klifs import setup_remote

from kissim.io import PocketBioPython, PocketDataFrame
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
        """
        if klifs_session is None:
            klifs_session = setup_remote()

        # Check if structure KLIFS ID exists
        if klifs_session._client:
            try:
                klifs_session.structures.by_structure_klifs_id(structure_klifs_id)
            except SwaggerMappingError as e:
                logger.warning(
                    f"Unknown structure KLIFS ID (remotely): {structure_klifs_id} (SwaggerMappingError: {e})"
                )
                return None
        else:
            try:
                klifs_session.structures.by_structure_klifs_id(structure_klifs_id)
            except ValueError as e:
                logger.warning(
                    f"Unknown structure KLIFS ID (locally): {structure_klifs_id} (ValueError: {e})"
                )
                return None

        fingerprint = cls()

        # Get pocket
        pocket_bp, pocket_df = fingerprint._get_pocket(structure_klifs_id, klifs_session)
        if pocket_bp is None or pocket_df is None:
            # If a pocket is None, fingerprint shall be None
            fingerprint = None
        else:
            # Check if residues are consistent between pockets
            if pocket_bp._residue_ids != pocket_df._residue_ids:
                raise ValueError(f"Residue PDB IDs are not the same for df and bp pockets.")
            if pocket_bp._residue_ixs != pocket_df._residue_ixs:
                raise ValueError(f"Residue indices are not the same for df and bp pockets.")
            # Set residue attributes
            fingerprint.structure_klifs_id = structure_klifs_id
            fingerprint.residue_ids = pocket_bp._residue_ids
            fingerprint.residue_ixs = pocket_bp._residue_ixs

            values_dict = {}
            values_dict["physicochemical"] = fingerprint._get_physicochemical_features_dict(pocket_bp)
            values_dict["spatial"] = fingerprint._get_spatial_features_dict(pocket_df)
            fingerprint.values_dict = values_dict

        return fingerprint

    def _get_pocket(self, structure_klifs_id, klifs_session):
        """
        Get DataFrame and BioPython-based pocket objects from a structure KLIFS ID.
        TODO do not fetch data from KLIFS twice!!!

        Parameters
        ----------
        structure_klifs_id : int
            Structure KLIFS ID.
        klifs_session : opencadd.databases.klifs.session.Session
            Local or remote KLIFS session.

        Returns
        -------
        pocket_bp : kissim.io.PocketBioPython
            Biopython-based pocket object.
        pocket_df : kissim.io.PocketDataFrame
            DataFrame-based pocket object.
        """

        # Set up BioPython-based pocket
        pocket_bp = PocketBioPython.from_structure_klifs_id(
            structure_klifs_id, klifs_session=klifs_session
        )
        # Set up DataFrame-based pocket
        pocket_df = PocketDataFrame.from_structure_klifs_id(
            structure_klifs_id, extension="pdb", klifs_session=klifs_session
        )
        return pocket_bp, pocket_df

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
        # Add solvent exposure feature
        feature = SolventExposureFeature.from_pocket(pocket_bp)
        features["exposure"] = feature.values

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
            Per-subpocket feature values (values) for distances and moments (keys).
        """

        # Set up spatial features
        features = {}
        # Add subpockets features
        feature = SubpocketsFeature.from_pocket(pocket_df)
        features["distances"] = feature._distances
        features["moments"] = feature._moments

        return features
