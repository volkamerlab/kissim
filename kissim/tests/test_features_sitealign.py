"""
Unit and regression test for kissim.encoding.features.sitealign class methods.
"""

import pytest

import numpy as np
import pandas as pd
from opencadd.databases.klifs import setup_remote

from kissim.encoding.features import SiteAlignFeature

REMOTE = setup_remote()


class TestsSiteAlignFeature:
    """
    Test SiteAlignFeature class methods.
    """

    @pytest.mark.parametrize(
        "structure_id, feature_name, remote",
        [
            (12347, "hba", REMOTE),
            (12347, "hbd", REMOTE),
            (12347, "size", REMOTE),
            (12347, "charge", REMOTE),
            (12347, "aliphatic", REMOTE),
            (12347, "aromatic", REMOTE),
        ],
    )
    def test_from_structure_klifs_id(self, structure_id, feature_name, remote):
        """
        Test if SiteAlignFeature can be set from KLIFS ID.
        """
        feature = SiteAlignFeature.from_structure_klifs_id(structure_id, feature_name, remote)
        assert isinstance(feature, SiteAlignFeature)
        # Test class attributes
        assert isinstance(feature._residue_ids, list)
        assert isinstance(feature._residue_names, list)
        assert isinstance(feature._categories, list)
        for residue_id, residue_name, category in zip(
            feature._residue_ids, feature._residue_names, feature._categories
        ):
            assert isinstance(residue_id, str)  # TODO: Cast this to int at some point
            assert isinstance(feature_name, str)
            assert isinstance(category, float)
        # Test class properties
        assert isinstance(feature.values, list)
        for value in feature.values:
            assert isinstance(value, float)
        assert isinstance(feature.details, pd.DataFrame)
        assert feature.details.columns.to_list() == ["residue.name", "sitealign.category"]

    @pytest.mark.parametrize(
        "structure_id, feature_name, remote",
        [(12347, "xxx", REMOTE)],
    )
    def test_from_structure_klifs_id_raises(self, structure_id, feature_name, remote):
        """
        Test if SiteAlignFeature raises error when passed an invalid feature name.
        """
        with pytest.raises(KeyError):
            SiteAlignFeature.from_structure_klifs_id(structure_id, feature_name, remote)

    @pytest.mark.parametrize(
        "residue_name, feature_name, value",
        [
            ("ALA", "size", 1),  # Size
            ("ASN", "size", 2),
            ("ARG", "size", 3),
            ("PTR", "size", 3),  # Converted non-standard
            ("MSE", "size", 2),  # Converted non-standard
            ("XXX", "size", None),  # Non-convertable non-standard
            ("ALA", "hbd", 0),
            ("ASN", "hbd", 1),
            ("ARG", "hbd", 3),
            ("XXX", "hbd", None),
            ("ALA", "hba", 0),
            ("ASN", "hba", 1),
            ("ASP", "hba", 2),
            ("XXX", "hba", None),
            ("ALA", "charge", 0),
            ("ARG", "charge", 1),
            ("ASP", "charge", -1),
            ("XXX", "charge", None),
            ("ALA", "aromatic", 0),
            ("HIS", "aromatic", 1),
            ("XXX", "aromatic", None),
            ("ARG", "aliphatic", 0),
            ("ALA", "aliphatic", 1),
            ("XXX", "aliphatic", None),
        ],
    )
    def test_residue_to_value(self, residue_name, feature_name, value):
        """
        Test function for retrieval of residue's size and pharmacophoric features
        (i.e. number of hydrogen bond donor,
        hydrogen bond acceptors, charge features, aromatic features or aliphatic features )

        Parameters
        ----------
        residue_name : str
            Three-letter code for residue.
        feature_name : str
            Feature type name.
        value : float or None
            Feature value.
        """

        feature = SiteAlignFeature()

        # Call feature from residue function
        value_calculated = feature._residue_to_value(residue_name, feature_name)
        if value_calculated:  # If not None
            assert isinstance(value_calculated, float)
        assert value == value_calculated
