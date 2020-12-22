"""
Unit and regression test for kissim.encoding.features.sitealign class methods.
"""

import pytest

import numpy as np
import pandas as pd
from opencadd.databases.klifs import setup_remote

from kissim.io import PocketBioPython
from kissim.encoding.features import SiteAlignFeature

REMOTE = setup_remote()


class TestsSiteAlignFeature:
    """
    Test SiteAlignFeature class methods.
    """

    @pytest.mark.parametrize(
        "structure_id, remote, feature_name",
        [
            (12347, REMOTE, "hba"),
            (12347, REMOTE, "hbd"),
            (12347, REMOTE, "size"),
            (12347, REMOTE, "charge"),
            (12347, REMOTE, "aliphatic"),
            (12347, REMOTE, "aromatic"),
        ],
    )
    def test_from_structure_klifs_id(self, structure_id, remote, feature_name):
        """
        Test if SiteAlignFeature can be set from KLIFS ID.
        """
        pocket = PocketBioPython.from_structure_klifs_id(structure_id, klifs_session=remote)
        feature = SiteAlignFeature.from_pocket(pocket, feature_name)
        assert isinstance(feature, SiteAlignFeature)
        # Test class attributes
        for residue_id, residue_ix, residue_name, category in zip(
            feature._residue_ids, feature._residue_ixs, feature._residue_names, feature._categories
        ):
            if residue_id is not None:
                assert isinstance(residue_id, int)
            assert isinstance(residue_ix, int)
            assert isinstance(feature_name, str)
            assert isinstance(category, float)
        # Test class properties
        assert isinstance(feature.values, list)
        for value in feature.values:
            assert isinstance(value, float)
        assert isinstance(feature.details, pd.DataFrame)
        assert feature.details.columns.to_list() == [
            "residue.id",
            "residue.name",
            "sitealign.category",
        ]

    @pytest.mark.parametrize(
        "structure_id, remote, feature_name",
        [(12347, REMOTE, "xxx")],
    )
    def test_from_structure_klifs_id_raises(self, structure_id, remote, feature_name):
        """
        Test if SiteAlignFeature raises error when passed an invalid feature name.
        """
        with pytest.raises(KeyError):
            pocket = PocketBioPython.from_structure_klifs_id(structure_id, klifs_session=remote)
            SiteAlignFeature.from_pocket(pocket, feature_name)

    @pytest.mark.parametrize(
        "residue_name, feature_name, value",
        [
            ("ALA", "size", np.float(1.0)),  # Size
            ("ASN", "size", np.float(2.0)),
            ("ARG", "size", np.float(3.0)),
            ("PTR", "size", np.float(3.0)),  # Converted non-standard
            ("MSE", "size", np.float(2.0)),  # Converted non-standard
            ("XXX", "size", np.nan),  # Non-convertable non-standard
            ("ALA", "hbd", np.float(0.0)),
            ("ASN", "hbd", np.float(1.0)),
            ("ARG", "hbd", np.float(3.0)),
            ("XXX", "hbd", np.nan),
            ("ALA", "hba", np.float(0.0)),
            ("ASN", "hba", np.float(1.0)),
            ("ASP", "hba", np.float(2.0)),
            ("XXX", "hba", np.nan),
            ("ALA", "charge", np.float(0.0)),
            ("ARG", "charge", np.float(1.0)),
            ("ASP", "charge", np.float(-1.0)),
            ("XXX", "charge", np.nan),
            ("ALA", "aromatic", np.float(0.0)),
            ("HIS", "aromatic", np.float(1.0)),
            ("XXX", "aromatic", np.nan),
            ("ARG", "aliphatic", np.float(0.0)),
            ("ALA", "aliphatic", np.float(1.0)),
            ("XXX", "aliphatic", np.nan),
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
        # Note: Cannot use == to compare np.nan values
        if np.isnan(value):
            assert np.isnan(value_calculated)
        else:
            assert value_calculated == value
