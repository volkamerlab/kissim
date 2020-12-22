"""
Unit and regression test for kissim.encoding.features.sco class methods.
"""

import pytest

import numpy as np
import pandas as pd
import Bio
from opencadd.databases.klifs import setup_remote

from kissim.io import PocketBioPython
from kissim.encoding.features import SideChainOrientationFeature

REMOTE = setup_remote()


class TestsSideChainOrientationFeature:
    """
    Test SideChainOrientationFeature class methods.
    """

    @pytest.mark.parametrize(
        "structure_id, remote",
        [(12347, REMOTE)],
    )
    def test_from_structure_klifs_id(self, structure_id, remote):
        """
        Test if SideChainOrientationFeature can be set from KLIFS ID.
        """
        pocket = PocketBioPython.from_structure_klifs_id(structure_id, remote)
        feature = SideChainOrientationFeature.from_pocket(pocket)
        assert isinstance(feature, SideChainOrientationFeature)
        # Test class attributes
        for residue_id, residue_ix, category, vertex_angle in zip(
            feature._residue_ids, feature._residue_ixs, feature._categories, feature._vertex_angles
        ):
            if residue_id is not None:
                assert isinstance(residue_id, int)
            assert isinstance(residue_ix, int)
            assert isinstance(category, float)
            assert isinstance(vertex_angle, float)
        assert isinstance(feature._pocket_center, Bio.PDB.vectors.Vector)
        assert isinstance(feature._ca_atoms, list)
        for ca_atom in feature._ca_atoms:
            if ca_atom:
                assert isinstance(ca_atom, Bio.PDB.vectors.Vector)
        assert isinstance(feature._sc_atoms, list)
        for sc_atom in feature._sc_atoms:
            if sc_atom:
                assert isinstance(sc_atom, Bio.PDB.vectors.Vector)

    @pytest.mark.parametrize(
        "structure_id, remote, values_mean",
        [(12347, REMOTE, 1.440678)],
    )
    def test_values(self, structure_id, remote, values_mean):
        """
        Test class property: side chain orientation values.
        The mean refers to the mean of non-NaN values.
        """
        pocket = PocketBioPython.from_structure_klifs_id(structure_id, remote)
        feature = SideChainOrientationFeature.from_pocket(pocket)
        assert isinstance(feature.values, list)
        values_mean_calculated = pd.Series(feature.values).dropna().mean()
        assert values_mean == pytest.approx(values_mean_calculated)

    @pytest.mark.parametrize(
        "structure_id, remote",
        [(12347, REMOTE)],
    )
    def test_details(self, structure_id, remote):
        """
        Test class property: side chain orientation details.
        """
        pocket = PocketBioPython.from_structure_klifs_id(structure_id, remote)
        feature = SideChainOrientationFeature.from_pocket(pocket)
        assert isinstance(feature.details, pd.DataFrame)
        assert feature.details.columns.to_list() == [
            "residue.id",
            "sco.category",
            "sco.angle",
            "ca.vector",
            "sc.vector",
            "pocket_center.vector",
        ]
        assert feature.details.index.to_list() == feature._residue_ixs

    @pytest.mark.parametrize(
        "vector1, vector2, vector3, vertex_angle",
        [
            (
                Bio.PDB.vectors.Vector(0, 1, 0),
                Bio.PDB.vectors.Vector(0, 0, 0),
                Bio.PDB.vectors.Vector(0, 0, 1),
                90.0,
            )
        ],
    )
    def test_calculate_vertex_angle(self, vector1, vector2, vector3, vertex_angle):
        """
        Test if vertex angles are calculated correctly.
        """
        feature = SideChainOrientationFeature()
        vertex_angle_calculated = feature._calculate_vertex_angle(vector1, vector2, vector3)
        assert vertex_angle == pytest.approx(vertex_angle_calculated)

    @pytest.mark.parametrize(
        "vertex_angle, category",
        [
            (0.0, 0.0),
            (1.0, 0.0),
            (45.0, 0.0),
            (46.0, 1.0),
            (90.0, 1.0),
            (91.0, 2.0),
            (180.0, 2.0),
            (np.nan, np.nan),
        ],
    )
    def test_get_category(self, vertex_angle, category):
        """
        Test tranformation of vertex angle to category (for side chain orientation).
        """

        feature = SideChainOrientationFeature()
        category_calculated = feature._get_category(vertex_angle)

        if not np.isnan(vertex_angle):
            assert isinstance(category_calculated, float)
            assert category == category_calculated
        else:
            assert np.isnan(category_calculated)
