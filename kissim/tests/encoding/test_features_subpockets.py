"""
Unit and regression test for the kissim.encoding.features.subpockets.SubpocketsFeature class.
"""

from pathlib import Path
import pytest

import numpy as np
import pandas as pd
from opencadd.databases.klifs import setup_local

from kissim.io import PocketDataFrame
from kissim.encoding.features import SubpocketsFeature

PATH_TEST_DATA = Path(__name__).parent / "kissim" / "tests" / "data"
LOCAL = setup_local(PATH_TEST_DATA / "KLIFS_download")


class TestsSubpocketsFeature:
    """
    Test SubpocketsFeature class methods.
    """

    @pytest.mark.parametrize(
        "structure_klifs_id, klifs_session",
        [
            (12347, LOCAL),
        ],
    )
    def test_from_pocket(self, structure_klifs_id, klifs_session):
        """
        Test if SubpocketsFeature can be set from a Pocket object.
        Test object attribues.
        """
        pocket = PocketDataFrame.from_structure_klifs_id(
            structure_klifs_id, klifs_session=klifs_session
        )
        feature = SubpocketsFeature.from_pocket(pocket)
        assert isinstance(feature, SubpocketsFeature)

        # Test class attributes
        # Note: _distances and _moments are tested separately below
        assert feature.name == structure_klifs_id
        for residue_id, residue_ix in zip(feature._residue_ids, feature._residue_ixs):
            if residue_id is not None:
                assert isinstance(residue_id, int)
            assert isinstance(residue_ix, int)

    @pytest.mark.parametrize(
        "structure_klifs_id, klifs_session, distances_mean, moments_mean, subpocket_centers_mean",
        [
            (
                12347,
                LOCAL,
                {
                    "hinge_region": 13.085004,
                    "dfg_region": 14.305939,
                    "front_pocket": 13.303858,
                    "center": 12.235948,
                },
                {
                    "hinge_region": 6.705098,
                    "dfg_region": 7.500417,
                    "front_pocket": 6.760902,
                    "center": 4.742250,
                },
                {
                    "hinge_region": 22.241111,
                    "dfg_region": 20.888666,
                    "front_pocket": 19.057000,
                    "center": 19.632547,
                },
            ),
        ],
    )
    def test_calculate_distances_and_moments(
        self,
        structure_klifs_id,
        klifs_session,
        distances_mean,
        moments_mean,
        subpocket_centers_mean,
    ):
        """
        Test calculation of distances and moments for all subpockets.

        We are testing here the class attributes _distances and _moments, whose values are the
        return values from the class methods calculate_distance and calculate_moments.
        """

        pocket = PocketDataFrame.from_structure_klifs_id(
            structure_klifs_id, klifs_session=klifs_session
        )
        feature = SubpocketsFeature.from_pocket(pocket)

        # Test distances
        for _, distances in feature._distances.items():
            assert isinstance(distances, list)
        distances_mean_calculated = {
            name: np.nanmean(distances) for name, distances in feature._distances.items()
        }
        assert pytest.approx(distances_mean_calculated, abs=1e-6) == distances_mean

        # Test moments
        for _, moments in feature._moments.items():
            assert isinstance(moments, list)
        moments_mean_calculated = {
            name: np.nanmean(moments) for name, moments in feature._moments.items()
        }
        assert pytest.approx(moments_mean_calculated, abs=1e-6) == moments_mean

        # Test subpocket centers
        for _, subpocket_center in feature._subpocket_centers.items():
            assert isinstance(subpocket_center, list)
        subpocket_centers_mean_calculated = {
            name: (np.nanmean(coordinates) if coordinates is not None else None)
            for name, coordinates in feature._subpocket_centers.items()
        }
        assert pytest.approx(subpocket_centers_mean_calculated, abs=1e-6) == subpocket_centers_mean

    @pytest.mark.parametrize(
        "structure_klifs_id, klifs_session",
        [(12347, LOCAL)],
    )
    def test_values(self, structure_klifs_id, klifs_session):
        """
        Test class property: values.
        """
        pocket = PocketDataFrame.from_structure_klifs_id(
            structure_klifs_id, klifs_session=klifs_session
        )
        feature = SubpocketsFeature.from_pocket(pocket)

        assert isinstance(feature.values, dict)
        assert feature.values == feature._moments
        # More tests on attribute _moments in test_calculate_distances_and_moments()

    @pytest.mark.parametrize(
        "structure_klifs_id, klifs_session",
        [(12347, LOCAL)],
    )
    def test_details(self, structure_klifs_id, klifs_session):
        """
        Test class property: details.
        """
        pocket = PocketDataFrame.from_structure_klifs_id(
            structure_klifs_id, klifs_session=klifs_session
        )
        feature = SubpocketsFeature.from_pocket(pocket)

        assert isinstance(feature.details, dict)

        # Distances
        assert isinstance(feature.details["distances"], pd.DataFrame)
        assert feature.details["distances"].index.to_list() == feature._residue_ixs
        # Test first column name and dtypes (the other columns depend on the input subpocket names)
        assert feature.details["distances"].columns.to_list()[0] == "residue.id"
        assert feature.details["distances"].dtypes[0] == "Int32"

        # Moments
        assert isinstance(feature.details["moments"], pd.DataFrame)
        assert feature.details["moments"].index.to_list() == [1, 2, 3]

    @pytest.mark.parametrize(
        "structure_klifs_id, klifs_session, subpockets, subpocket_center",
        [
            (
                12347,
                LOCAL,
                {
                    "anchor_residue.klifs_ids": [[12, 20]],
                    "subpocket.name": ["test"],
                    "subpocket.color": ["blue"],
                },
                [11.0105, 20.5705, 36.848],
            )
        ],
    )
    def test_add_subpockets(self, structure_klifs_id, klifs_session, subpockets, subpocket_center):
        """
        Test if subpockets are added correctly.
        """

        pocket = PocketDataFrame.from_structure_klifs_id(
            structure_klifs_id, klifs_session=klifs_session
        )
        feature = SubpocketsFeature()

        pocket = feature._add_subpockets(pocket, subpockets)
        assert pocket.subpockets.columns.to_list() == [
            "subpocket.name",
            "subpocket.color",
            "subpocket.center",
        ]
        assert pocket.subpockets["subpocket.name"].to_list() == subpockets["subpocket.name"]
        assert pocket.subpockets["subpocket.color"].to_list() == subpockets["subpocket.color"]
        subpocket_center_calculated = pocket.subpockets["subpocket.center"][0]
        for i, j in zip(subpocket_center_calculated, subpocket_center):
            assert pytest.approx(i, abs=1e-4) == j

    @pytest.mark.parametrize(
        "structure_klifs_id, klifs_session, subpocket_center, mean_distance",
        [
            (12347, LOCAL, [0, 0, 0], 43.866110),
            (12347, LOCAL, None, 43.866110),  # No center
        ],
    )
    def test_calculate_distances_to_center(
        self, structure_klifs_id, klifs_session, subpocket_center, mean_distance
    ):
        """
        Test calculation of distances between a subpocket center and all pocket residues.
        Test also the case that there is no subpocket center.
        """
        pocket = PocketDataFrame.from_structure_klifs_id(
            structure_klifs_id, klifs_session=klifs_session
        )
        feature = SubpocketsFeature.from_pocket(pocket)
        distances_calculated = feature._calculate_distances_to_center(pocket, subpocket_center)
        if subpocket_center is None:
            assert len(distances_calculated) == len(pocket._residue_ids)
            assert all(np.isnan(distances_calculated))
        else:
            mean_distance_calculated = np.nanmean(np.array(distances_calculated))
            assert pytest.approx(mean_distance_calculated, abs=1e-6) == mean_distance

    @pytest.mark.parametrize(
        "structure_klifs_id, klifs_session, residue_id, subpocket_center, distance",
        [
            (2542, LOCAL, 157, [0, 0, 0], 42.263037),  # Center and existing residue CA
            (2542, LOCAL, 1, [0, 0, 0], np.nan),  # Center and no residue (or residue CA)
            (2542, LOCAL, None, [0, 0, 0], np.nan),  # Center and residue is None
        ],
    )
    def test_calculate_distance_to_center(
        self, structure_klifs_id, klifs_session, residue_id, subpocket_center, distance
    ):
        """
        Test calculation of distances between a subpocket center and a pocket residues.
        """
        pocket = PocketDataFrame.from_structure_klifs_id(
            structure_klifs_id, klifs_session=klifs_session
        )
        feature = SubpocketsFeature.from_pocket(pocket)
        ca_atoms = pocket.ca_atoms
        distance_calculated = feature._calculate_distance_to_center(
            ca_atoms, residue_id, subpocket_center
        )
        if np.isnan(distance):
            assert np.isnan(distance_calculated)
        else:
            assert pytest.approx(distance_calculated, abs=1e-6) == distance
