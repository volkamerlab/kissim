"""
Unit and regression test for the kissim.encoding.features.subpockets.SubpocketsFeature class.
"""

import pytest

import numpy as np
import pandas as pd
from opencadd.databases.klifs import setup_remote

from kissim.io import PocketDataFrame
from kissim.encoding.features import SubpocketsFeature

REMOTE = setup_remote()


class TestsSubpocketsFeature:
    """
    Test SubpocketsFeature class methods.
    """

    @pytest.mark.parametrize(
        "structure_klifs_id, remote",
        [
            (12347, REMOTE),
        ],
    )
    def test_from_pocket(self, structure_klifs_id, remote):
        """
        Test if SubpocketsFeature can be set from a Pocket object.
        Test object attribues.
        """
        pocket = PocketDataFrame.from_structure_klifs_id(structure_klifs_id, klifs_session=remote)
        feature = SubpocketsFeature.from_pocket(pocket)
        assert isinstance(feature, SubpocketsFeature)

        # Test class attributes
        # Note: _distances and _moments are tested separately below
        for residue_id, residue_ix in zip(feature._residue_ids, feature._residue_ixs):
            if residue_id is not None:
                assert isinstance(residue_id, int)
            assert isinstance(residue_ix, int)

    @pytest.mark.parametrize(
        "structure_klifs_id, remote, distances_mean, moments_mean",
        [
            (
                12347,
                REMOTE,
                {
                    "hinge_region": 13.0850042,
                    "dfg_region": 14.3059388,
                    "front_pocket": 13.3038583,
                    "center": 12.2359484,
                },
                {
                    "hinge_region": 6.7050982,
                    "dfg_region": 7.5004165,
                    "front_pocket": 6.7609020,
                    "center": 4.7422497,
                },
            ),
        ],
    )
    def test_calculate_distances_and_moments(
        self, structure_klifs_id, remote, distances_mean, moments_mean
    ):
        """
        Test calculation of distances and moments for all subpockets.

        We are testing here the class attributes _distances and _moments, whose values are the
        return values from the class methods calculate_distance and calculate_moments.
        """

        pocket = PocketDataFrame.from_structure_klifs_id(structure_klifs_id, klifs_session=remote)
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

    @pytest.mark.parametrize(
        "structure_klifs_id, remote",
        [(12347, REMOTE)],
    )
    def test_values(self, structure_klifs_id, remote):
        """
        Test class property: values.
        """
        pocket = PocketDataFrame.from_structure_klifs_id(structure_klifs_id, klifs_session=remote)
        feature = SubpocketsFeature.from_pocket(pocket)

        assert isinstance(feature.values, dict)
        assert feature.values == feature._moments
        # More tests on attribute _moments in test_calculate_distances_and_moments()

    @pytest.mark.parametrize(
        "structure_klifs_id, remote",
        [(12347, REMOTE)],
    )
    def test_details(self, structure_klifs_id, remote):
        """
        Test class property: details.
        """
        pocket = PocketDataFrame.from_structure_klifs_id(structure_klifs_id, klifs_session=remote)
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
        "structure_klifs_id, remote, subpockets, subpocket_center",
        [
            (
                12347,
                REMOTE,
                {
                    "anchor_residue.klifs_ids": [[12, 20]],
                    "subpocket.name": ["test"],
                    "subpocket.color": ["blue"],
                },
                [11.0105, 20.5705, 36.848],
            )
        ],
    )
    def test_add_subpockets(self, structure_klifs_id, remote, subpockets, subpocket_center):
        """
        Test if subpockets are added correctly.
        """

        pocket = PocketDataFrame.from_structure_klifs_id(structure_klifs_id, klifs_session=remote)
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
        "structure_klifs_id, remote, subpocket_center, mean_distance",
        [
            (12347, REMOTE, [0, 0, 0], 43.866110),
        ],
    )
    def test_calculate_distances_to_center(
        self, structure_klifs_id, remote, subpocket_center, mean_distance
    ):
        """
        Test calculation of distances between a subpocket center and all pocket residues.
        """
        pocket = PocketDataFrame.from_structure_klifs_id(structure_klifs_id, klifs_session=remote)
        feature = SubpocketsFeature.from_pocket(pocket)
        distances_calculated = feature._calculate_distances_to_center(pocket, subpocket_center)
        mean_distance_calculated = np.nanmean(np.array(distances_calculated))
        assert pytest.approx(mean_distance_calculated, abs=1e-6) == mean_distance

    @pytest.mark.parametrize(
        "values, moments",
        [
            ([0], [0, 0, 0]),
            ([0, 0], [0, 0, 0]),
            ([1, 0], [0.5, 0.5, 0]),
            ([3, 0, 0], [1, 1.4142135, 1.2599210]),
        ],
    )
    def test_calculate_first_second_third_moment(self, values, moments):
        """
        Test static method that calculates the first three moments of a distribution.
        """
        feature = SubpocketsFeature()
        moments_calculated = feature.calculate_first_second_third_moments(values)
        assert pytest.approx(moments_calculated, abs=1e-6) == moments
