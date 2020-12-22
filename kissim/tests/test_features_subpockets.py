"""
Unit and regression test for kissim.encoding.features.subpockets class methods.
"""

import pytest

import numpy as np
import pandas as pd
from opencadd.databases.klifs import setup_remote

from kissim.io import PocketDataFrame
from kissim.definitions import SUBPOCKETS
from kissim.encoding.features import SubpocketsFeature

REMOTE = setup_remote()


class TestsSubpocketsFeature:
    """
    Test SubpocketsFeature class methods.
    """

    @pytest.mark.parametrize(
        "structure_id, remote, center_of_subpocket_centers",
        [
            (12347, REMOTE, [3.8522224, 19.975111, 38.359444]),
        ],
    )
    def test_from_structure_klifs_id(self, structure_id, remote, center_of_subpocket_centers):
        """
        Test if SubpocketsFeature can be set from KLIFS ID.
        """
        pocket = PocketDataFrame.from_structure_klifs_id(structure_id, klifs_session=remote)
        feature = SubpocketsFeature.from_pocket(pocket)
        assert isinstance(feature, SubpocketsFeature)
        # Test class attributes (_distances and _moments are tested separately)
        for residue_id, residue_ix in zip(feature._residue_ids, feature._residue_ixs):
            if residue_id is not None:
                assert isinstance(residue_id, int)
            assert isinstance(residue_ix, int)
        # Test subpockets
        assert pocket.subpockets["subpocket.name"].to_list() == list(SUBPOCKETS["subpocket.name"])
        assert pocket.subpockets["subpocket.color"].to_list() == list(
            SUBPOCKETS["subpocket.color"]
        )
        center_of_subpocket_centers_calculated = pocket.subpockets["subpocket.center"].mean()
        assert (
            pytest.approx(center_of_subpocket_centers_calculated, abs=1e-6)
            == center_of_subpocket_centers
        )
        # Test class properties
        assert isinstance(feature.values, dict)  # TODO checks for moments!
        assert isinstance(feature.details, dict)
        assert isinstance(feature.details["distances"], pd.DataFrame)
        assert isinstance(feature.details["moments"], pd.DataFrame)

    @pytest.mark.parametrize(
        "structure_id, remote, distances_mean, moments_mean",
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
    def test_distances_and_moments(self, structure_id, remote, distances_mean, moments_mean):
        """
        Test calculation of distances and moments for all subpockets.
        """

        pocket = PocketDataFrame.from_structure_klifs_id(structure_id, klifs_session=remote)
        feature = SubpocketsFeature.from_pocket(pocket)

        # Test distances
        distances_mean_calculated = {
            name: np.nanmean(distances) for name, distances in feature._distances.items()
        }
        assert pytest.approx(distances_mean_calculated, abs=1e-6) == distances_mean

        # Test moments
        moments_mean_calculated = {
            name: moments.mean() for name, moments in feature._moments.items()
        }
        assert pytest.approx(moments_mean_calculated, abs=1e-6) == moments_mean

    @pytest.mark.parametrize(
        "structure_id, remote, subpocket_center, mean_distance",
        [
            (12347, REMOTE, [0, 0, 0], 43.866110),
        ],
    )
    def test_calculate_distances_to_center(
        self, structure_id, remote, subpocket_center, mean_distance
    ):
        """
        Test calculation of distances between a subpocket center and all pocket residues.
        """
        pocket = PocketDataFrame.from_structure_klifs_id(structure_id, klifs_session=remote)
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
