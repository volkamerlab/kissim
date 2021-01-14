"""
Unit and regression test for kissim.encoding.fingerprint.Fingerprint and its
parent class kissim.encoding.base.FingerprintBase.
"""

from pathlib import Path
import pytest

import numpy as np
from opencadd.databases.klifs import setup_local, setup_remote

from kissim.utils import enter_temp_directory
from kissim.io import PocketBioPython, PocketDataFrame
from kissim.encoding import Fingerprint

PATH_TEST_DATA = Path(__name__).parent / "kissim" / "tests" / "data"
REMOTE = setup_remote()
LOCAL = setup_local(PATH_TEST_DATA / "KLIFS_download")

FEATURE_NAMES = ["physicochemical", "spatial"]
FEATURE_NAMES_PHYSICOCHEMICAL = [
    "size",
    "hbd",
    "hba",
    "charge",
    "aromatic",
    "aliphatic",
    "sco",
    "exposure",
]
FEATURE_NAMES_SPATIAL = ["distances", "moments"]
FEATURE_NAMES_DISTANCES_AND_MOMENTS = [
    "hinge_region",
    "dfg_region",
    "front_pocket",
    "center",
]


class TestFingerprint:
    """
    Test common functionalities in the PocketBioPython and PocketDataFrame classes.
    """

    @pytest.mark.parametrize(
        "structure_klifs_id",
        [109, 110, 118],
    )
    def test_from_structure_klifs_id(self, structure_klifs_id):
        """
        Test if Fingerprint can be set locally and remotely.
        """

        fingerprint1 = Fingerprint.from_structure_klifs_id(structure_klifs_id, LOCAL)
        fingerprint2 = Fingerprint.from_structure_klifs_id(structure_klifs_id, REMOTE)

        # Check if locally and remotely obtained fingerprints are the same
        # Use method values_array()
        assert np.allclose(
            fingerprint1.values_array(True, True, True),
            fingerprint2.values_array(True, True, True),
            rtol=0,
            atol=0,
            equal_nan=True,
        )

        # Test attributes
        # Attribute structure_klifs_id
        assert fingerprint1.structure_klifs_id == structure_klifs_id
        assert fingerprint2.structure_klifs_id == structure_klifs_id
        # Attribute values_dict
        assert list(fingerprint1.values_dict.keys()) == FEATURE_NAMES
        assert (
            list(fingerprint1.values_dict["physicochemical"].keys())
            == FEATURE_NAMES_PHYSICOCHEMICAL
        )
        assert list(fingerprint1.values_dict["spatial"].keys()) == FEATURE_NAMES_SPATIAL
        assert (
            list(fingerprint1.values_dict["spatial"]["distances"].keys())
            == FEATURE_NAMES_DISTANCES_AND_MOMENTS
        )
        assert (
            list(fingerprint1.values_dict["spatial"]["moments"].keys())
            == FEATURE_NAMES_DISTANCES_AND_MOMENTS
        )
        # Attribute residue_ids
        assert fingerprint1.residue_ids == fingerprint2.residue_ids
        # Attribute residue_ixs
        assert fingerprint1.residue_ixs == fingerprint2.residue_ixs

    @pytest.mark.parametrize(
        "structure_klifs_id, values_array_mean",
        [(109, 4.7811), (12347, 4.9610)],
    )
    def test_values_array(self, structure_klifs_id, values_array_mean):
        """
        Tets fingerprint values array.
        """

        fingerprint = Fingerprint.from_structure_klifs_id(structure_klifs_id, REMOTE)
        values_array_mean_calculated = np.nanmean(fingerprint.values_array(True, True, True))
        assert pytest.approx(values_array_mean_calculated, abs=1e-4) == values_array_mean

    @pytest.mark.parametrize(
        "structure_klifs_id",
        [109],
    )
    def test_physicochemical(self, structure_klifs_id):
        """
        Test DataFrame columns/index names.
        """

        fingerprint = Fingerprint.from_structure_klifs_id(structure_klifs_id, REMOTE)
        assert fingerprint.physicochemical.columns.to_list() == FEATURE_NAMES_PHYSICOCHEMICAL
        assert fingerprint.physicochemical.index.to_list() == list(range(1, 86))
        assert fingerprint.physicochemical.index.name == "residue.ix"

    @pytest.mark.parametrize(
        "structure_klifs_id",
        [109],
    )
    def test_distances(self, structure_klifs_id):
        """
        Test DataFrame columns/index names.
        """

        fingerprint = Fingerprint.from_structure_klifs_id(structure_klifs_id, REMOTE)
        assert fingerprint.distances.columns.to_list() == FEATURE_NAMES_DISTANCES_AND_MOMENTS
        assert fingerprint.distances.index.to_list() == list(range(1, 86))
        assert fingerprint.distances.index.name == "residue.ix"

    @pytest.mark.parametrize(
        "structure_klifs_id",
        [109],
    )
    def test_moments(self, structure_klifs_id):
        """
        Test DataFrame columns/index names.
        """

        fingerprint = Fingerprint.from_structure_klifs_id(structure_klifs_id, REMOTE)
        assert fingerprint.moments.columns.to_list() == FEATURE_NAMES_DISTANCES_AND_MOMENTS
        assert fingerprint.moments.index.to_list() == [1, 2, 3]
        assert fingerprint.moments.index.name == "moments"

    @pytest.mark.parametrize(
        "structure_klifs_id",
        [109],
    )
    def test_values_array(self, structure_klifs_id):
        """
        Test the different lengths of the final fingerprint based on the selection of
        physicochemical, distances and moments features.
        """

        fingerprint = Fingerprint.from_structure_klifs_id(structure_klifs_id, REMOTE)
        assert fingerprint.values_array(False, False, False).size == 0
        assert fingerprint.values_array(True, False, False).size == 680
        assert fingerprint.values_array(False, True, False).size == 340
        assert fingerprint.values_array(False, False, True).size == 12
        assert fingerprint.values_array(True, True, False).size == 1020
        assert fingerprint.values_array(True, False, True).size == 692
        assert fingerprint.values_array(False, True, True).size == 352
        assert fingerprint.values_array(True, True, True).size == 1032

    @pytest.mark.parametrize(
        "structure_klifs_id",
        [109],
    )
    def test_to_from_json(self, structure_klifs_id):
        """
        Test if saving/loading a fingerprint to/from a json file.
        """

        fingerprint = Fingerprint.from_structure_klifs_id(structure_klifs_id, REMOTE)
        json_filepath = Path("fingerprint.json")

        with enter_temp_directory():

            # Save json file
            fingerprint.to_json(json_filepath)
            assert json_filepath.exists()

            # Load json file
            fingerprint_reloaded = Fingerprint.from_json(json_filepath)
            # Test if class attributes from ID and from json are the same
            assert fingerprint.structure_klifs_id == fingerprint_reloaded.structure_klifs_id
            assert np.allclose(
                fingerprint.values_array(True, True, True),
                fingerprint_reloaded.values_array(True, True, True),
                rtol=0,
                atol=0,
                equal_nan=True,
            )
            assert fingerprint.residue_ids == fingerprint_reloaded.residue_ids
            assert fingerprint.residue_ixs == fingerprint_reloaded.residue_ixs

    @pytest.mark.parametrize(
        "structure_klifs_id",
        [109],
    )
    def test_get_pocket(self, structure_klifs_id):
        """
        Test pocket generation based on biopython and pandas.
        """

        fingerprint = Fingerprint()
        pocket_bp, pocket_df = fingerprint._get_pocket(structure_klifs_id, REMOTE)
        assert isinstance(pocket_bp, PocketBioPython)
        assert isinstance(pocket_df, PocketDataFrame)

    @pytest.mark.parametrize(
        "structure_klifs_id",
        [109],
    )
    def test_get_physicochemical_or_spatial_features_dict(self, structure_klifs_id):
        """
        Test if physicochemical an spatial features dictionary has correct keys.
        """

        fingerprint = Fingerprint()
        pocket_bp, pocket_df = fingerprint._get_pocket(structure_klifs_id, REMOTE)

        # Physicochemical features
        physicochemical_dict = fingerprint._get_physicochemical_features_dict(pocket_bp)
        assert isinstance(physicochemical_dict, dict)
        assert list(physicochemical_dict.keys()) == FEATURE_NAMES_PHYSICOCHEMICAL

        # Spatial features
        spatial_dict = fingerprint._get_spatial_features_dict(pocket_df)
        assert isinstance(spatial_dict, dict)
        assert list(spatial_dict.keys()) == FEATURE_NAMES_SPATIAL
        assert list(spatial_dict["distances"].keys()) == FEATURE_NAMES_DISTANCES_AND_MOMENTS
        assert list(spatial_dict["moments"].keys()) == FEATURE_NAMES_DISTANCES_AND_MOMENTS
