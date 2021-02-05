"""
Unit and regression test for kissim.encoding.FingerprintGenerator.
"""

from pathlib import Path
import pytest

import numpy as np
from opencadd.databases.klifs import setup_local, setup_remote

from kissim.utils import enter_temp_directory
from kissim.encoding import Fingerprint, FingerprintNormalized, FingerprintGenerator
from kissim.tests.encoding.schema import (
    FEATURE_NAMES_PHYSICOCHEMICAL,
    FEATURE_NAMES_PHYSICOCHEMICAL_DICT,
    FEATURE_NAMES_DISTANCES_AND_MOMENTS,
)

PATH_TEST_DATA = Path(__name__).parent / "kissim" / "tests" / "data"
REMOTE = setup_remote()
LOCAL = setup_local(PATH_TEST_DATA / "KLIFS_download")


class TestFingerprintGenerator:
    """
    Test common functionalities in the PocketBioPython and PocketDataFrame classes.
    """

    @pytest.mark.parametrize(
        "structure_klifs_ids, klifs_session, n_cores, fingerprints_values_array_sum",
        [
            ([110, 118], REMOTE, 1, 10152.4256),
            ([110, 118], REMOTE, 2, 10152.4256),
            ([110, 118], LOCAL, 1, 10152.4256),
            ([110, 118], LOCAL, 2, 10152.4256),
            ([110, 118], None, None, 10152.4256),
        ],
    )
    def test_from_structure_klifs_id(
        self, structure_klifs_ids, klifs_session, n_cores, fingerprints_values_array_sum
    ):
        """
        Test if fingerprints can be generated locally and remotely in sequence and in parallel.
        """

        fingerprints = FingerprintGenerator.from_structure_klifs_ids(
            structure_klifs_ids, klifs_session, n_cores
        )
        # Test attributes
        # Attribute: structure_klifs_id
        assert fingerprints.structure_klifs_ids == structure_klifs_ids
        # Attribute: klifs_session
        if klifs_session is not None:
            assert fingerprints.klifs_session == klifs_session
        else:
            # If no session was provided, use set up remote session
            assert fingerprints.klifs_session._client is not None
        # Attribute: data
        assert isinstance(fingerprints.data, dict)
        for key, value in fingerprints.data.items():
            assert isinstance(key, int)
            assert isinstance(value, Fingerprint)
        fingerprints_values_array_sum_calculated = sum(
            [
                np.nansum(fingerprint.values_array(True, True, True))
                for structure_klifs_id, fingerprint in fingerprints.data.items()
            ]
        )
        assert (
            pytest.approx(fingerprints_values_array_sum_calculated, abs=1e-4)
            == fingerprints_values_array_sum
        )
        # Attribute: data_normalized
        assert isinstance(fingerprints.data_normalized, dict)
        for key, value in fingerprints.data_normalized.items():
            assert isinstance(key, int)
            assert isinstance(value, FingerprintNormalized)

    @pytest.mark.parametrize(
        "n_cores",
        [1000000000000],
    )
    def test_get_n_cores_valueerror(self, n_cores):
        """
        Test if number of cores are set correctly.
        """

        fingerprint_generator = FingerprintGenerator()
        with pytest.raises(ValueError):
            fingerprint_generator._set_n_cores(n_cores)

    @pytest.mark.parametrize(
        "structure_klifs_ids, normalize, values_array_sum",
        [([110, 118], False, 10152.4256), ([110, 118], True, 10152.4256)],
    )
    def test_to_from_json(self, structure_klifs_ids, normalize, values_array_sum):
        """
        Test if saving/loading a fingerprint to/from a json file.
        """

        fingerprints = FingerprintGenerator.from_structure_klifs_ids(structure_klifs_ids, LOCAL, 1)
        json_filepath = Path("fingerprints.json")

        with enter_temp_directory():

            # Save json file
            fingerprints.to_json(json_filepath)
            assert json_filepath.exists()

            # Load json file
            fingerprints_reloaded = FingerprintGenerator.from_json(json_filepath, normalize)

        assert isinstance(fingerprints_reloaded, FingerprintGenerator)
        # Attribute data
        assert list(fingerprints.data.keys()) == list(fingerprints_reloaded.data.keys())
        if normalize:
            assert list(fingerprints.data_normalized.keys()) == list(
                fingerprints_reloaded.data_normalized.keys()
            )
        else:
            assert fingerprints_reloaded.data_normalized is None
        values_array_sum_calculated = sum(
            [
                np.nansum(fingerprint.values_array(True, True, True))
                for structure_klifs_id, fingerprint in fingerprints_reloaded.data.items()
            ]
        )
        assert pytest.approx(values_array_sum_calculated, abs=1e-4) == values_array_sum

    @pytest.mark.parametrize(
        "structure_klifs_ids, normalized",
        [([110, 118], True), ([110, 118], False)],
    )
    def test_physicochemical_distances_moments(self, structure_klifs_ids, normalized):
        """
        Test feature group extraction methods.

        Notes
        -----
        Input structure KLIFS IDs must be able to generate a valid fingerprint, otherwise test will
        fail.
        """

        fingerprints = FingerprintGenerator.from_structure_klifs_ids(structure_klifs_ids)

        physicochemical = fingerprints.physicochemical(normalized)
        assert physicochemical.index.to_list() == structure_klifs_ids
        if normalized:
            assert physicochemical.columns.to_list() == FEATURE_NAMES_PHYSICOCHEMICAL
        else:
            assert physicochemical.columns.to_list() == FEATURE_NAMES_PHYSICOCHEMICAL_DICT
        assert isinstance(physicochemical.iloc[0, 0], list)

        distances = fingerprints.distances(normalized)
        assert distances.index.to_list() == structure_klifs_ids
        assert distances.columns.to_list() == FEATURE_NAMES_DISTANCES_AND_MOMENTS
        assert isinstance(distances.iloc[0, 0], list)

        moments = fingerprints.moments(normalized)
        assert moments.index.to_list() == structure_klifs_ids
        assert moments.columns.to_list() == FEATURE_NAMES_DISTANCES_AND_MOMENTS
        assert isinstance(moments.iloc[0, 0], list)

    @pytest.mark.parametrize(
        "structure_klifs_ids, normalized",
        [([110, 118], True), ([110, 118], False)],
    )
    def test_physicochemical_distances_moments_exploded(self, structure_klifs_ids, normalized):
        """
        Test feature group extraction methods.

        Notes
        -----
        Input structure KLIFS IDs must be able to generate a valid fingerprint, otherwise test will
        fail.
        """

        fingerprints = FingerprintGenerator.from_structure_klifs_ids(structure_klifs_ids)

        def _index_structure_klifs_id(multiplier):
            index_structure_klifs_id = []
            for i in structure_klifs_ids:
                index_structure_klifs_id.extend([i] * multiplier)
            return index_structure_klifs_id

        index_residue_ix = list(range(1, 86)) * len(structure_klifs_ids)
        index_moment = list(range(1, 4)) * len(structure_klifs_ids)

        physicochemical = fingerprints.physicochemical_exploded(normalized)
        assert physicochemical.index.get_level_values(
            "structure_klifs_id"
        ).to_list() == _index_structure_klifs_id(85)
        assert physicochemical.index.get_level_values("residue_ix").to_list() == index_residue_ix
        if normalized:
            assert physicochemical.columns.to_list() == FEATURE_NAMES_PHYSICOCHEMICAL
        else:
            assert physicochemical.columns.to_list() == FEATURE_NAMES_PHYSICOCHEMICAL_DICT
        assert physicochemical.dtypes.unique() == "float64"

        distances = fingerprints.distances_exploded(normalized)
        assert distances.index.get_level_values(
            "structure_klifs_id"
        ).to_list() == _index_structure_klifs_id(85)
        assert distances.index.get_level_values("residue_ix").to_list() == index_residue_ix
        assert distances.columns.to_list() == FEATURE_NAMES_DISTANCES_AND_MOMENTS
        assert distances.dtypes.unique() == "float64"

        moments = fingerprints.moments_exploded(normalized)
        assert moments.index.get_level_values(
            "structure_klifs_id"
        ).to_list() == _index_structure_klifs_id(3)
        assert moments.index.get_level_values("moment").to_list() == index_moment
        assert moments.columns.to_list() == FEATURE_NAMES_DISTANCES_AND_MOMENTS
        assert moments.dtypes.unique() == "float64"
