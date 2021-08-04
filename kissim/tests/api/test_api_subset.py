"""
Unit and regression test for the kissim.api.subset module.
"""

from pathlib import Path

import numpy as np
import pytest
from opencadd.databases.klifs import setup_local

from kissim.utils import enter_temp_directory
from kissim.api import subset
from kissim.api.subset import _subset_fingerprint_generator_data
from kissim.encoding import FingerprintGenerator
from kissim.definitions import KLIFS_POCKET_RESIDUE_SUBSET

PATH_TEST_DATA = Path(__name__).parent / "kissim" / "tests" / "data"
LOCAL = setup_local(PATH_TEST_DATA / "KLIFS_download")


@pytest.mark.parametrize(
    "fingerprints_path, klifs_pocket_residue_subset_type, fingerprints_subset_path, klifs_pocket_residue_subset",
    [
        (
            (PATH_TEST_DATA / "fingerprints.json").absolute(),
            "dfg_all",
            None,
            KLIFS_POCKET_RESIDUE_SUBSET,
        ),
        (
            (PATH_TEST_DATA / "fingerprints.json").absolute(),
            "dfg_in",
            (PATH_TEST_DATA / "fingerprints_subset.json").absolute(),
            KLIFS_POCKET_RESIDUE_SUBSET,
        ),
        (
            (PATH_TEST_DATA / "fingerprints.json").absolute(),
            "dfg_out",
            None,
            KLIFS_POCKET_RESIDUE_SUBSET,
        ),
    ],
)
def test_subset(
    fingerprints_path,
    klifs_pocket_residue_subset_type,
    fingerprints_subset_path,
    klifs_pocket_residue_subset,
):

    with enter_temp_directory():

        # Generate regular fingerprints
        fingerprints_path = Path(fingerprints_path)
        fingerprint_generator = FingerprintGenerator.from_structure_klifs_ids([12347, 3835])
        fingerprint_generator.to_json(fingerprints_path)

        # Generate subset fingerprints
        fingerprint_generator_subset = subset(
            fingerprints_path,
            klifs_pocket_residue_subset_type,
            fingerprints_subset_path,
        )

        # Test FingerprintGenerator object
        assert isinstance(fingerprint_generator_subset, FingerprintGenerator)
        assert (
            fingerprint_generator_subset.structure_klifs_ids
            == fingerprint_generator.structure_klifs_ids
        )
        # Attribute `klifs_session` is set to None
        assert fingerprint_generator_subset.klifs_session is None

        # Test Fingerprint objects
        for fingerprint_id, fingerprint_subset in fingerprint_generator_subset.data.items():

            # Original fingerprint
            fingerprint = fingerprint_generator.data[fingerprint_id]

            # Is bit length correct
            n_residues = len(klifs_pocket_residue_subset[klifs_pocket_residue_subset_type])
            n_bits = len(fingerprint_subset.values_array())
            n_bits_theory = 8 * n_residues + 4 * n_residues + 12
            assert n_bits == n_bits_theory

            # Are lists of residues correct?
            assert (
                fingerprint_subset.residue_ixs
                == klifs_pocket_residue_subset[klifs_pocket_residue_subset_type]
            )
            assert len(fingerprint_subset.residue_ids) == n_residues

            # Is structure and kinase the same as in original fingerprint
            assert fingerprint_subset.structure_klifs_id == fingerprint.structure_klifs_id
            assert fingerprint_subset.kinase_name == fingerprint.kinase_name

        if fingerprints_subset_path is not None:
            fingerprints_subset_path = Path(fingerprints_subset_path)
            assert fingerprints_subset_path.exists()
            fingerprints_subset_path.unlink()

        fingerprints_path.unlink()


@pytest.mark.parametrize(
    "structure_klifs_id, klifs_session, subset_residue_ids, fp_subset_sum",
    [
        (110, LOCAL, [1, 2, 3], 250.981),
        (118, LOCAL, [10, 20, 30], 252.108),
    ],
)
def test_subset_fingerprint_generator_data(
    structure_klifs_id, klifs_session, subset_residue_ids, fp_subset_sum
):

    fingerprint_generator = FingerprintGenerator.from_structure_klifs_ids(
        [structure_klifs_id], klifs_session
    )
    fingerprint_generator_data = _subset_fingerprint_generator_data(
        fingerprint_generator, subset_residue_ids
    )
    fp_subset_sum_calculated = np.nansum(
        fingerprint_generator_data[structure_klifs_id].values_array()
    )
    print(fp_subset_sum_calculated)
    assert pytest.approx(fp_subset_sum_calculated, abs=1e-3) == fp_subset_sum
