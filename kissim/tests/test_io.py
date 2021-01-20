"""
Unit and regression test for kissim.io class functionalities that
PocketBioPython and PocketDataFrame have in common.
"""

from pathlib import Path

import pytest
import pandas as pd
from opencadd.databases.klifs import setup_local

from kissim.io import PocketBioPython, PocketDataFrame

PATH_TEST_DATA = Path(__name__).parent / "kissim" / "tests" / "data"
LOCAL = setup_local(PATH_TEST_DATA / "KLIFS_download")


class TestPocketBioPython:
    """
    Test common functionalities in the PocketBioPython and PocketDataFrame classes.
    """

    @pytest.mark.parametrize(
        "pocket_class, structure_klifs_id, klifs_session",
        [(PocketBioPython, 12347, None), (PocketDataFrame, 12347, LOCAL)],
    )
    def test_from_structure_klifs_id(self, pocket_class, structure_klifs_id, klifs_session):
        """
        Test if PocketBioPython can be set (from_structure_klifs_id()).
        Test attribute `name`.
        """
        pocket = pocket_class.from_structure_klifs_id(structure_klifs_id, klifs_session=klifs_session)
        assert isinstance(pocket, pocket_class)

        # Test attribute name
        assert pocket.name == structure_klifs_id

    @pytest.mark.parametrize(
        "pocket_class, structure_klifs_id, klifs_session, n_residues, n_residues_wo_na, residue_ids_sum, residue_ixs_sum",
        [
            (PocketBioPython, 12347, LOCAL, 85, 78, 41198, 3655),
            (PocketDataFrame, 12347, LOCAL, 85, 78, 41198, 3655),
        ],
    )
    def test_residues(
        self,
        pocket_class,
        structure_klifs_id,
        klifs_session,
        n_residues,
        n_residues_wo_na,
        residue_ids_sum,
        residue_ixs_sum,
    ):
        """
        Test the class
        - attribute (`_residue_ids`, `residue_ixs`) and
        - property (`residues`)
        regarding the residue IDs.
        """
        pocket = pocket_class.from_structure_klifs_id(structure_klifs_id, klifs_session=klifs_session)
        # Test property residues
        assert isinstance(pocket.residues, pd.DataFrame)
        assert len(pocket.residues) == n_residues
        assert len(pocket.residues.dropna(axis=0, subset=["residue.id"])) == n_residues_wo_na
        assert pocket.residues["residue.id"].sum() == residue_ids_sum
        assert pocket.residues["residue.ix"].sum() == residue_ixs_sum

        # Test attribute _residue_ids
        assert isinstance(pocket._residue_ids[0], int)
        assert len(pocket._residue_ids) == n_residues
        assert sum([i for i in pocket._residue_ids if i]) == residue_ids_sum

        # Test attribute _residue_ix
        assert isinstance(pocket._residue_ixs[0], int)
        assert len(pocket._residue_ixs) == n_residues
        assert sum([i for i in pocket._residue_ixs if i]) == residue_ixs_sum
