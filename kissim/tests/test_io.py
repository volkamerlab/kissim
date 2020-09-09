"""
Unit and regression test for kissim.io class methods.
"""

import pytest

import pandas as pd

from kissim.io import DataFrameToBiopythonChain


class TestsDataFrameToBiopythonChain:
    """
    Test DataFrameToBiopythonChain class.
    """

    @pytest.mark.parametrize(
        "residue_pdb_ids, residue_pdb_ids_out, residue_insertions_out",
        [
            (
                ["_12", "_12", "_5", "_5", "_4A", "3", "33A", "_40", "50"],
                [-12, -12, -5, -5, -4, 3, 33, 40, 50],
                [" ", " ", " ", " ", "A", " ", "A", " ", " "],
            )
        ],
    )
    def test_format_dataframe(
        self, residue_pdb_ids, residue_pdb_ids_out, residue_insertions_out
    ):

        dataframe = pd.DataFrame(
            [residue_pdb_ids], index=["residue.pdb_id"]
        ).transpose()

        bpy = DataFrameToBiopythonChain()
        dataframe = bpy._format_dataframe(dataframe)

        assert dataframe["residue.pdb_id"].to_list() == residue_pdb_ids_out
        assert dataframe["residue.insertion"].to_list() == residue_insertions_out

    @pytest.mark.parametrize(
        "residue_name, residue_pdb_id, residue_insertion, residue_id",
        [
            ("ALA", 33, "A", (" ", 33, "A")),
            ("XXX", 33, "A", ("H_XXX", 33, "A")),
            ("HOH", 33, " ", ("W", 33, " ")),
        ],
    )
    def test_residue(self, residue_name, residue_pdb_id, residue_insertion, residue_id):

        bpy = DataFrameToBiopythonChain()
        residue = bpy._residue(residue_name, residue_pdb_id, residue_insertion)

        assert residue.get_id() == residue_id
        assert residue.get_resname() == residue_name
