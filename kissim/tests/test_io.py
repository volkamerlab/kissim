"""
Unit and regression test for kissim.io class methods.
"""

import pytest
import numpy as np
import pandas as pd
import Bio
from opencadd.databases.klifs import setup_remote

from kissim.io import PocketBioPython, PocketDataFrame

REMOTE = setup_remote()


class TestPocketBioPython:
    """
    Test PocketBioPython class.
    """

    @pytest.mark.parametrize(
        "structure_id, remote",
        [(12347, None), (12347, REMOTE)],
    )
    def test_from_structure_klifs_id(self, structure_id, remote):
        """
        Test if PocketBioPython can be set remotely.
        """
        pocket_bp = PocketBioPython.from_structure_klifs_id(structure_id, remote)
        assert isinstance(pocket_bp, PocketBioPython)

    @pytest.mark.parametrize(
        "structure_id, remote, n_atoms_complex, n_atoms_pocket",
        [(12347, REMOTE, 1819, 577)],
    )
    def test_data(self, structure_id, remote, n_atoms_complex, n_atoms_pocket):
        """
        Test class attribute (complex data) and property (pocket data).
        """

        pocket_bp = PocketBioPython.from_structure_klifs_id(structure_id, remote)
        # Complex data
        assert isinstance(pocket_bp._data_complex, Bio.PDB.Chain.Chain)
        assert len(list(pocket_bp._data_complex.get_atoms())) == n_atoms_complex

    @pytest.mark.parametrize(
        "structure_id, remote, n_hse_ca_complex, n_hse_cb_complex, n_hse_ca_pocket, n_hse_cb_pocket",
        [(12347, REMOTE, 246, 254, 75, 78)],
    )
    def test_hse_ca_cb(
        self,
        structure_id,
        remote,
        n_hse_ca_complex,
        n_hse_cb_complex,
        n_hse_ca_pocket,
        n_hse_cb_pocket,
    ):
        """
        Test class attributes and properties regarding the HSExposure.
        """
        pocket_bp = PocketBioPython.from_structure_klifs_id(structure_id, remote)

        # HSE for full complex
        assert isinstance(pocket_bp._hse_ca_complex, Bio.PDB.HSExposure.HSExposureCA)
        assert len(pocket_bp._hse_ca_complex) == n_hse_ca_complex
        assert isinstance(pocket_bp._hse_cb_complex, Bio.PDB.HSExposure.HSExposureCB)
        assert len(pocket_bp._hse_cb_complex) == n_hse_cb_complex
        # HSE for pocket only
        assert isinstance(pocket_bp.hse_ca, dict)
        assert len(pocket_bp.hse_ca) == n_hse_ca_pocket
        assert isinstance(pocket_bp.hse_cb, dict)
        assert len(pocket_bp.hse_cb) == n_hse_cb_pocket

    @pytest.mark.parametrize(
        "structure_id, remote, n_residues, residue_ids_sum, residue_ixs_sum",
        [(12347, REMOTE, 78, 41198, 3381)],
    )
    def test_residues(self, structure_id, remote, n_residues, residue_ids_sum, residue_ixs_sum):
        """
        Test the class attribute and property regarding the residue IDs.
        """
        pocket_bp = PocketBioPython.from_structure_klifs_id(structure_id, remote)
        assert isinstance(pocket_bp.residues, pd.DataFrame)
        assert len(pocket_bp.residues) == n_residues
        assert pocket_bp.residues["residue.id"].sum() == residue_ids_sum
        assert pocket_bp.residues["residue.ix"].sum() == residue_ixs_sum
        assert isinstance(pocket_bp._residue_ids[0], int)
        assert len(pocket_bp._residue_ids) == n_residues
        assert sum(pocket_bp._residue_ids) == residue_ids_sum
        assert isinstance(pocket_bp._residue_ixs[0], int)
        assert len(pocket_bp._residue_ixs) == n_residues
        assert sum(pocket_bp._residue_ixs) == residue_ixs_sum

    @pytest.mark.parametrize(
        "structure_id, remote, pocket_centroid_mean",
        [(12347, REMOTE, 19.63254)],
    )
    def test_centroid(self, structure_id, remote, pocket_centroid_mean):
        """
        Test the class property regarding the pocket centroid.
        """
        pocket_bp = PocketBioPython.from_structure_klifs_id(structure_id, remote)
        assert isinstance(pocket_bp.center, Bio.PDB.vectors.Vector)
        assert pocket_bp.center.get_array().mean() == pytest.approx(pocket_centroid_mean)

    @pytest.mark.parametrize(
        "structure_id, remote, n_ca_atoms",
        [(12347, REMOTE, 78)],
    )
    def test_ca_atoms(self, structure_id, remote, n_ca_atoms):
        """
        Test the class property regarding the pocket's CA atoms.
        """
        pocket_bp = PocketBioPython.from_structure_klifs_id(structure_id, remote)
        assert pocket_bp.ca_atoms.shape == (n_ca_atoms, 3)
        assert pocket_bp.ca_atoms.columns.to_list() == ["residue.id", "ca.atom", "ca.vector"]
        assert pocket_bp.ca_atoms.dtypes.to_list() == ["int64", "object", "object"]
        for ca_atom in pocket_bp.ca_atoms["ca.atom"]:
            assert isinstance(ca_atom, Bio.PDB.Atom.Atom)
        for ca_vector in pocket_bp.ca_atoms["ca.vector"]:
            assert isinstance(ca_vector, Bio.PDB.vectors.Vector)

    @pytest.mark.parametrize(
        "structure_id, remote, residue_id, ca_atom_mean",
        [
            (5399, REMOTE, 1272, 18.5630),  # Residue has CA
            (5399, REMOTE, 1273, None),  # Residue has no CA
        ],
    )
    def test_ca_atom(self, structure_id, remote, residue_id, ca_atom_mean):
        """
        Test if CA atom is retrieved correctly from a residue ID (test if-else cases).
        """
        pocket_bp = PocketBioPython.from_structure_klifs_id(structure_id, remote)
        ca_atom_calculated = pocket_bp._ca_atom(residue_id)
        if ca_atom_mean:
            assert isinstance(ca_atom_calculated, Bio.PDB.Atom.Atom)
            ca_atom_mean_calculated = ca_atom_calculated.get_vector().get_array().mean()
            assert ca_atom_mean == pytest.approx(ca_atom_mean_calculated)
        else:
            assert ca_atom_mean == ca_atom_calculated

    @pytest.mark.parametrize(
        "structure_id, remote",
        [(12347, REMOTE)],
    )
    def test_pcb_atoms(self, structure_id, remote):
        """
        Test the class property regarding the pocket's pCB atoms.
        """
        pocket_bp = PocketBioPython.from_structure_klifs_id(structure_id, remote)
        assert pocket_bp.pcb_atoms.shape == (78, 2)
        assert pocket_bp.pcb_atoms.columns.to_list() == ["residue.id", "pcb.vector"]
        assert pocket_bp.pcb_atoms.dtypes.to_list() == ["int64", "object"]
        for pcb_vector in pocket_bp.pcb_atoms["pcb.vector"]:
            if pcb_vector is not None:
                assert isinstance(pcb_vector, Bio.PDB.vectors.Vector)

    @pytest.mark.parametrize(
        "structure_id, remote, residue_id, pcb_atom_mean",
        [(9122, REMOTE, 272, 0.706664)],  # GLY
    )
    def test_pcb_atom_from_gly(self, structure_id, remote, residue_id, pcb_atom_mean):
        """
        Test pseudo-CB calculation for GLY.
        """
        pocket_bp = PocketBioPython.from_structure_klifs_id(structure_id, remote)
        residue = pocket_bp._residue_from_residue_id(residue_id)
        pcb_atom_calculated = pocket_bp._pcb_atom_from_gly(residue)
        pcb_atom_mean_calculated = pcb_atom_calculated.get_array().mean()
        assert pcb_atom_mean == pytest.approx(pcb_atom_mean_calculated)

    @pytest.mark.parametrize(
        "structure_id, remote, residue_id",
        [
            (9122, REMOTE, 337),  # ALA
            (9122, REMOTE, 357),  # Non-standard residue
        ],
    )
    def test_pcb_atom_from_gly_valueerror(self, structure_id, remote, residue_id):
        """
        Test exceptions in pseudo-CB calculation for GLY.
        """
        pocket_bp = PocketBioPython.from_structure_klifs_id(structure_id, remote)
        residue = pocket_bp._residue_from_residue_id(residue_id)
        with pytest.raises(ValueError):
            pocket_bp._pcb_atom_from_gly(residue)

    @pytest.mark.parametrize(
        "structure_id, remote, residue_id, pcb_atom",
        [
            (9122, REMOTE, 272, np.array([12.223623, 8.544623, 32.441623])),  # GLY
            (
                9122,
                REMOTE,
                337,
                np.array([4.887966, 11.028965, 42.998965]),
            ),  # Residue with +- residue
            (9122, REMOTE, 261, None),  # Residue without + residue
        ],
    )
    def test_pcb_atoms(self, structure_id, remote, residue_id, pcb_atom):
        """
        Test pseudo-CB calculation for a residue.
        """

        pocket_bp = PocketBioPython.from_structure_klifs_id(structure_id, remote)
        pcb_atom_calculated = pocket_bp._pcb_atom(residue_id)

        if pcb_atom is None:
            assert pcb_atom_calculated is None
        else:
            pcb_atom_calculated = pcb_atom_calculated.get_array()
            assert pcb_atom[0] == pytest.approx(pcb_atom_calculated[0])
            assert pcb_atom[1] == pytest.approx(pcb_atom_calculated[1])
            assert pcb_atom[2] == pytest.approx(pcb_atom_calculated[2])

    @pytest.mark.parametrize(
        "structure_id, remote",
        [(12347, REMOTE)],
    )
    def test_side_chain_representatives(self, structure_id, remote):
        """
        Test the class property regarding the pocket's side chain representatives.
        """
        pocket_bp = PocketBioPython.from_structure_klifs_id(structure_id, remote)
        assert isinstance(pocket_bp.side_chain_representatives, pd.DataFrame)
        assert pocket_bp.side_chain_representatives.columns.to_list() == [
            "residue.id",
            "sc.atom",
            "sc.vector",
        ]
        assert pocket_bp.side_chain_representatives.dtypes.to_list() == [
            "int64",
            "object",
            "object",
        ]
        for sc_atom in pocket_bp.side_chain_representatives["sc.atom"]:
            if sc_atom is not None:
                assert isinstance(sc_atom, Bio.PDB.Atom.Atom)
        for sc_vector in pocket_bp.side_chain_representatives["sc.vector"]:
            if sc_vector is not None:
                assert isinstance(sc_vector, Bio.PDB.vectors.Vector)

    @pytest.mark.parametrize(
        "structure_id, remote, residue_id, sc_atom_mean",
        [
            (9122, REMOTE, 272, None),  # GLY
            (9122, REMOTE, 337, 20.31),  # ALA (with CB)
            (1641, REMOTE, 19, None),  # ALA (without CB)
            (9122, REMOTE, 336, 22.122666),  # PHE (with CZ)
            (9122, REMOTE, 357, 27.526666),  # MSE > MET (with CE)
        ],
    )
    def test_side_chain_representative_tmp(self, structure_id, remote, residue_id, sc_atom_mean):
        """
        Test if side chain representative is retrieved correctly from a residue.
        """
        pocket_bp = PocketBioPython.from_structure_klifs_id(structure_id, remote)
        sc_atom_calculated = pocket_bp._side_chain_representative(residue_id)

        if sc_atom_mean is not None:
            assert isinstance(sc_atom_calculated, Bio.PDB.Atom.Atom)
            sc_atom_mean_calculated = sc_atom_calculated.get_vector().get_array().mean()
            assert sc_atom_mean == pytest.approx(sc_atom_mean_calculated)
        else:
            assert sc_atom_calculated == None


class TestsPocketDataFrame:
    """
    Test PocketDataFrame class.
    """

    @pytest.mark.parametrize(
        "structure_id, remote",
        [(12347, None), (12347, REMOTE)],
    )
    def test_from_structure_klifs_id(self, structure_id, remote):
        """
        Test if PocketDataFrame can be set remotely.
        """
        pocket_df = PocketDataFrame.from_structure_klifs_id(structure_id, klifs_session=remote)
        assert isinstance(pocket_df, PocketDataFrame)

    @pytest.mark.parametrize(
        "structure_id, remote, n_atoms_complex, n_atoms_pocket",
        [(12347, REMOTE, 1819, 577)],
    )
    def test_data(self, structure_id, remote, n_atoms_complex, n_atoms_pocket):
        """
        Test the class attribute (complex data) and property (pocket data).
        """
        pocket_df = PocketDataFrame.from_structure_klifs_id(structure_id, klifs_session=remote)
        # Pocket data
        assert pocket_df.data.shape == (n_atoms_pocket, 7)

    @pytest.mark.parametrize(
        "structure_id, remote, n_residues",
        [(12347, REMOTE, 78)],
    )
    def test_residue_ids(self, structure_id, remote, n_residues):
        """
        Test the class attribute and property regarding the residue IDs.
        """
        pocket_df = PocketDataFrame.from_structure_klifs_id(structure_id, klifs_session=remote)
        assert isinstance(pocket_df._residue_ids[0], int)
        assert len(pocket_df._residue_ids) == n_residues
        assert len(pocket_df.residues["residue.id"]) == n_residues
        assert pocket_df.residues["residue.id"].to_list() == pocket_df._residue_ids

    @pytest.mark.parametrize(
        "structure_id, remote, pocket_centroid_mean",
        [(12347, REMOTE, 19.63254)],
    )
    def test_centroid(self, structure_id, remote, pocket_centroid_mean):
        """
        Test the class property regarding the pocket centroid.
        """
        pocket_df = PocketDataFrame.from_structure_klifs_id(structure_id, klifs_session=remote)
        assert np.array(pocket_df.center).mean() == pytest.approx(pocket_centroid_mean)

    @pytest.mark.parametrize(
        "structure_id, remote, n_ca_atoms",
        [(12347, REMOTE, 78)],
    )
    def test_ca_atoms(self, structure_id, remote, n_ca_atoms):
        """
        Test the class property regarding the pocket's CA atoms.
        """
        pocket_df = PocketDataFrame.from_structure_klifs_id(structure_id, klifs_session=remote)
        assert pocket_df.ca_atoms.shape == (n_ca_atoms, 7)
        assert pocket_df.ca_atoms.columns.to_list() == [
            "atom.id",
            "atom.name",
            "atom.x",
            "atom.y",
            "atom.z",
            "residue.id",
            "residue.name",
        ]
        assert pocket_df.ca_atoms.dtypes.to_list() == [
            "int32",
            "string",
            "float32",
            "float32",
            "float32",
            "int32",
            "string",
        ]
