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
    def test_from_remote(self, structure_id, remote):
        """
        Test if PocketBioPython can be set remotely.
        """
        pocket_bp = PocketBioPython.from_remote(structure_id, remote)
        assert isinstance(pocket_bp, PocketBioPython)

    @pytest.mark.parametrize(
        "structure_id, remote, n_atoms_complex, n_atoms_pocket",
        [(12347, REMOTE, 1819, 577)],
    )
    def test_data(self, structure_id, remote, n_atoms_complex, n_atoms_pocket):
        """
        Test class attribute (complex data) and property (pocket data).
        """

        pocket_bp = PocketBioPython.from_remote(structure_id, remote)
        # Complex data
        assert isinstance(pocket_bp._data_complex, Bio.PDB.Structure.Structure)
        assert len(list(pocket_bp._data_complex.get_atoms())) == n_atoms_complex
        # Pocket data
        with pytest.raises(NotImplementedError):
            pocket_bp.data

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
        pocket_bp = PocketBioPython.from_remote(structure_id, remote)

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
        "structure_id, remote, n_residues",
        [(12347, REMOTE, 78)],
    )
    def test_residue_ids(self, structure_id, remote, n_residues):
        """
        Test the class attribute and property regarding the residue IDs.
        """
        pocket_bp = PocketBioPython.from_remote(structure_id, remote)
        assert isinstance(pocket_bp._residue_ids[0], int)
        assert len(pocket_bp._residue_ids) == n_residues
        assert len(pocket_bp.residue_ids) == n_residues
        assert pocket_bp.residue_ids == pocket_bp._residue_ids

    @pytest.mark.parametrize(
        "structure_id, remote, pocket_centroid_mean",
        [(12347, REMOTE, 19.63254)],
    )
    def test_centroid(self, structure_id, remote, pocket_centroid_mean):
        """
        Test the class property regarding the pocket centroid.
        """
        pocket_bp = PocketBioPython.from_remote(structure_id, remote)
        assert isinstance(pocket_bp.centroid, Bio.PDB.vectors.Vector)
        assert pocket_bp.centroid.get_array().mean() == pytest.approx(pocket_centroid_mean)

    @pytest.mark.parametrize(
        "structure_id, remote, n_ca_atoms",
        [(12347, REMOTE, 78)],
    )
    def test_ca_atoms(self, structure_id, remote, n_ca_atoms):
        """
        Test the class property regarding the pocket's CA atoms.
        """
        pocket_bp = PocketBioPython.from_remote(structure_id, remote)
        assert pocket_bp.ca_atoms.shape == (n_ca_atoms, 3)
        assert pocket_bp.ca_atoms.columns.to_list() == ["residue.id", "ca.atom", "ca.vector"]
        assert pocket_bp.ca_atoms.dtypes.to_list() == ["int64", "object", "object"]
        for ca_atom in pocket_bp.ca_atoms["ca.atom"]:
            assert isinstance(ca_atom, Bio.PDB.Atom.Atom)
        for ca_vector in pocket_bp.ca_atoms["ca.vector"]:
            assert isinstance(ca_vector, Bio.PDB.vectors.Vector)

    @pytest.mark.parametrize(
        "structure_id, remote",
        [(12347, REMOTE)],
    )
    def test_pcb_atoms(self, structure_id, remote):
        """
        Test the class property regarding the pocket's pCB atoms.
        """
        pocket_bp = PocketBioPython.from_remote(structure_id, remote)
        assert pocket_bp.pcb_atoms.shape == (78, 2)
        assert pocket_bp.pcb_atoms.columns.to_list() == ["residue.id", "pcb.vector"]
        assert pocket_bp.pcb_atoms.dtypes.to_list() == ["int64", "object"]
        for pcb_vector in pocket_bp.pcb_atoms["pcb.vector"]:
            if pcb_vector is not None:
                assert isinstance(pcb_vector, Bio.PDB.vectors.Vector)

    @pytest.mark.parametrize(
        "structure_id, remote",
        [(12347, REMOTE)],
    )
    def test_side_chain_representatives(self, structure_id, remote):
        """
        Test the class property regarding the pocket's side chain representatives.
        """
        pocket_bp = PocketBioPython.from_remote(structure_id, remote)
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


class TestsPocketDataFrame:
    """
    Test PocketDataFrame class.
    """

    @pytest.mark.parametrize(
        "structure_id, remote",
        [(12347, None), (12347, REMOTE)],
    )
    def test_from_remote(self, structure_id, remote):
        """
        Test if PocketDataFrame can be set remotely.
        """
        pocket_df = PocketDataFrame.from_remote(structure_id, remote)
        assert isinstance(pocket_df, PocketDataFrame)

    @pytest.mark.parametrize(
        "structure_id, remote, n_atoms_complex, n_atoms_pocket",
        [(12347, REMOTE, 1819, 577)],
    )
    def test_data(self, structure_id, remote, n_atoms_complex, n_atoms_pocket):
        """
        Test the class attribute (complex data) and property (pocket data).
        """
        pocket_df = PocketDataFrame.from_remote(structure_id, remote)
        # Complex data
        assert pocket_df._data_complex.shape == (n_atoms_complex, 11)
        # Pocket data
        assert pocket_df.data.shape == (n_atoms_pocket, 11)

    @pytest.mark.parametrize(
        "structure_id, remote, n_residues",
        [(12347, REMOTE, 78)],
    )
    def test_residue_ids(self, structure_id, remote, n_residues):
        """
        Test the class attribute and property regarding the residue IDs.
        """
        pocket_df = PocketDataFrame.from_remote(structure_id, remote)
        assert isinstance(pocket_df._residue_ids[0], str)  # TODO in the future: cast to int?
        assert len(pocket_df._residue_ids) == n_residues
        assert len(pocket_df.residue_ids) == n_residues
        assert pocket_df.residue_ids == pocket_df._residue_ids

    @pytest.mark.parametrize(
        "structure_id, remote, pocket_centroid_mean",
        [(12347, REMOTE, 19.63254)],
    )
    def test_centroid(self, structure_id, remote, pocket_centroid_mean):
        """
        Test the class property regarding the pocket centroid.
        """
        pocket_df = PocketDataFrame.from_remote(structure_id, remote)
        assert np.array(pocket_df.centroid).mean() == pytest.approx(pocket_centroid_mean)

    @pytest.mark.parametrize(
        "structure_id, remote, n_ca_atoms",
        [(12347, REMOTE, 78)],
    )
    def test_ca_atoms(self, structure_id, remote, n_ca_atoms):
        """
        Test the class property regarding the pocket's CA atoms.
        """
        pocket_df = PocketDataFrame.from_remote(structure_id, remote)
        assert pocket_df.ca_atoms.shape == (n_ca_atoms, 11)
        assert pocket_df.ca_atoms.columns.to_list() == [
            "atom.id",
            "atom.name",
            "atom.x",
            "atom.y",
            "atom.z",
            "residue.id",
            "residue.name",
            "residue.klifs_id",
            "residue.klifs_region_id",
            "residue.klifs_region",
            "residue.klifs_color",
        ]
        assert pocket_df.ca_atoms.dtypes.to_list() == [
            "int32",
            "string",
            "float32",
            "float32",
            "float32",
            "object",
            "string",
            "Int64",
            "object",
            "object",
            "object",
        ]
