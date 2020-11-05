"""
Unit and regression test for kissim.encoding class methods.
"""

import math
from pathlib import Path

from Bio.PDB import Vector
import numpy as np
import pandas as pd
import pytest

from kissim.auxiliary import KlifsMoleculeLoader, PdbChainLoader
from kissim.encoding.schema import FEATURE_NAMES
from kissim.encoding.definitions import DISTANCE_CUTOFFS, MOMENT_CUTOFFS
from kissim.encoding.features import (
    PharmacophoreSizeFeatures,
    ExposureFeature,
    SideChainOrientationFeature,
    PhysicoChemicalFeatures,
    SpatialFeatures,
)
from kissim.encoding.api import Fingerprint

PATH_TEST_DATA = Path(__name__).parent / "kissim" / "tests" / "data"


class TestsPharmacophoreSizeFeatures:
    """
    Test PharmacophoreSizeFeatures class methods.
    """

    @pytest.mark.parametrize(
        "residue_name, feature_name, feature",
        [
            ("ALA", "size", 1),  # Size
            ("ASN", "size", 2),
            ("ARG", "size", 3),
            ("PTR", "size", 3),  # Converted non-standard
            ("MSE", "size", 2),  # Converted non-standard
            ("XXX", "size", None),  # Non-convertable non-standard
            ("ALA", "hbd", 0),
            ("ASN", "hbd", 1),
            ("ARG", "hbd", 3),
            ("XXX", "hbd", None),
            ("ALA", "hba", 0),
            ("ASN", "hba", 1),
            ("ASP", "hba", 2),
            ("XXX", "hba", None),
            ("ALA", "charge", 0),
            ("ARG", "charge", 1),
            ("ASP", "charge", -1),
            ("XXX", "charge", None),
            ("ALA", "aromatic", 0),
            ("HIS", "aromatic", 1),
            ("XXX", "aromatic", None),
            ("ARG", "aliphatic", 0),
            ("ALA", "aliphatic", 1),
            ("XXX", "aliphatic", None),
        ],
    )
    def test_from_residue(self, residue_name, feature_name, feature):
        """
        Test function for retrieval of residue's size and pharmacophoric features
        (i.e. number of hydrogen bond donor,
        hydrogen bond acceptors, charge features, aromatic features or aliphatic features )

        Parameters
        ----------
        residue_name : str
            Three-letter code for residue.
        feature_name : str
            Feature type name.
        feature : int or None
            Feature value.
        """

        pharmacophore_size_feature = PharmacophoreSizeFeatures()

        # Call feature from residue function
        feature_calculated = pharmacophore_size_feature.from_residue(residue_name, feature_name)

        if feature_calculated:  # If not None
            assert isinstance(feature_calculated, float)

        assert feature_calculated == feature

    @pytest.mark.parametrize(
        "path_klifs_metadata, path_mol2, molecule_code, shape",
        [
            (
                PATH_TEST_DATA / "klifs_metadata.csv",
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/AAK1/4wsq_altA_chainB/pocket.mol2",
                "HUMAN/AAK1_4wsq_altA_chainB",
                (85, 6),
            ),
            (
                PATH_TEST_DATA / "klifs_metadata.csv",
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ABL1/2g2i_chainA/pocket.mol2",
                "HUMAN/ABL1_2g2i_chainA",
                (82, 6),
            ),  # Contains not full KLIFS positions
        ],
    )
    def test_pharmacophoresizefeatures_from_residue(
        self, path_klifs_metadata, path_mol2, molecule_code, shape
    ):
        """
        Test PharmacophoreSizeFeatures class attributes.

        Parameters
        ----------
        path_klifs_metadata : pathlib.Path
            Path to unfiltered KLIFS metadata.
        path_mol2 : str
            Path to file originating from test data folder.
        molecule_code : str
            Molecule code as defined by KLIFS in mol2 file.
        """

        # Load molecule
        klifs_molecule_loader = KlifsMoleculeLoader()
        klifs_molecule_loader.from_file(path_mol2, path_klifs_metadata)

        molecule = klifs_molecule_loader.molecule

        # Get pharmacophore and size features
        pharmacophore_size_feature = PharmacophoreSizeFeatures()
        pharmacophore_size_feature.from_molecule(molecule)

        assert pharmacophore_size_feature.molecule_code == molecule_code
        assert pharmacophore_size_feature.features.shape == shape


class TestsExposureFeature:
    """
    Test ExposureFeature class methods.
    """

    @pytest.mark.parametrize(
        "path_pdb, chain_id, radius, method, n_residues, up_mean, down_mean, index_mean",
        [
            (
                PATH_TEST_DATA
                / "KLIFS_download"
                / "HUMAN/AAK1/4wsq_altA_chainB/protein_pymol.pdb",
                "B",
                12.0,
                "HSExposureCA",
                308,
                12.3636,
                16.4708,
                187.5,
            ),
            (
                PATH_TEST_DATA
                / "KLIFS_download"
                / "HUMAN/AAK1/4wsq_altA_chainB/protein_pymol.pdb",
                "B",
                12.0,
                "HSExposureCB",
                310,
                13.1355,
                15.5806,
                187.5,
            ),
        ],
    )
    def test_get_exposure_by_method(
        self, path_pdb, chain_id, radius, method, n_residues, up_mean, down_mean, index_mean
    ):
        """
        Test half sphere exposure and exposure ratio calculation as well as the result format.

        Parameters
        ----------
        path_pdb : str
            Path to cif file.
        chain_id : str
            Chain ID.
        radius : float
            Sphere radius to be used for half sphere exposure calculation.
        method : str
            Half sphere exposure method name: HSExposureCA or HSExposureCB.
        n_residues : int
            Number of residues in exposure calculation result.
        up_mean : float
            Mean of all exposure up values.
        down_mean : float
            Mean of all exposure down values.
        index_mean : float
            Mean of all residue IDs.
        """

        # Load pdb file
        pdb_chain_loader = PdbChainLoader()
        pdb_chain_loader.from_file(path_pdb, chain_id)

        # Get exposure
        feature = ExposureFeature()
        exposures_calculated = feature.get_molecule_exposure_by_method(
            pdb_chain_loader.chain, radius, method
        )

        # Get method prefix, i.e. ca or cb
        prefix = method[-2:].lower()

        # Test DataFrame length
        assert len(exposures_calculated) == n_residues

        # Test column names
        columns = [f"{prefix}_{i}" for i in ["up", "down", "angle_CB-CA-pCB", "exposure"]]
        assert list(exposures_calculated.columns) == columns

        # Test exposure up values (mean)
        up_mean_calculated = exposures_calculated[f"{prefix}_up"].mean()
        assert np.isclose(up_mean_calculated, up_mean, rtol=1e-05)

        # Test exposure down values (mean)
        down_mean_calculated = exposures_calculated[f"{prefix}_down"].mean()
        assert np.isclose(down_mean_calculated, down_mean, rtol=1e-05)

        # Test residue IDs (mean)
        index_mean_calculated = np.array(exposures_calculated.index).mean()
        assert np.isclose(index_mean_calculated, index_mean, rtol=1e-05)

        # Test for example residue the exposure ratio calculation
        example_residue = exposures_calculated.iloc[0]
        ratio = example_residue[f"{prefix}_exposure"]
        ratio_calculated = example_residue[f"{prefix}_up"] / (
            example_residue[f"{prefix}_up"] + example_residue[f"{prefix}_down"]
        )
        assert np.isclose(ratio_calculated, ratio, rtol=1e-04)

    @pytest.mark.parametrize(
        "path_pdb, chain_id, radius, n_residues, missing_exposure",
        [
            (
                PATH_TEST_DATA
                / "KLIFS_download"
                / "HUMAN/AAK1/4wsq_altA_chainB/protein_pymol.pdb",
                "B",
                12.0,
                310,
                {"ca": [33, 342], "cb": []},
            ),
        ],
    )
    def test_get_molecule_exposures(
        self, path_pdb, chain_id, radius, n_residues, missing_exposure
    ):
        """
        Test join of HSExposureCA and HSExposureCB data.

        Parameters
        ----------
        path_pdb : str
            Path to cif file.
        chain_id : str
            Chain ID.
        radius : float
            Sphere radius to be used for half sphere exposure calculation.
        n_residues : int
            Number of residues in exposure calculation result.
        missing_exposure : dict of list of int
            Residue IDs with missing exposures for HSExposureCA and HSExposureCB calculation.
        """

        # Load pdb file
        pdb_chain_loader = PdbChainLoader()
        pdb_chain_loader.from_file(path_pdb, chain_id)

        # Get exposure
        feature = ExposureFeature()
        exposures_calculated = feature.get_molecule_exposures(pdb_chain_loader.chain, radius)

        # Test DataFrame length
        assert len(exposures_calculated) == n_residues

        # Test column names
        column_names_ca = ["ca_up", "ca_down", "ca_angle_CB-CA-pCB", "ca_exposure"]
        column_names_cb = ["cb_up", "cb_down", "cb_angle_CB-CA-pCB", "cb_exposure"]
        column_names = column_names_ca + column_names_cb
        assert list(exposures_calculated.columns) == column_names

        # Test missing residues in HSExposureCA and HSExposureCB calculation
        missing_residues_calculated = dict()

        missing_residues_calculated["ca"] = list(
            exposures_calculated[exposures_calculated.ca_up.isna()].index
        )
        assert missing_residues_calculated["ca"] == missing_exposure["ca"]

        missing_residues_calculated["cb"] = list(
            exposures_calculated[exposures_calculated.cb_up.isna()].index
        )
        assert missing_residues_calculated["cb"] == missing_exposure["cb"]

    @pytest.mark.parametrize(
        "path_klifs_metadata, path_mol2, path_pdb, chain_id, radius, n_residues, missing_exposure",
        [
            (
                PATH_TEST_DATA / "klifs_metadata.csv",
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ABL1/2g2i_chainA/pocket.mol2",
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ABL1/2g2i_chainA/protein_pymol.pdb",
                "A",
                12.0,
                82,
                {"ca": [5, 6], "cb": []},
            ),
            (
                PATH_TEST_DATA / "klifs_metadata.csv",
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/CHK1/3nlb_chainA/pocket.mol2",
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/CHK1/3nlb_chainA/protein_pymol.pdb",
                "A",
                12.0,
                85,
                {"ca": [], "cb": [7]},
            ),
        ],
    )
    def test_from_molecule(
        self,
        path_klifs_metadata,
        path_mol2,
        path_pdb,
        chain_id,
        radius,
        n_residues,
        missing_exposure,
    ):
        """
        Test KLIFS ID subset of molecule exposure values and correct selection of HSExposureCB and
        HSExposureCA values as
        final exposure value (use CB, but if not available use CA).

        Parameters
        ----------
        path_klifs_metadata : pathlib.Path
            Path to unfiltered KLIFS metadata.
        path_mol2 : pathlib.Path
            Path to mol2 file.
        path_pdb : pathlib.Path
            Path to cif file.
        chain_id : str
            Chain ID.
        radius : float
            Sphere radius to be used for half sphere exposure calculation.
        n_residues : int
            Number of pocket residues in exposure calculation result.
        missing_exposure : dict of list of int
            Residue IDs with missing exposures for HSExposureCA and HSExposureCB calculation.
        """

        # Load pdb and mol2 files
        klifs_molecule_loader = KlifsMoleculeLoader()
        klifs_molecule_loader.from_file(path_mol2, path_klifs_metadata)
        pdb_chain_loader = PdbChainLoader()
        pdb_chain_loader.from_file(path_pdb, chain_id)

        # Get exposure
        exposure_feature = ExposureFeature()
        exposure_feature.from_molecule(
            klifs_molecule_loader.molecule, pdb_chain_loader.chain, radius
        )

        # Test number of pocket residues in exposure calculation
        assert len(exposure_feature.features) == n_residues
        assert len(exposure_feature.features_verbose) == n_residues

        # Test column names in class attribute "features"
        column_names = ["exposure"]
        assert list(exposure_feature.features.columns) == column_names

        # Test column names in class attribute "features_verbose"
        column_names_ca = ["ca_up", "ca_down", "ca_angle_CB-CA-pCB", "ca_exposure"]
        column_names_cb = ["cb_up", "cb_down", "cb_angle_CB-CA-pCB", "cb_exposure"]
        column_names = ["res_id"] + column_names_ca + column_names_cb + ["exposure"]
        assert list(exposure_feature.features_verbose.columns) == column_names

        # Test for residues with missing exposures
        exposures_calculated = exposure_feature.features_verbose
        missing_residues_calculated = dict()
        missing_residues_calculated["ca"] = list(
            exposures_calculated[exposures_calculated.ca_up.isna()].index
        )
        missing_residues_calculated["cb"] = list(
            exposures_calculated[exposures_calculated.cb_up.isna()].index
        )
        assert missing_residues_calculated["ca"] == missing_exposure["ca"]
        assert missing_residues_calculated["cb"] == missing_exposure["cb"]

        # Test resulting exposure (HSExposureCB values, unless they are missing, then set HSExposureCA values)
        for index, row in exposures_calculated.iterrows():

            if index in missing_exposure["cb"]:
                assert (
                    exposures_calculated.loc[index].exposure
                    == exposures_calculated.loc[index].ca_exposure
                )
            else:
                assert (
                    exposures_calculated.loc[index].exposure
                    == exposures_calculated.loc[index].cb_exposure
                )


class TestsSideChainOrientationFeature:
    """
    Test SideChainOrientationFeature class methods.
    """

    @pytest.mark.parametrize(
        "path_klifs_metadata, path_mol2, path_pdb, chain_id, res_id_mean, n_pocket_atoms",
        [
            (
                PATH_TEST_DATA / "klifs_metadata.csv",
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ABL1/2g2i_chainA/pocket.mol2",
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ABL1/2g2i_chainA/protein_pymol.pdb",
                "A",
                315.95,
                659,
            )
        ],
    )
    def test_get_pocket_residues(
        self, path_klifs_metadata, path_mol2, path_pdb, chain_id, res_id_mean, n_pocket_atoms
    ):
        """
        Test the mean of the pocket's PDB residue IDs and the number of pocket atoms.

        Parameters
        ----------
        path_klifs_metadata : pathlib.Path
            Path to unfiltered KLIFS metadata.
        path_mol2 : pathlib.Path
            Path to mol2 file.
        path_pdb : pathlib.Path
            Path to cif file.
        chain_id : str
            Chain ID.
        res_id_mean : float
            Mean of pocket's PDB residue IDs.
        n_pocket_atoms : int
            Number of pocket atoms.
        """

        klifs_molecule_loader = KlifsMoleculeLoader()
        klifs_molecule_loader.from_file(path_mol2, path_klifs_metadata)
        pdb_chain_loader = PdbChainLoader()
        pdb_chain_loader.from_file(path_pdb, chain_id)

        feature = SideChainOrientationFeature()
        pocket_residues = feature._get_pocket_residues(
            klifs_molecule_loader.molecule, pdb_chain_loader.chain
        )

        # Get and test the mean of pocket PDB residue IDs and the number of pocket atoms
        res_id_mean_calculated = pocket_residues.res_id.mean()

        pocket_atoms = []
        for residue in pocket_residues.pocket_residues:
            for atom in residue:
                if not atom.get_name().startswith("H"):  # Count only non-hydrogen atoms
                    pocket_atoms.append(atom.get_name())
        n_pocket_atoms_calculated = len(pocket_atoms)

        assert np.isclose(res_id_mean_calculated, res_id_mean, rtol=1e-03)
        assert n_pocket_atoms_calculated == n_pocket_atoms

    @pytest.mark.parametrize(
        "path_pdb, chain_id, residue_id, ca",
        [
            (
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ALK/2yjr_altA_chainA/protein_pymol.pdb",
                "A",
                1272,
                [5.78, 18.76, 31.15],
            ),  # Residue has CA
            (
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ALK/2yjr_altA_chainA/protein_pymol.pdb",
                "A",
                1273,
                None,
            ),  # Residue has no CA
        ],
    )
    def test_get_ca(self, path_pdb, chain_id, residue_id, ca):
        """
        Test if CA atom is retrieved correctly from a residue (test if-else cases).

        Parameters
        ----------
        path_pdb : pathlib.Path
            Path to cif file.
        chain_id : str
            Chain ID.
        residue_id : int
            Residue ID.
        ca : list of int or None
            3D coordinates of CA atom.
        """

        pdb_chain_loader = PdbChainLoader()
        pdb_chain_loader.from_file(path_pdb, chain_id)

        chain = pdb_chain_loader.chain
        residue = chain[residue_id]

        feature = SideChainOrientationFeature()
        ca_calculated = feature._get_ca(residue)

        if ca_calculated and ca:
            # Check only x coordinate
            assert np.isclose(list(ca_calculated)[0], ca[0], rtol=1e-03)
            assert isinstance(ca_calculated, Vector)
        else:
            assert ca_calculated == ca

    @pytest.mark.parametrize(
        "path_pdb, chain_id, residue_id",
        [
            (
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ADCK3/5i35_chainA/protein_pymol.pdb",
                "A",
                337,
            ),  # ALA
            (
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ADCK3/5i35_chainA/protein_pymol.pdb",
                "A",
                357,
            ),  # Non-standard residue
        ],
    )
    def test_get_pcb_from_gly_valueerror(self, path_pdb, chain_id, residue_id):
        """
        Test exceptions in pseudo-CB calculation for GLY.

        Parameters
        ----------
        path_pdb : pathlib.Path
            Path to cif file.
        chain_id : str
            Chain ID.
        residue_id : int
            Residue ID.
        """

        pdb_chain_loader = PdbChainLoader()
        pdb_chain_loader.from_file(path_pdb, chain_id)

        chain = pdb_chain_loader.chain
        try:
            residue = chain[residue_id]
        except KeyError:
            # For non-standard residue MSE indexing did not work, thus use this workaround
            residue = [i for i in chain.get_list() if i.get_id()[1] == residue_id][0]

        with pytest.raises(ValueError):
            feature = SideChainOrientationFeature()
            feature._get_pcb_from_gly(residue)

    @pytest.mark.parametrize(
        "path_pdb, chain_id, residue_id, ca_pcb",
        [
            (
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ADCK3/5i35_chainA/protein_pymol.pdb",
                "A",
                272,
                np.array([12.22, 8.37, 31.38]),
            ),  # GLY
        ],
    )
    def test_get_pcb_from_gly(self, path_pdb, chain_id, residue_id, ca_pcb):
        """
        Test pseudo-CB calculation for GLY.

        Parameters
        ----------
        path_pdb : pathlib.Path
            Path to cif file.
        chain_id : str
            Chain ID.
        residue_id : int
            Residue ID.
        ca_pcb : numpy.array
            Pseudo-CB atom coordinates.
        """

        pdb_chain_loader = PdbChainLoader()
        pdb_chain_loader.from_file(path_pdb, chain_id)

        chain = pdb_chain_loader.chain
        try:
            residue = chain[residue_id]
        except KeyError:
            # For non-standard residue MSE indexing did not work, thus use this workaround
            residue = [i for i in chain.get_list() if i.get_id()[1] == residue_id][0]

        feature = SideChainOrientationFeature()
        ca_pcb_calculated = feature._get_pcb_from_gly(residue)

        assert np.isclose(ca_pcb_calculated.get_array().mean(), ca_pcb.mean(), rtol=1e-04)

    @pytest.mark.parametrize(
        "path_pdb, chain_id, residue_id, ca_pcb",
        [
            (
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ADCK3/5i35_chainA/protein_pymol.pdb",
                "A",
                272,
                np.array([12.22, 8.37, 31.38]),
            ),  # GLY
            (
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ADCK3/5i35_chainA/protein_pymol.pdb",
                "A",
                337,
                np.array([4.89, 12.19, 43.60]),
            ),  # Residue with +- residue
            (
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/CHK1/4jik_chainA/protein_pymol.pdb",
                "A",
                19,
                None,
            ),  # Residue without + residue
        ],
    )
    def test_get_pcb_from_residue(self, path_pdb, chain_id, residue_id, ca_pcb):
        """
        Test pseudo-CB calculation for a residue.

        Parameters
        ----------
        path_pdb : pathlib.Path
            Path to cif file.
        chain_id : str
            Chain ID.
        residue_id : int
            Residue ID.
        ca_pcb : numpy.array
            Pseudo-CB atom coordinates.
        """

        pdb_chain_loader = PdbChainLoader()
        pdb_chain_loader.from_file(path_pdb, chain_id)

        chain = pdb_chain_loader.chain
        try:
            residue = chain[residue_id]
        except KeyError:
            # For non-standard residue MSE indexing did not work, thus use this workaround
            residue = [i for i in chain.get_list() if i.get_id()[1] == residue_id][0]

        feature = SideChainOrientationFeature()
        feature.chain = chain  # Set chain value artificially
        ca_pcb_calculated = feature._get_pcb_from_residue(residue, chain)

        if ca_pcb is None:
            assert ca_pcb_calculated is None
        else:
            assert np.isclose(ca_pcb_calculated.get_array().mean(), ca_pcb.mean(), rtol=1e-04)

    @pytest.mark.parametrize(
        "path_pdb, chain_id, residue_id, side_chain_centroid",
        [
            (
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ADCK3/5i35_chainA/protein_pymol.pdb",
                "A",
                272,
                np.array([12.22, 8.37, 31.38]),
            ),  # GLY with pCB
            (
                PATH_TEST_DATA
                / "KLIFS_download"
                / "HUMAN/CHK1/3ot8_altA_chainA/protein_pymol.pdb",
                "A",
                18,
                None,
            ),  # GLY without pCB (missing C and CA)
            (
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ADCK3/5i35_chainA/protein_pymol.pdb",
                "A",
                337,
                np.array([4.73, 12.85, 43.35]),
            ),  # ALA with CB
            (
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/CHK1/3nlb_chainA/protein_pymol.pdb",
                "A",
                19,
                np.array([5.47, 13.78, 32.29]),
            ),  # ALA with pCB
            (
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/CHK1/4jik_chainA/protein_pymol.pdb",
                "A",
                19,
                None,
            ),  # ALA without pCB (missing CA)
            (
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ADCK3/5i35_chainA/protein_pymol.pdb",
                "A",
                336,
                np.array([4.48, 15.79, 46.66]),
            ),  # Standard residue (side chain) with enough atoms
            (
                PATH_TEST_DATA
                / "KLIFS_download"
                / "HUMAN/AAK1/5l4q_altA_chainA/protein_pymol.pdb",
                "A",
                130,
                np.array([-5.11, 20.31, 49.99]),
            ),  # Standard residue with too few atoms but CB atom
            (
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/BTK/4yhf_altA_chainB/protein_pymol.pdb",
                "B",
                412,
                np.array([3.42, 12.33, 35.24]),
            ),  # Standard residue with too few atoms, no CB atom, but pCB atom
            (
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/CHK1/4jik_chainA/protein_pymol.pdb",
                "A",
                51,
                None,
            ),  # Standard residue with too few atoms and no CB and pCB atom
            (
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ADCK3/5i35_chainA/protein_pymol.pdb",
                "A",
                357,
                np.array([12.94, 22.55, 44.96]),
            ),  # Non-standard residue with enough atoms (>0)
            (
                PATH_TEST_DATA
                / "KLIFS_download"
                / "HUMAN/AAK1/5l4q_altA_chainA/protein_pymol.pdb",
                "A",
                57,
                np.array([10.44, 12.84, 31.17]),
            ),  # Side chain containing H atoms
            # ('some.pdb', 'X', 0, None),  # Non-standard residue side chain with no atoms
        ],
    )
    def test_get_side_chain_centroid(self, path_pdb, chain_id, residue_id, side_chain_centroid):
        """
        Test if side chain centroid is retrieved correctly from a residue (test if-else cases).

        Parameters
        ----------
        path_pdb : pathlib.Path
            Path to cif file.
        chain_id : str
            Chain ID.
        residue_id : int
            Residue ID.
        side_chain_centroid : list of int or None
            3D coordinates of CA atom.
        """

        pdb_chain_loader = PdbChainLoader()
        pdb_chain_loader.from_file(path_pdb, chain_id)

        chain = pdb_chain_loader.chain
        try:
            residue = chain[residue_id]
        except KeyError:
            # For non-standard residue MSE indexing did not work, thus use this workaround
            residue = [i for i in chain.get_list() if i.get_id()[1] == residue_id][0]

        feature = SideChainOrientationFeature()
        side_chain_centroid_calculated = feature._get_side_chain_centroid(residue, chain)
        print(side_chain_centroid_calculated)

        if side_chain_centroid is not None:
            # Check only x coordinate
            assert np.isclose(
                side_chain_centroid_calculated.get_array().mean(),
                side_chain_centroid.mean(),
                rtol=1e-03,
            )
            assert isinstance(side_chain_centroid_calculated, Vector)
        else:
            assert side_chain_centroid_calculated == side_chain_centroid

    @pytest.mark.parametrize(
        "path_klifs_metadata, path_mol2, path_pdb, chain_id, pocket_centroid",
        [
            (
                PATH_TEST_DATA / "klifs_metadata.csv",
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ABL1/2g2i_chainA/pocket.mol2",
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ABL1/2g2i_chainA/protein_pymol.pdb",
                "A",
                np.array([0.99, 21.06, 36.70]),
            )
        ],
    )
    def test_get_pocket_centroid(
        self, path_klifs_metadata, path_mol2, path_pdb, chain_id, pocket_centroid
    ):
        """
        Test pocket centroid calculation.

        Parameters
        ----------
        path_klifs_metadata : pathlib.Path
            Path to unfiltered KLIFS metadata.
        path_mol2 : pathlib.Path
            Path to mol2 file.
        path_pdb : pathlib.Path
            Path to cif file.
        chain_id : str
            Chain ID.
        pocket_centroid : list of float
            Pocket centroid coordinates.
        """

        klifs_molecule_loader = KlifsMoleculeLoader()
        klifs_molecule_loader.from_file(path_mol2, path_klifs_metadata)
        pdb_chain_loader = PdbChainLoader()
        pdb_chain_loader.from_file(path_pdb, chain_id)

        feature = SideChainOrientationFeature()
        pocket_residues = feature._get_pocket_residues(
            klifs_molecule_loader.molecule, pdb_chain_loader.chain
        )
        pocket_centroid_calculated = feature._get_pocket_centroid(pocket_residues)

        if pocket_centroid is not None:
            assert np.isclose(
                pocket_centroid_calculated.get_array().mean(), pocket_centroid.mean(), rtol=1e-03
            )
            assert isinstance(pocket_centroid_calculated, Vector)
        else:
            assert pocket_centroid_calculated == pocket_centroid

    @pytest.mark.parametrize(
        "path_klifs_metadata, path_mol2, path_pdb, chain_id, n_vectors",
        [
            (
                PATH_TEST_DATA / "klifs_metadata.csv",
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ABL1/2g2i_chainA/pocket.mol2",
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ABL1/2g2i_chainA/protein_pymol.pdb",
                "A",
                82,
            )
        ],
    )
    def test_get_pocket_vectors(
        self, path_klifs_metadata, path_mol2, path_pdb, chain_id, n_vectors
    ):
        """
        Test if returned DataFrame for pocket vectors contains correct column names and correct number of vectors
        (= number of pocket residues).

        Parameters
        ----------
        path_klifs_metadata : pathlib.Path
            Path to unfiltered KLIFS metadata.
        path_mol2 : pathlib.Path
            Path to mol2 file.
        path_pdb : pathlib.Path
            Path to cif file.
        chain_id : str
            Chain ID.
        n_vectors : int
            Number of vectors (= number of pocket residues)
        """

        klifs_molecule_loader = KlifsMoleculeLoader()
        klifs_molecule_loader.from_file(path_mol2, path_klifs_metadata)
        pdb_chain_loader = PdbChainLoader()
        pdb_chain_loader.from_file(path_pdb, chain_id)

        feature = SideChainOrientationFeature()
        pocket_residues = feature._get_pocket_residues(
            klifs_molecule_loader.molecule, pdb_chain_loader.chain
        )
        pocket_vectors = feature._get_pocket_vectors(pocket_residues, pdb_chain_loader.chain)

        # Test if DataFrame contains correct columns
        pocket_vectors_columns = [
            "klifs_id",
            "res_id",
            "res_name",
            "ca",
            "side_chain_centroid",
            "pocket_centroid",
        ]
        assert list(pocket_vectors.columns) == pocket_vectors_columns
        assert len(pocket_vectors) == n_vectors

    @pytest.mark.parametrize(
        "path_klifs_metadata, path_mol2, path_pdb, chain_id, angles_mean",
        [
            (
                PATH_TEST_DATA / "klifs_metadata.csv",
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ABL1/2g2i_chainA/pocket.mol2",
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ABL1/2g2i_chainA/protein_pymol.pdb",
                "A",
                95.07,
            )
        ],
    )
    def test_get_vertex_angles(
        self, path_klifs_metadata, path_mol2, path_pdb, chain_id, angles_mean
    ):
        """
        Test if vertex angles are calculated correctly (check mean angle), and if returned
        DataFrame contains correct column
        name.

        Parameters
        ----------
        path_klifs_metadata : pathlib.Path
            Path to unfiltered KLIFS metadata.
        path_mol2 : pathlib.Path
            Path to mol2 file.
        path_pdb : pathlib.Path
            Path to cif file.
        chain_id : str
            Chain ID.
        angles_mean : float
            Mean of non-None angles.
        """

        klifs_molecule_loader = KlifsMoleculeLoader()
        klifs_molecule_loader.from_file(path_mol2, path_klifs_metadata)
        pdb_chain_loader = PdbChainLoader()
        pdb_chain_loader.from_file(path_pdb, chain_id)

        feature = SideChainOrientationFeature()
        pocket_residues = feature._get_pocket_residues(
            klifs_molecule_loader.molecule, pdb_chain_loader.chain
        )
        pocket_vectors = feature._get_pocket_vectors(pocket_residues, pdb_chain_loader.chain)
        angles_calculated = feature._get_vertex_angles(pocket_vectors)

        assert list(angles_calculated.columns) == ["vertex_angle"]

        # Calculate and test mean of all angles (excluding NaN values)
        angles_mean_calculated = angles_calculated.vertex_angle.mean()
        assert np.isclose(angles_mean_calculated, angles_mean, rtol=1e-03)

    @pytest.mark.parametrize(
        "vertex_angles",
        [pd.DataFrame([0.0] * 85, index=range(1, 86), columns=["sco"])],  # Wrong column
    )
    def test_get_categories_valueerror(self, vertex_angles):
        """
        Test if exception are raised.

        Parameters
        ----------
        vertex_angles : pandas.DataFrame
            Vertex angles (column) for up to 85 residues (rows).
        """

        with pytest.raises(ValueError):
            feature = SideChainOrientationFeature()
            feature._get_categories(vertex_angles)

    @pytest.mark.parametrize(
        "vertex_angles, categories",
        [
            (
                pd.DataFrame([0.0] * 85, index=range(1, 86), columns=["vertex_angle"]),
                pd.DataFrame([0.0] * 85, index=range(1, 86), columns=["sco"]),
            )
        ],
    )
    def test_get_categories(self, vertex_angles, categories):
        """
        Test transformation of vertex angles to categories (for side chain orientation).

        Parameters
        ----------
        vertex_angles : pandas.DataFrame
            Vertex angles (column) for up to 85 residues (rows).
        categories : pandas.DataFrame
            Side chain orientation categories (column) for up to 85 residues (rows).
        """

        feature = SideChainOrientationFeature()
        categories_calculated = feature._get_categories(vertex_angles)

        assert categories_calculated.equals(categories)

    @pytest.mark.parametrize(
        "vertex_angle, category",
        [
            (0.0, 0.0),
            (1.0, 0.0),
            (45.0, 0.0),
            (46.0, 1.0),
            (90.0, 1.0),
            (91.0, 2.0),
            (180.0, 2.0),
            (np.nan, np.nan),
        ],
    )
    def test_get_category_from_vertex_angle(self, vertex_angle, category):
        """
        Test tranformation of vertex angle to category (for side chain orientation).

        Parameters
        ----------
        vertex_angle : float
            Vertex angle between a residue's CA atom (vertex), side chain centroid and pocket
            centroid. Ranges between 0.0 and 180.0.
        category : float
            Side chain orientation towards pocket:
            Inwards (category 0.0), intermediate (category 1.0), and outwards (category 2.0).
        """

        feature = SideChainOrientationFeature()
        category_calculated = feature._get_category_from_vertex_angle(vertex_angle)

        if not np.isnan(vertex_angle):
            assert isinstance(category_calculated, float)
            assert category_calculated == category
        else:
            assert np.isnan(category_calculated)

    @pytest.mark.parametrize(
        "path_klifs_metadata, path_mol2, path_pdb, chain_id",
        [
            (
                PATH_TEST_DATA / "klifs_metadata.csv",
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ABL1/2g2i_chainA/pocket.mol2",
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ABL1/2g2i_chainA/protein_pymol.pdb",
                "A",
            )
        ],
    )
    def test_from_molecule(self, path_klifs_metadata, path_mol2, path_pdb, chain_id):
        """
        Test if SideChainOrientation attributes features and features_verbose contain the correct
        column names.
        Values are tested already in other test_sidechainorientation_* unit tests.

        Parameters
        ----------
        path_klifs_metadata : pathlib.Path
            Path to unfiltered KLIFS metadata.
        path_mol2 : pathlib.Path
            Path to mol2 file.
        path_pdb : pathlib.Path
            Path to cif file.
        chain_id : str
            Chain ID.
        """

        klifs_molecule_loader = KlifsMoleculeLoader()
        klifs_molecule_loader.from_file(path_mol2, path_klifs_metadata)
        pdb_chain_loader = PdbChainLoader()
        pdb_chain_loader.from_file(path_pdb, chain_id)

        feature = SideChainOrientationFeature()
        feature.from_molecule(klifs_molecule_loader.molecule, pdb_chain_loader.chain)

        # Check column names
        features_columns = ["sco"]
        features_verbose_columns = [
            "klifs_id",
            "res_id",
            "res_name",
            "ca",
            "side_chain_centroid",
            "pocket_centroid",
            "vertex_angle",
            "sco",
        ]

        # Test column names
        assert list(feature.features.columns) == features_columns
        assert list(feature.features_verbose.columns) == features_verbose_columns


class TestsPhysicoChemicalFeatures:
    """
    Test PhysicoChemicalFeatures class methods.
    """

    @pytest.mark.parametrize(
        "path_klifs_metadata, path_mol2, path_pdb, chain_id",
        [
            (
                PATH_TEST_DATA / "klifs_metadata.csv",
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ABL1/2g2i_chainA/pocket.mol2",
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ABL1/2g2i_chainA/protein_pymol.pdb",
                "A",
            )
        ],
    )
    def test_from_molecule(self, path_klifs_metadata, path_mol2, path_pdb, chain_id):
        """
        Test length (85 rows for 85 KLIFS residues) and columns names of features DataFrame.
        Values are tested already in respective feature unit test.

        Parameters
        ----------
        path_klifs_metadata : pathlib.Path
            Path to unfiltered KLIFS metadata.
        path_mol2 : pathlib.Path
            Path to mol2 file.
        path_pdb : pathlib.Path
            Path to cif file.
        chain_id : str
            Chain ID.
        """

        klifs_molecule_loader = KlifsMoleculeLoader()
        klifs_molecule_loader.from_file(path_mol2, path_klifs_metadata)
        pdb_chain_loader = PdbChainLoader()
        pdb_chain_loader.from_file(path_pdb, chain_id)

        physicochemicalfeatures = PhysicoChemicalFeatures()
        physicochemicalfeatures.from_molecule(
            klifs_molecule_loader.molecule, pdb_chain_loader.chain
        )
        features = physicochemicalfeatures.features

        physiochemicalfeatures_columns = (
            "size hbd hba charge aromatic aliphatic sco exposure".split()
        )
        assert list(features.columns) == physiochemicalfeatures_columns
        assert len(features) == 85


class TestsSpatialFeatures:
    """
    Test SpatialFeatures class methods.
    """

    @pytest.mark.parametrize(
        "path_klifs_metadata, path_mol2",
        [
            (
                PATH_TEST_DATA / "klifs_metadata.csv",
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ABL1/2g2i_chainA/pocket.mol2",
            )
        ],
    )
    def test_from_molecule(self, path_klifs_metadata, path_mol2):
        """
        Test length (85 rows for 85 KLIFS residues) and columns names of features DataFrame.
        Values are tested already in respective feature unit test.

        Parameters
        ----------
        path_klifs_metadata : pathlib.Path
            Path to unfiltered KLIFS metadata.
        path_mol2 : pathlib.Path
            Path to mol2 file.
        """

        klifs_molecule_loader = KlifsMoleculeLoader()
        klifs_molecule_loader.from_file(path_mol2, path_klifs_metadata)

        spatialfeatures = SpatialFeatures()
        spatialfeatures.from_molecule(klifs_molecule_loader.molecule)
        features = spatialfeatures.features

        spatialfeatures_columns = [
            "distance_to_centroid",
            "distance_to_hinge_region",
            "distance_to_dfg_region",
            "distance_to_front_pocket",
        ]
        assert list(features.columns) == spatialfeatures_columns
        assert len(features) == 85

    @pytest.mark.parametrize(
        "path_klifs_metadata, path_mol2, reference_point_name, anchor_residue_klifs_ids, x_coordinate",
        [
            (
                PATH_TEST_DATA / "klifs_metadata.csv",
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/AAK1/4wsq_altA_chainB/pocket.mol2",
                "hinge_region",
                [16, 47, 80],
                6.25545,
            ),
            (
                PATH_TEST_DATA / "klifs_metadata.csv",
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/AAK1/4wsq_altA_chainB/pocket.mol2",
                "dfg_region",
                [20, 23, 81],
                11.6846,
            ),
            (
                PATH_TEST_DATA / "klifs_metadata.csv",
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/AAK1/4wsq_altA_chainB/pocket.mol2",
                "front_pocket",
                [6, 48, 75],
                float("nan"),
            ),
        ],
    )
    def test_get_anchor_atoms(
        self,
        path_klifs_metadata,
        path_mol2,
        reference_point_name,
        anchor_residue_klifs_ids,
        x_coordinate,
    ):
        """
        Test function that calculates the anchor atoms for different scenarios
        (missing anchor residues, missing neighbors)

        Parameters
        ----------
        path_klifs_metadata : pathlib.Path
            Path to unfiltered KLIFS metadata.
        path_mol2 : pathlib.Path
            Path to file originating from test data folder.
        reference_point_name : str
            Reference point name, e.g. 'hinge_region'.
        anchor_residue_klifs_ids : list of int
            List of KLIFS IDs that are used to calculate a given reference point.
        x_coordinate: float
            X coordinate of first anchor atom.
        """

        klifs_molecule_loader = KlifsMoleculeLoader()
        klifs_molecule_loader.from_file(path_mol2, path_klifs_metadata)
        molecule = klifs_molecule_loader.molecule

        # Delete residues

        # Case: Missing anchor residue but neighboring residues available
        molecule.df.drop(molecule.df[molecule.df.klifs_id == 16].index, inplace=True)

        # Case: Missing anchor residue but neighboring residues available
        molecule.df.drop(molecule.df[molecule.df.klifs_id.isin([18, 19])].index, inplace=True)

        # Case: Missing anchor residue but neighboring residues available
        molecule.df.drop(molecule.df[molecule.df.klifs_id.isin([24, 25])].index, inplace=True)

        # Case: Missing anchor and neighboring residues
        molecule.df.drop(molecule.df[molecule.df.klifs_id.isin([5, 6, 7])].index, inplace=True)

        # Get spatial features
        spatial_features = SpatialFeatures()
        spatial_features.from_molecule(molecule)

        # Get anchor atoms
        anchors = spatial_features.get_anchor_atoms(molecule)

        assert list(anchors[reference_point_name].index) == anchor_residue_klifs_ids

        # Ugly workaround to test NaN values in anchors
        if math.isnan(x_coordinate):
            assert math.isnan(anchors[reference_point_name].loc[anchor_residue_klifs_ids[0], "x"])
        else:
            assert (
                anchors[reference_point_name].loc[anchor_residue_klifs_ids[0], "x"] == x_coordinate
            )

    @pytest.mark.parametrize(
        "path_klifs_metadata, path_mol2, x_coordinate",
        [
            (
                PATH_TEST_DATA / "klifs_metadata.csv",
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/AAK1/4wsq_altA_chainB/pocket.mol2",
                1.02664,
            )
        ],
    )
    def test_get_reference_points(self, path_klifs_metadata, path_mol2, x_coordinate):
        """
        Test calculation of reference point "centroid" of a pocket.

        Parameters
        ----------
        path_klifs_metadata : pathlib.Path
            Path to unfiltered KLIFS metadata.
        path_mol2 : pathlib.Path
            Path to file originating from test data folder.
        x_coordinate: float
            X coordinate of the centroid.
        """

        klifs_molecule_loader = KlifsMoleculeLoader()
        klifs_molecule_loader.from_file(path_mol2, path_klifs_metadata)
        molecule = klifs_molecule_loader.molecule

        # Get spatial features
        spatial_features = SpatialFeatures()
        spatial_features.from_molecule(molecule)

        # Get reference points
        reference_points = spatial_features.get_reference_points(molecule)
        print(reference_points.centroid.x)

        assert np.isclose(reference_points.centroid.x, x_coordinate, rtol=1e-04)


class TestsFingerprint:
    """
    Test Fingerprint class methods.
    """

    @pytest.mark.parametrize(
        "path_klifs_metadata, path_mol2, path_pdb, chain_id",
        [
            (
                PATH_TEST_DATA / "klifs_metadata.csv",
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ABL1/2g2i_chainA/pocket.mol2",
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ABL1/2g2i_chainA/protein_pymol.pdb",
                "A",
            )
        ],
    )
    def test_from_molecule(self, path_klifs_metadata, path_mol2, path_pdb, chain_id):
        """
        Test if Fingerprint class attributes (accessed via property function) have correct
        DataFrame shape, column and index names.

        Parameters
        ----------
        path_klifs_metadata : pathlib.Path
            Path to unfiltered KLIFS metadata.
        path_mol2 : pathlib.Path
            Path to mol2 file.
        path_pdb : pathlib.Path
            Path to cif file.
        chain_id : pathlib.Path
            Chain ID.
        """

        klifs_molecule_loader = KlifsMoleculeLoader()
        klifs_molecule_loader.from_file(path_mol2, path_klifs_metadata)
        pdb_chain_loader = PdbChainLoader()
        pdb_chain_loader.from_file(path_pdb, chain_id)

        fingerprint = Fingerprint()
        fingerprint.from_molecule(klifs_molecule_loader.molecule, pdb_chain_loader.chain)

        klifs_ids = list(range(1, 86))

        # Non-normalized
        assert fingerprint.physicochemical.shape == (85, 8)
        assert list(fingerprint.physicochemical.index) == klifs_ids
        assert list(fingerprint.physicochemical.columns) == FEATURE_NAMES["physicochemical"]

        assert fingerprint.distances.shape == (85, 4)
        assert list(fingerprint.distances.index) == klifs_ids
        assert list(fingerprint.distances.columns) == FEATURE_NAMES["distances"]

        assert fingerprint.moments.shape == (4, 3)
        assert list(fingerprint.moments.index) == FEATURE_NAMES["distances"]
        assert list(fingerprint.moments.columns) == FEATURE_NAMES["moments"]

        assert (
            list(fingerprint.physicochemical_distances.keys())
            == "physicochemical distances".split()
        )
        assert (
            list(fingerprint.physicochemical_moments.keys()) == "physicochemical moments".split()
        )

        # Normalized
        assert fingerprint.physicochemical_normalized.shape == (85, 8)
        assert list(fingerprint.physicochemical_normalized.index) == klifs_ids
        assert (
            list(fingerprint.physicochemical_normalized.columns)
            == FEATURE_NAMES["physicochemical"]
        )

        assert fingerprint.distances.shape == (85, 4)
        assert list(fingerprint.distances_normalized.index) == klifs_ids
        assert list(fingerprint.distances_normalized.columns) == FEATURE_NAMES["distances"]

        assert fingerprint.moments.shape == (4, 3)
        assert list(fingerprint.moments_normalized.index) == FEATURE_NAMES["distances"]
        assert list(fingerprint.moments_normalized.columns) == FEATURE_NAMES["moments"]

        assert (
            list(fingerprint.physicochemical_distances_normalized.keys())
            == "physicochemical distances".split()
        )
        assert (
            list(fingerprint.physicochemical_moments_normalized.keys())
            == "physicochemical moments".split()
        )

    @pytest.mark.parametrize(
        "physicochemical, physicochemical_normalized",
        [
            (
                pd.DataFrame(
                    [[3.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                    columns=FEATURE_NAMES["physicochemical"],
                ),
                pd.DataFrame(
                    [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0]],
                    columns=FEATURE_NAMES["physicochemical"],
                ),
            ),
            (
                pd.DataFrame(
                    [[2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.8]],
                    columns=FEATURE_NAMES["physicochemical"],
                ),
                pd.DataFrame(
                    [[0.5, 0.3333, 0.5, 0.5, 0.0, 0.0, 1.0, 0.8]],
                    columns=FEATURE_NAMES["physicochemical"],
                ),
            ),
            (
                pd.DataFrame([[np.nan] * 8], columns=FEATURE_NAMES["physicochemical"]),
                pd.DataFrame([[np.nan] * 8], columns=FEATURE_NAMES["physicochemical"]),
            ),
        ],
    )
    def test_normalize_physicochemical_bits(self, physicochemical, physicochemical_normalized):
        """
        Test normalization of physicochemical bits.

        Parameters
        ----------
        physicochemical : pandas.DataFrame
            Physicochemical bits.
        physicochemical_normalized : pandas.DataFrame
            Normalized physicochemical bits.
        """

        fingerprint = Fingerprint()
        fingerprint.molecule_code = "molecule"
        fingerprint.fingerprint["physicochemical"] = physicochemical

        physicochemical_normalized_calculated = fingerprint._normalize_physicochemical_bits()

        for feature in FEATURE_NAMES["physicochemical"]:

            if np.isnan(physicochemical.iloc[0, 0]):
                assert np.isnan(physicochemical_normalized_calculated[feature][0]) and np.isnan(
                    physicochemical_normalized[feature][0]
                )
            else:
                assert np.isclose(
                    physicochemical_normalized_calculated[feature][0],
                    physicochemical_normalized[feature][0],
                    rtol=1e-03,
                )

    @pytest.mark.parametrize(
        "distances, distances_normalized",
        [
            (
                pd.DataFrame([[1, 1, 1, 1]], columns=FEATURE_NAMES["distances"]),
                pd.DataFrame([[0.0, 0.0, 0.0, 0.0]], columns=FEATURE_NAMES["distances"]),
            ),
            (
                pd.DataFrame(
                    [
                        [
                            DISTANCE_CUTOFFS["distance_to_centroid"][0],
                            DISTANCE_CUTOFFS["distance_to_hinge_region"][0],
                            DISTANCE_CUTOFFS["distance_to_dfg_region"][0],
                            DISTANCE_CUTOFFS["distance_to_front_pocket"][0],
                        ]
                    ],
                    columns=FEATURE_NAMES["distances"],
                ),
                pd.DataFrame([[0.0, 0.0, 0.0, 0.0]], columns=FEATURE_NAMES["distances"]),
            ),
            (
                pd.DataFrame([[10, 10, 10, 10]], columns=FEATURE_NAMES["distances"]),
                pd.DataFrame(
                    [[0.3792, 0.3110, 0.2438, 0.2510]], columns=FEATURE_NAMES["distances"]
                ),
            ),
            (
                pd.DataFrame(
                    [
                        [
                            DISTANCE_CUTOFFS["distance_to_centroid"][1],
                            DISTANCE_CUTOFFS["distance_to_hinge_region"][1],
                            DISTANCE_CUTOFFS["distance_to_dfg_region"][1],
                            DISTANCE_CUTOFFS["distance_to_front_pocket"][1],
                        ]
                    ],
                    columns=FEATURE_NAMES["distances"],
                ),
                pd.DataFrame([[1.0, 1.0, 1.0, 1.0]], columns=FEATURE_NAMES["distances"]),
            ),
            (
                pd.DataFrame([[30, 30, 30, 30]], columns=FEATURE_NAMES["distances"]),
                pd.DataFrame([[1.0, 1.0, 1.0, 1.0]], columns=FEATURE_NAMES["distances"]),
            ),
            (
                pd.DataFrame([[np.nan] * 4], columns=FEATURE_NAMES["distances"]),
                pd.DataFrame([[np.nan] * 4], columns=FEATURE_NAMES["distances"]),
            ),
        ],
    )
    def test_normalize_distances_bits(self, distances, distances_normalized):
        """
        Test normalization of distances bits.

        Parameters
        ----------
        distances : pandas.DataFrame
            Distances.
        distances_normalized : pandas.DataFrame
            Normalized distances.
        """

        fingerprint = Fingerprint()
        fingerprint.molecule_code = "molecule"
        fingerprint.fingerprint["distances"] = distances

        distances_normalized_calculated = fingerprint._normalize_distances_bits()

        for feature in FEATURE_NAMES["distances"]:

            if np.isnan(distances.iloc[0, 0]):
                assert np.isnan(distances_normalized_calculated[feature][0]) and np.isnan(
                    distances_normalized[feature][0]
                )
            else:
                assert np.isclose(
                    distances_normalized_calculated[feature][0],
                    distances_normalized[feature][0],
                    rtol=1e-03,
                )

    @pytest.mark.parametrize(
        "moments, moments_normalized",
        [
            (
                pd.DataFrame([[11, 3, -2]], columns=FEATURE_NAMES["moments"]),
                pd.DataFrame([[0.0, 0.0, 0.0]], columns=FEATURE_NAMES["moments"]),
            ),
            (
                pd.DataFrame(
                    [
                        [
                            MOMENT_CUTOFFS["moment1"][0],
                            MOMENT_CUTOFFS["moment2"][0],
                            MOMENT_CUTOFFS["moment3"][0],
                        ]
                    ],
                    columns=FEATURE_NAMES["moments"],
                ),
                pd.DataFrame([[0.0, 0.0, 0.0]], columns=FEATURE_NAMES["moments"]),
            ),
            (
                pd.DataFrame([[12, 4, 1]], columns=FEATURE_NAMES["moments"]),
                pd.DataFrame([[0.1301, 0.355, 0.4030]], columns=FEATURE_NAMES["moments"]),
            ),
            (
                pd.DataFrame(
                    [
                        [
                            MOMENT_CUTOFFS["moment1"][1],
                            MOMENT_CUTOFFS["moment2"][1],
                            MOMENT_CUTOFFS["moment3"][1],
                        ]
                    ],
                    columns=FEATURE_NAMES["moments"],
                ),
                pd.DataFrame([[1.0, 1.0, 1.0]], columns=FEATURE_NAMES["moments"]),
            ),
            (
                pd.DataFrame([[15, 6, 5]], columns=FEATURE_NAMES["moments"]),
                pd.DataFrame([[1.0, 1.0, 1.0]], columns=FEATURE_NAMES["moments"]),
            ),
            (
                pd.DataFrame([[np.nan] * 3], columns=FEATURE_NAMES["moments"]),
                pd.DataFrame([[np.nan] * 3], columns=FEATURE_NAMES["moments"]),
            ),
        ],
    )
    def test_normalize_moments_bits(self, moments, moments_normalized):

        fingerprint = Fingerprint()
        fingerprint.molecule_code = "molecule"
        fingerprint.fingerprint["moments"] = moments

        moments_normalized_calculated = fingerprint._normalize_moments_bits()

        for feature in FEATURE_NAMES["moments"]:

            if np.isnan(moments.iloc[0, 0]):
                assert np.isnan(moments_normalized_calculated[feature][0]) and np.isnan(
                    moments_normalized[feature][0]
                )
            else:
                assert np.isclose(
                    moments_normalized_calculated[feature][0],
                    moments_normalized[feature][0],
                    rtol=1e-03,
                )

    @pytest.mark.parametrize(
        "distances, moments",
        [
            (
                pd.DataFrame([[1, 1], [4, 4], [4, 4]]),
                pd.DataFrame(
                    [[3.00, 1.41, -1.26], [3.00, 1.41, -1.26]],
                    columns="moment1 moment2 moment3".split(),
                ),
            ),
            (
                pd.DataFrame([[1, 2]]),
                pd.DataFrame(
                    [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], columns="moment1 moment2 moment3".split()
                ),
            ),
        ],
    )
    def test_calc_moments(self, distances, moments):

        fingerprint = Fingerprint()
        moments_calculated = fingerprint._calc_moments(distances)

        print(len(distances))
        print(moments_calculated)

        assert np.isclose(moments_calculated.moment1[0], moments.moment1[0], rtol=1e-02)
        assert np.isclose(moments_calculated.moment2[0], moments.moment2[0], rtol=1e-02)
        assert np.isclose(moments_calculated.moment3[0], moments.moment3[0], rtol=1e-02)

    @pytest.mark.parametrize(
        "value, minimum, maximum, value_normalized",
        [
            (1.00, 2.00, 3.00, 0.00),
            (2.00, 2.00, 3.00, 0.00),
            (2.50, 2.00, 3.00, 0.50),
            (3.00, 2.00, 3.00, 1.00),
            (4.00, 2.00, 3.00, 1.00),
            (np.nan, 2.00, 3.00, np.nan),
        ],
    )
    def test_normalize(self, value, minimum, maximum, value_normalized):
        """
        Test value normalization.

        Parameters
        ----------
        value : float or int
                Value to be normalized.
        minimum : float or int
            Minimum value for normalization, values equal or greater than this minimum are set
            to 0.0.
        maximum : float or int
            Maximum value for normalization, values equal or greater than this maximum are set
            to 1.0.
        value_normalized : float
            Normalized value.
        """

        fingerprint = Fingerprint()
        value_normalized_calculated = fingerprint._normalize(value, minimum, maximum)

        if np.isnan(value):
            assert np.isnan(value_normalized_calculated) and np.isnan(value_normalized)
        else:
            assert np.isclose(value_normalized_calculated, value_normalized, rtol=1e-06)
