"""
Unit and regression test for kissim.preprocessing class methods.
"""

from pathlib import Path

import pytest
import pandas as pd

from kissim.preprocessing import (
    KlifsMetadataLoader,
    KlifsMetadataFilter,
    Mol2KlifsToPymolConverter,
    Mol2ToPdbConverter,
)

PATH_TEST_DATA = Path(__name__).parent / "kissim" / "tests" / "data"


class TestsKlifsMetadataLoader:
    """
    Test KlifsMetadataLoader class methods.
    """

    @pytest.mark.parametrize(
        "kinase_names_string, kinase_names_list",
        [
            ("a", ["a"]),
            ("a (b)", ["a", "b"]),
            ("a (b", ["a", "b"]),
            ("a b)", ["a", "b"]),
            ("a (b, c)", ["a", "b", "c"]),
            ("a (b c)", ["a", "b", "c"]),
        ],
    )
    def test_format_kinase_name(self, kinase_names_string, kinase_names_list):
        """
        Test formatting of kinase name(s): One or multiple kinase names (additional names in brackets) are formated to list
        of kinase names.

        Parameters
        ----------
        kinase_names_string : str
            String, here kinase name(s).
        kinase_names_list : list of str
            List of strings, here list of kinase name(s).
        """

        klifs_metadata_loader = KlifsMetadataLoader()
        kinase_names_list_calculated = klifs_metadata_loader._format_kinase_name(
            kinase_names_string
        )

        assert kinase_names_list == kinase_names_list_calculated

    @pytest.mark.parametrize(
        "klifs_export_file, n_rows",
        [(PATH_TEST_DATA / "KLIFS_download" / "KLIFS_export.csv", 10469)],
    )
    def test_from_klifs_export_file(self, klifs_export_file, n_rows):
        """
        Test DataFrame for KLIFS_export.csv file from KLIFS database download.

        Parameters
        ----------
        klifs_export_file : pathlib.Path or str
            Path to KLIFS_export.csv file from KLIFS database download.
        n_rows : int
            Number of rows in DataFrame for KLIFS_export.csv file from KLIFS database download.
        """

        klifs_metadata_loader = KlifsMetadataLoader()
        klifs_export = klifs_metadata_loader._from_klifs_export_file(klifs_export_file)

        klifs_export_columns = [
            "kinase",
            "kinase_all",
            "family",
            "groups",
            "pdb_id",
            "chain",
            "alternate_model",
            "species",
            "ligand_orthosteric_name",
            "ligand_orthosteric_pdb_id",
            "ligand_allosteric_name",
            "ligand_allosteric_pdb_id",
            "dfg",
            "ac_helix",
        ]

        assert len(klifs_export) == n_rows
        assert list(klifs_export.columns) == klifs_export_columns

    @pytest.mark.parametrize(
        "klifs_overview_file, n_rows",
        [(PATH_TEST_DATA / "KLIFS_download" / "overview.csv", 10469)],
    )
    def test_from_klifs_overview_file(self, klifs_overview_file, n_rows):
        """
        Test DataFrame for overview.csv file from KLIFS database download.

        Parameters
        ----------
        klifs_overview_file : pathlib.Path or str
            Path to overview.csv file from KLIFS database download.
        n_rows : int
            Number of rows in DataFrame for overview.csv file from KLIFS database download.
        """

        klifs_metadata_loader = KlifsMetadataLoader()
        klifs_overview = klifs_metadata_loader._from_klifs_overview_file(klifs_overview_file)

        klifs_overview_columns = [
            "species",
            "kinase",
            "pdb_id",
            "alternate_model",
            "chain",
            "ligand_orthosteric_pdb_id",
            "ligand_allosteric_pdb_id",
            "rmsd1",
            "rmsd2",
            "qualityscore",
            "pocket",
            "resolution",
            "missing_residues",
            "missing_atoms",
            "full_ifp",
            "fp_i",
            "fp_ii",
            "bp_i_a",
            "bp_i_b",
            "bp_ii_in",
            "bp_ii_a_in",
            "bp_ii_b_in",
            "bp_ii_out",
            "bp_ii_b",
            "bp_iii",
            "bp_iv",
            "bp_v",
        ]

        assert len(klifs_overview) == n_rows
        assert list(klifs_overview.columns) == klifs_overview_columns

    @pytest.mark.parametrize(
        "klifs_overview_file, klifs_export_file",
        [
            (
                PATH_TEST_DATA / "KLIFS_download" / "overview.csv",
                PATH_TEST_DATA / "KLIFS_download" / "KLIFS_export.csv",
            )
        ],
    )
    def test_from_files(self, klifs_overview_file, klifs_export_file):
        """
        Test shape of KLIFS metadata as DataFrame.

        Parameters
        ----------
        klifs_overview_file : pathlib.Path or str
            Path to overview.csv file from KLIFS database download.
        klifs_export_file : pathlib.Path or str
            Path to KLIFS_export.csv file from KLIFS database download.
        """

        klifs_metadata_loader = KlifsMetadataLoader()
        klifs_metadata_loader.from_files(klifs_overview_file, klifs_export_file)

        assert klifs_metadata_loader.data.shape == (10469, 35)

    @pytest.mark.parametrize(
        "klifs_overview_file, klifs_export_file",
        [
            (
                PATH_TEST_DATA / "KLIFS_download" / "overview.csv",
                PATH_TEST_DATA / "KLIFS_download" / "KLIFS_export.csv",
            )
        ],
    )
    def test_data_essential(self, klifs_overview_file, klifs_export_file):
        """
        Test class attribute, i.e. column-reduced version of full DataFrame.

        Parameters
        ----------
        klifs_overview_file : pathlib.Path or str
            Path to overview.csv file from KLIFS database download.
        klifs_export_file : pathlib.Path or str
            Path to KLIFS_export.csv file from KLIFS database download.
        """

        klifs_metadata_loader = KlifsMetadataLoader()
        klifs_metadata_loader.from_files(klifs_overview_file, klifs_export_file)

        metadata_columns = [
            "pdb_id",
            "alternate_model",
            "chain",
            "kinase",
            "kinase_all",
            "family",
            "groups",
            "species",
            "dfg",
            "ac_helix",
            "pocket",
            "rmsd1",
            "rmsd2",
            "qualityscore",
            "resolution",
            "missing_residues",
            "missing_atoms",
            "filepath",
        ]

        assert list(klifs_metadata_loader.data_essential.columns) == metadata_columns


class TestsKlifsMetadataFilter:
    """
    Test KlifsMetadataFilter class methods.
    """

    @pytest.mark.parametrize(
        "klifs_overview_file, klifs_export_file, n_rows",
        [
            (
                PATH_TEST_DATA / "KLIFS_download" / "overview.csv",
                PATH_TEST_DATA / "KLIFS_download" / "KLIFS_export.csv",
                9964,
            )
        ],
    )
    def test_get_species(self, klifs_overview_file, klifs_export_file, n_rows):
        """
        Test filtering by species.

        Parameters
        ----------
        klifs_overview_file : str or pathlib.Path
            Path to KLIFS download file `overview.csv` containing mainly KLIFS alignment-related metadata.
        klifs_export_file : str or pathlib.Path
            Path to KLIFS download file `KLIFS_download/KLIFS_export.csv` containing mainly structure-related metadata.
        n_rows : int
            Number of rows (structures) after filtering.
        """

        klifs_metadata_loader = KlifsMetadataLoader()
        klifs_metadata_loader.from_files(klifs_overview_file, klifs_export_file)

        klifs_metadata = klifs_metadata_loader.data_essential

        klifs_metadata_filter = KlifsMetadataFilter()
        klifs_metadata_filter.filtered = klifs_metadata
        klifs_metadata_filter._get_species("Human")

        assert klifs_metadata_filter.filtered.shape[0] == n_rows

    @pytest.mark.parametrize(
        "klifs_overview_file, klifs_export_file, n_rows",
        [
            (
                PATH_TEST_DATA / "KLIFS_download" / "overview.csv",
                PATH_TEST_DATA / "KLIFS_download" / "KLIFS_export.csv",
                9088,
            )
        ],
    )
    def test_get_dfg(self, klifs_overview_file, klifs_export_file, n_rows):
        """
        Test filtering by DFG conformation.

        Parameters
        ----------
        klifs_overview_file : str or pathlib.Path
            Path to KLIFS download file `overview.csv` containing mainly KLIFS alignment-related metadata.
        klifs_export_file : str or pathlib.Path
            Path to KLIFS download file `KLIFS_download/KLIFS_export.csv` containing mainly structure-related metadata.
        n_rows : int
            Number of rows (structures) after filtering.
        """

        klifs_metadata_loader = KlifsMetadataLoader()
        klifs_metadata_loader.from_files(klifs_overview_file, klifs_export_file)

        klifs_metadata = klifs_metadata_loader.data_essential

        klifs_metadata_filter = KlifsMetadataFilter()
        klifs_metadata_filter.filtered = klifs_metadata
        klifs_metadata_filter._get_dfg("in")

        assert klifs_metadata_filter.filtered.shape[0] == n_rows

    @pytest.mark.parametrize(
        "klifs_overview_file, klifs_export_file, n_rows",
        [
            (
                PATH_TEST_DATA / "KLIFS_download" / "overview.csv",
                PATH_TEST_DATA / "KLIFS_download" / "KLIFS_export.csv",
                10416,
            )
        ],
    )
    def test_get_resolution(self, klifs_overview_file, klifs_export_file, n_rows):
        """
        Test filtering by structural resolution.

        Parameters
        ----------
        klifs_overview_file : str or pathlib.Path
            Path to KLIFS download file `overview.csv` containing mainly KLIFS alignment-related metadata.
        klifs_export_file : str or pathlib.Path
            Path to KLIFS download file `KLIFS_download/KLIFS_export.csv` containing mainly structure-related metadata.
        n_rows : int
            Number of rows (structures) after filtering.
        """

        klifs_metadata_loader = KlifsMetadataLoader()
        klifs_metadata_loader.from_files(klifs_overview_file, klifs_export_file)

        klifs_metadata = klifs_metadata_loader.data_essential

        klifs_metadata_filter = KlifsMetadataFilter()
        klifs_metadata_filter.filtered = klifs_metadata
        klifs_metadata_filter._get_resolution(4)

        assert klifs_metadata_filter.filtered.shape[0] == n_rows

    @pytest.mark.parametrize(
        "klifs_overview_file, klifs_export_file, n_rows",
        [
            (
                PATH_TEST_DATA / "KLIFS_download" / "overview.csv",
                PATH_TEST_DATA / "KLIFS_download" / "KLIFS_export.csv",
                10399,
            )
        ],
    )
    def test_get_qualityscore(self, klifs_overview_file, klifs_export_file, n_rows):
        """
        Test filtering by KLIFS quality score.

        Parameters
        ----------
        klifs_overview_file : str or pathlib.Path
            Path to KLIFS download file `overview.csv` containing mainly KLIFS alignment-related metadata.
        klifs_export_file : str or pathlib.Path
            Path to KLIFS download file `KLIFS_download/KLIFS_export.csv` containing mainly structure-related metadata.
        n_rows : int
            Number of rows (structures) after filtering.
        """

        klifs_metadata_loader = KlifsMetadataLoader()
        klifs_metadata_loader.from_files(klifs_overview_file, klifs_export_file)

        klifs_metadata = klifs_metadata_loader.data_essential

        klifs_metadata_filter = KlifsMetadataFilter()
        klifs_metadata_filter.filtered = klifs_metadata
        klifs_metadata_filter._get_qualityscore(4)

        assert klifs_metadata_filter.filtered.shape[0] == n_rows

    @pytest.mark.parametrize(
        "klifs_overview_file, klifs_export_file, path_klifs_download, n_rows",
        [
            (
                PATH_TEST_DATA / "KLIFS_download" / "overview.csv",
                PATH_TEST_DATA / "KLIFS_download" / "KLIFS_export.csv",
                PATH_TEST_DATA / "KLIFS_download",
                2,  # Folders with both protein_pymol.mol2 and pocket.mol2
            )
        ],
    )
    def test_get_existing_mol2s(
        self, klifs_overview_file, klifs_export_file, path_klifs_download, n_rows
    ):
        """
        Test filtering by existing pocket mol2 files.

        Parameters
        ----------
        klifs_overview_file : str or pathlib.Path
            Path to KLIFS download file `overview.csv` containing mainly KLIFS alignment-related metadata.
        klifs_export_file : str or pathlib.Path
            Path to KLIFS download file `KLIFS_download/KLIFS_export.csv` containing mainly structure-related metadata.
        path_klifs_download : pathlib.Path or str
            Path to directory of KLIFS dataset files.
        n_rows : int
            Number of rows (structures) after filtering.
        """

        klifs_metadata_loader = KlifsMetadataLoader()
        klifs_metadata_loader.from_files(klifs_overview_file, klifs_export_file)

        klifs_metadata = klifs_metadata_loader.data_essential

        klifs_metadata_filter = KlifsMetadataFilter()
        klifs_metadata_filter.filtered = klifs_metadata
        klifs_metadata_filter._get_existing_mol2s(path_klifs_download)

        assert klifs_metadata_filter.filtered.shape[0] == n_rows

    @pytest.mark.parametrize(
        "klifs_overview_file, klifs_export_file, n_rows",
        [
            (
                PATH_TEST_DATA / "KLIFS_download" / "overview.csv",
                PATH_TEST_DATA / "KLIFS_download" / "KLIFS_export.csv",
                4908,
            )
        ],
    )
    def test_get_unique_kinase_pdbid_pair(self, klifs_overview_file, klifs_export_file, n_rows):
        """
        Test filtering by unique kinase-PDB ID pairs.

        Parameters
        ----------
        klifs_overview_file : str or pathlib.Path
            Path to KLIFS download file `overview.csv` containing mainly KLIFS alignment-related metadata.
        klifs_export_file : str or pathlib.Path
            Path to KLIFS download file `KLIFS_download/KLIFS_export.csv` containing mainly structure-related metadata.
        n_rows : int
            Number of rows (structures) after filtering.
        """

        klifs_metadata_loader = KlifsMetadataLoader()
        klifs_metadata_loader.from_files(klifs_overview_file, klifs_export_file)

        klifs_metadata = klifs_metadata_loader.data_essential

        klifs_metadata_filter = KlifsMetadataFilter()
        klifs_metadata_filter.filtered = klifs_metadata
        klifs_metadata_filter._get_unique_kinase_pdbid_pair()

        assert klifs_metadata_filter.filtered.shape[0] == n_rows


class TestsMol2KlifsToPymolConverter:
    """
    Test Mol2KlifsToPymolConverter class methods.
    """

    @pytest.mark.parametrize(
        "path_mol2, path_mol2_pymol, n_converted_lines",
        [
            (
                PATH_TEST_DATA / "Mol2KlifsToPymolConverter" / "protein.mol2",
                PATH_TEST_DATA / "Mol2KlifsToPymolConverter" / "protein_pymol.mol2",
                15,
            )
        ],
    )
    def test_convert_mol2(self, path_mol2, path_mol2_pymol, n_converted_lines):
        """
        Test if conversion from KLIFS mol2 file to PyMol readable file is correct (replace underscores with minus signs
        for residue IDs).

        Parameters
        ----------
        path_mol2 : pathlib.Path or str
            Path to KLIFS mol2 file.
        path_mol2_pymol : pathlib.Path or str
            Path to PyMol readable KLIFS mol2 file (after conversion).
        """

        # Load KLIFS mol2 file
        with open(path_mol2, "r") as f:
            lines_mol2 = f.readlines()

        # Load PyMol readable KLIFS mol2 file
        with open(path_mol2_pymol, "r") as f:
            lines_mol2_pymol = f.readlines()

        # Convert KLIFS mol2 file
        converter = Mol2KlifsToPymolConverter()
        lines_new_calculated, lines_converted_calculated = converter._convert_mol2(
            lines_mol2, "molecule1"
        )

        # Test if conversion is correct
        assert lines_new_calculated == lines_mol2_pymol

        # Test if converted lines are correctly assigned to class attribute
        assert len(lines_converted_calculated) == n_converted_lines


class TestsMol2ToPdbConverter:
    """
    Test Mol2ToPdbConverter class methods.
    """

    @pytest.mark.parametrize(
        "path_mol2",
        [PATH_TEST_DATA / "KLIFS_download" / "HUMAN/AAK1/4wsq_altA_chainA/pocket.mol2"],
    )
    def test_set_path_mol2(self, path_mol2):
        """
        Test if mol2 path is set correctly.

        Parameters
        ----------
        path_mol2 : pathlib.Path
            Path to mol2 file.
        """

        converter = Mol2ToPdbConverter()

        assert path_mol2 == converter._set_path_mol2(path_mol2)

    @pytest.mark.parametrize("path_mol2", [PATH_TEST_DATA / "KLIFS_download" / "xxx.mol2"])
    def test_set_path_mol2_filenotfounderror(self, path_mol2):
        """
        Test if mol2 path is set correctly (in this case raises error because file does not exist).

        Parameters
        ----------
        path_mol2 : pathlib.Path
            Path to mol2 file.
        """

        with pytest.raises(FileNotFoundError):
            converter = Mol2ToPdbConverter()
            converter._set_path_mol2(path_mol2)

    @pytest.mark.parametrize(
        "path_mol2",
        [
            PATH_TEST_DATA
            / "KLIFS_download"
            / "HUMAN/ADCK3/5i35_chainA/protein_correct.pdb"  # Existing file but no mol2
        ],
    )
    def test_set_path_mol2_valueerror(self, path_mol2):
        """
        Test if mol2 path is set correctly (in this case raises error because file is no mol2 file).

        Parameters
        ----------
        path_mol2 : pathlib.Path
            Path to mol2 file.
        """

        with pytest.raises(ValueError):
            converter = Mol2ToPdbConverter()
            converter._set_path_mol2(path_mol2)

    @pytest.mark.parametrize(
        "path_mol2_input, path_pdb_input, path_pdb_output",
        [
            (
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/AAK1/4wsq_altA_chainA/pocket.mol2",
                None,  # Do not define pdb path - will be set automatically
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/AAK1/4wsq_altA_chainA/pocket.pdb",
            ),
            (
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/AAK1/4wsq_altA_chainA/pocket.mol2",
                PATH_TEST_DATA
                / "KLIFS_download"
                / "HUMAN/AAK1/4wsq_altA_chainA/pocket2.pdb",  # Define some pdb path
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/AAK1/4wsq_altA_chainA/pocket2.pdb",
            ),
        ],
    )
    def test_set_path_pdb(self, path_mol2_input, path_pdb_input, path_pdb_output):
        """
        Test if pdb path is set correctly, given a mol2 path and an optional pdb path as input.

        Parameters
        ----------
        path_mol2_input : pathlib.Path
            Path to mol2 file.
        path_pdb_input : pathlib.Path or None
            Path to pdb file (None: Automatically set pdb path in same folder as mol2 file)
        path_pdb_output : pathlib.Path
            Path to pdb file which is returned from function.
        """

        converter = Mol2ToPdbConverter()
        converter.path_mol2 = path_mol2_input

        path_pdb_output_calculated = converter._set_path_pdb(path_mol2_input, path_pdb_input)

        assert path_pdb_output_calculated == path_pdb_output

    @pytest.mark.parametrize(
        "path_mol2_input, path_pdb_input",
        [
            (
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/AAK1/4wsq_altA_chainA/pocket.mol2",
                PATH_TEST_DATA / "KLIFS_download" / "xxx/pocket.pdb",
            )
        ],
    )
    def test_set_path_pdb_filenotfounderror(self, path_mol2_input, path_pdb_input):
        """
        Test if pdb path is set correctly, given a mol2 path and an optional pdb path as input (in this case raises error
        because input pdb path, i.e. directory, does not exist).

        Parameters
        ----------
        path_mol2_input : pathlib.Path
            Path to mol2 file.
        path_pdb_input : pathlib.Path or None
            Path to pdb file (None: Automatically set pdb path in same folder as mol2 file)
        """

        with pytest.raises(FileNotFoundError):
            converter = Mol2ToPdbConverter()
            converter.path_mol2 = path_mol2_input

            converter._set_path_pdb(path_mol2_input, path_pdb_input)

    @pytest.mark.parametrize(
        "path_mol2_input, path_pdb_input",
        [
            (None, PATH_TEST_DATA / "KLIFS_download" / "HUMAN/AAK1/4wsq_altA_chainA/pocket.pdb"),
            (
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/AAK1/4wsq_altA_chainA/pocket.mol2",
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/AAK1/4wsq_altA_chainA/pocket.mol2",
            ),
            (
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/AAK1/4wsq_altA_chainA/pocket.mol2",
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/AAK1/4wsq_altA_chainA/pocket",
            ),
            (
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/AAK1/4wsq_altA_chainA/pocket.mol2",
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/AAK1/4wsq_altA_chainA/pocket/",
            ),
        ],
    )
    def test_set_path_pdb_valueerror(self, path_mol2_input, path_pdb_input):
        """
        Test if pdb path is set correctly, given a mol2 path and an optional pdb path as input (in this case raises error
        because input file path incorrect/incomplete).

        Parameters
        ----------
        path_mol2_input : pathlib.Path
            Path to mol2 file.
        path_pdb_input : pathlib.Path or None
            Path to pdb file (None: Automatically set pdb path in same folder as mol2 file)
        """

        with pytest.raises(ValueError):
            converter = Mol2ToPdbConverter()
            converter.path_mol2 = path_mol2_input

            converter._set_path_pdb(path_mol2_input, path_pdb_input)

    @pytest.mark.parametrize(
        "path_mol2, path_pdb",
        [
            (
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ADCK3/5i35_chainA/protein.mol2",
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ADCK3/5i35_chainA/protein_correct.pdb",
            )
        ],
    )
    def test_report_inconsistent_conversion(self, path_mol2, path_pdb):
        """
        Test if mol2 and pdb file contain same information.

        Parameters
        ----------
        path_mol2 : pathlib.Path
            Path to mol2 file.
        path_pdb : pathlib.Path or None
            Path to pdb file (= converted mol2 file). Directory must exist.
            Default is None - saves pdb file next to the mol2 file in same directory with the same filename.
        """

        converter = Mol2ToPdbConverter()

        assert converter._report_inconsistent_conversion(path_mol2, path_pdb) is None

    @pytest.mark.parametrize(
        "path_mol2, path_pdb, inconsistency",
        [
            (
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ADCK3/5i35_chainA/protein.mol2",
                PATH_TEST_DATA
                / "KLIFS_download"
                / "HUMAN/ADCK3/5i35_chainA/protein_irregular_n_atoms.pdb",
                pd.DataFrame(
                    [
                        [
                            "HUMAN/ADCK3/5i35_chainA",
                            "Unequal number of atoms",
                            {"mol2": 6194, "pdb": 6193},
                        ]
                    ],
                    columns=["filepath", "inconsistency", "details"],
                ),
            ),
            (
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ADCK3/5i35_chainA/protein.mol2",
                PATH_TEST_DATA
                / "KLIFS_download"
                / "HUMAN/ADCK3/5i35_chainA/protein_irregular_x_mean.pdb",
                pd.DataFrame(
                    [["HUMAN/ADCK3/5i35_chainA", "Unequal x coordinate mean", -0.14]],
                    columns=["filepath", "inconsistency", "details"],
                ),
            ),
            (
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ADCK3/5i35_chainA/protein.mol2",
                PATH_TEST_DATA
                / "KLIFS_download"
                / "HUMAN/ADCK3/5i35_chainA/protein_irregular_y_mean.pdb",
                pd.DataFrame(
                    [["HUMAN/ADCK3/5i35_chainA", "Unequal y coordinate mean", -0.15]],
                    columns=["filepath", "inconsistency", "details"],
                ),
            ),
            (
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ADCK3/5i35_chainA/protein.mol2",
                PATH_TEST_DATA
                / "KLIFS_download"
                / "HUMAN/ADCK3/5i35_chainA/protein_irregular_z_mean.pdb",
                pd.DataFrame(
                    [["HUMAN/ADCK3/5i35_chainA", "Unequal z coordinate mean", -0.14]],
                    columns=["filepath", "inconsistency", "details"],
                ),
            ),
            (
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ADCK3/5i35_chainA/protein.mol2",
                PATH_TEST_DATA
                / "KLIFS_download"
                / "HUMAN/ADCK3/5i35_chainA/protein_irregular_record_name.pdb",
                pd.DataFrame(
                    [
                        [
                            "HUMAN/ADCK3/5i35_chainA",
                            "Non-ATOM entries",
                            {"record_name": {"HETATM"}, "residue_name": {"GLU"}},
                        ]
                    ],
                    columns=["filepath", "inconsistency", "details"],
                ),
            ),
            (
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ADCK3/5i35_chainA/protein.mol2",
                PATH_TEST_DATA
                / "KLIFS_download"
                / "HUMAN/ADCK3/5i35_chainA/protein_irregular_residue_details.pdb",
                pd.DataFrame(
                    [
                        [
                            "HUMAN/ADCK3/5i35_chainA",
                            "Unequal residue ID/name",
                            {"mol2": {"GLU261"}, "pdb": {"GLU999"}},
                        ]
                    ],
                    columns=["filepath", "inconsistency", "details"],
                ),
            ),
        ],
    )
    def test_report_inconsistent_conversion_valueerror(self, path_mol2, path_pdb, inconsistency):
        """
        Test if mol2 and pdb file contain same information.

        Parameters
        ----------
        path_mol2 : pathlib.Path
            Path to mol2 file.
        path_pdb : pathlib.Path or None
            Path to pdb file (= converted mol2 file). Directory must exist.
            Default is None - saves pdb file next to the mol2 file in same directory with the same filename.
        """

        converter = Mol2ToPdbConverter()
        converter._report_inconsistent_conversion(path_mol2, path_pdb)

        assert converter.inconsistent_conversions.equals(inconsistency)

    @pytest.mark.parametrize(
        "path_mol2_input, path_pdb_input, path_pdb_output",
        [
            (
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/AAK1/4wsq_altA_chainA/pocket.mol2",
                None,
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/AAK1/4wsq_altA_chainA/pocket.pdb",
            )
        ],
    )
    def test_from_file(self, path_mol2_input, path_pdb_input, path_pdb_output):
        """
        Test if pdb file is created.

        Parameters
        ----------
        path_mol2_input : pathlib.Path
            Path to input mol2 file.
        path_pdb_input : pathlib.Path
            Path to input pdb file.
        path_pdb_output : pathlib.Path
            Path to output pdb file.
        """

        converter = Mol2ToPdbConverter()
        converter.from_file(path_mol2_input, path_pdb_input)

        # Test if pdb file exists
        assert path_pdb_output.exists()

        # Remove pdb file
        path_pdb_output.unlink()

        # Test if pdb file does not exist
        assert not path_pdb_output.exists()

    @pytest.mark.parametrize(
        "klifs_metadata, path_klifs_download, path_pdb",
        [
            (
                pd.DataFrame(["HUMAN/ADCK3/5i35_chainA"], columns=["filepath"]),
                PATH_TEST_DATA / "KLIFS_download",
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ADCK3/5i35_chainA/protein_pymol.pdb",
            )
        ],
    )
    def ttest_from_klifs_metadata(self, klifs_metadata, path_klifs_download, path_pdb):
        # TODO PyMol cannot be launched mulitple times...
        """
        Test if mol2 to pdb conversion works if KLIFS metadata is given.

        Parameters
        ----------
        klifs_metadata : pandas.DataFrame
            KLIFS metadata describing the KLIFS dataset.
        path_klifs_download : pathlib.Path
            Path to directory of KLIFS dataset files.
        path_pdb : pathlib.Path
            Pdb path to converted pdb file.
        """

        converter = Mol2ToPdbConverter()
        converter.from_klifs_metadata(klifs_metadata, path_klifs_download)

        # Test if pdb file exists
        assert path_pdb.exists()

        # Remove pdb file
        path_pdb.unlink()

        # Test if pdb file does not exist
        assert not path_pdb.exists()
