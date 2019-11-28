"""
Unit and regression test for kinsim_structure.preprocessing.Mol2ToPdbConverter methods.
"""

from pathlib import Path

import pandas as pd
import pytest

from kinsim_structure.preprocessing import Mol2ToPdbConverter

PATH_TEST_DATA = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data'


@pytest.mark.parametrize('path_mol2', [
    PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainA/pocket.mol2'
])
def test_set_path_mol2(path_mol2):
    """
    Test if mol2 path is set correctly.

    Parameters
    ----------
    path_mol2 : pathlib.Path
        Path to mol2 file.
    """

    converter = Mol2ToPdbConverter()

    assert path_mol2 == converter._set_path_mol2(path_mol2)


@pytest.mark.parametrize('path_mol2', [
    PATH_TEST_DATA / 'KLIFS_download' / 'xxx.mol2'
])
def test_set_path_mol2_filenotfounderror(path_mol2):
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


@pytest.mark.parametrize('path_mol2', [
    PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ADCK3/5i35_chainA/protein_correct.pdb'  # Existing file but no mol2
])
def test_set_path_mol2_valueerror(path_mol2):
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


@pytest.mark.parametrize('path_mol2_input, path_pdb_input, path_pdb_output', [
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainA/pocket.mol2',
        None,  # Do not define pdb path - will be set automatically
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainA/pocket.pdb'
    ),
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainA/pocket.mol2',
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainA/pocket2.pdb',  # Define some pdb path
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainA/pocket2.pdb'
    )
])
def test_set_path_pdb(path_mol2_input, path_pdb_input, path_pdb_output):
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


@pytest.mark.parametrize('path_mol2_input, path_pdb_input', [
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainA/pocket.mol2',
        PATH_TEST_DATA / 'KLIFS_download' / 'xxx/pocket.pdb'
    )
])
def test_set_path_pdb_filenotfounderror(path_mol2_input, path_pdb_input):
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


@pytest.mark.parametrize('path_mol2_input, path_pdb_input', [
    (
        None,
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainA/pocket.pdb'
    ),
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainA/pocket.mol2',
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainA/pocket.mol2'
    ),
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainA/pocket.mol2',
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainA/pocket'
    ),
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainA/pocket.mol2',
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainA/pocket/'
    )
])
def test_set_path_pdb_valueerror(path_mol2_input, path_pdb_input):
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


@pytest.mark.parametrize('path_mol2, path_pdb', [
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ADCK3/5i35_chainA/protein.mol2',
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ADCK3/5i35_chainA/protein_correct.pdb'
    )
])
def test_report_inconsistent_conversion(path_mol2, path_pdb):
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


@pytest.mark.parametrize('path_mol2, path_pdb, inconsistency', [
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ADCK3/5i35_chainA/protein.mol2',
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ADCK3/5i35_chainA/protein_irregular_n_atoms.pdb',
        pd.DataFrame(
            [
                ['HUMAN/ADCK3/5i35_chainA', 'Unequal number of atoms', {'mol2': 6194, 'pdb': 6193}]
            ],
            columns=['filepath', 'inconsistency', 'details']
        )
    ),
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ADCK3/5i35_chainA/protein.mol2',
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ADCK3/5i35_chainA/protein_irregular_x_mean.pdb',
        pd.DataFrame(
            [
                ['HUMAN/ADCK3/5i35_chainA', 'Unequal x coordinate mean', -0.14]
            ],
            columns=['filepath', 'inconsistency', 'details']
        )
    ),
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ADCK3/5i35_chainA/protein.mol2',
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ADCK3/5i35_chainA/protein_irregular_y_mean.pdb',
        pd.DataFrame(
            [
                ['HUMAN/ADCK3/5i35_chainA', 'Unequal y coordinate mean', -0.15]
            ],
            columns=['filepath', 'inconsistency', 'details']
        )
    ),
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ADCK3/5i35_chainA/protein.mol2',
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ADCK3/5i35_chainA/protein_irregular_z_mean.pdb',
        pd.DataFrame(
            [
                ['HUMAN/ADCK3/5i35_chainA', 'Unequal z coordinate mean', -0.14]
            ],
            columns=['filepath', 'inconsistency', 'details']
        )
    ),
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ADCK3/5i35_chainA/protein.mol2',
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ADCK3/5i35_chainA/protein_irregular_record_name.pdb',
        pd.DataFrame(
            [
                ['HUMAN/ADCK3/5i35_chainA', 'Non-ATOM entries', {'record_name': {'HETATM'}, 'residue_name': {'GLU'}}]
            ],
            columns=['filepath', 'inconsistency', 'details']
        )
    ),
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ADCK3/5i35_chainA/protein.mol2',
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ADCK3/5i35_chainA/protein_irregular_residue_details.pdb',
        pd.DataFrame(
            [
                ['HUMAN/ADCK3/5i35_chainA', 'Unequal residue ID/name', {'mol2': {'GLU261'}, 'pdb': {'GLU999'}}]
            ],
            columns=['filepath', 'inconsistency', 'details']
        )
    )
])
def test_report_inconsistent_conversion_valueerror(path_mol2, path_pdb, inconsistency):
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


@pytest.mark.parametrize('path_mol2_input, path_pdb_input, path_pdb_output', [
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainA/pocket.mol2',
        None,
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainA/pocket.pdb'
    )
])
def test_from_file(path_mol2_input, path_pdb_input, path_pdb_output):
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


@pytest.mark.parametrize('klifs_metadata, path_klifs_download, path_pdb', [
    (
            pd.DataFrame(['HUMAN/ADCK3/5i35_chainA'], columns=['filepath']),
            PATH_TEST_DATA / 'KLIFS_download',
            PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ADCK3/5i35_chainA/protein_pymol.pdb'
    )
])
def ttest_from_klifs_metadata(klifs_metadata, path_klifs_download, path_pdb):  # TODO PyMol cannot be launched mulitple times...
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
