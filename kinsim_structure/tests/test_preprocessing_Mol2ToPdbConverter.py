"""
Unit and regression test for kinsim_structure.preprocessing.Mol2ToPdbConverter methods.
"""

from pathlib import Path

import pandas as pd
import pytest

from kinsim_structure.preprocessing import Mol2ToPdbConverter


@pytest.mark.parametrize('mol2_path', [
    'KLIFS_download/HUMAN/AAK1/4wsq_altA_chainA/pocket.mol2'
])
def test_set_mol2_path(mol2_path):

    mol2_path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / mol2_path

    converter = Mol2ToPdbConverter()

    assert mol2_path == converter._set_mol2_path(mol2_path)


@pytest.mark.parametrize('mol2_path', [
    'xxx.mol2'
])
def test_set_mol2_path_filenotfounderror(mol2_path):

    mol2_path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / mol2_path

    with pytest.raises(FileNotFoundError):
        converter = Mol2ToPdbConverter()
        converter._set_mol2_path(mol2_path)


@pytest.mark.parametrize('mol2_path', [
    'PDB_download/2g2i.cif'  # Existing file but no mol2
])
def test_set_mol2_path_valueerror(mol2_path):

    mol2_path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / mol2_path

    with pytest.raises(ValueError):
        converter = Mol2ToPdbConverter()
        converter._set_mol2_path(mol2_path)


@pytest.mark.parametrize('mol2_path_input, pdb_path_input, pdb_path_output', [
    (
        'KLIFS_download/HUMAN/AAK1/4wsq_altA_chainA/pocket.mol2',
        None,
        'KLIFS_download/HUMAN/AAK1/4wsq_altA_chainA/pocket.pdb'
    ),
    (
        'KLIFS_download/HUMAN/AAK1/4wsq_altA_chainA/pocket.mol2',
        'KLIFS_download/HUMAN/AAK1/4wsq_altA_chainA/pocket2.pdb',
        'KLIFS_download/HUMAN/AAK1/4wsq_altA_chainA/pocket2.pdb'
    )
])
def test_set_pdb_path(mol2_path_input, pdb_path_input, pdb_path_output):

    mol2_path_input = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / mol2_path_input
    pdb_path_output = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / pdb_path_output

    if pdb_path_input is not None:
        pdb_path_input = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / pdb_path_input

    converter = Mol2ToPdbConverter()
    converter.mol2_path = mol2_path_input

    pdb_path_output_calculated = converter._set_pdb_path(pdb_path_input)

    assert pdb_path_output_calculated == pdb_path_output


@pytest.mark.parametrize('mol2_path_input, pdb_path_input', [
    (
        'KLIFS_download/HUMAN/AAK1/4wsq_altA_chainA/pocket.mol2',
        'xxx/pocket.pdb'
    )
])
def test_set_pdb_path_filenotfounderror(mol2_path_input, pdb_path_input):

    if mol2_path_input is not None:
        mol2_path_input = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / mol2_path_input

    if pdb_path_input is not None:
        pdb_path_input = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / pdb_path_input

    with pytest.raises(FileNotFoundError):

        converter = Mol2ToPdbConverter()
        converter.mol2_path = mol2_path_input

        converter._set_pdb_path(pdb_path_input)


@pytest.mark.parametrize('mol2_path_input, pdb_path_input', [
    (
        None,
        'KLIFS_download/HUMAN/AAK1/4wsq_altA_chainA/pocket.pdb'
    ),
    (
        'KLIFS_download/HUMAN/AAK1/4wsq_altA_chainA/pocket.mol2',
        'KLIFS_download/HUMAN/AAK1/4wsq_altA_chainA/pocket.mol2'
    ),
    (
        'KLIFS_download/HUMAN/AAK1/4wsq_altA_chainA/pocket.mol2',
        'KLIFS_download/HUMAN/AAK1/4wsq_altA_chainA/pocket'
    ),
    (
        'KLIFS_download/HUMAN/AAK1/4wsq_altA_chainA/pocket.mol2',
        'KLIFS_download/HUMAN/AAK1/4wsq_altA_chainA/pocket/'
    )
])
def test_set_pdb_path_valueerror(mol2_path_input, pdb_path_input):

    if mol2_path_input is not None:
        mol2_path_input = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / mol2_path_input

    if pdb_path_input is not None:
        pdb_path_input = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / pdb_path_input

    with pytest.raises(ValueError):

        converter = Mol2ToPdbConverter()
        converter.mol2_path = mol2_path_input

        converter._set_pdb_path(pdb_path_input)


@pytest.mark.parametrize('mol2_path, pdb_path', [
    (
        'KLIFS_download/HUMAN/ADCK3/5i35_chainA/protein.mol2',
        'KLIFS_download/HUMAN/ADCK3/5i35_chainA/protein_correct.pdb'
    )
])
def test_raise_conversion_error(mol2_path, pdb_path):
    """
    Test if mol2 and pdb file contain same information.

    Parameters
    ----------
    mol2_path : pathlib.Path or str
        Path to mol2 file.
    pdb_path : None or pathlib.Path or str
        Path to pdb file (= converted mol2 file). Directory must exist.
        Default is None - saves pdb file next to the mol2 file in same directory with the same filename.
    """

    mol2_path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / mol2_path
    pdb_path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / pdb_path

    converter = Mol2ToPdbConverter()
    converter.mol2_path = mol2_path
    converter.pdb_path = pdb_path

    assert converter._raise_conversion_error() is None


@pytest.mark.parametrize('mol2_path, pdb_path', [
    (
        'KLIFS_download/HUMAN/ADCK3/5i35_chainA/protein.mol2',
        'KLIFS_download/HUMAN/ADCK3/5i35_chainA/protein_incorrect_n_atoms.pdb'
    ),
    (
        'KLIFS_download/HUMAN/ADCK3/5i35_chainA/protein.mol2',
        'KLIFS_download/HUMAN/ADCK3/5i35_chainA/protein_incorrect_x_mean.pdb'
    ),
    (
        'KLIFS_download/HUMAN/ADCK3/5i35_chainA/protein.mol2',
        'KLIFS_download/HUMAN/ADCK3/5i35_chainA/protein_incorrect_y_mean.pdb'
    ),
    (
        'KLIFS_download/HUMAN/ADCK3/5i35_chainA/protein.mol2',
        'KLIFS_download/HUMAN/ADCK3/5i35_chainA/protein_incorrect_z_mean.pdb'
    ),
    (
        'KLIFS_download/HUMAN/ADCK3/5i35_chainA/protein.mol2',
        'KLIFS_download/HUMAN/ADCK3/5i35_chainA/protein_incorrect_record_name.pdb'
    ),
    (
        'KLIFS_download/HUMAN/ADCK3/5i35_chainA/protein.mol2',
        'KLIFS_download/HUMAN/ADCK3/5i35_chainA/protein_incorrect_residue_details.pdb'
    )
])
def test_raise_conversion_error_ValueError(mol2_path, pdb_path):
    """
    Test if mol2 and pdb file contain same information.

    Parameters
    ----------
    mol2_path : pathlib.Path or str
        Path to mol2 file.
    pdb_path : None or pathlib.Path or str
        Path to pdb file (= converted mol2 file). Directory must exist.
        Default is None - saves pdb file next to the mol2 file in same directory with the same filename.
    """

    mol2_path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / mol2_path
    pdb_path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / pdb_path

    converter = Mol2ToPdbConverter()
    converter.mol2_path = mol2_path
    converter.pdb_path = pdb_path

    with pytest.raises(ValueError):
        converter._raise_conversion_error()


@pytest.mark.parametrize('mol2_path_input, pdb_path_input, pdb_path_output', [
    (
        'KLIFS_download/HUMAN/AAK1/4wsq_altA_chainA/pocket.mol2',
        'KLIFS_download/HUMAN/AAK1/4wsq_altA_chainA/pocket.pdb',
        'KLIFS_download/HUMAN/AAK1/4wsq_altA_chainA/pocket.pdb'
    ),
    (
        'KLIFS_download/HUMAN/AAK1/4wsq_altA_chainA/pocket.mol2',
        None,
        'KLIFS_download/HUMAN/AAK1/4wsq_altA_chainA/pocket.pdb'
    )
])
def test_from_file(mol2_path_input, pdb_path_input, pdb_path_output):
    """
    Test if pdb file is created.

    Parameters
    ----------
    mol2_path_input : str
        Path to input mol2 file.
    pdb_path_input : str
        Path to input pdb file.
    pdb_path_output : str
        Path to output pdb file.
    """

    mol2_path_input = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / mol2_path_input
    pdb_path_output = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / pdb_path_output

    if pdb_path_input is not None:
        pdb_path_input = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / pdb_path_input

    converter = Mol2ToPdbConverter()
    converter.from_file(mol2_path_input, pdb_path_input)

    # Test class attributes
    assert converter.mol2_path == mol2_path_input
    assert converter.pdb_path == pdb_path_output

    # Test if pdb file exists
    assert pdb_path_output.exists()

    # Remove pdb file
    pdb_path_output.unlink()

    # Test if pdb file does not exist
    assert not pdb_path_output.exists()


@pytest.mark.parametrize('klifs_metadata_entry, path_to_klifs_download, mol2_path, pdb_path', [
    (
            pd.Series(['HUMAN/ADCK3/5i35_chainA'], index=['filepath']),
            '.',
            'KLIFS_download/HUMAN/ADCK3/5i35_chainA/protein.mol2',
            'KLIFS_download/HUMAN/ADCK3/5i35_chainA/protein.pdb'
    )
])
def test_from_klifs_metadata_entry(klifs_metadata_entry, path_to_klifs_download, mol2_path, pdb_path):

    mol2_path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / mol2_path
    pdb_path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / pdb_path
    path_to_klifs_download = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / path_to_klifs_download

    converter = Mol2ToPdbConverter()
    converter.from_klifs_metadata_entry(klifs_metadata_entry, path_to_klifs_download)

    # Test class attributes
    assert converter.mol2_path == mol2_path
    assert converter.pdb_path == pdb_path

    # Test if pdb file exists
    assert pdb_path.exists()

    # Remove pdb file
    pdb_path.unlink()

    # Test if pdb file does not exist
    assert not pdb_path.exists()
