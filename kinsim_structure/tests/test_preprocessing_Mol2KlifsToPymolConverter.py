"""
Unit and regression test for kinsim_structure.preprocessing.Mol2KlifsToPymolConverter methods.
"""

from pathlib import Path

import pytest

from kinsim_structure.preprocessing import Mol2KlifsToPymolConverter

PATH_TEST_DATA = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data'


@pytest.mark.parametrize('path_mol2, path_mol2_pymol, n_converted_lines', [
    (
        PATH_TEST_DATA / 'Mol2KlifsToPymolConverter' / 'protein.mol2',
        PATH_TEST_DATA / 'Mol2KlifsToPymolConverter' / 'protein_pymol.mol2',
        15
    )
])
def test_convert_mol2(path_mol2, path_mol2_pymol, n_converted_lines):
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
    with open(path_mol2, 'r') as f:
        lines_mol2 = f.readlines()

    # Load PyMol readable KLIFS mol2 file
    with open(path_mol2_pymol, 'r') as f:
        lines_mol2_pymol = f.readlines()

    # Convert KLIFS mol2 file
    converter = Mol2KlifsToPymolConverter()
    lines_new_calculated, lines_converted_calculated = converter._convert_mol2(lines_mol2, 'molecule1')

    # Test if conversion is correct
    assert lines_new_calculated == lines_mol2_pymol

    # Test if converted lines are correctly assigned to class attribute
    assert len(lines_converted_calculated) == n_converted_lines
