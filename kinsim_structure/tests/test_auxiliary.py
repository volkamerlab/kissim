"""
Unit and regression test for auxiliary classes and functions in the kinsim_structure.auxiliary module of the
kinsim_structure package.
"""

from pathlib import Path

import numpy as np
import pytest

from kinsim_structure.auxiliary import MoleculeLoader


@pytest.mark.parametrize('filename, code, n_atoms, centroid', [
(
        'AAK1_4wsq_altA_chainA.mol2',
        ['HUMAN/AAK1_4wsq_altA_chainA'],
        [1326],
        [[1.4157, 20.9502, 36.0209]]
    )
])
def test_molecule_loader(filename, code, n_atoms, centroid):
    """
    Test MoleculeLoader class.

    Parameters
    ---------
    filename : str
        Name of molecule file.
    code : str
        Name of molecule code.
    n_atoms : int
        Number of atoms (i.e. number of DataFrame rows).
    centroid : list of lists
        Centroid(s) of molecule.
    """

    # Load molecule
    molecule_path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / filename
    molecule_loader = MoleculeLoader(molecule_path)

    assert len(molecule_loader.molecules) == len(code)

    for c, v in enumerate(molecule_loader.molecules):
        assert v.code == code[c]
        assert v.df.shape == (n_atoms[c], 9)
        assert list(v.df.columns) == ['atom_id', 'atom_name', 'res_id', 'res_name', 'subst_name', 'x', 'y', 'z', 'charge']
        assert np.isclose(v.df['x'].mean(), centroid[c][0], rtol=1e-04)
        assert np.isclose(v.df['y'].mean(), centroid[c][1], rtol=1e-04)
        assert np.isclose(v.df['z'].mean(), centroid[c][2], rtol=1e-04)