"""
Unit and regression test for auxiliary classes and functions in the kinsim_structure.auxiliary module of the
kinsim_structure package.
"""

from pathlib import Path
import pytest

from kinsim_structure.auxiliary import split_klifs_code
from kinsim_structure.auxiliary import KlifsMoleculeLoader


@pytest.mark.parametrize('klifs_code, result', [
    (
        'HUMAN/AAK1_4wsq_altA_chainA',
        {'species': 'HUMAN', 'group': 'AAK1', 'pdb_id': '4wsq', 'alternate_id': 'A', 'chain_id': 'A'}
    ),
    (
        'HUMAN/AAK1_4wsq',
        {'species': 'HUMAN', 'group': 'AAK1', 'pdb_id': '4wsq', 'alternate_id': None, 'chain_id': None}
    )

])
def test_split_klifs_code(klifs_code, result):

    assert split_klifs_code(klifs_code) == result


@pytest.mark.parametrize('mol2_path, result', [
    ('AAK1/4wsq_altA_chainB/pocket.mol2', None)
])
def test_klifsmoleculeloader_fromfile(mol2_path, result):

    mol2_path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / mol2_path
    print(mol2_path)

    klifs_molecule_loader = KlifsMoleculeLoader()
    molecule = klifs_molecule_loader.from_file(mol2_path)

    assert molecule.df.kinase[0] == result
