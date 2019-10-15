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
