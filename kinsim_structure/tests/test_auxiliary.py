"""
Unit and regression test for auxiliary classes and functions in the kinsim_structure.auxiliary module of the
kinsim_structure package.
"""

import pytest

from kinsim_structure.auxiliary import format_klifs_code


@pytest.mark.parametrize('klifs_code, split_klifs_code', [
    (
        'HUMAN/AAK1_4wsq_altA_chainA',
        {'species': 'HUMAN', 'group': 'AAK1', 'pdb_id': '4wsq', 'alternate_id': 'A', 'chain_id': 'A'}
    ),
    (
        'HUMAN/AAK1_4wsq',
        {'species': 'HUMAN', 'group': 'AAK1', 'pdb_id': '4wsq', 'alternate_id': None, 'chain_id': None}
    )

])
def test_format_klifs_code(klifs_code, split_klifs_code):

    assert format_klifs_code(klifs_code) == split_klifs_code
