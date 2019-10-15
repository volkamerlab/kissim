"""
Unit and regression test for auxiliary classes and functions in the kinsim_structure.auxiliary module of the
kinsim_structure package.
"""

import pytest

from kinsim_structure.auxiliary import split_klifs_code


@pytest.mark.parametrize('klifs_code, result', [
    (
        'HUMAN/AAK1_4wsq_altA_chainA',
        {'species': 'HUMAN', 'kinase': 'AAK1', 'pdb_id': '4wsq', 'alternate_model': 'A', 'chain': 'A'}
    ),
    (
        'HUMAN/AAK1_4wsq',
        {'species': 'HUMAN', 'kinase': 'AAK1', 'pdb_id': '4wsq', 'alternate_model': None, 'chain': None}
    )

])
def test_split_klifs_code(klifs_code, result):
    """
    Rest if KLIFS code can be split into its properties.

    Parameters
    ----------
    klifs_code : str
        Unique KLIFS code: SPECIES/kinase_pdbid_alternatemodel_chain.
    result : dict
        KLIFS attributes the KLIFS code should be split into.
    """

    assert split_klifs_code(klifs_code) == result
