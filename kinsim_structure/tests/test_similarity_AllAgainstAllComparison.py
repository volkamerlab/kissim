"""
Unit and regression test for kinsim_structure.similarity.AllAgainstAllComparison methods.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kinsim_structure.auxiliary import KlifsMoleculeLoader, PdbChainLoader
from kinsim_structure.encoding import Fingerprint
from kinsim_structure.similarity import AllAgainstAllComparison


@pytest.mark.parametrize('fingerprints, empty_fingerprints', [
    ({'a': Fingerprint(), 'b': None}, {'a': Fingerprint()}),
    ({'a': Fingerprint()}, {'a': Fingerprint()})
])
def test_remove_empty_fingerprints(fingerprints, empty_fingerprints):
    """
    Test removal of empty fingerprints (None) from fingerprints dictionary.

    Parameters
    ----------
    fingerprints : dict of (kinsim_structure.encoding.Fingerprint or None)
        Dictionary of fingerprints: Keys are molecule codes and values are fingerprint data.
    empty_fingerprints : dict of kinsim_structure.encoding.Fingerprint
        Dictionary of non-empty fingerprints: Keys are molecule codes and values are fingerprint data.
    """

    comparison = AllAgainstAllComparison()
    empty_fingerprints_calculated = comparison._remove_empty_fingerprints(fingerprints)

    assert empty_fingerprints_calculated.keys() == empty_fingerprints.keys()


@pytest.mark.parametrize('fingerprints, pairs', [
    ({'a': Fingerprint(), 'b': Fingerprint(), 'c': Fingerprint()}, [['a', 'b'], ['a', 'c'], ['b', 'c']])
])
def test_get_fingerprint_pairs(fingerprints, pairs):
    """
    Test calculation of all fingerprint pair combinations from fingerprints dictionary.

    Parameters
    ----------
    fingerprints : dict of (kinsim_structure.encoding.Fingerprint or None)
        Dictionary of fingerprints: Keys are molecule codes and values are fingerprint data.
    pairs : list of list of str
        List of molecule code pairs (list).
    """

    comparison = AllAgainstAllComparison()
    pairs_calculated = comparison._get_fingerprint_pairs(fingerprints)

    for pair_calculated, pair in zip(pairs_calculated, pairs):
        assert pair_calculated == pair


