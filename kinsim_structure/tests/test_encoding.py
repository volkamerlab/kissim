
from biopandas.mol2 import PandasMol2
import numpy as np
from pathlib import Path
import pytest

from kinsim_structure.encoding import get_feature_size
from kinsim_structure.encoding import get_feature_hbd, get_feature_hba, get_feature_charge, get_feature_aromatic, get_feature_aliphatic
from kinsim_structure.encoding import get_side_chain_orientation


@pytest.mark.parametrize('residue, feature', [
    ('ALA', 1),
    ('ASN', 2),
    ('ARG', 3),
    ('XXX', None)
])
def test_get_feature_size(residue, feature):

    assert get_feature_size(residue) == feature


@pytest.mark.parametrize('residue, feature', [
    ('ALA', 0),
    ('ASN', 1),
    ('ARG', 3),
    ('XXX', None)
])
def test_get_feature_hdb(residue, feature):

    assert get_feature_hbd(residue) == feature


@pytest.mark.parametrize('residue, feature', [
    ('ALA', 0),
    ('ASN', 1),
    ('ASP', 2),
    ('XXX', None)
])
def test_get_feature_hda(residue, feature):

    assert get_feature_hba(residue) == feature


@pytest.mark.parametrize('residue, feature', [
    ('ALA', 0),
    ('ARG', 1),
    ('ASP', -1),
    ('XXX', None)
])
def test_get_feature_charge(residue, feature):

    assert get_feature_charge(residue) == feature


@pytest.mark.parametrize('residue, feature', [
    ('ALA', 0),
    ('HIS', 1),
    ('XXX', None)
])
def test_get_feature_aromatic(residue, feature):

    assert get_feature_aromatic(residue) == feature


@pytest.mark.parametrize('residue, feature', [
    ('ARG', 0),
    ('ALA', 1),
    ('XXX', None)
])
def test_get_feature_aliphatic(residue, feature):

    assert get_feature_aliphatic(residue) == feature


@pytest.mark.parametrize('filename, side_chain_orientation', [
    ('AAK1_4wsq_altA_chainA.mol2', np.array([1.0]*85))
])
def test_get_side_chain_orientation(filename, side_chain_orientation):

    path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / filename
    path = str(path)

    pmol = PandasMol2().read_mol2(path=path,
                                  columns = {0: ('atom_id', int),
                                        1: ('atom_name', str),
                                        2: ('x', float),
                                        3: ('y', float),
                                        4: ('z', float),
                                        5: ('atom_type', str),
                                        6: ('subst_id', str),
                                        7: ('subst_name', str),
                                        8: ('charge', float),
                                        9: ('status_bit', str)}
    )
    assert all(get_side_chain_orientation(pmol) == side_chain_orientation)

