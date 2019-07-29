
import pytest

from kinsim_structure.encoding import get_feature_size, get_feature_hbd, get_feature_hba, get_feature_charge, get_feature_aromatic, get_feature_aliphatic


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

