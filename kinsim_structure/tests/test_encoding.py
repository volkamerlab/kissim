
import pytest

from kinsim_structure.encoding import get_feature_size

@pytest.mark.parametrize('residue, feature', [
    ('ALA', 1),
    ('ASN', 2),
    ('ARG', 3),
    ('XXX', None)
])
def test_get_feature_size(residue, feature):

    assert get_feature_size(residue) == feature
