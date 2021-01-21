"""
Unit and regression test for the kissim.encoding.features.base.BaseFeature class.
"""

import pytest

from kissim.encoding.features import BaseFeature

class TestsBaseFeature:
    """
    Test BaseFeature class methods.
    """

    @pytest.mark.parametrize(
        "structure_klifs_id",
        [12347],
    )
    def test_from_structure_klifs_id_notimplementederror(self, structure_klifs_id):
        with pytest.raises(NotImplementedError):
            BaseFeature.from_structure_klifs_id(structure_klifs_id)

    @pytest.mark.parametrize(
        "pocket",
        [None],
    )
    def test_from_pocket_notimplementederror(self, pocket):
        with pytest.raises(NotImplementedError):
            BaseFeature.from_pocket(pocket)

    def test_values(self):
        with pytest.raises(NotImplementedError):
            feature = BaseFeature()
            feature.values

    def test_details(self):
        with pytest.raises(NotImplementedError):
            feature = BaseFeature()
            feature.details
