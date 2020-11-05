"""
Unit and regression test for kissim.io class methods.
"""

import pytest

from kissim.io import Complex, Pocket


class TestComplex:
    """
    Test Complex class.
    """

    @pytest.mark.parametrize(
        "structure_id",
        [12347],
    )
    def test_from_structure_id(self, structure_id):
        """
        Test if Complex is set correctly.
        """

        complex = Complex.from_structure_id(structure_id)
        assert isinstance(complex, Complex)
        # TODO add more tests


class TestsPocket:
    """
    Test Pocket class.
    """

    @pytest.mark.parametrize(
        "structure_id",
        [12347],
    )
    def test_from_structure_id(self, structure_id):
        """
        Test if Pocket is set correctly.
        """

        pocket = Pocket.from_structure_id(structure_id)
        assert isinstance(pocket, Pocket)
        # TODO add more tests
