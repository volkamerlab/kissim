"""
Unit and regression test for kissim's compare CLI.
"""

from argparse import Namespace
import logging
from pathlib import Path
import pytest

from kissim.utils import enter_temp_directory
from kissim.cli.compare import compare_from_cli

PATH_TEST_DATA = Path(__name__).parent / "kissim" / "tests" / "data"


@pytest.mark.parametrize(
    "args",
    [
        Namespace(
            input=(PATH_TEST_DATA / "fingerprints_test.json").absolute(),
            output=".",
            weights=None,
            ncores=None,
        ),
        Namespace(
            input=(PATH_TEST_DATA / "fingerprints_test.json").absolute(),
            output=".",
            weights=None,
            ncores=2,
        ),
        Namespace(
            input=(PATH_TEST_DATA / "fingerprints_test.json").absolute(),
            output=".",
            weights=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ncores=2,
        ),
    ],
)
def test_compare_from_cli(args):

    with enter_temp_directory():
        compare_from_cli(args)

        # Feature distances CSV there?
        assert Path("feature_distances.csv").exists()

        # Fingerprint distance CSV there?
        assert Path("fingerprint_distances.csv").exists()

        # Distances LOG there?
        assert Path("distances.log").exists()


@pytest.mark.parametrize(
    "args, error",
    [
        (
            Namespace(
                input=None,  # Missing
                output=None,  # Missing
                weights=None,
                ncores=None,
            ),
            TypeError,
        ),
        (
            Namespace(
                input=(PATH_TEST_DATA / "fingerprints_test.json").absolute(),
                output=None,  # Missing
                weights=None,
                ncores=None,
            ),
            TypeError,
        ),
        (
            Namespace(
                input=".",  # Not a file
                output=".",
                weights=None,
                ncores=None,
            ),
            IsADirectoryError,
        ),
        (
            Namespace(
                input="fp.json",  # File does not exist
                output=".",
                weights=None,
                ncores=None,
            ),
            FileNotFoundError,
        ),
        # TODO incorrect file
        (
            Namespace(
                input=(PATH_TEST_DATA / "fingerprints_test.json").absolute(),
                output=".",
                weights=[1.0],  # Must be 15 floats
                ncores=None,
            ),
            ValueError,
        ),
        (
            Namespace(
                input=(PATH_TEST_DATA / "fingerprints_test.json").absolute(),
                output=".",
                weights=[
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],  # Weights sum must be 1
                ncores=None,
            ),
            ValueError,
        ),
        (
            Namespace(
                input=(PATH_TEST_DATA / "fingerprints_test.json").absolute(),
                output=".",
                weights=None,
                ncores=10000,  # Too many
            ),
            ValueError,
        ),
    ],
)
def test_compare_from_cli_error(args, error):

    with pytest.raises(error):
        compare_from_cli(args)
