"""
Unit and regression test for kissim's compare CLI.
"""

from argparse import Namespace
from pathlib import Path
import pytest

from kissim.utils import enter_temp_directory
from kissim.cli.encode import encode_from_cli

PATH_TEST_DATA = Path(__name__).parent / "kissim" / "tests" / "data"


@pytest.mark.parametrize(
    "args",
    [
        Namespace(
            input=["12347"],
            output="fingerprints.json",
            local=None,
            ncores=None,
        ),
        Namespace(
            input=["12347", "109"],
            output="fingerprints.json",
            local=None,
            ncores=None,
        ),
        Namespace(
            input=[str((PATH_TEST_DATA / "structure_klifs_ids.txt").absolute())],
            output="fingerprints.json",
            local=None,
            ncores=None,
        ),
        Namespace(
            input=["12347"],
            output="fingerprints.json",
            local=None,
            ncores=2,
        ),
        Namespace(
            input=["12347"],
            output="fingerprints.json",
            local=str((PATH_TEST_DATA / "KLIFS_download").absolute()),
            ncores=None,
        ),
    ],
)
def test_encode_from_cli(args):

    with enter_temp_directory():
        encode_from_cli(args)
        assert Path("fingerprints.json").exists()
        assert Path("fingerprints.log").exists()


@pytest.mark.parametrize(
    "args, error",
    [
        (
            Namespace(
                input=None,
                output=None,
                local=None,
                ncores=None,
            ),
            TypeError,
        ),
        (
            Namespace(
                input=["12347"],
                output=None,
                local=None,
                ncores=None,
            ),
            TypeError,
        ),
        (
            Namespace(
                input=["12347"],
                output="fingerprints.json",
                local=None,
                ncores=10000,
            ),
            ValueError,
        ),
    ],
)
def test_encode_from_cli_error(args, error):

    with pytest.raises(error):
        encode_from_cli(args)
