"""
Unit and regression test for the kissim.cli.encode/compare module.
"""

from argparse import Namespace
import pytest

from kissim.utils import enter_temp_directory
from kissim.cli import encode_from_cli, compare_from_cli


@pytest.mark.parametrize(
    "encode_args, compare_args",
    [
        (
            Namespace(
                func=compare_from_cli, input=["12347"], local=None, ncores=1, output="fps.json"
            ),
            Namespace(
                distance="scaled_euclidean",
                func=compare_from_cli,
                input="fps.json",
                ncores=1,
                output="matrix.csv",
                weights="001",
            ),
        )
    ],
)
def test_encode_from_cli(encode_args, compare_args):

    with enter_temp_directory():
        encode_from_cli(encode_args)
        compare_from_cli(compare_args)
