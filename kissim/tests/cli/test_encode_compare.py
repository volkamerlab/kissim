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
            Namespace(input=["12347"], output="fps.json", local=None, ncores=1),
            Namespace(
                input="fps.json",
                output="matrix.csv",
                distance="scaled_euclidean",
                weights="001",
                ncores=1,
            ),
        )
    ],
)
def ttest_encode_from_cli(encode_args, compare_args):

    with enter_temp_directory():
        encode_from_cli(encode_args)
        compare_from_cli(compare_args)
