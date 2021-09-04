"""
Unit and regression test for kissim's normalized CLI.
"""

from pathlib import Path
import platform
import pytest
import subprocess

from kissim.utils import enter_temp_directory

PATH_TEST_DATA = Path(__name__).parent / "kissim" / "tests" / "data"
# PATH_TEST_DATA = Path(__file__).parent / "../data"


@pytest.mark.parametrize(
    "args",
    [
        f"kissim normalize -i {(PATH_TEST_DATA / 'fingerprints_test.json').absolute()} -o fingerprints_normalized_test.json",
        f"kissim normalize -i {(PATH_TEST_DATA / 'fingerprints_test.json').absolute()} -o fingerprints_normalized_test.json -f",
        f"kissim normalize -i {(PATH_TEST_DATA / 'fingerprints_test.json').absolute()} -o fingerprints_normalized_test.json -m min_max",
    ],
)
def test_main_normalize(args):
    """
    Test CLI for normalize using subprocesses.
    """

    output = Path("fingerprints_normalized_test.json")
    args = args.split()

    with enter_temp_directory():
        subprocess.run(args, check=True)

        # Json file there?
        assert output.exists()
        # Log file there?
        if platform.system() != "Windows":
            assert Path(f"{output.stem}.log").exists()
