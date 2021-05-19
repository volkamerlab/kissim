"""
Unit and regression test for kissim's outliers CLI.
"""

from pathlib import Path
import platform
import pytest
import subprocess

from kissim.utils import enter_temp_directory

PATH_TEST_DATA = Path(__name__).parent / "kissim" / "tests" / "data"
PATH_TEST_DATA = Path(__file__).parent / "../data"


@pytest.mark.parametrize(
    "args",
    [
        f"kissim outliers -i {(PATH_TEST_DATA / 'fingerprints_test.json').absolute()} -d 5 -o fingerprints_clean_test.json",
    ],
)
def test_main_outliers(args):
    """
    Test CLI for outliers using subprocesses.
    """

    output = Path("fingerprints_clean_test.json")
    args = args.split()

    with enter_temp_directory():
        subprocess.run(args, check=True)

        # Json file there?
        assert output.exists()
        # Log file there?
        if platform.system() != "Windows":
            assert Path(f"{output.stem}.log").exists()
