"""
Unit and regression test for kissim's encoding CLI.
"""

from pathlib import Path
import platform
import pytest
import subprocess

from kissim.utils import enter_temp_directory
from kissim.encoding import FingerprintBase, FingerprintGenerator

PATH_TEST_DATA = Path(__name__).parent / "kissim" / "tests" / "data"


@pytest.mark.parametrize(
    "args",
    [
        "kissim encode -i 12347 -o fingerprints.json",
        "kissim encode -i 12347 109 -o fingerprints.json",
        f"kissim encode -i {(PATH_TEST_DATA / 'structure_klifs_ids.txt').absolute()} -o fingerprints.json",
        "kissim encode -i 12347 109 -o fingerprints.json -c 2",
        f"kissim encode -i 12347 109 -o fingerprints.json -l {(PATH_TEST_DATA / 'KLIFS_download').absolute()}",
    ],
)
def test_main_encode(args):
    """
    Test CLI for encoding using subprocesses.
    """

    output = Path("fingerprints.json")
    args = args.split()

    with enter_temp_directory():
        subprocess.run(args, check=True)

        # Json file there?
        assert output.exists()
        # Log file there?
        if platform.system() != "Windows":
            assert Path(f"{output.stem}.log").exists()

        # Json file can be loaded as FingerprintGenerator object?
        fingerprint_generator = FingerprintGenerator.from_json(output)
        assert isinstance(fingerprint_generator, FingerprintGenerator)
        assert isinstance(list(fingerprint_generator.data.values())[0], FingerprintBase)


@pytest.mark.parametrize(
    "args",
    [
        "kissim encode -i 12347 109",
        "kissim encode -i 12347 109 -o fingerprints.json -c 1000",
    ],
)
def test_encode_error(args):
    """
    Test if input arguments cause error.
    """

    args = args.split()

    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(args, check=True)
