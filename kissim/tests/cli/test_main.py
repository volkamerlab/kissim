"""
Unit and regression test for kissim's cli modules.


    # Test input options
    kissim encode
    kissim encode -i 12347 -o "fingerprints.json"
    kissim encode -i 12347 109 -o "fingerprints.json"
    kissim encode -i "structure_klifs_ids.txt" -o "fingerprints.json"

    # Test output options
    kissim encode -i 12347 109

    # Test number of cores
    kissim encode -i 12347 109 -o "fingerprints.json" -c 4

    # Test local KLIFS session
    kissim encode -i 12347 109 -o "fingerprints.json" -l "KLIFS_download"
    kissim encode -i 109 110 -o "fingerprints.json" -l "KLIFS_download"

# COMPARE
kissim compare
kissim compare -i "fingerprints.json" -o "distances.csv" 
"""

from pathlib import Path
import pytest
import subprocess

from kissim.utils import enter_temp_directory
from kissim.encoding import Fingerprint, FingerprintGenerator

PATH_TEST_DATA = Path(__name__).parent / "kissim" / "tests" / "data"


def pytest_args_to_subprocess_args(input, output, local, ncores):

    args = ["kissim", "encode"]
    if input:
        if isinstance(input, list):
            args_input = [str(i) for i in input]
            args.extend(["--input", *args_input])
        else:
            args_input = str(input)
            args.extend(["--input", args_input])
    if output:
        args_output = str(output)
        args.extend(["--output", args_output])
    if local:
        print(local)
        assert local.exists()
        args_local = str(local)
        args.extend(["--local", args_local])
    if ncores:
        args_ncores = str(ncores)
        args.extend(["--ncores", args_ncores])
    print(" ".join(args))
    return args


@pytest.mark.parametrize(
    "input, output, local, ncores",
    [
        ([12347], "fingerprints.json", None, None),
        ([12347, 109], "fingerprints.json", None, 2),  # Test parallelization
        # ([12347, 109], "fingerprints.json", PATH_TEST_DATA / "KLIFS_download", None),  # TODO: Test local KLIFS session
        (
            "structure_klifs_ids.txt",
            "fingerprints.json",
            None,
            None,
        ),  # Test IDs from file
    ],
)
def test_encode(input, output, local, ncores):
    """
    Test CLI for encoding using subprocesses.
    """

    args = pytest_args_to_subprocess_args(input, output, local, ncores)

    with enter_temp_directory():

        # TODO This is a workaround: Local file in kissim/tests/data cannot be read for some reason,
        # so I am creating this file in the working directory instead
        if isinstance(input, str):
            with open(input, "w") as f:
                f.write("12346\n109")

        subprocess.run(args, check=True)

        if output:
            # Json file there?
            assert Path(output).exists()
            # Log file there?
            assert Path(f"{Path(output).stem}.log").exists()

            # Json file can be loaded as FingerprintGenerator object?
            fingerprint_generator = FingerprintGenerator.from_json(output)
            assert isinstance(list(fingerprint_generator.data.values())[0], Fingerprint)


@pytest.mark.parametrize(
    "input, output, local, ncores",
    [
        (None, None, None, None),  # Missing input/output
        ([12347], None, None, None),  # Missing output
    ],
)
def test_encode_error(input, output, local, ncores):
    """
    Test if input arguments cause error.
    """

    args = pytest_args_to_subprocess_args(input, output, local, ncores)

    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(args, check=True)
