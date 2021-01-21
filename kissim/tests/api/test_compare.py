"""
Unit and regression test for the kissim.api.compare module.
"""

from pathlib import Path

import pytest

from kissim.api import compare

PATH_TEST_DATA = Path(__name__).parent / "kissim/tests/data/KLIFS_download"