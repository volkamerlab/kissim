"""
Unit and regression test for the kissim package.
"""

# Import package, test suite, and other packages as needed
import kissim
import pytest
import sys

def test_kissim_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "kissim" in sys.modules
