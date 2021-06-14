"""
Unit and regression test for the kissim.viewer.structure.StructureViewer class.
"""

from pathlib import Path

import pytest
from opencadd.databases.klifs import setup_local

from kissim.viewer import StructureViewer

PATH_TEST_DATA = Path(__name__).parent / "kissim" / "tests" / "data"
KLIFS_SESSION = setup_local(PATH_TEST_DATA / "KLIFS_download")


class TestStructureViewer:
    """Test structure viewer."""

    @pytest.mark.parametrize(
        "structure_klifs_id, klifs_session",
        [
            (12347, None),
            (12347, KLIFS_SESSION),
        ],
    )
    def test_from_structure_klifs_id(self, structure_klifs_id, klifs_session):

        viewer = StructureViewer.from_structure_klifs_id(structure_klifs_id, klifs_session)
        assert isinstance(viewer, StructureViewer)

    @pytest.mark.parametrize(
        "feature_name, show_side_chains",
        [
            ("size", True),
            ("size", False),
            ("sco", True),
            ("sco", False),
            ("hinge_region", True),
            ("hinge_region", False),
        ],
    )
    def test_show(self, structure_viewer, feature_name, show_side_chains):

        structure_viewer._show(feature_name, show_side_chains)
