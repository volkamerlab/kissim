"""
Unit and regression test for the kissim.viewer.pair.StructurePairViewer class.
"""

from pathlib import Path

import pytest
from opencadd.databases.klifs import setup_local

from kissim.viewer import StructurePairViewer

PATH_TEST_DATA = Path(__name__).parent / "kissim" / "tests" / "data"
KLIFS_SESSION = setup_local(PATH_TEST_DATA / "KLIFS_download")


class TestStructurePairViewer:
    """Test structure viewer."""

    @pytest.mark.parametrize(
        "structure_klifs_id1, structure_klifs_id2, klifs_session",
        [
            (12347, 3833, None),
            (12347, 3833, KLIFS_SESSION),
        ],
    )
    def test_from_structure_pair_viewer(
        self, structure_klifs_id1, structure_klifs_id2, klifs_session
    ):
        viewer = StructurePairViewer.from_structure_klifs_ids(
            structure_klifs_id1, structure_klifs_id2, klifs_session
        )
        assert isinstance(viewer, StructurePairViewer)

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
    def test_show(self, structure_pair_viewer, feature_name, show_side_chains):
        structure_pair_viewer._show(feature_name, show_side_chains)
