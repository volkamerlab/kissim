"""
Unit and regression test for the kissim.viewer.kinase.KinaseViewer class.
"""

from pathlib import Path

import pytest
from opencadd.databases.klifs import setup_local

from kissim.viewer import KinaseViewer

PATH_TEST_DATA = Path(__name__).parent / "kissim" / "tests" / "data"
KLIFS_SESSION = setup_local(PATH_TEST_DATA / "KLIFS_download")


class TestKinaseViewer:
    """Test kinase viewer."""

    @pytest.mark.parametrize(
        "kinase_klifs_id, klifs_session, example_structure_klifs_id, n_sampled",
        [
            (393, KLIFS_SESSION, None, None),
            (393, None, None, 3),  # Sample 3 structures
            (393, KLIFS_SESSION, None, 3000),  # Sample 3000 structures; defaults back to 9
            (393, KLIFS_SESSION, 109, 3),  # Set example structure
        ],
    )
    def test_from_kinase_klifs_id(
        self, kinase_klifs_id, klifs_session, example_structure_klifs_id, n_sampled
    ):

        viewer = KinaseViewer.from_kinase_klifs_id(
            kinase_klifs_id, klifs_session, example_structure_klifs_id, n_sampled
        )
        assert isinstance(viewer, KinaseViewer)

    @pytest.mark.parametrize(
        "kinase_klifs_id, klifs_session, example_structure_klifs_id, n_sampled",
        [
            (393, KLIFS_SESSION, 0, None),  # Structure unknown to kinase
            (121, KLIFS_SESSION, 0, None),  # Kinase as only 1 structure
            (393, KLIFS_SESSION, None, 0),  # Selected too few structures
            (393, KLIFS_SESSION, None, 1),  # Selected too few structures
            (393, KLIFS_SESSION, 1, None),  # Example structure unknown to kinase
        ],
    )
    def test_from_kinase_klifs_id_raises(
        self, kinase_klifs_id, klifs_session, example_structure_klifs_id, n_sampled
    ):

        with pytest.raises(ValueError):
            KinaseViewer.from_kinase_klifs_id(
                kinase_klifs_id, klifs_session, example_structure_klifs_id, n_sampled
            )

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
    def test_show(self, kinase_viewer, feature_name, show_side_chains):

        kinase_viewer._show(feature_name, show_side_chains)
