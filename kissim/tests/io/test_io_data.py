"""
Unit and regression test for the KlifsToKissimData class.
"""

from pathlib import Path
import pytest

from opencadd.databases.klifs import setup_local, setup_remote

from kissim.io import KlifsToKissimData

PATH_TEST_DATA = Path(__name__).parent / "kissim/tests/data/KLIFS_download"

# Set local and remote KLIFS session
LOCAL = setup_local(PATH_TEST_DATA)
REMOTE = setup_remote()


class TestKlifsToKissimData:
    """
    Test KlifsToKissimData class.
    """

    @pytest.mark.parametrize(
        "structure_klifs_id, klifs_session, exists",
        [
            (118, LOCAL, True),
            (12347, REMOTE, True),
            (100000, LOCAL, False),  # Invalid ID
            (117, LOCAL, False),  # Missing complex.pdb
            (100000, REMOTE, False),  # Invalid ID
            (13623, LOCAL, False),  # Incomplete pocket sequence
            (1243, LOCAL, False),  # Pocket length not the same in sequence and structure
        ],
    )
    def test_from_structure_klifs_id(self, structure_klifs_id, klifs_session, exists):
        data = KlifsToKissimData.from_structure_klifs_id(structure_klifs_id, klifs_session)
        if exists:
            assert isinstance(data, KlifsToKissimData)
            assert data.structure_klifs_id == structure_klifs_id
            assert isinstance(data.text, str)
            assert data.extension == "pdb"
            assert len(data.residue_ids) == len(data.residue_ixs)
        else:
            assert data is None

    @pytest.mark.parametrize(
        "structure_klifs_id, klifs_session, exists",
        [
            (118, LOCAL, True),
            (12347, REMOTE, True),
            (100000, LOCAL, False),
            (100000, REMOTE, False),
        ],
    )
    def test_structure_klifs_id_exists(self, structure_klifs_id, klifs_session, exists):
        data = KlifsToKissimData()
        data.structure_klifs_id = structure_klifs_id
        data.klifs_session = klifs_session
        assert data._structure_klifs_id_exists() == exists

    @pytest.mark.parametrize(
        "structure_klifs_id, exists",
        [
            (118, True),  # Structure KLIFS ID and files exist
            (117, False),  # Structure KLIFS ID but not file(s) exist
        ],
    )
    def test_local_session_files_exist(self, structure_klifs_id, exists):
        data = KlifsToKissimData()
        data.structure_klifs_id = structure_klifs_id
        data.klifs_session = LOCAL
        assert data._local_session_files_exist() == exists

    @pytest.mark.parametrize(
        "structure_klifs_id, klifs_session",
        [(118, LOCAL), (12347, REMOTE)],
    )
    def test_get_text_and_extension(self, structure_klifs_id, klifs_session):
        data = KlifsToKissimData()
        data.structure_klifs_id = structure_klifs_id
        data.klifs_session = klifs_session
        text, extension = data._get_text_and_extension()
        assert isinstance(text, str)
        assert extension == "pdb"

    @pytest.mark.parametrize(
        "structure_klifs_id, klifs_session",
        [(118, LOCAL), (12347, REMOTE)],
    )
    def test_get_pocket_residue_ids_and_ixs(self, structure_klifs_id, klifs_session):
        data = KlifsToKissimData()
        data.structure_klifs_id = structure_klifs_id
        data.klifs_session = klifs_session
        residue_ids, residue_ixs = data._get_pocket_residue_ids_and_ixs()
        assert len(residue_ids) == len(residue_ixs)
