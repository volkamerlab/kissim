"""
Unit and regression test for kinsim_structure.preprocessing.KlifsMetadataLoader methods.
"""

from pathlib import Path

import pytest

from kinsim_structure.preprocessing import KlifsMetadataLoader, KlifsMetadataFilter

PATH_TEST_DATA = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data'


@pytest.mark.parametrize('klifs_overview_file, klifs_export_file, n_rows', [
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'overview.csv',
        PATH_TEST_DATA / 'KLIFS_download' / 'KLIFS_export.csv',
        9964
    )
])
def test_get_species(klifs_overview_file, klifs_export_file, n_rows):
    """
    Test filtering by species.

    Parameters
    ----------
    klifs_overview_file : str or pathlib.Path
        Path to KLIFS download file `overview.csv` containing mainly KLIFS alignment-related metadata.
    klifs_export_file : str or pathlib.Path
        Path to KLIFS download file `KLIFS_download/KLIFS_export.csv` containing mainly structure-related metadata.
    n_rows : int
        Number of rows (structures) after filtering.
    """

    klifs_metadata_loader = KlifsMetadataLoader()
    klifs_metadata_loader.from_files(klifs_overview_file, klifs_export_file)

    klifs_metadata = klifs_metadata_loader.data_essential

    klifs_metadata_filter = KlifsMetadataFilter()
    klifs_metadata_filter.filtered = klifs_metadata
    klifs_metadata_filter._get_species('Human')

    assert klifs_metadata_filter.filtered.shape[0] == n_rows


@pytest.mark.parametrize('klifs_overview_file, klifs_export_file, n_rows', [
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'overview.csv',
        PATH_TEST_DATA / 'KLIFS_download' / 'KLIFS_export.csv',
        9088
    )
])
def test_get_dfg(klifs_overview_file, klifs_export_file, n_rows):
    """
    Test filtering by DFG conformation.

    Parameters
    ----------
    klifs_overview_file : str or pathlib.Path
        Path to KLIFS download file `overview.csv` containing mainly KLIFS alignment-related metadata.
    klifs_export_file : str or pathlib.Path
        Path to KLIFS download file `KLIFS_download/KLIFS_export.csv` containing mainly structure-related metadata.
    n_rows : int
        Number of rows (structures) after filtering.
    """

    klifs_metadata_loader = KlifsMetadataLoader()
    klifs_metadata_loader.from_files(klifs_overview_file, klifs_export_file)

    klifs_metadata = klifs_metadata_loader.data_essential

    klifs_metadata_filter = KlifsMetadataFilter()
    klifs_metadata_filter.filtered = klifs_metadata
    klifs_metadata_filter._get_dfg('in')

    assert klifs_metadata_filter.filtered.shape[0] == n_rows


@pytest.mark.parametrize('klifs_overview_file, klifs_export_file, n_rows', [
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'overview.csv',
        PATH_TEST_DATA / 'KLIFS_download' / 'KLIFS_export.csv',
        10416
    )
])
def test_get_resolution(klifs_overview_file, klifs_export_file, n_rows):
    """
    Test filtering by structural resolution.

    Parameters
    ----------
    klifs_overview_file : str or pathlib.Path
        Path to KLIFS download file `overview.csv` containing mainly KLIFS alignment-related metadata.
    klifs_export_file : str or pathlib.Path
        Path to KLIFS download file `KLIFS_download/KLIFS_export.csv` containing mainly structure-related metadata.
    n_rows : int
        Number of rows (structures) after filtering.
    """

    klifs_metadata_loader = KlifsMetadataLoader()
    klifs_metadata_loader.from_files(klifs_overview_file, klifs_export_file)

    klifs_metadata = klifs_metadata_loader.data_essential

    klifs_metadata_filter = KlifsMetadataFilter()
    klifs_metadata_filter.filtered = klifs_metadata
    klifs_metadata_filter._get_resolution(4)

    assert klifs_metadata_filter.filtered.shape[0] == n_rows


@pytest.mark.parametrize('klifs_overview_file, klifs_export_file, n_rows', [
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'overview.csv',
        PATH_TEST_DATA / 'KLIFS_download' / 'KLIFS_export.csv',
        10399
    )
])
def test_get_qualityscore(klifs_overview_file, klifs_export_file, n_rows):
    """
    Test filtering by KLIFS quality score.

    Parameters
    ----------
    klifs_overview_file : str or pathlib.Path
        Path to KLIFS download file `overview.csv` containing mainly KLIFS alignment-related metadata.
    klifs_export_file : str or pathlib.Path
        Path to KLIFS download file `KLIFS_download/KLIFS_export.csv` containing mainly structure-related metadata.
    n_rows : int
        Number of rows (structures) after filtering.
    """

    klifs_metadata_loader = KlifsMetadataLoader()
    klifs_metadata_loader.from_files(klifs_overview_file, klifs_export_file)

    klifs_metadata = klifs_metadata_loader.data_essential

    klifs_metadata_filter = KlifsMetadataFilter()
    klifs_metadata_filter.filtered = klifs_metadata
    klifs_metadata_filter._get_qualityscore(4)

    assert klifs_metadata_filter.filtered.shape[0] == n_rows


@pytest.mark.parametrize('klifs_overview_file, klifs_export_file, path_klifs_download, n_rows', [
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'overview.csv',
        PATH_TEST_DATA / 'KLIFS_download' / 'KLIFS_export.csv',
        PATH_TEST_DATA / 'KLIFS_download',
        2  # Folders with both protein_pymol.mol2 and pocket.mol2
    )
])
def test_get_existing_mol2s(klifs_overview_file, klifs_export_file, path_klifs_download, n_rows):
    """
    Test filtering by existing pocket mol2 files.

    Parameters
    ----------
    klifs_overview_file : str or pathlib.Path
        Path to KLIFS download file `overview.csv` containing mainly KLIFS alignment-related metadata.
    klifs_export_file : str or pathlib.Path
        Path to KLIFS download file `KLIFS_download/KLIFS_export.csv` containing mainly structure-related metadata.
    path_klifs_download : pathlib.Path or str
        Path to directory of KLIFS dataset files.
    n_rows : int
        Number of rows (structures) after filtering.
    """

    klifs_metadata_loader = KlifsMetadataLoader()
    klifs_metadata_loader.from_files(klifs_overview_file, klifs_export_file)

    klifs_metadata = klifs_metadata_loader.data_essential

    klifs_metadata_filter = KlifsMetadataFilter()
    klifs_metadata_filter.filtered = klifs_metadata
    klifs_metadata_filter._get_existing_mol2s(path_klifs_download)

    assert klifs_metadata_filter.filtered.shape[0] == n_rows


@pytest.mark.parametrize('klifs_overview_file, klifs_export_file, n_rows', [
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'overview.csv',
        PATH_TEST_DATA / 'KLIFS_download' / 'KLIFS_export.csv',
        4908
    )
])
def test_get_unique_kinase_pdbid_pair(klifs_overview_file, klifs_export_file, n_rows):
    """
    Test filtering by unique kinase-PDB ID pairs.

    Parameters
    ----------
    klifs_overview_file : str or pathlib.Path
        Path to KLIFS download file `overview.csv` containing mainly KLIFS alignment-related metadata.
    klifs_export_file : str or pathlib.Path
        Path to KLIFS download file `KLIFS_download/KLIFS_export.csv` containing mainly structure-related metadata.
    n_rows : int
        Number of rows (structures) after filtering.
    """

    klifs_metadata_loader = KlifsMetadataLoader()
    klifs_metadata_loader.from_files(klifs_overview_file, klifs_export_file)

    klifs_metadata = klifs_metadata_loader.data_essential

    klifs_metadata_filter = KlifsMetadataFilter()
    klifs_metadata_filter.filtered = klifs_metadata
    klifs_metadata_filter._get_unique_kinase_pdbid_pair()

    assert klifs_metadata_filter.filtered.shape[0] == n_rows
