"""
Unit and regression test for kinsim_structure.preprocessing.KlifsMetadataLoader methods.
"""

from pathlib import Path

import pytest

from kinsim_structure.preprocessing import KlifsMetadataLoader

PATH_TEST_DATA = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data'


@pytest.mark.parametrize('kinase_names_string, kinase_names_list', [
    ('a', ['a']),
    ('a (b)', ['a', 'b']),
    ('a (b', ['a', 'b']),
    ('a b)', ['a', 'b']),
    ('a (b, c)', ['a', 'b', 'c']),
    ('a (b c)', ['a', 'b', 'c']),

])
def test_format_kinase_name(kinase_names_string, kinase_names_list):
    """
    Test formatting of kinase name(s): One or multiple kinase names (additional names in brackets) are formated to list
    of kinase names.

    Parameters
    ----------
    kinase_names_string : str
        String, here kinase name(s).
    kinase_names_list : list of str
        List of strings, here list of kinase name(s).
    """

    klifs_metadata_loader = KlifsMetadataLoader()
    kinase_names_list_calculated = klifs_metadata_loader._format_kinase_name(kinase_names_string)

    assert kinase_names_list == kinase_names_list_calculated


@pytest.mark.parametrize('klifs_export_file, n_rows', [
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'KLIFS_export.csv',
        10469
    )
])
def test_from_klifs_export_file(klifs_export_file, n_rows):
    """
    Test DataFrame for KLIFS_export.csv file from KLIFS database download.

    Parameters
    ----------
    klifs_export_file : pathlib.Path or str
        Path to KLIFS_export.csv file from KLIFS database download.
    n_rows : int
        Number of rows in DataFrame for KLIFS_export.csv file from KLIFS database download.
    """

    klifs_metadata_loader = KlifsMetadataLoader()
    klifs_export = klifs_metadata_loader._from_klifs_export_file(klifs_export_file)

    klifs_export_columns = [
        'kinase',
        'kinase_all',
        'family',
        'groups',
        'pdb_id',
        'chain',
        'alternate_model',
        'species',
        'ligand_orthosteric_name',
        'ligand_orthosteric_pdb_id',
        'ligand_allosteric_name',
        'ligand_allosteric_pdb_id',
        'dfg',
        'ac_helix'
    ]

    assert len(klifs_export) == n_rows
    assert list(klifs_export.columns) == klifs_export_columns


@pytest.mark.parametrize('klifs_overview_file, n_rows', [
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'overview.csv',
        10469
    )
])
def test_from_klifs_overview_file(klifs_overview_file, n_rows):
    """
    Test DataFrame for overview.csv file from KLIFS database download.

    Parameters
    ----------
    klifs_overview_file : pathlib.Path or str
        Path to overview.csv file from KLIFS database download.
    n_rows : int
        Number of rows in DataFrame for overview.csv file from KLIFS database download.
    """

    klifs_metadata_loader = KlifsMetadataLoader()
    klifs_overview = klifs_metadata_loader._from_klifs_overview_file(klifs_overview_file)

    klifs_overview_columns = [
        'species',
        'kinase',
        'pdb_id',
        'alternate_model',
        'chain',
        'ligand_orthosteric_pdb_id',
        'ligand_allosteric_pdb_id',
        'rmsd1',
        'rmsd2',
        'qualityscore',
        'pocket',
        'resolution',
        'missing_residues',
        'missing_atoms',
        'full_ifp',
        'fp_i',
        'fp_ii',
        'bp_i_a',
        'bp_i_b',
        'bp_ii_in',
        'bp_ii_a_in',
        'bp_ii_b_in',
        'bp_ii_out',
        'bp_ii_b',
        'bp_iii',
        'bp_iv',
        'bp_v'
    ]

    assert len(klifs_overview) == n_rows
    assert list(klifs_overview.columns) == klifs_overview_columns


@pytest.mark.parametrize('klifs_overview_file, klifs_export_file', [
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'overview.csv',
        PATH_TEST_DATA / 'KLIFS_download' / 'KLIFS_export.csv'
    )
])
def test_from_files(klifs_overview_file, klifs_export_file):
    """
    Test shape of KLIFS metadata as DataFrame.

    Parameters
    ----------
    klifs_overview_file : pathlib.Path or str
        Path to overview.csv file from KLIFS database download.
    klifs_export_file : pathlib.Path or str
        Path to KLIFS_export.csv file from KLIFS database download.
    """

    klifs_metadata_loader = KlifsMetadataLoader()
    klifs_metadata_loader.from_files(klifs_overview_file, klifs_export_file)

    assert klifs_metadata_loader.data.shape == (10469, 35)


@pytest.mark.parametrize('klifs_overview_file, klifs_export_file', [
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'overview.csv',
        PATH_TEST_DATA / 'KLIFS_download' / 'KLIFS_export.csv'
    )
])
def test_data_essential(klifs_overview_file, klifs_export_file):
    """
    Test class attribute, i.e. column-reduced version of full DataFrame.

    Parameters
    ----------
    klifs_overview_file : pathlib.Path or str
        Path to overview.csv file from KLIFS database download.
    klifs_export_file : pathlib.Path or str
        Path to KLIFS_export.csv file from KLIFS database download.
    """

    klifs_metadata_loader = KlifsMetadataLoader()
    klifs_metadata_loader.from_files(klifs_overview_file, klifs_export_file)

    metadata_columns = [
            'pdb_id',
            'alternate_model',
            'chain',
            'kinase',
            'kinase_all',
            'family',
            'groups',
            'species',
            'dfg',
            'ac_helix',
            'pocket',
            'rmsd1',
            'rmsd2',
            'qualityscore',
            'resolution',
            'missing_residues',
            'missing_atoms',
            'filepath'
        ]

    assert list(klifs_metadata_loader.data_essential.columns) == metadata_columns

