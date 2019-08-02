"""
preprocessing.py

Subpocket-based structural fingerprint for kinase pocket comparison.

Handles the primary functions for the preprocessing of the KLIFS dataset.
"""

import logging
from pathlib import Path

from Bio.PDB import PDBList
import pandas as pd

logger = logging.getLogger(__name__)


def get_klifs_metadata_from_files(klifs_overview_file, klifs_export_file, remove_subpocket_columns=True):
    """
    Get KLIFS metadata as DataFrame.
    
    1. Load KLIFS download files ...
    2. Unify column names and column cell formatting.
    3. Merge into one DataFrame.
    4. Optional: Remove subpocket columns.
    
    Parameters
    ----------
    klifs_overview_file : str or pathlib.Path
        Path to KLIFS download file `overview.csv` containing mainly KLIFS alignment-related metadata.
    klifs_export_file : str or pathlib.Path
        Path to KLIFS download file `KLIFS_download/KLIFS_export.csv` containing mainly structure-related metadata.
    remove_subpocket_columns : bool
        Remove subpocket columns by default.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing merged metadate from both input KLIFS tables.
    """
    
    # Load KLIFS files 
    # - `overview.csv` (KLIFS alignment metadata) and 
    # - `KLIFS_export.csv` (structural metadata on PDB files)
    
    klifs_overview_file = Path(klifs_overview_file)
    klifs_export_file = Path(klifs_export_file)
    
    klifs_overview = pd.read_csv(klifs_overview_file)
    klifs_export = pd.read_csv(klifs_export_file)
    
    # Both tables contain some columns with the same information, such as:
    # - Species
    # - Kinase
    # - PDB ID
    # - Chain
    # - Alternate model (alternate conformation)
    # - Orthosteric ligand PDB ID
    # - Allosteric ligand PDB ID
    
    klifs_overview.rename(
        columns={
            'pdb': 'pdb_id',
            'alt': 'alternate_model',
            'orthosteric_PDB': 'ligand_orthosteric_pdb_id',
            'allosteric_PDB': 'ligand_allosteric_pdb_id',
        },
        inplace=True
        )

    klifs_export.rename(
        columns={
            'NAME': 'kinase',
            'FAMILY': 'family',
            'GROUPS': 'groups',
            'PDB': 'pdb_id',
            'CHAIN': 'chain',
            'ALTERNATE_MODEL': 'alternate_model',
            'SPECIES': 'species',
            'LIGAND': 'ligand_orthosteric_name',
            'PDB_IDENTIFIER': 'ligand_orthosteric_pdb_id',
            'ALLOSTERIC_NAME': 'ligand_allosteric_name',
            'ALLOSTERIC_PDB': 'ligand_allosteric_pdb_id',
            'DFG': 'dfg',
            'AC_HELIX': 'ac_helix',
        },
        inplace=True
    )
    
    # Check if PDB IDs occur in one file but not the other
    
    not_in_export = klifs_export[~klifs_export.pdb_id.isin(klifs_overview.pdb_id)]
    not_in_overview = klifs_overview[~klifs_overview.pdb_id.isin(klifs_export.pdb_id)]
    
    if not_in_export.size > 0:
        raise ValueError(f'Number of PDBs in overview but not in export table: {not_in_export.size}.\n')

    if not_in_overview.size > 0:
        raise(f'Number of PDBs in export but not in overview table: {not_in_overview.size}.'
              f'PDB codes are probably updated because structures are deprecated')
    
    # Unify column 'alternate model'
    
    klifs_overview.alternate_model.replace(' ', '-', inplace=True)
    
    # Unify column 'kinase'
    # If kinase names in brackets, extract only this and remove the rest
    # Example: 'CSNK2A1 (CK2a1)' results in 'CK2a1'
    
    klifs_export.kinase = klifs_export.kinase.apply(lambda x: x[x.find('(')+1:x.find(')')] if '(' in x else x)
    
    # Merge on mutual columns
    
    klifs_metadata = klifs_export.merge(
        right=klifs_overview,
        how='outer',
        on=['species',
            'kinase',
            'pdb_id',
            'chain',
            'alternate_model',
            'ligand_orthosteric_pdb_id',
            'ligand_allosteric_pdb_id']
    )
    
    if not klifs_overview.shape[0] == klifs_export.shape[0] == klifs_metadata.shape[0]:
        raise ValueError(f'Output table has incorrect number of rows:'
                         f'KLIFS overview table has shape: {klifs_overview.shape}'
                         f'KLIFS export table has shape: {klifs_export.shape}'
                         f'KLIFS merged table has shape: {klifs_metadata.shape}')
    
    if not (klifs_overview.shape[1] + klifs_export.shape[1] - 7) == klifs_metadata.shape[1]:
        raise ValueError(f'Output table has incorrect number of columns'
                         f'KLIFS overview table has shape: {klifs_overview.shape}'
                         f'KLIFS export table has shape: {klifs_export.shape}'
                         f'KLIFS merged table has shape: {klifs_metadata.shape}')
        
    # Remove subpocket columns
    if remove_subpocket_columns:
        klifs_metadata.drop(labels=klifs_metadata.columns[21:], axis=1, inplace=True)
        
    return klifs_metadata


def get_species(klifs_metadata, species='Human'):
    """
    Filter KLIFS dataset by species.

    Parameters
    ----------
    klifs_metadata : pandas.DataFrame
        DataFrame containing merged metadate from both input KLIFS tables.
    species : str
        String for species name.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing merged metadata from both input KLIFS tables filtered by certain criteria.
    """

    klifs_metadata_filtered = klifs_metadata.copy()

    # Select species
    if species in klifs_metadata_filtered.species.unique():
        klifs_metadata_filtered.drop(
            klifs_metadata_filtered[klifs_metadata_filtered.species != species].index,
            inplace=True
        )
    else:
        raise ValueError(f'Species {species} not in species list: '
                         f'{", ".join(klifs_metadata_filtered.species.unique())}')

    return klifs_metadata_filtered


def get_dfg(klifs_metadata, dfg='in'):
    """
    Filter KLIFS dataset by DFG loop position.

    Parameters
    ----------
    klifs_metadata : pandas.DataFrame
        DataFrame containing merged metadate from both input KLIFS tables.
    dfg : str
        String for DFG loop position.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing merged metadata from both input KLIFS tables filtered by certain criteria.
    """

    klifs_metadata_filtered = klifs_metadata.copy()

    # Select DFG conformation
    if dfg in klifs_metadata_filtered.dfg.unique():
        klifs_metadata_filtered.drop(
            klifs_metadata_filtered[klifs_metadata_filtered.dfg != dfg].index,
            inplace=True
        )
    else:
        raise ValueError(f'DFG position {dfg} not in DFG list: {", ".join(klifs_metadata_filtered.dfg.unique())}')

    return klifs_metadata_filtered


def get_unique_pdbid_per_kinase(klifs_metadata):
    """
    Filter KLIFS dataset by keeping only the KLIFS entry per kinase-PDB ID combination with the best quality score.
    
    Parameters
    ----------
    klifs_metadata : pandas.DataFrame
        DataFrame containing merged metadate from both input KLIFS tables.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing merged metadata from both input KLIFS tables filtered by certain criteria.
    """
    
    klifs_metadata_filtered = klifs_metadata.copy()
        
    # For each kinase and PDB IDs with multiple KLIFS entries (structures),
    # select entry with the best quality score

    # Sort by kinase, PDB ID and quality score
    # (so that for multiple equal kinase-pdb_id combos, highest quality score will come first)
    klifs_metadata_filtered.sort_values(
        by=['kinase', 'pdb_id', 'qualityscore'],
        ascending=[True, True, False],
        inplace=True
    )
    # Drop duplicate kinase-pdb_id combos and keep only first (with highest quality score)
    klifs_metadata_filtered.drop_duplicates(
        subset=['kinase', 'pdb_id'],
        keep='first',
        inplace=True
    )
    # Reset DataFrame indices
    klifs_metadata_filtered.reset_index(inplace=True)

    return klifs_metadata_filtered


def drop_missing_mol2s(klifs_metadata, path_to_data):
    """
    Drop entries in KLIFS metadata that have no corresponding mol2 file.

    Parameters
    ----------
    klifs_metadata : pandas.DataFrame
        DataFrame containing merged metadate from both input KLIFS tables.
    path_to_data : str or pathlib.Path
        Path to directory of KLIFS dataset files.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing merged metadata from both input KLIFS tables filtered by certain criteria.
    """

    path_to_data = Path(path_to_data) / 'raw' / 'KLIFS_download'

    klifs_metadata_filtered = klifs_metadata.copy()

    indices = []

    for index, row in klifs_metadata_filtered.iterrows():

        # Depending on whether alternate model and chain ID is given build file path:
        mol2_path = path_to_data / row.species.upper() / row.kinase

        if row.alternate_model != '-' and row.chain != '-':
            mol2_path = mol2_path / f'{row.pdb_id}_alt{row.alternate_model}_chain{row.chain}' / 'pocket.mol2'
        elif row.alternate_model == '-' and row.chain != '-':
            mol2_path = mol2_path / f'{row.pdb_id}_chain{row.chain}' / 'pocket.mol2'
        elif row.alternate_model == '-' and row.chain == '-':
            mol2_path = mol2_path / f'{row.pdb_id}' / 'pocket.mol2'
        else:
            raise ValueError(f'{row.alternate_model}, {row.chain}')

        # Not all paths exist - save list with missing paths
        if not mol2_path.exists():
            indices.append(index)

    klifs_metadata_filtered.drop(indices, inplace=True)

    return klifs_metadata_filtered


def drop_missing_pdbs(klifs_metadata, path_to_data):
    """
    Drop entries in KLIFS metadata that have no corresponding pdb file.

    Parameters
    ----------
    klifs_metadata : pandas.DataFrame
        DataFrame containing merged metadate from both input KLIFS tables.

    path_to_data : str or pathlib.Path
        Path to directory of KLIFS dataset files.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing merged metadata from both input KLIFS tables filtered by certain criteria.
    """

    path_to_data = Path(path_to_data) / 'raw' / 'PDB_download'

    klifs_metadata_filtered = klifs_metadata.copy()

    indices = []

    for index, row in klifs_metadata_filtered.iterrows():

        # Depending on whether alternate model and chain ID is given build file path:
        cif_path = path_to_data / f'{row.pdb_id}.cif'

        # Not all paths exist - save list with missing paths
        if not cif_path.exists():
            indices.append(index)

    klifs_metadata_filtered.drop(indices, inplace=True)

    return klifs_metadata_filtered


def calculate_gap_rate(klifs_metadata):
    """
    Calculate gap rate at every KLIFS MSA position across the filtered kinase data set
    
    Parameters
    ----------
    klifs_metadata : pandas.DataFrame
        DataFrame containing merged metadate from both input KLIFS tables.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing gap rates for each position in the KLIFS alignment for the input data.
    """

    gaps = [0] * 85
    
    for pocket in klifs_metadata.pocket:
        for klifs_position, residue in enumerate(pocket):
            if residue == '_':
                gaps[klifs_position] += 1

    gap_rate = [round(i/float(klifs_metadata.shape[0]), 4) for i in gaps]
    
    return pd.concat([pd.Series(range(1, 86), name='klifs_position'),
                      pd.Series(gaps, name='gaps'),  
                      pd.Series(gap_rate, name='gap_rate')], 
                     axis=1)


def download_from_pdb(klifs_metadata, output_path):
    """
    Download structure files from the PDB for KLIFS dataset.
    
    Parameters
    ----------
    klifs_metadata : pandas.DataFrame
        DataFrame containing merged metadate from both input KLIFS tables.
    output_path : str or pathlib.Path
        Path to output directory.
    """
    
    pdbfile = PDBList()
    
    for index, row in klifs_metadata.iterrows():
        if not (Path(output_path) / f'{row.pdb_id}.cif').exists():
            print(Path(output_path) / f'{row.pdb_id}.cif')
            pdbfile.retrieve_pdb_file(row.pdb_id, pdir=output_path)
        else:
            continue