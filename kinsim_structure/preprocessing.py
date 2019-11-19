"""
preprocessing.py

Subpocket-based structural fingerprint for kinase pocket comparison.

Handles the primary functions for the preprocessing of the KLIFS dataset.
"""

import logging
from pathlib import Path

from Bio.PDB import PDBList, MMCIFParser
import pandas as pd

from kinsim_structure.auxiliary import KlifsMoleculeLoader, get_klifs_regions

logger = logging.getLogger(__name__)


class KlifsMetadataLoader:

    def __init__(self):
        self.data = None

    def from_files(self, klifs_overview_file, klifs_export_file):
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

        Returns
        -------
        pandas.DataFrame
            DataFrame containing merged metadata from both input KLIFS tables.
        """

        klifs_overview = self._from_klifs_overview_file(Path(klifs_overview_file))
        klifs_export = self._from_klifs_export_file(Path(klifs_export_file))

        klifs_metadata = self._merge_files(klifs_overview, klifs_export)
        klifs_metadata = self._add_filepaths(klifs_metadata)

        self.data = klifs_metadata

    @property
    def data_reduced(self):
        """
        Reduced number of columns for metadata: structure and kinase information on kinase only.

        Returns
        -------
        pandas.DataFrame
            Metadata without subpocket columns.
        """

        klifs_metadata = self.data.copy()

        sorted_columns = [
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

        klifs_metadata = klifs_metadata[sorted_columns]

        return klifs_metadata

    def _from_klifs_export_file(self, klifs_export_file):
        """
        Read KLIFS_export.csv file from KLIFS database download as DataFrame and unify format with overview.csv format.

        Parameters
        ----------
        klifs_export_file : pathlib.Path or str
            Path to KLIFS_export.csv file from KLIFS database download.

        Returns
        -------
        pandas.DataFrame
            Data loaded and formatted: KLIFS_export.csv file from KLIFS database download.
        """

        klifs_export = pd.read_csv(Path(klifs_export_file))

        # Unify column names with column names in overview.csv
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

        # Unify column 'kinase': Sometimes several kinase names are available, e.g. "EPHA7 (EphA7)"
        # Column "kinase": Retain only first kinase name, e.g. EPHA7
        # Column "kinase_all": Save all kinase names as list, e.g. [EPHA7, EphA7]
        kinase_names = [self._format_kinase_name(i) for i in klifs_export.kinase]
        klifs_export.kinase = [i[0] for i in kinase_names]
        klifs_export.insert(
            loc=1,
            column='kinase_all',
            value=kinase_names
        )

        return klifs_export

    @staticmethod
    def _from_klifs_overview_file(klifs_overview_file):
        """
        Read overview.csv file from KLIFS database download as DataFrame and unify format with KLIFS_export.csv format.

        Parameters
        ----------
        klifs_overview_file : pathlib.Path or str
            Path to overview.csv file from KLIFS database download.

        Returns
        -------
        pandas.DataFrame
            Data loaded and formatted: overview.csv file from KLIFS database download.
        """

        klifs_overview = pd.read_csv(Path(klifs_overview_file))

        # Unify column names with column names in KLIFS_export.csv
        klifs_overview.rename(
            columns={
                'pdb': 'pdb_id',
                'alt': 'alternate_model',
                'orthosteric_PDB': 'ligand_orthosteric_pdb_id',
                'allosteric_PDB': 'ligand_allosteric_pdb_id',
            },
            inplace=True
        )

        # Unify column 'alternate model' with corresponding column in KLIFS_export.csv
        klifs_overview.alternate_model.replace(' ', '-', inplace=True)

        return klifs_overview

    @staticmethod
    def _format_kinase_name(kinase_name):
        """
        Format kinase name(s): One or multiple kinase names (additional names in brackets) are formated to list of
        kinase names.

        Examples:
        Input: "EPHA7 (EphA7)", output: ["EPHA7", "EphA7"].
        Input: "ITK", output: ["ITK"].

        Parameters
        ----------
        kinase_name : str
            String, here kinase name(s).

        Returns
        -------
        List of str
            List of strings, here list of kinase name(s).
        """

        kinase_name = kinase_name.replace('(', '')
        kinase_name = kinase_name.replace(')', '')
        kinase_name = kinase_name.replace(',', '')
        kinase_name = kinase_name.split()

        return kinase_name

    @staticmethod
    def _merge_files(klifs_export, klifs_overview):
        """

        Parameters
        ----------
        klifs_export : pandas.DataFrame
            Data contained in KLIFS_export.csv file from KLIFS database download.
        klifs_overview : pandas.DataFrame
            Data contained in overview.csv file from KLIFS database download.

        Returns
        -------
        pandas.DataFrame
            Merged data contained in overview.csv and KLIFS_export.csv files from KLIFS database download.
        """

        # Check if PDB IDs occur in one file but not the other
        not_in_export = klifs_export[~klifs_export.pdb_id.isin(klifs_overview.pdb_id)]
        not_in_overview = klifs_overview[~klifs_overview.pdb_id.isin(klifs_export.pdb_id)]

        if not_in_export.size > 0:
            raise ValueError(f'Number of PDBs in overview but not in export table: {not_in_export.size}.\n')
        if not_in_overview.size > 0:
            raise (f'Number of PDBs in export but not in overview table: {not_in_overview.size}.'
                   f'PDB codes are probably updated because structures are deprecated.')

        # Merge on mutual columns:
        # Species, kinase, PDB ID, chain, alternate model, orthosteric and allosteric ligand PDB ID

        mutual_columns = [
            'species',
            'pdb_id',
            'chain',
            'alternate_model'
        ]

        klifs_metadata = klifs_export.merge(
            right=klifs_overview,
            how='inner',
            on=mutual_columns
        )

        klifs_metadata.drop(
            columns=['ligand_orthosteric_pdb_id_y', 'ligand_allosteric_pdb_id_y', 'kinase_y'],
            inplace=True
        )

        klifs_metadata.rename(
            columns={
                'ligand_orthosteric_pdb_id_x': 'ligand_orthosteric_pdb_id',
                'ligand_allosteric_pdb_id_x': 'ligand_allosteric_pdb_id',
                'kinase_x': 'kinase'
            },
            inplace=True
        )

        if not (klifs_overview.shape[1] + klifs_export.shape[1] - 7) == klifs_metadata.shape[1]:
            raise ValueError(f'Output table has incorrect number of columns\n'
                             f'KLIFS overview table has shape: {klifs_overview.shape}\n'
                             f'KLIFS export table has shape: {klifs_export.shape}\n'
                             f'KLIFS merged table has shape: {klifs_metadata.shape}')

        if not klifs_overview.shape[0] == klifs_export.shape[0] == klifs_metadata.shape[0]:
            raise ValueError(f'Output table has incorrect number of rows:\n'
                             f'KLIFS overview table has shape: {klifs_overview.shape}\n'
                             f'KLIFS export table has shape: {klifs_export.shape}\n'
                             f'KLIFS merged table has shape: {klifs_metadata.shape}')

        return klifs_metadata

    @staticmethod
    def _add_filepaths(klifs_metadata):

        filepaths = []

        for index, row in klifs_metadata.iterrows():

            # Depending on whether alternate model and chain ID is given build file path:
            mol2_path = Path('.') / row.species.upper() / row.kinase

            if row.alternate_model != '-' and row.chain != '-':
                mol2_path = mol2_path / f'{row.pdb_id}_alt{row.alternate_model}_chain{row.chain}'
            elif row.alternate_model == '-' and row.chain != '-':
                mol2_path = mol2_path / f'{row.pdb_id}_chain{row.chain}'
            elif row.alternate_model == '-' and row.chain == '-':
                mol2_path = mol2_path / f'{row.pdb_id}'
            else:
                raise ValueError(f'Incorrect metadata entry {index}: {row.alternate_model}, {row.chain}')

            filepaths.append(mol2_path)

        klifs_metadata['filepath'] = filepaths

        return klifs_metadata

class KlifsMetadataFilter:

    def __init__(self):

        self.unfiltered = None
        self.filtered = None
        self.filtering_statistics = pd.DataFrame(
            [],
            columns=['filtering_step', 'n_filtered', 'n_remained']
        )

    def from_klifs_metadata(self, klifs_metadata, path_to_klifs_download):

        self.unfiltered = klifs_metadata
        self.filtered = klifs_metadata

        # Add filtering statistics
        filtering_step = 'Unfiltered'
        n_filtered = 0
        n_remained = len(self.unfiltered)
        self._add_filtering_statistics(filtering_step, n_filtered, n_remained)

        # Perform filtering steps
        self._get_species(species='Human')
        self._get_dfg(dfg='in')
        self._get_unique_kinase_pdbid_pair()
        self._get_existing_mol2s(path_to_klifs_download)

    def _add_filtering_statistics(self, filtering_step, n_filtered, n_remained):
        """
        Add filtering step data to filtering statistics (class attribute).

        Parameters
        ----------
        filtering_step : str
            Name of filtering step
        n_filtered : int
            Number of filtered rows (structures).
        n_remained : int
            Number of remaining rows (structures).
        """

        self.filtering_statistics = self.filtering_statistics.append(
            {
                'filtering_step': filtering_step,
                'n_filtered': n_filtered,
                'n_remained': n_remained
            },
            ignore_index=True
        )

    def _get_species(self, species='Human'):
        """
        Filter KLIFS dataset by species.

        Parameters
        ----------
        species : str
            String for species name.
        """

        klifs_metadata = self.filtered.copy()

        if species not in klifs_metadata.species.unique():
            raise ValueError(f'Species {species} not in species list: '
                             f'{", ".join(klifs_metadata.species.unique())}')

        indices_to_be_dropped = klifs_metadata[klifs_metadata.species != species].index

        klifs_metadata.drop(
            indices_to_be_dropped,
            inplace=True
        )

        # Add filtering statistics
        filtering_step = f'Only {species}'
        n_filtered = len(indices_to_be_dropped)
        n_remained = len(klifs_metadata)
        self._add_filtering_statistics(filtering_step, n_filtered, n_remained)

        self.filtered = klifs_metadata

    def _get_dfg(self, dfg='in'):
        """
        Filter KLIFS dataset by DFG region position.

        Parameters
        ----------
        dfg : str
            String for DFG region position.
        """

        klifs_metadata = self.filtered.copy()

        if dfg not in klifs_metadata.dfg.unique():
            raise ValueError(f'DFG {dfg} not in DFG list: '
                             f'{", ".join(klifs_metadata.dfg.unique())}')

        indices_to_be_dropped = klifs_metadata[klifs_metadata.dfg != dfg].index

        klifs_metadata.drop(
            indices_to_be_dropped,
            inplace=True
        )

        # Add filtering statistics
        filtering_step = f'Only DFG {dfg}'
        n_filtered = len(indices_to_be_dropped)
        n_remained = len(klifs_metadata)
        self._add_filtering_statistics(filtering_step, n_filtered, n_remained)

        self.filtered = klifs_metadata

    def _get_resolution(self, resolution=4):
        """
        Filter KLIFS dataset by structures with a resolution value lower or equal to given value.

        Parameters
        ----------
        resolution : int
            Maximum resolution value.
        """

        klifs_metadata = self.filtered.copy()

        indices_to_be_dropped = klifs_metadata[klifs_metadata.resolution > resolution].index

        klifs_metadata.drop(
            indices_to_be_dropped,
            inplace=True
        )

        # Add filtering statistics
        filtering_step = f'Only resolution <= {resolution}'
        n_filtered = len(indices_to_be_dropped)
        n_remained = len(klifs_metadata)
        self._add_filtering_statistics(filtering_step, n_filtered, n_remained)

        self.filtered = klifs_metadata

    def _get_qualityscore(self, qualityscore=4):
        """
        Filter KLIFS dataset by structures with a KLIFS quality score higher or equal to given value.

        Parameters
        ----------
        qualityscore : int
            Minimum KLIFS quality score value.
        """

        klifs_metadata = self.filtered.copy()

        indices_to_be_dropped = klifs_metadata[klifs_metadata.qualityscore < qualityscore].index

        klifs_metadata.drop(
            indices_to_be_dropped,
            inplace=True
        )

        # Add filtering statistics
        filtering_step = f'Only resolution >= {qualityscore}'
        n_filtered = len(indices_to_be_dropped)
        n_remained = len(klifs_metadata)
        self._add_filtering_statistics(filtering_step, n_filtered, n_remained)

        self.filtered = klifs_metadata

    def _get_existing_mol2s(self, path_to_klifs_download):
        """
        Drop entries in KLIFS metadata that have no corresponding pocket and protein mol2 file.

        Parameters
        ----------
        path_to_klifs_download : pathlib.Path or str
            Path to directory of KLIFS dataset files.
        """

        klifs_metadata = self.filtered.copy()

        indices_to_be_dropped = []

        for index, row in klifs_metadata.iterrows():

            # Depending on whether alternate model and chain ID is given build file path:
            pocket_mol2_path = Path(path_to_klifs_download) / 'KLIFS_download' / row.filepath / 'pocket.mol2'
            protein_mol2_path = Path(path_to_klifs_download) / 'KLIFS_download' / row.filepath / 'protein.mol2'

            # Not all paths exist - save list with missing paths
            if not pocket_mol2_path.exists():
                indices_to_be_dropped.append(index)
                logger.info(f'Missing pocket.mol2: {pocket_mol2_path}')
            elif not protein_mol2_path.exists():
                indices_to_be_dropped.append(index)
                logger.info(f'Missing protein.mol2: {protein_mol2_path}')
            else:
                pass

        klifs_metadata.drop(
            indices_to_be_dropped,
            inplace=True
        )

        # Add filtering statistics
        filtering_step = f'Only existing mol2 files'
        n_filtered = len(indices_to_be_dropped)
        n_remained = len(klifs_metadata)
        self._add_filtering_statistics(filtering_step, n_filtered, n_remained)

        self.filtered = klifs_metadata
    def _get_unique_kinase_pdbid_pair(self):
        """
        Filter KLIFS dataset by keeping only the KLIFS entry per kinase-PDB ID combination with the best quality score.
        """

        klifs_metadata = self.filtered.copy()

        # For each kinase and PDB IDs with multiple KLIFS entries (structures),
        # select entry with the best quality score

        # Sort by kinase, PDB ID and quality score
        # (so that for multiple equal kinase-pdb_id combos, highest quality score will come first)
        klifs_metadata.sort_values(
            by=['kinase', 'pdb_id', 'qualityscore'],
            ascending=[True, True, False],
            inplace=True
        )
        # Drop duplicate kinase-pdb_id combos and keep only first (with highest quality score)
        klifs_metadata.drop_duplicates(
            subset=['kinase', 'pdb_id'],
            keep='first',
            inplace=True
        )
        # Reset DataFrame indices
        klifs_metadata.reset_index(inplace=True)

        # Add filtering statistics
        filtering_step = f'Only unique kinase-PDB ID pairs'
        print(self.filtered.shape)
        n_filtered = len(self.filtered) - len(klifs_metadata)
        n_remained = len(klifs_metadata)
        self._add_filtering_statistics(filtering_step, n_filtered, n_remained)

        self.filtered = klifs_metadata


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


def download_from_pdb(klifs_metadata, path_to_data):
    """
    Download structure files from the PDB for KLIFS dataset.

    Parameters
    ----------
    klifs_metadata : pandas.DataFrame
        DataFrame containing merged metadate from both input KLIFS tables.
    path_to_data : str or pathlib.Path
        Path to directory of KLIFS dataset files.
    """

    path_to_data = Path(path_to_data) / 'raw' / 'PDB_download'

    pdbfile = PDBList()

    for index, row in klifs_metadata.iterrows():
        if not (Path(path_to_data) / f'{row.pdb_id}.cif').exists():
            print(Path(path_to_data) / f'{row.pdb_id}.cif')
            pdbfile.retrieve_pdb_file(row.pdb_id, pdir=path_to_data)
        else:
            continue


def drop_unparsable_pdbs(klifs_metadata, path_to_data):
    """
    Drop entries in KLIFS metadata where PDB parsing with biopython fails.

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

    path_to_data = Path(path_to_data)

    klifs_metadata_filtered = klifs_metadata.copy()

    pdb_ids_parsing_fails = []

    for pdb_id in klifs_metadata_filtered.pdb_id.unique():
        print(pdb_id)
        try:
            parser = MMCIFParser()
            structure = parser.get_structure(
                structure_id=pdb_id,
                filename=path_to_data / 'raw' / 'PDB_download' / f'{pdb_id}.cif'
            )
        except ValueError:
            pdb_ids_parsing_fails.append(pdb_id)

    print(f'Parsing failed for {len(pdb_ids_parsing_fails)} PDB IDs: {", ".join(pdb_ids_parsing_fails)}')

    with open(path_to_data / 'preprocessed' / 'pdb_ids_parsing_fails.txt', 'w') as f:
        for i in pdb_ids_parsing_fails:
            f.write(i)

    indices = klifs_metadata_filtered[klifs_metadata_filtered.pdb_id.isin(pdb_ids_parsing_fails)].index

    if list(indices):
        klifs_metadata_filtered.drop(indices, inplace=True)
    else:
        pass

    return klifs_metadata_filtered


def drop_underscored_residue_ids(klifs_metadata):
    """
    Drop entries in KLIFS metadata that are linked to mol2 files that contain underscores in their residue IDs.

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

    ids_with_underscored_residues = []

    for index, row in klifs_metadata_filtered.iterrows():

        print(f'{index + 1}/{len(klifs_metadata_filtered)}')

        ml = KlifsMoleculeLoader(klifs_metadata_entry=row)
        molecule = ml.molecule

        # Get first entry of each residue ID
        firsts = molecule.df.groupby(by='res_id', as_index=False).first()

        # Originally in mol2 file '_', but converted to '-' during mol2 file loading, see auxiliary.MoleculeLoader
        if any([i < 0 for i in firsts.res_id]):
            print(f'Contains underscored residue(s): metadata_index {index}')
            ids_with_underscored_residues.append([index, molecule])

    klifs_metadata_filtered.drop([i[0] for i in ids_with_underscored_residues], inplace=True)

    return klifs_metadata_filtered


def drop_residue_x(klifs_metadata):

    # Get important KLIFS

    klifs_regions = get_klifs_regions()

    important_klifs_regions = klifs_regions[
        klifs_regions.region_name.isin(
            ['x', 'DFG', 'GK', 'hinge', 'g.I']
        )
    ]

    klifs_metadata_filtered = klifs_metadata.copy()

    for index, row in klifs_metadata_filtered.iterrows():

        # If pocket contains residue X
        if 'X' in row.pocket:

            # Get pocket residue(s) with X
            pocket = pd.Series(list(row.pocket))
            pocket_x = pocket[pocket == 'X']

            # If this residues sits in an important KLIFS region, drop the PDB structure
            shared_residues = set(pocket_x.index) & set(important_klifs_regions.index)
            if shared_residues:
                print(index)
                klifs_metadata_filtered.drop(index, inplace=True)
                print(f'Drop PDB ID: {row.pdb_id}')

    return klifs_metadata_filtered
