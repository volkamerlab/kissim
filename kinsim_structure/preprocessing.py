"""
preprocessing.py

Subpocket-based structural fingerprint for kinase pocket comparison.

Handles the primary functions for the preprocessing of the KLIFS dataset.
"""

import logging
from pathlib import Path
import time
import sys

from biopandas.mol2 import PandasMol2
from biopandas.pdb import PandasPdb
from Bio.PDB import PDBList, PDBParser
import numpy as np
import pandas as pd
import pymol

from kinsim_structure.auxiliary import MoleculeLoader, KlifsMoleculeLoader, get_klifs_regions, AMINO_ACIDS

logger = logging.getLogger(__name__)


class KlifsMetadataLoader:
    """
    Load metadata of KLIFS download by merging details of two KLIFS metadata files,
    i.e. KLIFS_export.csv and overview.csv.

    Attributes
    ----------
    data : pandas.DataFrame
        Metadata of KLIFS download, merged from two metadata files.
    """

    def __init__(self):
        self.data = None

    def from_files(self, klifs_overview_file, klifs_export_file):
        """
        Get KLIFS metadata as DataFrame.

        1. Load KLIFS download metadata files.
        2. Unify column names and column cell formatting.
        3. Merge into one DataFrame.

        Parameters
        ----------
        klifs_overview_file : str or pathlib.Path
            Path to KLIFS download file `overview.csv` containing mainly KLIFS alignment-related metadata.
        klifs_export_file : str or pathlib.Path
            Path to KLIFS download file `KLIFS_download/KLIFS_export.csv` containing mainly structure-related metadata.

        Returns
        -------
        pandas.DataFrame
            Metadata of KLIFS download, merged from two KLIFS metadata files.
        """

        klifs_overview = self._from_klifs_overview_file(Path(klifs_overview_file))
        klifs_export = self._from_klifs_export_file(Path(klifs_export_file))

        klifs_metadata = self._merge_files(klifs_overview, klifs_export)
        klifs_metadata = self._add_filepaths(klifs_metadata)

        self.data = klifs_metadata

    @property
    def data_essential(self):
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
        Merge data contained in overview.csv and KLIFS_export.csv files from KLIFS database download.

        Parameters
        ----------
        klifs_export : pandas.DataFrame
            Metadata contained in KLIFS_export.csv file from KLIFS database download.
        klifs_overview : pandas.DataFrame
            Metadata contained in overview.csv file from KLIFS database download.

        Returns
        -------
        pandas.DataFrame
            Metadata for KLIFS download.
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
        """

        Parameters
        ----------
        klifs_metadata : pandas.DataFrame
            Metadata for KLIFS download.

        Returns
        -------
        pandas.DataFrame
            Metadata for KLIFS download plus column with file paths.
        """

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
    """
    Filter KLIFS metadata by different criteria such as species (HUMAN), DFG conformation (in), resolution (<=4),
    KLIFS quality score (>=4) and existence/usability of mol2 and pdb files.

    Attributes
    ----------
    unfiltered : pandas.DataFrame
        Unfiltered metadata for KLIFS download.
    filtered : pandas.DataFrame
        Filtered metadata for KLIFS download.
    filtering_statistics : pandas.DataFrame
        Filtering statistics for each filtering step.
    filtered_indices : dict of list of int
        List of filtered indices for selected filtering steps.
    """

    def __init__(self):

        self.unfiltered = None
        self.filtered = None
        self.filtering_statistics = pd.DataFrame(
            [],
            columns=['filtering_step', 'n_filtered', 'n_remained']
        )
        self.filtered_indices = {}

    def from_klifs_metadata(self, klifs_metadata, path_klifs_download):
        """
        Filter KLIFS metadata by different criteria such as species (HUMAN), DFG conformation (in), resolution (<=4),
        KLIFS quality score (>=4) and existence/usability of mol2 and pdb files.

        Parameters
        ----------
        klifs_metadata : pandas.DataFrame
            KLIFS metadata describing the KLIFS dataset.
        path_klifs_download : pathlib.Path or str
            Path to directory of KLIFS dataset files.

        Returns
        -------

        """

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
        self._get_resolution(resolution=4)
        self._get_qualityscore(qualityscore=4)
        self._get_existing_mol2s(path_klifs_download)
        self._get_existing_pdbs(path_klifs_download)
        self._get_parsable_pdbs(path_klifs_download)  # Takes time
        self._get_clean_residue_ids(path_klifs_download)  # Takes time
        self._get_existing_important_klifs_regions()  # Takes time
        self._get_unique_kinase_pdbid_pair()

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
        self._add_filtering_statistics(
            f'Only {species}',
            len(indices_to_be_dropped),
            len(klifs_metadata)
        )

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
        self._add_filtering_statistics(
            f'Only DFG {dfg}',
            len(indices_to_be_dropped),
            len(klifs_metadata)
        )

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
        self._add_filtering_statistics(
            f'Only resolution <= {resolution}',
            len(indices_to_be_dropped),
            len(klifs_metadata)
        )

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
        self._add_filtering_statistics(
            f'Only quality score >= {qualityscore}',
            len(indices_to_be_dropped),
            len(klifs_metadata)
        )

        self.filtered = klifs_metadata

    def _get_existing_mol2s(self, path_klifs_download):
        """
        Drop entries in KLIFS metadata that have no corresponding pocket and protein mol2 file.

        Parameters
        ----------
        path_klifs_download : pathlib.Path or str
            Path to directory of KLIFS dataset files.
        """

        klifs_metadata = self.filtered.copy()

        indices_to_be_dropped = []

        for index, row in klifs_metadata.iterrows():

            # Depending on whether alternate model and chain ID is given build file path:
            pocket_mol2_path = Path(path_klifs_download) / row.filepath / 'pocket.mol2'
            protein_mol2_path = Path(path_klifs_download) / row.filepath / 'protein.mol2'

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
        self._add_filtering_statistics(
            f'Only existing pocket/protein.mol2 files',
            len(indices_to_be_dropped),
            len(klifs_metadata)
        )

        # Save filtered indices
        self.filtered_indices['non_existing_mol2s'] = indices_to_be_dropped

        self.filtered = klifs_metadata

    def _get_existing_pdbs(self, path_klifs_download):
        """
        Drop entries in KLIFS metadata that have no corresponding pdb file.

        Parameters
        ----------
        path_klifs_download : str or pathlib.Path
            Path to directory of KLIFS dataset files.
        """

        klifs_metadata = self.filtered.copy()

        indices_to_be_dropped = []

        for index, row in klifs_metadata.iterrows():

            pdb_path = Path(path_klifs_download) / row.filepath / 'protein.pdb'

            # Not all paths exist - save list with missing paths
            if not pdb_path.exists():
                indices_to_be_dropped.append(index)
                logger.info(f'Missing protein.pdb: {pdb_path}')
            else:
                pass

        klifs_metadata.drop(
            indices_to_be_dropped,
            inplace=True
        )

        # Add filtering statistics
        self._add_filtering_statistics(
            f'Only existing protein.pdb files',
            len(indices_to_be_dropped),
            len(klifs_metadata)
        )

        # Save filtered indices
        self.filtered_indices['non_existing_pdbs'] = indices_to_be_dropped

        self.filtered = klifs_metadata

    def _get_parsable_pdbs(self, path_klifs_download):
        """
        Drop entries in KLIFS metadata where PDB parsing with biopython fails.

        Parameters
        ----------
        path_klifs_download : str or pathlib.Path
            Path to directory of KLIFS dataset files.
        """

        klifs_metadata = self.filtered.copy()

        indices_to_be_dropped = []

        for index, row in klifs_metadata.iterrows():

            pdb_path = Path(path_klifs_download) / row.filepath / 'protein.pdb'

            try:
                parser = PDBParser()
                parser.get_structure(
                    id=index,
                    file=pdb_path
                )
            except ValueError:
                indices_to_be_dropped.append(index)
                logger.info(f'Parsing failed for: {pdb_path}')

        klifs_metadata.drop(
            indices_to_be_dropped,
            inplace=True
        )

        # Add filtering statistics
        self._add_filtering_statistics(
            f'Only parsable protein.pdb files',
            len(indices_to_be_dropped),
            len(klifs_metadata)
        )

        # Save filtered indices
        self.filtered_indices['non_parsable_pdbs'] = indices_to_be_dropped

        self.filtered = klifs_metadata

    def _get_clean_residue_ids(self, path_klifs_download):
        """
        Drop entries in KLIFS metadata that are linked to mol2 files that contain underscores in their residue IDs.

        Parameters
        ----------
        path_klifs_download : str or pathlib.Path
            Path to directory of KLIFS dataset files.
        """

        klifs_metadata = self.filtered.copy()

        indices_to_be_dropped = []

        for index, row in klifs_metadata.iterrows():

            ml = KlifsMoleculeLoader(path_klifs_download)
            ml.from_metadata_entry(klifs_metadata_entry=row)
            molecule = ml.molecule

            # Get first entry of each residue ID
            firsts = molecule.df.groupby(by='res_id', as_index=False).first()

            # Originally in mol2 file '_', but converted to '-' during mol2 file loading, see auxiliary.MoleculeLoader
            if any([i < 0 for i in firsts.res_id]):
                indices_to_be_dropped.append(index)
                logger.info(f'Contains underscored residue(s): {row.filepath}')

        klifs_metadata.drop(
            indices_to_be_dropped,
            inplace=True
        )

        # Add filtering statistics
        self._add_filtering_statistics(
            f'Only clean residue IDs (no underscores in pocket!)',
            len(indices_to_be_dropped),
            len(klifs_metadata)
        )

        # Save filtered indices
        self.filtered_indices['with_underscored_residues'] = indices_to_be_dropped

        self.filtered = klifs_metadata

    def _get_existing_important_klifs_regions(self):
        """
        Get all KLIFS metadata that have no X residue (modified residue) at important KLIFS position.
        Important KLIFS positions are xDFG (DFG-motif plus one preceding amino acid residue), GK (gatekeeper), hinge
        (hinge region), and g.l (G-rich loop).
        """

        # Get important KLIFS
        klifs_regions = get_klifs_regions()
        important_klifs_regions = klifs_regions[
            klifs_regions.region_name.isin(
                ['x', 'DFG', 'GK', 'hinge', 'g.l']
            )
        ]

        klifs_metadata = self.filtered.copy()

        indices_to_be_dropped = []

        for index, row in klifs_metadata.iterrows():

            # If pocket contains residue X
            if 'X' in row.pocket:

                # Get pocket residue(s) with X
                pocket = pd.Series(list(row.pocket))
                pocket_x = pocket[pocket == 'X']

                # If this residues sits in an important KLIFS region, drop the PDB structure
                shared_residues = set(pocket_x.index) & set(important_klifs_regions.index)
                if shared_residues:
                    indices_to_be_dropped.append(index)
                    logger.info(f'Contains modified residue (X) at important KLIFS position: {row.filepath}')

        klifs_metadata.drop(
            indices_to_be_dropped,
            inplace=True
        )

        # Add filtering statistics
        self._add_filtering_statistics(
            f'Only without X residues at important KLIFS positions',
            len(indices_to_be_dropped),
            len(klifs_metadata)
        )

        # Save filtered indices
        self.filtered_indices['with_x_residue_at_important_klifs_position'] = indices_to_be_dropped

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
        self._add_filtering_statistics(
            f'Only unique kinase-PDB ID pairs',
            len(self.filtered) - len(klifs_metadata),
            len(klifs_metadata)
        )

        self.filtered = klifs_metadata


class Mol2FormatScreener:
    """
    Screen all structures for irregular residues, i.e. underscored residues, non-standard residues, and residues
    with duplicated atom names.

    Attributes
    ----------
    structures_irregular : dict of DataFrames
        Irregular residues in all structures.
    """

    def __init__(self):

        self.structures_irregular = {
            'residues_underscored': None,
            'residues_non_standard': None,
            'residues_duplicated_atom_names': None
        }

    def from_metadata(self, klifs_metadata, path_klifs_download):
        """
        Screen structures for irregular residues.

        Parameters
        ----------
        klifs_metadata : pandas.DataFrame
            KLIFS metadata describing the KLIFS dataset.
        path_klifs_download : pathlib.Path or str
            Path to directory of KLIFS dataset files.
        """

        klifs_metadata = klifs_metadata.copy()

        self.structures_irregular = {
            'residues_underscored': [],
            'residues_non_standard': [],
            'residues_duplicated_atom_names': []
        }

        for index, row in klifs_metadata.iterrows():

            if index % 100 == 0:
                print(f'Progress: {index}/{len(klifs_metadata)}')

            # Load molecule
            ml = MoleculeLoader(path_klifs_download / row.filepath / 'protein.mol2')
            molecule = ml.molecules[0]

            # Get underscored residues
            self._get_underscored_residues(molecule)

            # Get non-standard residues
            self._get_non_standard_residues(molecule)

            # Get duplicated atom names per residue
            self._get_structures_with_duplicated_residue_atom_names(molecule)

        # Cast lists to DataFrame and log results
        if len(self.structures_irregular['residues_underscored']) > 0:
            self.structures_irregular['residues_underscored'] = pd.concat(
                self.structures_irregular['residues_underscored']
            )
            logger.info(f'Number of structures with underscored residues: '
                        f'{len(self.structures_irregular["residues_underscored"].groupby("molecule_code"))}')
        else:
            self.structures_irregular['residues_underscored'] = None

        if len(self.structures_irregular['residues_non_standard']) > 0:
            self.structures_irregular['residues_non_standard'] = pd.concat(
                self.structures_irregular['residues_non_standard']
            )
            logger.info(f'Number of structures with non-standard residues: '
                        f'{len(self.structures_irregular["residues_non_standard"].groupby("molecule_code"))}')
        else:
            self.structures_irregular['residues_non_standard'] = None

        if len(self.structures_irregular['residues_duplicated_atom_names']) > 0:
            self.structures_irregular['residues_duplicated_atom_names'] = pd.concat(
                self.structures_irregular['residues_duplicated_atom_names']
            )
            logger.info(f'Number of structures with residues with duplicated atom names: '
                        f'{len(self.structures_irregular["residues_duplicated_atom_names"].groupby("molecule_code"))}')
        else:
            self.structures_irregular['residues_duplicated_atom_names'] = None

    def _get_underscored_residues(self, molecule):
        """
        Screen structure for underscored residues.

        Parameters
        ----------
        molecule : auxiliary.MoleculeLoader
            Mol2 file content for structure.
        """

        residues_irregular = molecule.df[
            molecule.df.res_id < 0
        ].groupby('res_id').first().copy()

        if len(residues_irregular) > 0:
            residues_irregular.reset_index(inplace=True)
            residues_irregular['molecule_code'] = molecule.code
            residues_irregular = residues_irregular[['molecule_code', 'res_id', 'res_name', 'subst_name']]

            self.structures_irregular['residues_underscored'].append(residues_irregular)

    def _get_non_standard_residues(self, molecule):
        """
        Screen structures for non-standard residues.

        Parameters
        ----------
        molecule : auxiliary.MoleculeLoader
            Mol2 file content for structure.
        """

        residues_irregular = molecule.df[
            ~molecule.df.res_name.isin(AMINO_ACIDS.aa_three)
        ].groupby('res_id').first().copy()

        if len(residues_irregular) > 0:
            residues_irregular.reset_index(inplace=True)
            residues_irregular['molecule_code'] = molecule.code
            residues_irregular = residues_irregular[['molecule_code', 'res_id', 'res_name', 'subst_name']]

            self.structures_irregular['residues_non_standard'].append(residues_irregular)

    def _get_structures_with_duplicated_residue_atom_names(self, molecule):
        """
        Screen structures for residues with duplicated atom names.

        Parameters
        ----------
        molecule : auxiliary.MoleculeLoader
            Mol2 file content for structure.
        """

        residues_irregular = molecule.df.groupby(
            ['res_id', 'atom_name']
        ).filter(
            lambda x: len(x) > 1
        ).copy()

        if len(residues_irregular) > 0:
            residues_irregular.reset_index(inplace=True)
            residues_irregular['molecule_code'] = molecule.code
            residues_irregular = residues_irregular[['molecule_code', 'res_id', 'res_name', 'subst_name', 'atom_name']]

            self.structures_irregular['residues_duplicated_atom_names'].append(residues_irregular)


class Mol2KlifsToPymolConverter:
    """
    Convert KLIFS mol2 files to PyMol readable mol2 files, i.e. replace underscored with negative residue IDs.

    Attributes
    ----------
    pymol_mol2_path : list of str
        List of PyMol readable mol2 files.
    """

    def __init__(self):

        self.pymol_mol2_path = []

    def from_metadata(self, klifs_metadata, path_klifs_download):
        """
        Convert KLIFS mol2 files to PyMol readable mol2 files, i.e. replace underscored with negative residue IDs.

        Parameters
        ----------
        klifs_metadata : pandas.DataFrame
            KLIFS metadata describing the KLIFS dataset.
        path_klifs_download : pathlib.Path or str
            Path to directory of KLIFS dataset files.
        """

        for index, row in klifs_metadata.iterrows():

            mol2_path = path_klifs_download / row.filepath / 'protein.mol2'
            self._rewrite_mol2_file(mol2_path)

    def _rewrite_mol2_file(self, mol2_path):
        """
        Convert KLIFS mol2 file to PyMol readable mol2 file, i.e. replace underscored with negative residue IDs.

        Parameters
        ----------
        mol2_path : pathlib.Path or str
            Path to KLIFS mol2 file
        """

        mol2_path = Path(mol2_path)

        # Load file
        with open(mol2_path, 'r') as f:
            lines = f.readlines()

        headers = {
            '@<TRIPOS>MOLECULE': False,
            '@<TRIPOS>ATOM': False,
            '@<TRIPOS>BOND': False,
            '@<TRIPOS>SUBSTRUCTURE': False
        }

        lines_new = []
        unexpected_targets = []

        for line in lines:

            if line.startswith('@<TRIPOS>MOLECULE'):
                headers['@<TRIPOS>MOLECULE'] = True

            if line.startswith('@<TRIPOS>ATOM'):
                headers['@<TRIPOS>ATOM'] = True

            if line.startswith('@<TRIPOS>BOND'):
                headers['@<TRIPOS>BOND'] = True

            if line.startswith('@<TRIPOS>SUBSTRUCTURE'):
                headers['@<TRIPOS>SUBSTRUCTURE'] = True

            if '_' in line:

                # In what section are we?

                if headers['@<TRIPOS>MOLECULE'] and not headers['@<TRIPOS>ATOM'] and not headers['@<TRIPOS>BOND'] and not headers['@<TRIPOS>SUBSTRUCTURE']:

                    if line.startswith('USER_CHARGES'):
                        lines_new.append(line)
                    else:
                        unexpected_targets.append(line)

                elif headers['@<TRIPOS>MOLECULE'] and headers['@<TRIPOS>ATOM'] and not headers['@<TRIPOS>BOND'] and not headers['@<TRIPOS>SUBSTRUCTURE']:

                    if '_' in line[56:59]:
                        lines_new.append(line.replace('_', '-'))
                    else:
                        unexpected_targets.append(line)

                elif headers['@<TRIPOS>MOLECULE'] and headers['@<TRIPOS>ATOM'] and headers['@<TRIPOS>BOND'] and not headers['@<TRIPOS>SUBSTRUCTURE']:

                    unexpected_targets.append(line)

                else:

                    if '_' in line[7:10]:
                        lines_new.append(line.replace('_', '-'))
                    elif line.startswith('# MOE 2012.10 (io_trps.svl 2012.10)'):
                        lines_new.append(line)
                    else:
                        unexpected_targets.append(line)
            else:

                lines_new.append(line)

            if len(unexpected_targets) > 0:

                raise ValueError(f'Unknown underscores were transformed, please check: {unexpected_targets}')

        # Write new mol2 file
        pymol_mol2_path = Path(mol2_path).parent / 'protein_pymol.mol2'
        self.pymol_mol2_path.append(pymol_mol2_path)

        with open(pymol_mol2_path, 'w') as f:
            f.writelines(lines_new)


class Mol2ToPdbConverter:
    """
    Convert mol2 file to pdb file and save pdb file locally.
    """

    def from_klifs_metadata(self, klifs_metadata, path_klifs_download):
        """
        Convert mol2 file to pdb file and save pdb file locally.

        Parameters
        ----------
        klifs_metadata : pandas.DataFrame
            KLIFS metadata describing the KLIFS dataset.
        path_klifs_download : pathlib.Path or str
            Path to directory of KLIFS dataset files.
        """

        # Launch PyMol once
        self._pymol_launch()

        for index, row in klifs_metadata.iterrows():
            print(index, row.filepath)

            mol2_path = Path(path_klifs_download) / row.filepath / 'protein.mol2'

            # Set mol2 path
            mol2_path = self._set_mol2_path(mol2_path)

            # Set pdb path
            pdb_path = self._set_pdb_path(mol2_path, pdb_path=None)

            # PyMol mol2 to pdb conversion
            self._pymol_mol2_to_pdb_conversion(mol2_path, pdb_path)
            self._raise_conversion_error(mol2_path, pdb_path)  # Check if files are equivalent

        self._pymol_quit()

    def from_file(self, mol2_path, pdb_path=None):
        """
        Convert mol2 file to pdb file and save pdb file locally.

        Parameters
        ----------
        mol2_path : pathlib.Path or str
            Path to mol2 file.
        pdb_path : None or pathlib.Path or str
            Path to pdb file (= converted mol2 file). Directory must exist.
            Default is None - saves pdb file next to the mol2 file in same directory with the same filename.
        """

        # Set mol2 path
        mol2_path = self._set_mol2_path(mol2_path)

        # Set pdb path
        pdb_path = self._set_pdb_path(mol2_path, pdb_path)

        # PyMol mol2 to pdb conversion
        self._pymol_launch()
        self._pymol_mol2_to_pdb_conversion(mol2_path, pdb_path)
        self._raise_conversion_error(mol2_path, pdb_path)  # Check if files are equivalent
        self._pymol_quit()

    @staticmethod
    def _pymol_launch():

        stdin = sys.stdin
        stdout = sys.stdout
        stderr = sys.stderr

        pymol.finish_launching(['pymol', '-qc'])

        sys.stdin = stdin
        sys.stdout = stdout
        sys.stderr = stderr

    @staticmethod
    def _pymol_mol2_to_pdb_conversion(mol2_path, pdb_path):
        """
        Convert a mol2 file to a pdb file and save locally (using pymol).
        """

        pymol.cmd.reinitialize()
        pymol.cmd.load(mol2_path)
        pymol.cmd.save(pdb_path)

    @staticmethod
    def _pymol_quit():

        pymol.cmd.quit()

    @staticmethod
    def _raise_conversion_error(mol2_path, pdb_path):
        """
        Raise ValueError if mol2 and pdb file are not equivalent.
        """

        klifs_metadata_filepath = '/'.join(mol2_path.parts[-4:-1])  # Mimic filepath column in metadata

        pmol2 = PandasMol2().read_mol2(
            str(mol2_path),
            columns={
                0: ('atom_id', int),
                1: ('atom_name', str),
                2: ('x', float),
                3: ('y', float),
                4: ('z', float),
                5: ('atom_type', str),
                6: ('subst_id', str),
                7: ('subst_name', str),
                8: ('charge', float),
                9: ('status_bit', str)
            }

        )

        # Wait until pdb file is written to disc
        attempt = 0
        while attempt <= 10:
            if pdb_path.exists():
                break
            else:
                if attempt == 10:
                    logger.info(f'{klifs_metadata_filepath}: PDB file does not exist: {pdb_path}')
                else:
                    time.sleep(1)
                    attempt += 1

        ppdb = PandasPdb().read_pdb(str(pdb_path))

        mol2_df = pmol2.df
        pdb_df = pd.concat([ppdb.df['ATOM'], ppdb.df['HETATM']])

        # Number of atoms?
        n_atoms_mol2 = len(mol2_df)
        n_atoms_pdb = len(pdb_df)
        if not n_atoms_mol2 == n_atoms_pdb:
            logger.info(f'{klifs_metadata_filepath}: Unequal number of atoms in mol2 ({n_atoms_mol2}) '
                        f'and pdb ({n_atoms_pdb}) file.')

        # Record name of PDB file is always ATOM (although containing non-standard residues)
        if not set(pdb_df.record_name) == {'ATOM'}:
            logger.info(f'{klifs_metadata_filepath}: PDB file contains non-ATOM entries.')

        # x, y, and z mean values are the same
        if not np.isclose(mol2_df.x.mean(), pdb_df.x_coord.mean(), rtol=1e-05):
            logger.info(f'{klifs_metadata_filepath}: Coordinates are not the same (x means differ).')

        if not np.isclose(mol2_df.y.mean(), pdb_df.y_coord.mean(), rtol=1e-05):
            logger.info(f'{klifs_metadata_filepath}: Coordinates are not the same (y means differ).')

        if not np.isclose(mol2_df.z.mean(), pdb_df.z_coord.mean(), rtol=1e-05):
            logger.info(f'{klifs_metadata_filepath}: Coordinates are not the same (z means differ).')

        # Residue ID and name are the same
        residue_set_mol2 = set(mol2_df.subst_name)
        pdb_df.residue_number = pdb_df.residue_number.astype(str)
        residue_set_pdb = set(pdb_df.residue_name + pdb_df.residue_number)
        residue_set_diff = (residue_set_mol2 - residue_set_pdb)|(residue_set_pdb - residue_set_mol2)
        if not len(residue_set_diff) == 0:
            logger.info(f'{klifs_metadata_filepath}: Residue ID/name differs. '
                        f'In mol2: {(residue_set_mol2 - residue_set_pdb)}. '
                        f'In pdb: {(residue_set_pdb - residue_set_mol2)}.')

    @staticmethod
    def _set_mol2_path(mol2_path):
        """
        Test if input mol2 file exists - and if so, set as class attribute.

        Parameters
        ----------
        mol2_path : pathlib.Path or str
            Path to mol2 file.

        Returns
        -------
        pathlib.Path
            Path to mol2 file.
        """

        mol2_path = Path(mol2_path)

        if not mol2_path.exists():
            raise FileNotFoundError(f'File path does not exist: {mol2_path}')
        if not mol2_path.suffix == '.mol2':
            raise ValueError(f'Incorrect file suffix: {mol2_path.suffix}, must be mol2.')

        return mol2_path

    @staticmethod
    def _set_pdb_path(mol2_path, pdb_path):
        """
        Test if input pdb file path exists and has a pdb suffix - and if so, set as class attribute.

        Parameters
        ----------
        pdb_path : None or pathlib.Path or str
            Path to pdb file (= converted mol2 file). Directory must exist.
            Default is None - saves pdb file next to the mol2 file in same directory with the same filename.

        Returns
        -------
        pathlib.Path
            Path to pdb file (= converted mol2 file).
        """

        # If mol2 path is not set, do not continue to set pdb path (otherwise we cannot convert mol2 to pdb)
        if mol2_path is None:
            raise ValueError(f'Set mol2 path (class attribute) first.')

        else:
            pass

        if pdb_path is None:
            pdb_path = mol2_path.parent / f'{mol2_path.stem}.pdb'

        else:
            pdb_path = Path(pdb_path)

            if not pdb_path.parent.exists():
                raise FileNotFoundError(f'Directory does not exist: {pdb_path}')
            if not pdb_path.suffix == '.pdb':
                raise ValueError(f'Missing file name or incorrect file type: {pdb_path}. '
                                 f'Must have the form: /path/to/existing/directory/filename.pdb')

        return pdb_path


class PdbDownloader:

    def from_klifs_metadata(self, klifs_metadata, path_pdb_download):
        """
        Download structure files from the PDB for KLIFS dataset.

        Parameters
        ----------
        klifs_metadata : pandas.DataFrame
            KLIFS metadata describing the KLIFS dataset.
        path_pdb_download : pathlib.Path or str
            Path to directory of files of PDB download.
        """

        path_to_pdb_download = Path(path_pdb_download)
        path_to_pdb_download.mkdir(parents=True, exist_ok=True)

        pdbfile = PDBList()

        for index, row in klifs_metadata.iterrows():
            if not (Path(path_to_pdb_download) / f'{row.pdb_id}.cif').exists():
                pdbfile.retrieve_pdb_file(row.pdb_id, pdir=path_to_pdb_download)
            else:
                logger.info(f'Pdb file could not be downloaded: {row.pdb_id}.')
