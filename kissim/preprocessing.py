"""
preprocessing.py

Subpocket-based structural fingerprint for kinase pocket comparison.

Handles the primary functions for the preprocessing of the KLIFS dataset.
"""

import datetime
from multiprocessing import cpu_count, Pool
import logging
from pathlib import Path
import time
import sys

from biopandas.mol2 import PandasMol2
from biopandas.pdb import PandasPdb
from Bio.PDB import PDBParser
import numpy as np
import pandas as pd
import pymol

from kissim.auxiliary import MoleculeLoader, KlifsMoleculeLoader, get_klifs_regions, AMINO_ACIDS

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

    def from_files(self, path_klifs_overview, path_klifs_export):
        """
        Get KLIFS metadata as DataFrame.

        1. Load KLIFS download metadata files.
        2. Unify column names and column cell formatting.
        3. Merge into one DataFrame.

        Parameters
        ----------
        path_klifs_overview : pathlib.Path or str
            Path to KLIFS download file `overview.csv` containing mainly KLIFS alignment-related metadata.
        path_klifs_export : pathlib.Path or str
            Path to KLIFS download file `KLIFS_download/KLIFS_export.csv` containing mainly structure-related metadata.

        Returns
        -------
        pandas.DataFrame
            Metadata of KLIFS download, merged from two KLIFS metadata files.
        """

        logger.info(f"PREPROCESSING: KlifsMetadataLoader")

        klifs_overview = self._from_klifs_overview_file(Path(path_klifs_overview))
        klifs_export = self._from_klifs_export_file(Path(path_klifs_export))

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
            "pdb_id",
            "alternate_model",
            "chain",
            "kinase",
            "kinase_all",
            "family",
            "groups",
            "species",
            "dfg",
            "ac_helix",
            "pocket",
            "rmsd1",
            "rmsd2",
            "qualityscore",
            "resolution",
            "missing_residues",
            "missing_atoms",
            "filepath",
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
                "NAME": "kinase",
                "FAMILY": "family",
                "GROUPS": "groups",
                "PDB": "pdb_id",
                "CHAIN": "chain",
                "ALTERNATE_MODEL": "alternate_model",
                "SPECIES": "species",
                "LIGAND": "ligand_orthosteric_name",
                "PDB_IDENTIFIER": "ligand_orthosteric_pdb_id",
                "ALLOSTERIC_NAME": "ligand_allosteric_name",
                "ALLOSTERIC_PDB": "ligand_allosteric_pdb_id",
                "DFG": "dfg",
                "AC_HELIX": "ac_helix",
            },
            inplace=True,
        )

        # Unify column 'kinase': Sometimes several kinase names are available, e.g. "EPHA7 (EphA7)"
        # Column "kinase": Retain only first kinase name, e.g. EPHA7
        # Column "kinase_all": Save all kinase names as list, e.g. [EPHA7, EphA7]
        kinase_names = [self._format_kinase_name(i) for i in klifs_export.kinase]
        klifs_export.kinase = [i[0] for i in kinase_names]
        klifs_export.insert(loc=1, column="kinase_all", value=kinase_names)

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
                "pdb": "pdb_id",
                "alt": "alternate_model",
                "orthosteric_PDB": "ligand_orthosteric_pdb_id",
                "allosteric_PDB": "ligand_allosteric_pdb_id",
            },
            inplace=True,
        )

        # Unify column 'alternate model' with corresponding column in KLIFS_export.csv
        klifs_overview.alternate_model.replace(" ", "-", inplace=True)

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

        kinase_name = kinase_name.replace("(", "")
        kinase_name = kinase_name.replace(")", "")
        kinase_name = kinase_name.replace(",", "")
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
            raise ValueError(
                f"Number of PDBs in overview but not in export table: {not_in_export.size}.\n"
            )
        if not_in_overview.size > 0:
            raise (
                f"Number of PDBs in export but not in overview table: {not_in_overview.size}."
                f"PDB codes are probably updated because structures are deprecated."
            )

        # Merge on mutual columns:
        # Species, kinase, PDB ID, chain, alternate model, orthosteric and allosteric ligand PDB ID

        mutual_columns = ["species", "pdb_id", "chain", "alternate_model"]

        klifs_metadata = klifs_export.merge(right=klifs_overview, how="inner", on=mutual_columns)

        klifs_metadata.drop(
            columns=["ligand_orthosteric_pdb_id_y", "ligand_allosteric_pdb_id_y", "kinase_y"],
            inplace=True,
        )

        klifs_metadata.rename(
            columns={
                "ligand_orthosteric_pdb_id_x": "ligand_orthosteric_pdb_id",
                "ligand_allosteric_pdb_id_x": "ligand_allosteric_pdb_id",
                "kinase_x": "kinase",
            },
            inplace=True,
        )

        if not (klifs_overview.shape[1] + klifs_export.shape[1] - 7) == klifs_metadata.shape[1]:
            raise ValueError(
                f"Output table has incorrect number of columns\n"
                f"KLIFS overview table has shape: {klifs_overview.shape}\n"
                f"KLIFS export table has shape: {klifs_export.shape}\n"
                f"KLIFS merged table has shape: {klifs_metadata.shape}"
            )

        if not klifs_overview.shape[0] == klifs_export.shape[0] == klifs_metadata.shape[0]:
            raise ValueError(
                f"Output table has incorrect number of rows:\n"
                f"KLIFS overview table has shape: {klifs_overview.shape}\n"
                f"KLIFS export table has shape: {klifs_export.shape}\n"
                f"KLIFS merged table has shape: {klifs_metadata.shape}"
            )

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
            path_mol2 = Path(".") / row.species.upper() / row.kinase

            if row.alternate_model != "-" and row.chain != "-":
                path_mol2 = path_mol2 / f"{row.pdb_id}_alt{row.alternate_model}_chain{row.chain}"
            elif row.alternate_model == "-" and row.chain != "-":
                path_mol2 = path_mol2 / f"{row.pdb_id}_chain{row.chain}"
            elif row.alternate_model == "-" and row.chain == "-":
                path_mol2 = path_mol2 / f"{row.pdb_id}"
            else:
                raise ValueError(
                    f"Incorrect metadata entry {index}: {row.alternate_model}, {row.chain}"
                )

            filepaths.append(path_mol2)

        klifs_metadata["filepath"] = filepaths

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
            [], columns=["filtering_step", "n_removed", "n_retained"]
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
        """

        logger.info(f"PREPROCESSING: KlifsMetadataFilter")
        logger.info(f"Number of metadata entries: {len(klifs_metadata)}")

        self.unfiltered = klifs_metadata
        self.filtered = klifs_metadata

        # Add filtering statistics
        filtering_step = "Unfiltered"
        n_removed = 0
        n_retained = len(self.unfiltered)
        self._add_filtering_statistics(filtering_step, n_removed, n_retained)

        # Perform filtering steps
        self._get_species(species="Human")
        self._get_dfg(dfg="in")
        self._get_resolution(resolution=4)
        self._get_qualityscore(qualityscore=4)
        self._get_existing_mol2s(path_klifs_download)
        self._get_existing_pdbs(path_klifs_download)
        self._get_parsable_pdbs(path_klifs_download)  # Takes time
        self._get_clean_residue_ids(path_klifs_download)  # Takes time
        self._get_existing_important_klifs_regions()  # Takes time
        self._get_unique_kinase_pdbid_pair()

        logger.info(
            f"Number of unfiltered metadata entries: {len(self.unfiltered)}, "
            f"representing {len(self.unfiltered.kinase.unique())} kinases."
        )
        logger.info(
            f"Number of filtered metadata entries: {len(self.filtered)} "
            f"representing {len(self.filtered.kinase.unique())} kinases."
        )

    def _add_filtering_statistics(self, filtering_step, n_removed, n_retained):
        """
        Add filtering step data to filtering statistics (class attribute).

        Parameters
        ----------
        filtering_step : str
            Name of filtering step
        n_removed : int
            Number of removed rows (structures).
        n_retained : int
            Number of retained rows (structures).
        """

        logger.info(
            f"Filtering step: {filtering_step}: {n_removed} removed, {n_retained} retained."
        )

        self.filtering_statistics = self.filtering_statistics.append(
            {"filtering_step": filtering_step, "n_removed": n_removed, "n_retained": n_retained},
            ignore_index=True,
        )

    def _get_species(self, species="Human"):
        """
        Filter KLIFS dataset by species.

        Parameters
        ----------
        species : str
            String for species name.
        """

        klifs_metadata = self.filtered.copy()

        if species not in klifs_metadata.species.unique():
            raise ValueError(
                f"Species {species} not in species list: "
                f'{", ".join(klifs_metadata.species.unique())}'
            )

        indices_to_be_dropped = klifs_metadata[klifs_metadata.species != species].index

        klifs_metadata.drop(indices_to_be_dropped, inplace=True)

        # Add filtering statistics
        self._add_filtering_statistics(
            f"Only {species}", len(indices_to_be_dropped), len(klifs_metadata)
        )

        self.filtered = klifs_metadata

    def _get_dfg(self, dfg="in"):
        """
        Filter KLIFS dataset by DFG region position.

        Parameters
        ----------
        dfg : str
            String for DFG region position.
        """

        klifs_metadata = self.filtered.copy()

        if dfg not in klifs_metadata.dfg.unique():
            raise ValueError(
                f"DFG {dfg} not in DFG list: " f'{", ".join(klifs_metadata.dfg.unique())}'
            )

        indices_to_be_dropped = klifs_metadata[klifs_metadata.dfg != dfg].index

        klifs_metadata.drop(indices_to_be_dropped, inplace=True)

        # Add filtering statistics
        self._add_filtering_statistics(
            f"Only DFG {dfg}", len(indices_to_be_dropped), len(klifs_metadata)
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

        klifs_metadata.drop(indices_to_be_dropped, inplace=True)

        # Add filtering statistics
        self._add_filtering_statistics(
            f"Only resolution <= {resolution}", len(indices_to_be_dropped), len(klifs_metadata)
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

        klifs_metadata.drop(indices_to_be_dropped, inplace=True)

        # Add filtering statistics
        self._add_filtering_statistics(
            f"Only quality score >= {qualityscore}",
            len(indices_to_be_dropped),
            len(klifs_metadata),
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
            pocket_path_mol2 = Path(path_klifs_download) / row.filepath / "pocket.mol2"
            protein_path_mol2 = Path(path_klifs_download) / row.filepath / "protein_pymol.mol2"

            # Not all paths exist - save list with missing paths
            if not pocket_path_mol2.exists():
                indices_to_be_dropped.append(index)
                logger.info(f"Missing pocket mol2 files: {pocket_path_mol2}")
            elif not protein_path_mol2.exists():
                indices_to_be_dropped.append(index)
                logger.info(f"Missing protein mol2 files: {protein_path_mol2}")
            else:
                pass

        klifs_metadata.drop(indices_to_be_dropped, inplace=True)

        # Add filtering statistics
        self._add_filtering_statistics(
            f"Only existing pocket/protein mol2 files",
            len(indices_to_be_dropped),
            len(klifs_metadata),
        )

        # Save filtered indices
        self.filtered_indices["non_existing_mol2s"] = indices_to_be_dropped

        self.filtered = klifs_metadata

    def _get_existing_pdbs(self, path_klifs_download):
        """
        Drop entries in KLIFS metadata that have no corresponding pdb file.

        Parameters
        ----------
        path_klifs_download : pathlib.Path or str
            Path to directory of KLIFS dataset files.
        """

        klifs_metadata = self.filtered.copy()

        indices_to_be_dropped = []

        for index, row in klifs_metadata.iterrows():

            path_pdb = Path(path_klifs_download) / row.filepath / "protein_pymol.pdb"

            # Not all paths exist - save list with missing paths
            if not path_pdb.exists():
                indices_to_be_dropped.append(index)
                logger.info(f"Missing protein pdb files: {path_pdb}")
            else:
                pass

        klifs_metadata.drop(indices_to_be_dropped, inplace=True)

        # Add filtering statistics
        self._add_filtering_statistics(
            f"Only existing protein pdb files", len(indices_to_be_dropped), len(klifs_metadata)
        )

        # Save filtered indices
        self.filtered_indices["non_existing_pdbs"] = indices_to_be_dropped

        self.filtered = klifs_metadata

    def _get_parsable_pdbs(self, path_klifs_download):
        """
        Drop entries in KLIFS metadata where PDB parsing with biopython fails.

        Parameters
        ----------
        path_klifs_download : pathlib.Path or str
            Path to directory of KLIFS dataset files.
        """

        klifs_metadata = self.filtered.copy()

        indices_to_be_dropped = []

        for index, row in klifs_metadata.iterrows():

            path_pdb = Path(path_klifs_download) / row.filepath / "protein_pymol.pdb"

            try:
                parser = PDBParser()
                parser.get_structure(id=index, file=path_pdb)
            except ValueError:
                indices_to_be_dropped.append(index)
                logger.info(f"Parsing failed for: {path_pdb}")

        klifs_metadata.drop(indices_to_be_dropped, inplace=True)

        # Add filtering statistics
        self._add_filtering_statistics(
            f"Only parsable protein pdb files", len(indices_to_be_dropped), len(klifs_metadata)
        )

        # Save filtered indices
        self.filtered_indices["non_parsable_pdbs"] = indices_to_be_dropped

        self.filtered = klifs_metadata

    def _get_clean_residue_ids(self, path_klifs_download):
        """
        Drop entries in KLIFS metadata that are linked to mol2 files that contain underscores in their residue IDs.

        Parameters
        ----------
        path_klifs_download : pathlib.Path or str
            Path to directory of KLIFS dataset files.
        """

        klifs_metadata = self.filtered.copy()

        indices_to_be_dropped = []

        for index, row in klifs_metadata.iterrows():

            ml = KlifsMoleculeLoader()
            ml.from_metadata_entry(
                klifs_metadata_entry=row, path_klifs_download=path_klifs_download
            )
            molecule = ml.molecule

            # Get first entry of each residue ID
            firsts = molecule.df.groupby(by="res_id", as_index=False).first()

            # Originally in mol2 file '_', but converted to '-' during mol2 file loading, see auxiliary.MoleculeLoader
            if any([i < 0 for i in firsts.res_id]):
                indices_to_be_dropped.append(index)
                logger.info(f"Contains underscored residue(s): {row.filepath}")

        klifs_metadata.drop(indices_to_be_dropped, inplace=True)

        # Add filtering statistics
        self._add_filtering_statistics(
            f"Only clean residue IDs (no underscores in pocket!)",
            len(indices_to_be_dropped),
            len(klifs_metadata),
        )

        # Save filtered indices
        self.filtered_indices["with_underscored_residues"] = indices_to_be_dropped

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
            klifs_regions.region_name.isin(["x", "DFG", "GK", "hinge", "g.l"])
        ]

        klifs_metadata = self.filtered.copy()

        indices_to_be_dropped = []

        for index, row in klifs_metadata.iterrows():

            # If pocket contains residue X
            if "X" in row.pocket:

                # Get pocket residue(s) with X
                pocket = pd.Series(list(row.pocket))
                pocket_x = pocket[pocket == "X"]

                # If this residues sits in an important KLIFS region, drop the PDB structure
                shared_residues = set(pocket_x.index) & set(important_klifs_regions.index)
                if shared_residues:
                    indices_to_be_dropped.append(index)
                    logger.info(
                        f"Contains modified residue (X) at important KLIFS position: {row.filepath}"
                    )

        klifs_metadata.drop(indices_to_be_dropped, inplace=True)

        # Add filtering statistics
        self._add_filtering_statistics(
            f"Only without X residues at important KLIFS positions",
            len(indices_to_be_dropped),
            len(klifs_metadata),
        )

        # Save filtered indices
        self.filtered_indices["with_x_residue_at_important_klifs_position"] = indices_to_be_dropped

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
            by=["kinase", "pdb_id", "qualityscore"], ascending=[True, True, False], inplace=True
        )
        # Drop duplicate kinase-pdb_id combos and keep only first (with highest quality score)
        klifs_metadata.drop_duplicates(subset=["kinase", "pdb_id"], keep="first", inplace=True)
        # Reset DataFrame indices
        klifs_metadata.reset_index(inplace=True)

        # Add filtering statistics
        self._add_filtering_statistics(
            f"Only unique kinase-PDB ID pairs",
            len(self.filtered) - len(klifs_metadata),
            len(klifs_metadata),
        )

        self.filtered = klifs_metadata


class Mol2FormatScreener:
    """
    Screen all structures for irregular residues, i.e. underscored residues, non-standard residues, and residues
    with duplicated atom names.

    Attributes
    ----------
    structures_irregular : pandas.DataFrame
        Irregular residues in all structures.
    path_klifs_download : pathlib.Path or str
        Path to directory of KLIFS dataset files.
    """

    def __init__(self):

        self.path_klifs_download = None
        self.structures_irregular = None

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

        start = datetime.datetime.now()

        logger.info(f"PREPROCESSING: Mol2FormatScreener")
        logger.info(f"Number of metadata entries: {len(klifs_metadata)}")

        path_klifs_download = Path(path_klifs_download)
        if path_klifs_download.exists():
            self.path_klifs_download = path_klifs_download

        # Number of CPUs on machine
        num_cores = cpu_count() - 1
        logger.info(f"Number of cores used: {num_cores}")

        # Create pool with `num_processes` processes
        pool = Pool(processes=num_cores)

        # Get KLIFS entries as list
        entry_list = [j for i, j in klifs_metadata.iterrows()]

        # Apply function to each chunk in list
        structures_irregular = pool.map(self._screen_mol2_format, entry_list)
        structures_irregular = pd.concat(
            structures_irregular, axis=0, sort=False, ignore_index=True
        )
        self.structures_irregular = structures_irregular

        # Close and join pool
        pool.close()
        pool.join()

        end = datetime.datetime.now()

        logger.info(f"Start of mol2 format screening: {start}")
        logger.info(f"End of mol2 format screening: {end}")

    def _screen_mol2_format(self, klifs_metadata_entry):
        """
        Screen mol file for irregular residues.

        Parameters
        ----------
        klifs_metadata_entry : pandas.Series
            KLIFS metadata describing one entry in the KLIFS dataset.

        Returns
        -------
        pandas.DataFrame
            Irregular residues.
        """

        # Load molecule
        ml = MoleculeLoader(
            self.path_klifs_download / klifs_metadata_entry.filepath / "protein.mol2"
        )
        molecule = ml.molecules[0]

        # Get underscored residues
        residues_underscored = self._get_underscored_residues(molecule)

        # Get non-standard residues
        residues_non_standard = self._get_non_standard_residues(molecule)

        # Get duplicated atom names per residue
        residues_duplicated_atom_names = self._get_structures_with_duplicated_residue_atom_names(
            molecule
        )

        # Concat results
        return pd.concat(
            [residues_underscored, residues_non_standard, residues_duplicated_atom_names],
            axis=0,
            sort=False,
            ignore_index=True,
        )

    @staticmethod
    def _get_underscored_residues(molecule):
        """
        Screen structure for underscored residues.

        Parameters
        ----------
        molecule : auxiliary.MoleculeLoader
            Mol2 file content for structure.

        Returns
        -------
        pandas.DataFrame
            Irregular residues.
        """

        residues_irregular = molecule.df[molecule.df.res_id < 0].groupby("res_id").first().copy()

        residues_irregular.reset_index(inplace=True)
        residues_irregular["molecule_code"] = molecule.code
        residues_irregular = residues_irregular[
            ["molecule_code", "res_id", "res_name", "subst_name"]
        ]
        residues_irregular.insert(loc=0, column="irregularity", value="residue_underscored")

        return residues_irregular

    @staticmethod
    def _get_non_standard_residues(molecule):
        """
        Screen structures for non-standard residues.

        Parameters
        ----------
        molecule : auxiliary.MoleculeLoader
            Mol2 file content for structure.

        Returns
        -------
        pandas.DataFrame
            Irregular residues.
        """

        residues_irregular = (
            molecule.df[~molecule.df.res_name.isin(AMINO_ACIDS.aa_three)]
            .groupby("res_id")
            .first()
            .copy()
        )

        residues_irregular.reset_index(inplace=True)
        residues_irregular["molecule_code"] = molecule.code
        residues_irregular = residues_irregular[
            ["molecule_code", "res_id", "res_name", "subst_name"]
        ]
        residues_irregular.insert(loc=0, column="irregularity", value="residue_non_standard")

        return residues_irregular

    @staticmethod
    def _get_structures_with_duplicated_residue_atom_names(molecule):
        """
        Screen structures for residues with duplicated atom names.

        Parameters
        ----------
        molecule : auxiliary.MoleculeLoader
            Mol2 file content for structure.

        Returns
        -------
        pandas.DataFrame
            Irregular residues.
        """

        residues_irregular = (
            molecule.df.groupby(["res_id", "atom_name"]).filter(lambda x: len(x) > 1).copy()
        )

        residues_irregular.reset_index(inplace=True)
        residues_irregular["molecule_code"] = molecule.code
        residues_irregular = residues_irregular[
            ["molecule_code", "res_id", "res_name", "subst_name", "atom_name"]
        ]
        residues_irregular.insert(
            loc=0, column="irregularity", value="residues_duplicated_atom_names"
        )

        return residues_irregular


class Mol2KlifsToPymolConverter:
    """
    Convert KLIFS mol2 files to PyMol readable mol2 files, i.e. replace underscored with negative residue IDs.

    Attributes
    ----------
    path_klifs_download : pathlib.Path or str
        Path to directory of KLIFS dataset files.
    lines_converted : pandas.DataFrame
        Converted mol2 file lines.
    """

    def __init__(self):

        self.path_klifs_download = None
        self.lines_converted = None

    def from_metadata(self, klifs_metadata, path_klifs_download):
        """
        Convert KLIFS mol2 files to PyMol readable mol2 files, e.g. replace underscored with negative residue IDs.

        Parameters
        ----------
        klifs_metadata : pandas.DataFrame
            KLIFS metadata describing the KLIFS dataset.
        path_klifs_download : pathlib.Path or str
            Path to directory of KLIFS dataset files.
        """

        start = datetime.datetime.now()

        logger.info(f"PREPROCESSING: Mol2KlifsToPymolConverter")
        logger.info(f"Number of metadata entries: {len(klifs_metadata)}")

        path_klifs_download = Path(path_klifs_download)
        if path_klifs_download.exists():
            self.path_klifs_download = path_klifs_download

        # Number of CPUs on machine
        num_cores = cpu_count() - 1
        logger.info(f"Number of cores used: {num_cores}")

        # Create pool with `num_processes` processes
        pool = Pool(processes=num_cores)

        # Get KLIFS entries as list
        entry_list = [j for i, j in klifs_metadata.iterrows()]

        # Apply function to each chunk in list
        lines_converted = pool.map(self._convert_mol2_file, entry_list)
        self.lines_converted = pd.concat(lines_converted, axis=0, sort=False, ignore_index=True)

        # Close and join pool
        pool.close()
        pool.join()

        end = datetime.datetime.now()

        logger.info(f"Start of mol2 KLIFS to PyMol conversion: {start}")
        logger.info(f"End of mol2 KLIFS to PyMol conversion: {end}")

    def _convert_mol2_file(self, klifs_metadata_entry):
        """
        Load, convert, and save mol2 file. Return all converted lines.

        Parameters
        ----------
        klifs_metadata_entry : pandas.Series
            KLIFS metadata describing one entry in the KLIFS dataset.

        Returns
        -------
        pandas.DataFrame
            Converted mol2 file lines.
        """

        path_mol2 = Path(self.path_klifs_download) / klifs_metadata_entry.filepath / "protein.mol2"
        path_mol2_pymol = Path(path_mol2).parent / "protein_pymol.mol2"

        # Load lines from mol2 file
        with open(path_mol2, "r") as f:
            lines = f.readlines()

        # Convert lines
        lines_new, lines_converted = self._convert_mol2(lines, klifs_metadata_entry.filepath)

        # Write new lines to new mol2 file
        with open(path_mol2_pymol, "w") as f:
            f.writelines(lines_new)

        return lines_converted

    @staticmethod
    def _convert_mol2(lines_mol2, filepath=None):
        """
        Convert KLIFS mol2 file to PyMol readable mol2 file, i.e. replace underscored with negative residue IDs.

        Parameters
        ----------
        lines_mol2 : list of str
            Lines from KLIFS mol2 file.
        filepath : str
            Molecule file path name (default None).

        Returns
        -------
        tuple of list of str and pandas.DataFrame
            New lines (original and converted) from KLIFS mol2 file, and converted lines for reporting purposes.
        """

        headers = {
            "@<TRIPOS>MOLECULE": False,
            "@<TRIPOS>ATOM": False,
            "@<TRIPOS>BOND": False,
            "@<TRIPOS>SUBSTRUCTURE": False,
        }

        # Lines for new mol2 file (including converted lines)
        lines_new = []

        # Details on converted lines (for reporting purposes).
        lines_converted = []

        # KLIFS mol2 file have 4 sections: MOLECULE, ATOM, BOND, SUBSTRUCTURE
        for line in lines_mol2:

            # Set flag to sections that have been visited, lasted True flag is currently visited section
            if line.startswith("@<TRIPOS>MOLECULE"):
                headers["@<TRIPOS>MOLECULE"] = True

            if line.startswith("@<TRIPOS>ATOM"):
                headers["@<TRIPOS>ATOM"] = True

            if line.startswith("@<TRIPOS>BOND"):
                headers["@<TRIPOS>BOND"] = True

            if line.startswith("@<TRIPOS>SUBSTRUCTURE"):
                headers["@<TRIPOS>SUBSTRUCTURE"] = True

            # In MOLECULE section
            # - No conversions necessary
            if (
                headers["@<TRIPOS>MOLECULE"]
                and not headers["@<TRIPOS>ATOM"]
                and not headers["@<TRIPOS>BOND"]
                and not headers["@<TRIPOS>SUBSTRUCTURE"]
            ):

                lines_new.append(line)

            # In ATOM section
            # - Underscored residues > residues with minus sign
            elif (
                headers["@<TRIPOS>MOLECULE"]
                and headers["@<TRIPOS>ATOM"]
                and not headers["@<TRIPOS>BOND"]
                and not headers["@<TRIPOS>SUBSTRUCTURE"]
            ):

                try:

                    if "_" in line.split()[7]:  # Substructure name, e.g. GLY_1
                        line_new = line.replace("_", "-")
                        lines_new.append(line_new)
                        lines_converted.append(
                            pd.DataFrame(
                                [
                                    [
                                        filepath,
                                        "ATOM_underscored_residue",
                                        line,
                                        line_new,
                                        "ATOM section: Underscored residue ID",
                                    ]
                                ],
                                columns=["path_mol2", "conversion", "line", "line_new", "details"],
                            )
                        )

                    else:  # Substructure name, e.g. GLY1
                        lines_new.append(line)

                except IndexError:

                    lines_new.append(line)

            # In BOND section
            # - No conversions necessary
            elif (
                headers["@<TRIPOS>MOLECULE"]
                and headers["@<TRIPOS>ATOM"]
                and headers["@<TRIPOS>BOND"]
                and not headers["@<TRIPOS>SUBSTRUCTURE"]
            ):

                lines_new.append(line)

            # In SUBSTRUCTURE section
            # - Underscored residues > residues with minus sign
            # - Irregular chain IDs, e.g. A1 > A
            else:

                try:

                    # In case of underscored residue IDs
                    if "_" in line.split()[1] and not len(line.split()[5]) > 1:

                        line_new = line.replace("_", "-")

                        lines_new.append(line_new)
                        lines_converted.append(
                            pd.DataFrame(
                                [
                                    [
                                        filepath,
                                        "SUBSTRUCTURE_residue_underscored",
                                        line,
                                        line_new,
                                        "SUBSTRUCTURE section: Underscored residue ID",
                                    ]
                                ],
                                columns=["path_mol2", "conversion", "line", "line_new", "details"],
                            )
                        )

                    # In case of irregular chain IDs
                    elif "_" not in line.split()[1] and len(line.split()[5]) > 1:

                        chain_id = line.split()[5]
                        # Spaces necessary, so that ALA145 is not converted to ALA45
                        line_new = line.replace(" " + chain_id, " " + chain_id[0])

                        lines_new.append(line_new)
                        lines_converted.append(
                            pd.DataFrame(
                                [
                                    [
                                        filepath,
                                        "SUBSTRUCTURE_chain_irregular",
                                        line,
                                        line_new,
                                        "SUBSTRUCTURE section: Irregular chain ID",
                                    ]
                                ],
                                columns=["path_mol2", "conversion", "line", "line_new", "details"],
                            )
                        )

                    # In case of underscored residue IDs AND irregular chain IDs
                    elif "_" in line.split()[1] and len(line.split()[5]) > 1:

                        # In case of underscored residue IDs
                        line_new = line.replace("_", "-")

                        # In case of irregular chain IDs
                        chain_id = line.split()[5]
                        # Spaces necessary, so that ALA145 is not converted to ALA45
                        line_new = line_new.replace(" " + chain_id, " " + chain_id[0])

                        lines_new.append(line_new)
                        lines_converted.append(
                            pd.DataFrame(
                                [
                                    [
                                        filepath,
                                        "SUBSTRUCTURE_residue_underscored_chain_irregular",
                                        line,
                                        line_new,
                                        "SUBSTRUCTURE section: Underscored residue and irregular chain ID",
                                    ]
                                ],
                                columns=["path_mol2", "conversion", "line", "line_new", "details"],
                            )
                        )

                    else:

                        lines_new.append(line)

                except IndexError:

                    lines_new.append(line)

        if len(lines_converted) > 0:
            lines_converted = pd.concat(lines_converted, axis=0, sort=False, ignore_index=True)
        else:
            lines_converted = pd.DataFrame(
                [], columns=["path_mol2", "conversion", "line", "line_new"]
            )

        return lines_new, lines_converted


class Mol2ToPdbConverter:
    """
    Convert mol2 file to pdb file and save pdb file locally.

    Attributes
    ----------
    inconsistent_conversions : pandas.DataFrame
        Inconsistencies between mol2 and pdb file, i.e. unequal number of atoms, unequal coordinates,
        differing residue IDs/names.
    """

    def __init__(self):
        self.inconsistent_conversions = pd.DataFrame(
            [], columns=["filepath", "inconsistency", "details"]
        )

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

        start = datetime.datetime.now()

        logger.info(f"PREPROCESSING: Mol2ToPdbConverter")
        logger.info(f"Number of metadata entries: {len(klifs_metadata)}")

        # Launch PyMol once
        self._pymol_launch()

        for index, row in klifs_metadata.iterrows():

            if index % 1000 == 0:
                logger.info(f"Progress: {index}/{len(klifs_metadata)}")

            try:

                path_mol2 = Path(path_klifs_download) / row.filepath / "protein_pymol.mol2"

                # Set mol2 path
                path_mol2 = self._set_path_mol2(path_mol2)

                # Set pdb path
                path_pdb = self._set_path_pdb(path_mol2, path_pdb=None)

                # PyMol mol2 to pdb conversion
                self._pymol_mol2_to_pdb_conversion(path_mol2, path_pdb)
                self._report_inconsistent_conversion(
                    path_mol2, path_pdb
                )  # Check if files are equivalent

            except FileNotFoundError:
                pass

        self._pymol_quit()

        end = datetime.datetime.now()

        logger.info(f"Start of mol2 to pdb conversion: {start}")
        logger.info(f"End of mol2 to pdb conversion: {end}")

    def from_file(self, path_mol2, path_pdb=None):
        """
        Convert mol2 file to pdb file and save pdb file locally.

        Parameters
        ----------
        path_mol2 : pathlib.Path or str
            Path to mol2 file.
        path_pdb : None or pathlib.Path or str
            Path to pdb file (= converted mol2 file). Directory must exist.
            Default is None - saves pdb file next to the mol2 file in same directory with the same filename.
        """

        # Set mol2 path
        path_mol2 = self._set_path_mol2(path_mol2)

        # Set pdb path
        path_pdb = self._set_path_pdb(path_mol2, path_pdb)

        # PyMol mol2 to pdb conversion
        self._pymol_launch()
        self._pymol_mol2_to_pdb_conversion(path_mol2, path_pdb)
        self._report_inconsistent_conversion(path_mol2, path_pdb)  # Check if files are equivalent
        self._pymol_quit()

    @staticmethod
    def _pymol_launch():

        stdin = sys.stdin
        stdout = sys.stdout
        stderr = sys.stderr

        pymol.finish_launching(["pymol", "-qc"])

        sys.stdin = stdin
        sys.stdout = stdout
        sys.stderr = stderr

    @staticmethod
    def _pymol_mol2_to_pdb_conversion(path_mol2, path_pdb):
        """
        Convert a mol2 file to a pdb file and save locally (using pymol).
        """

        pymol.cmd.reinitialize()
        pymol.cmd.load(path_mol2)
        pymol.cmd.save(path_pdb)

    @staticmethod
    def _pymol_quit():

        pymol.cmd.quit()

    def _report_inconsistent_conversion(self, path_mol2, path_pdb):
        """
        Report inconsistent mol2 and pdb conversion.
        """

        klifs_metadata_filepath = "/".join(
            path_mol2.parts[-4:-1]
        )  # Mimic filepath column in metadata

        # Wait until pdb file is written to disc and check if it is there
        attempt = 0

        while attempt <= 10:

            if path_pdb.exists():
                break

            else:

                if attempt == 10:
                    self.inconsistent_conversions = self.inconsistent_conversions.append(
                        pd.DataFrame(
                            [[klifs_metadata_filepath, "Non-existing pdb file", path_pdb]],
                            columns=["filepath", "inconsistency", "details"],
                        )
                    )

                else:
                    time.sleep(1)
                    attempt += 1

        # Load mol2 file
        pmol2 = PandasMol2().read_mol2(
            str(path_mol2),
            columns={
                0: ("atom_id", int),
                1: ("atom_name", str),
                2: ("x", float),
                3: ("y", float),
                4: ("z", float),
                5: ("atom_type", str),
                6: ("subst_id", str),
                7: ("subst_name", str),
                8: ("charge", float),
                9: ("status_bit", str),
            },
        )
        mol2_df = pmol2.df

        # Load pdb file
        ppdb = PandasPdb().read_pdb(str(path_pdb))
        pdb_df = pd.concat([ppdb.df["ATOM"], ppdb.df["HETATM"]])

        # Number of atoms?
        n_atoms_mol2 = len(mol2_df)
        n_atoms_pdb = len(pdb_df)

        if not n_atoms_mol2 == n_atoms_pdb:

            self.inconsistent_conversions = self.inconsistent_conversions.append(
                pd.DataFrame(
                    [
                        [
                            klifs_metadata_filepath,
                            "Unequal number of atoms",
                            {"mol2": n_atoms_mol2, "pdb": n_atoms_pdb},
                        ]
                    ],
                    columns=["filepath", "inconsistency", "details"],
                )
            )

        else:

            # x, y, and z mean values are the same
            if not np.isclose(mol2_df.x.mean(), pdb_df.x_coord.mean(), rtol=1e-03):

                self.inconsistent_conversions = self.inconsistent_conversions.append(
                    pd.DataFrame(
                        [
                            [
                                klifs_metadata_filepath,
                                "Unequal x coordinate mean",
                                round(mol2_df.x.mean() - pdb_df.x_coord.mean(), 2),
                            ]
                        ],
                        columns=["filepath", "inconsistency", "details"],
                    )
                )

            if not np.isclose(mol2_df.y.mean(), pdb_df.y_coord.mean(), rtol=1e-03):

                self.inconsistent_conversions = self.inconsistent_conversions.append(
                    pd.DataFrame(
                        [
                            [
                                klifs_metadata_filepath,
                                "Unequal y coordinate mean",
                                round(mol2_df.y.mean() - pdb_df.y_coord.mean(), 2),
                            ]
                        ],
                        columns=["filepath", "inconsistency", "details"],
                    )
                )

            if not np.isclose(mol2_df.z.mean(), pdb_df.z_coord.mean(), rtol=1e-03):

                self.inconsistent_conversions = self.inconsistent_conversions.append(
                    pd.DataFrame(
                        [
                            [
                                klifs_metadata_filepath,
                                "Unequal z coordinate mean",
                                round(mol2_df.z.mean() - pdb_df.z_coord.mean(), 2),
                            ]
                        ],
                        columns=["filepath", "inconsistency", "details"],
                    )
                )

        # Record name of PDB file is always ATOM (although containing non-standard residues)
        if not set(pdb_df.record_name) == {"ATOM"}:

            non_atom_entries = pdb_df[pdb_df.record_name != "ATOM"]

            self.inconsistent_conversions = self.inconsistent_conversions.append(
                pd.DataFrame(
                    [
                        [
                            klifs_metadata_filepath,
                            "Non-ATOM entries",
                            {
                                "record_name": set(non_atom_entries.record_name),
                                "residue_name": set(non_atom_entries.residue_name),
                            },
                        ]
                    ],
                    columns=["filepath", "inconsistency", "details"],
                )
            )

        # Residue ID and name are the same
        residue_set_mol2 = set(mol2_df.subst_name)
        pdb_df.residue_number = pdb_df.residue_number.astype(str)
        residue_set_pdb = set(pdb_df.residue_name + pdb_df.residue_number)
        residue_set_diff = (residue_set_mol2 - residue_set_pdb) | (
            residue_set_pdb - residue_set_mol2
        )

        if not len(residue_set_diff) == 0:

            self.inconsistent_conversions = self.inconsistent_conversions.append(
                pd.DataFrame(
                    [
                        [
                            klifs_metadata_filepath,
                            "Unequal residue ID/name",
                            {
                                "mol2": (residue_set_mol2 - residue_set_pdb),
                                "pdb": (residue_set_pdb - residue_set_mol2),
                            },
                        ]
                    ],
                    columns=["filepath", "inconsistency", "details"],
                )
            )

    @staticmethod
    def _set_path_mol2(path_mol2):
        """
        Test if input mol2 file exists - and if so, return to be set as class attribute.

        Parameters
        ----------
        path_mol2 : pathlib.Path or str
            Path to mol2 file.

        Returns
        -------
        pathlib.Path
            Path to mol2 file.
        """

        path_mol2 = Path(path_mol2)

        if not path_mol2.exists():
            raise FileNotFoundError(f"File path does not exist: {path_mol2}")
        if not path_mol2.suffix == ".mol2":
            raise ValueError(f"Incorrect file suffix: {path_mol2.suffix}, must be mol2.")

        return path_mol2

    @staticmethod
    def _set_path_pdb(path_mol2, path_pdb):
        """
        Test if input directory for pdb file exists and has a pdb suffix - and if so, return to be set as class
        attribute.

        Parameters
        ----------
        path_pdb : None or pathlib.Path or str
            Path to pdb file (= converted mol2 file). Directory must exist.
            Default is None - saves pdb file next to the mol2 file in same directory with the same filename.

        Returns
        -------
        pathlib.Path
            Path to pdb file (= converted mol2 file).
        """

        # If mol2 path is not set, do not continue to set pdb path (otherwise we cannot convert mol2 to pdb)
        if path_mol2 is None:
            raise ValueError(f"Set mol2 path (class attribute) first.")

        else:
            pass

        if path_pdb is None:
            path_pdb = path_mol2.parent / f"{path_mol2.stem}.pdb"

        else:
            path_pdb = Path(path_pdb)

            if not path_pdb.parent.exists():
                raise FileNotFoundError(f"Directory does not exist: {path_pdb}")
            if not path_pdb.suffix == ".pdb":
                raise ValueError(
                    f"Missing file name or incorrect file type: {path_pdb}. "
                    f"Must have the form: /path/to/existing/directory/filename.pdb"
                )

        return path_pdb


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print("preprocessing.py executed from CLI.")
