"""
analysis.py

Subpocket-based structural fingerprint for kinase pocket comparison.

Handles the primary functions for the structural kinase fingerprint analysis.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from kinsim_structure.auxiliary import KlifsMoleculeLoader
from kinsim_structure.encoding import SideChainOrientationFeature

logger = logging.getLogger(__name__)


class GapRate:
    """
    ...

    Attributes
    ----------
    data : pandas.DataFrame
        DataFrame containing gap rates for each position in the KLIFS alignment for the input data.

    Parameters
    ----------
    klifs_metadata : pandas.DataFrame
        DataFrame containing merged metadate from both input KLIFS tables.
    """

    def __init__(self, klifs_metadata):

        self.data = self.calculate_gap_rate(klifs_metadata)
        self.n_structures = klifs_metadata.shape[0]

    @staticmethod
    def calculate_gap_rate(klifs_metadata):
        """
        Calculate gap rate at every KLIFS MSA position across the filtered kinase data set.

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
        gap_rate = [round(i / float(klifs_metadata.shape[0]), 4) for i in gaps]

        data = pd.concat(
            [
                pd.Series(range(1, 86), name='klifs_id'),
                pd.Series(gaps, name='gaps'),
                pd.Series(gap_rate, name='gap_rate')
            ],
            axis=1
        )

        data.set_index('klifs_id', inplace=True, drop=False)

        return data

    def plot_gap_rate(self, path_to_results):
        """
        Plot gap rate for KLIFS IDs (positions).

        Parameters
        ----------
        path_to_results : str or pathlib.Path
            Path to directory where plot shall be saved.
        """

        path_to_results = Path(path_to_results)

        plt.figure(figsize=(15, 6))
        ax = sns.barplot(x='klifs_id',
                         y='gap_rate',
                         data=self.data,
                         color='steelblue')
        ax.set_title(f'KLIFS sequence alignment: '
                     f'Gap rate for the 85 residue positions ({self.n_structures} KLIFS entries)',
                     fontsize=20)
        ax.set_xlabel('Alignment residue position')
        ax.set_ylabel('Gap rate')
        ax.xaxis.set_ticks(np.arange(0, 85, 5));
        ax.set_xticklabels(np.arange(0, 85, 5));

        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(20)

        plt.savefig(path_to_results / 'plot_gap_rate.png', dpi=300)


def get_non_standard_amino_acids_in_klifs(klifs_metadata):
    """
    For a given set of mol2 files, collect all non-standard amino acids.

    Parameters
    ----------
    klifs_metadata : pandas.DataFrame
        KLIFS metadata describing every pocket entry in the KLIFS dataset.

    Returns
    -------
    dict
        List of non-standard amino acids (value) for each structure contained in the input mol2 file (key).
    """

    # Get list of molecules linked to metadata

    molecules = []
    for index, row in klifs_metadata.iterrows():
        klifs_molecule_loader = KlifsMoleculeLoader(metadata_entry=row)
        molecule = klifs_molecule_loader.molecule
        molecules.append(molecule)

    # 20 standard amino acids
    standard_aminoacids = 'ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split()

    # Will contain per structure (key) list of non-standard amino acids (values)
    non_standard_aminoacids = {}

    for molecule in molecules:

        # Get deduplicated amino acid names
        res_names = set(molecule.df.res_name)

        # Retain only non-standard amino acids
        non_standard_aminoacids_per_file = [res_name for res_name in res_names if res_name not in standard_aminoacids]

        # If structure contains non-standard amino acids, add to dict
        if non_standard_aminoacids_per_file:
            non_standard_aminoacids[molecule.code] = non_standard_aminoacids_per_file

    return non_standard_aminoacids


def get_side_chain_orientation_stats(klifs_metadata):
    """

    Parameters
    ----------
    klifs_metadata : pandas.DataFrame
        KLIFS metadata describing every pocket entry in the KLIFS dataset.

    Returns
    -------
    pandas.DataFrame
        CA, CB and centroid points for each residue for all molecules described in the KLIFS metadata.
    """

    stats = []

    for index, row in klifs_metadata.iterrows():

        print(f'{index+1}/{len(klifs_metadata)}')

        klifs_molecule_loader = KlifsMoleculeLoader(metadata_entry=row)
        molecule = klifs_molecule_loader.molecule

        side_chain_orientation_feature = SideChainOrientationFeature(molecule)
        points = side_chain_orientation_feature.from_molecule(molecule, verbose=True)

        points['klifs_code'] = molecule.code

        stats.append(points)

    return pd.concat(stats)
