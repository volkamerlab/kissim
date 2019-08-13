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


class ResidueConservation:
    """
    Occurrence of each residue per KLIFS ID over all KLIFS structures.

    Attributes
    ----------
    data : pandas.DataFrame
        Occurrence of each residue (rows) over all structures per KLIFS ID (columns).

    Parameters
    ----------
    klifs_metadata : pandas.DataFrame
        DataFrame containing merged metadate from both input KLIFS tables.
    """

    def __init__(self, klifs_metadata):
        self.data = self.get_residue_occurrence(klifs_metadata)

    @staticmethod
    def get_residue_occurrence(klifs_metadata):

        # Get DataFrame containing residue per KLIFS ID (columns) for all structure pockets (rows).
        pockets = pd.DataFrame(
            list(klifs_metadata.pocket.apply(lambda x: list(x))),
            columns=range(1, 86)
        )

        # Get DataFrame containing per KLIFS ID (columns) the occurrence of each residue (rows) over all structures.
        residues_occurrence = pockets.apply(lambda x: x.value_counts(), axis=0)

        # Convert float and nan values into int and 0 values
        residues_occurrence.fillna(0, inplace=True)
        residues_occurrence = residues_occurrence.astype('int64')

        return residues_occurrence


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

        self.n_structures = klifs_metadata.shape[0]
        self.data = self.calculate_gap_rate(klifs_metadata)

    def calculate_gap_rate(self, klifs_metadata):
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

        gap_rate = [round(i / float(self.n_structures), 4) for i in gaps]

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
        ax.set_xlabel('KLIFS position ID')
        ax.set_ylabel('Gap rate')
        ax.xaxis.set_ticks(np.arange(4, 85, 5))
        ax.set_xticklabels(np.arange(5, 86, 5))

        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(20)

        plt.savefig(path_to_results / 'gap_rate.png', dpi=300)


class SideChainOrientationStatistics:

    def __init__(self):
        self.n_structures = None
        self.data = None
        self.missing_residues_ca_cb = None

    def from_metadata(self, klifs_metadata):
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

        # Set number of structures in dataset
        self.n_structures = klifs_metadata.shape[0]

        stats = []

        for index, row in klifs_metadata.iterrows():
            print(f'{index + 1}/{len(klifs_metadata)}')

            klifs_molecule_loader = KlifsMoleculeLoader(metadata_entry=row)
            molecule = klifs_molecule_loader.molecule

            side_chain_orientation_feature = SideChainOrientationFeature()
            side_chain_orientation_feature.from_molecule(molecule, verbose=True)

            side_chain_orientation_feature.features['klifs_code'] = molecule.code

            stats.append(side_chain_orientation_feature.features)

        self.data = pd.concat(stats)

    def get_missing_residues_ca_cb(self, gap_rate):

        # Get number of missing atoms per KLIFS position
        missing_ca = self.data[self.data.ca.isna()].groupby(by='klifs_id').size()
        missing_cb = self.data[self.data.cb.isna()].groupby(by='klifs_id').size()
        missing_ca_cb = self.data[(self.data.ca.isna()) & (self.data.cb.isna())].groupby(by='klifs_id', sort=False).size()

        missing_positions = pd.DataFrame(
            [
                missing_ca,
                missing_cb,
                missing_ca_cb,
                gap_rate.data.gaps
            ],
            index='ca cb ca_cb gaps'.split()
        ).transpose()
        missing_positions.fillna(value=0, inplace=True)
        missing_positions = missing_positions.astype('int64')
        missing_positions['klifs_id'] = missing_positions.index

        self.missing_residues_ca_cb = missing_positions

    def get_mean_median(self, output_path=None, from_file=None):
        """
        Get mean and median of side chain orientation angles for each amino acid in dataset.
        Add angle of 0 to GLY.

        Parameters
        ----------
        output_path : str or pathlib.Path
            Path to directory where data file should be saved
        from_file : None or str or pathlib.Path
            Default is None, optionally can take path to file containing respective data.

        Returns
        -------
        pandas.DataFrame
            Mean and median of side chain orientation angles for each amino acid in dataset.
        """

        if isinstance(self.data, pd.DataFrame):
            sco_stats_std = self.data.copy()
        elif from_file:
            with open(Path(from_file), 'rb') as f:
                sco_stats_std = pickle.load(f)
        else:
            raise ValueError(f'No data available for mean/median value calculation.')

        # Calculate mean and median
        sco_mean_median = pd.DataFrame(
            [
                sco_stats_std.groupby('residue_name').mean()['sco'],
                sco_stats_std.groupby('residue_name').median()['sco']
            ]
        ).transpose()
        sco_mean_median.columns = ['sco_mean', 'sco_median']
        sco_mean_median = sco_mean_median.round(2)

        # Add value: GLY=0
        sco_mean_median.loc['GLY'] = 0

        if output_path:
            sco_mean_median.to_csv(output_path / 'side_chain_orientation_mean_median.csv')

        return sco_mean_median

    def plot_missing_residues_ca_cb(self, path_to_results):

        ax = plt.gca()

        self.missing_residues_ca_cb.loc[:, ['cb', 'gaps']].plot(
            figsize=(20, 6),
            kind='bar',
            stacked=True,
            rot=1,
            ax=ax
        )

        ax.set_title(f'Missing residues and Cb atoms ({self.n_structures} KLIFS entries)',
                     fontsize=20)

        ax.set_xlabel('KLIFS position ID')
        ax.set_ylabel('Number of missing data')
        ax.xaxis.set_ticks(np.arange(4, 85, 5))
        ax.set_xticklabels(np.arange(5, 86, 5))

        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(20)

        plt.savefig(path_to_results / 'missing_residues_ca_cb.png', dpi=300)

    def plot_side_chain_orientation_distribution(self, path_to_results, kind='violin'):

        kinds = 'violin histograms'.split()

        # 20 standard amino acids
        standard_aminoacids = 'GLY ALA SER PRO VAL THR CYS ILE LEU ASN ASP GLN LYS GLU MET HIS PHE ARG TYR TRP'.split()

        if kind == kinds[0]:

            # Plot standard amino acids
            plt.figure(figsize=(25, 8))

            ax = sns.violinplot(
                x='residue_name',
                y='sco',
                data=self.data[self.data.residue_name.isin(standard_aminoacids)],
                order=standard_aminoacids
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
            ax.set_xlabel('Residue name (sorted by molecular weight)')
            ax.set_ylabel('Side chain orientation angle')

            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(20)

            plt.savefig(path_to_results / 'sco_violin_standard.png', dpi=300)

            # Plot non-standard amino acids
            plt.figure(figsize=(20, 8))

            ax = sns.violinplot(
                x='residue_name',
                y='sco',
                data=self.data[~self.data.residue_name.isin(standard_aminoacids)]
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
            ax.set_xlabel('Residue name')
            ax.set_ylabel('Side chain orientation angle')

            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(20)

            plt.savefig(path_to_results / 'sco_violin_nonstandard.png', dpi=300)

        elif kind == kinds[1]:

            plt.figure(figsize=(20, 30))

            for index, residue in enumerate(standard_aminoacids, 1):
                group = self.data.groupby(by='residue_name').get_group(residue)
                plt.subplot(7, 3, index)
                group.sco.plot(kind='hist', title=residue, xlim=(0, 180))

            plt.savefig(path_to_results / 'sco_histograms.png', dpi=300, bbox_inches='tight')

        else:

            raise ValueError(f'Plot kind unknown. Please choose from: {", ".join(kinds)}')


class NonStandardKlifsAminoAcids:

    def __init__(self, klifs_metadata):
        self.data = self.get_non_standard_amino_acids_in_klifs(klifs_metadata)

    @staticmethod
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



