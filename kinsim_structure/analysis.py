"""
analysis.py

Subpocket-based structural fingerprint for kinase pocket comparison.

Handles the primary functions for the structural kinase fingerprint analysis.
"""

import logging
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from kinsim_structure.auxiliary import KlifsMoleculeLoader, PdbChainLoader
from kinsim_structure.encoding import SideChainAngleFeature

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


class SideChainAngleStatistics:

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

            klifs_molecule_loader = KlifsMoleculeLoader(klifs_metadata_entry=row)
            molecule = klifs_molecule_loader.molecule

            pdb_chain_loader = PdbChainLoader(klifs_metadata_entry=row)
            chain = pdb_chain_loader.chain

            side_chain_angle_feature = SideChainAngleFeature()
            side_chain_angle_feature.from_molecule(molecule, chain=chain, verbose=True)

            side_chain_angle_feature.features['klifs_code'] = molecule.code

            stats.append(side_chain_angle_feature.features)

        self.data = pd.concat(stats)

    def get_missing_residues_ca_cb(self, gap_rate):

        # Get number of missing atoms per KLIFS position
        missing_ca = self.data[
            self.data.ca.isna()
        ].groupby(by='klifs_id').size()
        missing_cb = self.data[
            self.data.cb.isna()
        ].groupby(by='klifs_id').size()
        missing_ca_cb = self.data[
            (self.data.ca.isna()) & (self.data.cb.isna())
        ].groupby(by='klifs_id', sort=False).size()

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
        Get mean and median of side chain angles for each amino acid in dataset.
        Add angle of 0 to GLY.

        Parameters
        ----------
        output_path : str or pathlib.Path
            Path to directory where data file should be saved.
        from_file : None or str or pathlib.Path
            Default is None, optionally can take path to file containing respective data.

        Returns
        -------
        pandas.DataFrame
            Mean and median of side chain angles for each amino acid in dataset.
        """

        if isinstance(self.data, pd.DataFrame):
            sca_stats_std = self.data.copy()
        elif from_file:
            with open(Path(from_file), 'rb') as f:
                sca_stats_std = pickle.load(f)
        else:
            raise ValueError(f'No data available for mean/median value calculation.')

        # Calculate mean and median
        sca_mean_median = pd.DataFrame(
            [
                sca_stats_std.groupby('residue_name').mean()['sca'],
                sca_stats_std.groupby('residue_name').median()['sca']
            ]
        ).transpose()
        sca_mean_median.columns = ['sca_mean', 'sca_median']
        sca_mean_median = sca_mean_median.round(2)

        # Add value: GLY=0
        sca_mean_median.loc['GLY'] = 0

        if output_path:
            sca_mean_median.to_csv(output_path / 'side_chain_angle_mean_median.csv')

        return sca_mean_median

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

    def plot_side_chain_angle_distribution(self, path_to_results, kind='violin'):

        kinds = 'violin histograms'.split()

        # 20 standard amino acids
        standard_aminoacids = 'GLY ALA SER PRO VAL THR CYS ILE LEU ASN ASP GLN LYS GLU MET HIS PHE ARG TYR TRP'.split()

        if kind == kinds[0]:

            # Plot standard amino acids
            plt.figure(figsize=(25, 8))

            ax = sns.violinplot(
                x='residue_name',
                y='sca',
                data=self.data[self.data.residue_name.isin(standard_aminoacids)],
                order=standard_aminoacids
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
            ax.set_xlabel('Residue name (sorted by molecular weight)')
            ax.set_ylabel('Side chain angle')

            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(20)

            plt.savefig(path_to_results / 'sca_violin_standard.png', dpi=300)

            # Plot non-standard amino acids
            plt.figure(figsize=(20, 8))

            ax = sns.violinplot(
                x='residue_name',
                y='sca',
                data=self.data[~self.data.residue_name.isin(standard_aminoacids)]
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
            ax.set_xlabel('Residue name')
            ax.set_ylabel('Side chain angle')

            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(20)

            plt.savefig(path_to_results / 'sca_violin_nonstandard.png', dpi=300)

        elif kind == kinds[1]:

            plt.figure(figsize=(20, 30))

            for index, residue in enumerate(standard_aminoacids, 1):
                group = self.data.groupby(by='residue_name').get_group(residue)
                plt.subplot(7, 3, index)
                group.sca.plot(kind='hist', title=residue, xlim=(0, 180))

            plt.savefig(path_to_results / 'sca_histograms.png', dpi=300, bbox_inches='tight')

        else:

            raise ValueError(f'Plot kind unknown. Please choose from: {", ".join(kinds)}')


class NonStandardKlifsAminoAcids:

    def __init__(self):
        self.data = None

    def get_non_standard_amino_acids_in_klifs(self, klifs_metadata):
        """
        For a given set of mol2 files, collect all non-standard amino acids.

        Parameters
        ----------
        klifs_metadata : pandas.DataFrame
            KLIFS metadata describing every pocket entry in the KLIFS dataset.

        Returns
        -------
        pandas.DataFrame
            Non-standard amino acids (residue ID and residue name in mol2 file and KLIFS metadata)
            for each structure listed in the KLIFS metadata.
        """

        standard_aminoacids = 'ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split()

        # Initialize list for non-standard amino acid entries in dataset
        non_std_list = []

        for index, row in klifs_metadata.iterrows():

            # Load molecule from metadata
            klifs_molecule_loader = KlifsMoleculeLoader(klifs_metadata_entry=row)
            molecule = klifs_molecule_loader.molecule

            # Get first entry per residue
            firsts = molecule.df.groupby(by='res_id', as_index=False).first()

            non_std = firsts[~firsts.res_name.isin(standard_aminoacids)].copy()
            non_std = non_std[['res_name', 'res_id', 'klifs_id']]

            # If non-standard amino acids are present in molecule...
            if len(non_std) > 0:

                # ... add molecule code
                non_std['code'] = molecule.code

                # ... add KLIFS residue names
                non_std['klifs_res_name'] = non_std.apply(
                    lambda x: row.pocket[x.klifs_id - 1],
                    axis=1
                )

                # Add to non-standard amino acid entry list
                non_std_list.append(non_std)

            # If no non-standard amino acids are present, do nothing
            else:
                continue

        non_std_all = pd.concat(non_std_list)
        non_std_all.reset_index(inplace=True, drop=True)

        self.data = non_std_all
