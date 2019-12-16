"""
analysis.py

Subpocket-based structural fingerprint for kinase pocket comparison.

Handles the primary functions for the structural kinase fingerprint analysis.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


logger = logging.getLogger(__name__)


class KlifsMetadataAnalyser:
    """
    xxx

    """

    def __init__(self):
        pass

    @staticmethod
    def plot_residue_sequence_occurrency(klifs_metadata, residue_name, path_output):
        """
        xxx

        Parameters
        ----------
        klifs_metadata : pandas.DataFrame
            DataFrame containing merged metadate from both input KLIFS tables.
        residue_name : str
            One letter residue code.

        Returns
        -------
        xxx
            xxx
        """

        sequence = pd.DataFrame([list(i) for i in klifs_metadata.pocket], columns=range(1, 86))

        fig, ax = plt.subplots(figsize=(20, 4))

        axes = sequence.apply(
            lambda x: x == residue_name
        ).apply(
            lambda x: sum(x)
        ).plot(kind='bar', ax=ax)

        axes.set_ylabel('Number of KLIFS entries')
        axes.set_xlabel('KLIFS position')
        axes.set_title(f'Positional occurrency of residue {residue_name} in KLIFS pocket')
        plt.suptitle('')

        if residue_name == '_':
            residue_name = 'gap'

        axes.get_figure().savefig(
            Path(path_output) / f'sequence_occurrence_of_{residue_name}.png',
            dpi=300,
        )

        return axes


class FeatureDistributions:

    def __init__(self):
        pass

    def plot_boxplot(self, fingerprints, features_type, features_type_label, color, path_output):
        """
        Generate boxplot describing the feature distribution per feature name.

        Parameters
        ----------
        fingerprints : list of kissim.encoding.Fingerprint
            Fingerprints.
        features_type : str
            Type of fingerprint feature.
        features_type_label : str
            Label name for type of fingerprint feature.
        path_output : pathlib.Path or str
            Path to output folder.

        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot
            Boxplot axes.
        """

        feature = self._get_features_by_type(fingerprints, features_type)

        axes = feature.plot(
            figsize=(8.5, 6),
            kind='box',
            title=features_type_label,
            grid=False,
            color=dict(boxes=color, whiskers=color, medians='grey', caps=color),
            boxprops=dict(linestyle='-', linewidth=1.5),
            flierprops=dict(linestyle='none', marker='o', markerfacecolor='none', markersize=3,  markeredgecolor='grey'),
            medianprops=dict(linestyle='-', linewidth=1.5),
            whiskerprops=dict(linestyle='-', linewidth=1.5),
            capprops=dict(linestyle='-', linewidth=1.5),
            showfliers=True
        )

        # https://www.drawingfromdata.com/how-to-rotate-axis-labels-in-seaborn-and-matplotlib
        #axes.set_xticklabels(axes.get_xticklabels(), rotation=45, ha='right')

        axes.get_figure().savefig(
            Path(path_output) / f'feature_distribution_boxplot_{features_type}.png',
            dpi=300,
            #bbox_inches='tight'
        )

        return axes

    def plot_violinplot(self, fingerprints, features_type, features_type_label, color, path_output):

        plt.figure(figsize=(8.5, 6))

        # Melt data
        melted_data = self._melt_features_by_type(fingerprints, features_type, features_type_label)

        if ('physicochemical' in features_type) or ('distances' in features_type):

            axes = sns.violinplot(
                x=features_type_label,
                y='Feature value',
                data=melted_data,
                color=color,
                rot=90
            )

        elif 'moments' in features_type:

            axes = sns.violinplot(
                x=features_type_label,
                y='Feature value',
                hue='Distance to',
                data=melted_data,
                color=color,
                rot=90
            )

        else:
            print(f'Input did not match. Check again.')
            axes = None

        # https://www.drawingfromdata.com/how-to-rotate-axis-labels-in-seaborn-and-matplotlib
        #axes.set_xticklabels(axes.get_xticklabels(), rotation=45, ha='right')

        axes.get_figure().savefig(
            path_output / f'feature_distribution_violinplot_{features_type}.png',
            dpi=300
        )

        return axes

    def _melt_features_by_type(self, fingerprints, features_type, features_type_label):
        """
        Melt data for fingerprint feature (as preparation for plotting).

        Parameters
        ----------
        fingerprints : list of kissim.encoding.Fingerprint
            Fingerprints.
        features_type : str
            Type of fingerprint feature.
        features_type_label : str
            Label name for type of fingerprint feature.

        Returns
        -------
        pandas.DataFrame
            Melted fingerprint features of certain type (label names as column/row names).
        """

        if ('physicochemical' in features_type) or ('distances' in features_type):

            return self._get_features_by_type(fingerprints, features_type).melt(
                var_name=features_type_label,
                value_name='Feature value'
            )

        elif 'moments' in features_type:

            return self._get_features_by_type(fingerprints, features_type).reset_index().melt(
                id_vars=['Distance to', 'index'],
                var_name=features_type_label,
                value_name='Feature value'
            )

        else:
            print(f'Input did not match. Check again.')

    @staticmethod
    def _get_features_by_type(fingerprints, features_type):
        """
        Get fingerprint features by feature type and return with updated columns/indices suitable as plot label names.

        Parameters
        ----------
        fingerprints : list of kissim.encoding.Fingerprint
            Fingerprints.
        features_type : str
            Type of fingerprint feature.

        Returns
        -------
        pandas.DataFrame
            Fingerprint features of certain type (label names as column/row names).
        """

        # Set label names for plots
        physicochemical_columns = {
            'size': 'Size',
            'hbd': 'HBD',
            'hba': 'HBA',
            'charge': 'Charge',
            'aromatic': 'Aromatic',
            'aliphatic': 'Aliphatic',
            'sco': 'SCO',
            'exposure': 'Exposure'
        }

        distances_columns = {
            'distance_to_centroid': 'Centroid',
            'distance_to_hinge_region': 'Hinge region',
            'distance_to_dfg_region': 'DFG region',
            'distance_to_front_pocket': 'Front pocket'
        }

        moments_columns = {
            'index': 'Distance to',
            'moment1': 'Moment 1',
            'moment2': 'Moment 2',
            'moment3': 'Moment 3'
        }

        # Get all fingerprints of certain type
        features = pd.concat(
            [getattr(i, features_type) for i in fingerprints],
            axis=0
        )

        # Rename columns/indices to label names
        if 'physicochemical' in features_type:

            features.rename(
                columns=physicochemical_columns,
                inplace=True
            )

        elif 'distances' in features_type:

            features.rename(
                columns=distances_columns,
                inplace=True
            )

        elif 'moments' in features_type:

            features.rename(
                columns=moments_columns,
                inplace=True
            )

            features.reset_index(inplace=True)

            features.rename(
                columns=moments_columns,
                inplace=True
            )

            features['Distance to'] = [distances_columns[i] for i in features['Distance to']]

        else:
            raise ValueError(f'Input fingerprint type did not match. Check again.')

        return features


class SideChainOrientationDistribution:

    def __init__(self):
        pass

    @staticmethod
    def plot_sco_boxplot(sco_df, path_output):

        fig, ax = plt.subplots(figsize=(20, 6))

        axes = sco_df.boxplot(
            column='vertex_angle',
            by='klifs_id',
            ax=ax,
            grid=False
        )

        axes.set_ylabel('Side chain orientation (vertex angle)')
        axes.set_xlabel('KLIFS position')
        axes.set_title('Side chain orientation towards pocket centroid')
        plt.suptitle('')

        axes.get_figure().savefig(
            Path(path_output) / f'side_chain_orientation_per_klifs_position_boxplot.png',
            dpi=300,
        )

        return axes

    @staticmethod
    def plot_sco_barplot(sco_df, path_output):

        fig, ax = plt.subplots(figsize=(20, 6))
        plt.suptitle('')

        axes = sco_df.groupby(['klifs_id', 'sco']).size().unstack().plot(
            kind='bar',
            stacked=True,
            ax=ax,
            color=['steelblue', 'lightgrey', 'skyblue'],
            legend=['a', 'b', 'c'],
            ylim=[0, 5000]
        )

        axes.set_ylabel('Number of KLIFS structures')
        axes.set_xlabel('KLIFS position')
        axes.set_title('Side chain orientation towards pocket centroid')

        legend = plt.legend()
        legend.get_texts()[0].set_text('0.0 - Inwards (angle [0, 45])')
        legend.get_texts()[1].set_text('1.0 - Intermediate (angle ]45, 90])')
        legend.get_texts()[2].set_text('2.0 - Outwards (angle ]90, 180])')

        axes.get_figure().savefig(
            Path(path_output) / f'side_chain_orientation_per_klifs_position_barplot.png',
            dpi=300,
        )

        return axes

    @staticmethod
    def get_sco_data(fingerprint_generator):

        sco_list = []

        for molecule_code, fingerprint in fingerprint_generator.data.items():

            sco = fingerprint.features_verbose['side_chain_orientation']
            sco['molecule_code'] = molecule_code
            sco_list.append(sco)

        sco_df = pd.concat(sco_list, sort=False)
        sco_df.reset_index(inplace=True, drop=True)

        return sco_df


class ExposureDistribution:

    def __init__(self):
        pass

    @staticmethod
    def get_exposure_data(fingerprint_generator):
        exposure_list = []

        for molecule_code, fingerprint in fingerprint_generator.data.items():
            exposure = fingerprint.features_verbose['exposure']
            exposure['molecule_code'] = molecule_code
            exposure_list.append(exposure)

        exposure_df = pd.concat(exposure_list, sort=False)
        exposure_df.reset_index(inplace=True, drop=True)

        return exposure_df
