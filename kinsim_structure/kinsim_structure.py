"""
kinsim_structure.py
Subpocket-based structural fingerprint for kinase pocket comparison

Handles the primary functions
"""

"""
kinsim_structure.py
Subpocket-based structural fingerprint for kinase pocket comparison

Handles the primary functions
"""

import logging
from pathlib import Path
import pickle

from kinsim_structure.preprocessing import KlifsMetadataLoader, KlifsMetadataFilter, \
    Mol2FormatScreener, Mol2KlifsToPymolConverter, Mol2ToPdbConverter
from kinsim_structure.encoding import FingerprintGenerator
from kinsim_structure.similarity import FeatureDistancesGenerator, FingerprintDistanceGenerator

logger = logging.getLogger(__name__)


class Preprocessing:
    """
    Preprocess KLIFS dataset:
    1. Load KLIFS metadata from files.
    2. Screen KLIFS protein mol files for irregular file rows.
    3. Convert KLIFS protein.mol files to PyMol readable protein_pymol.mol2 files
     (residues with underscores are transformed to residues with negative sign).
    4. Convert protein_pymol.mol2 file to protein_pymol.pdb file.
    5. Filter KLIFS metadata by different criteria such as species (HUMAN), DFG conformation (in), resolution
    (<=4), KLIFS quality score (>=4) and existent/parsable mol2 and pdb files.

    Attributes
    ----------
    path_klifs_overview : str or pathlib.Path or None
        Path to KLIFS download file `overview.csv` containing mainly KLIFS alignment-related metadata.
    path_klifs_export : str or pathlib.Path or None
        Path to KLIFS download file `KLIFS_download/KLIFS_export.csv` containing mainly structure-related metadata.
    path_klifs_download : pathlib.Path or str or None
        Path to directory of KLIFS dataset files.
    path_results : pathlib.Path or str or None
        Path to results folder.
    """

    def __init__(self):

        self.path_klifs_overview = None
        self.path_klifs_export = None
        self.path_klifs_download = None
        self.path_results = None

    def execute(self, path_klifs_overview, path_klifs_export, path_klifs_download, path_results):
        """
        Preprocess KLIFS dataset:
        1. Load KLIFS metadata from files.
        2. Screen KLIFS protein mol files for irregular file rows.
        3. Convert KLIFS protein.mol files to PyMol readable protein_pymol.mol2 files
         (residues with underscores are transformed to residues with negative sign).
        4. Convert protein_pymol.mol2 file to protein_pymol.pdb file.
        5. Filter KLIFS metadata by different criteria such as species (HUMAN), DFG conformation (in), resolution
        (<=4), KLIFS quality score (>=4) and existent/parsable mol2 and pdb files.

        Parameters
        ----------
        path_klifs_overview : str or pathlib.Path
            Path to KLIFS download file `overview.csv` containing mainly KLIFS alignment-related metadata.
        path_klifs_export : str or pathlib.Path
            Path to KLIFS download file `KLIFS_download/KLIFS_export.csv` containing mainly structure-related metadata.
        path_klifs_download : pathlib.Path or str
            Path to directory of KLIFS dataset files.
        path_results : pathlib.Path or str
            Path to results folder.

        Returns
        -------
        klifs_metadata_filter : preprocessing.KlifsMetadataFilter
            Filtered KLIFS metadata.
        """

        # Check if input files and directories exist.
        if not self.path_klifs_overview.exists():
            raise FileNotFoundError(f'File not found: {self.path_klifs_overview}')
        if not self.path_klifs_export.exists():
            raise FileNotFoundError(f'File not found: {self.path_klifs_export}')
        if not self.path_klifs_download.exists():
            raise FileNotFoundError(f'Directory not found: {self.path_klifs_download}')
        if not self.path_results.exists():
            raise FileNotFoundError(f'Directory not found: {self.path_results}')

        # Set as class attributes
        self.path_klifs_overview = Path(path_klifs_overview)
        self.path_klifs_export = Path(path_klifs_export)
        self.path_klifs_download = Path(path_klifs_download)
        self.path_results = Path(path_results)

        # Load KLIFS metadata (unfiltered)
        klifs_metadata_loader = self._load_klifs_metadata()

        # Screen all KLIFS protein.mol2 files for irregular rows
        mol2_format_screener = self._screen_mol2_format(klifs_metadata_loader)

        # Convert all KLIFS protein.mol2 files to PyMol readable protein_pymol.mol2 files
        mol2_klifs_to_pymol_converter = self._convert_mol2_klifs_to_pymol(klifs_metadata_loader)

        # Convert all protein_pymol.mol2 to protein_pymol.pdb files
        mol2_to_pdb_converter = self._convert_mol2_to_pdb(klifs_metadata_loader)

        # Filter KLIFS metadata
        klifs_metadata_filter = self._filter_klifs_metadata(klifs_metadata_loader)

        # Save class objects as files
        with open(self.path_results / 'preprocessing_klifs_metadata_loader.p', 'wb') as f:
            pickle.dump(klifs_metadata_loader, f)
        with open(self.path_results / 'preprocessing_mol2_format_screener.p', 'wb') as f:
            pickle.dump(mol2_format_screener, f)
        with open(self.path_results / 'preprocessing_mol2_klifs_to_pymol_converter.p', 'wb') as f:
            pickle.dump(mol2_klifs_to_pymol_converter, f)
        with open(self.path_results / 'preprocessing_mol2_to_pdb_converter.p', 'wb') as f:
            pickle.dump(mol2_to_pdb_converter, f)
        with open(self.path_results / 'preprocessing_klifs_metadata_filter.p', 'wb') as f:
            pickle.dump(klifs_metadata_filter, f)

        return klifs_metadata_filter

    def _load_klifs_metadata(self):
        """
        Load KLIFS metadata from metadata files and save KlifsMetadataLoader class object to results folder.
        """

        klifs_metadata_loader = KlifsMetadataLoader()
        klifs_metadata_loader.from_files(
            self.path_klifs_overview,
            self.path_klifs_export
        )

        return klifs_metadata_loader

    def _screen_mol2_format(self, klifs_metadata_loader):
        """
        Screen KLIFS protein mol2 file for irregular row formats and save Mol2FormatScreener class object to results
        folder.

        Parameters
        ----------
        klifs_metadata_loader
        """

        mol2_format_screener = Mol2FormatScreener()
        mol2_format_screener.from_metadata(
            klifs_metadata_loader.data_essential,
            self.path_klifs_download
        )

        return mol2_format_screener

    def _convert_mol2_klifs_to_pymol(self, klifs_metadata_loader):
        """
        Convert KLIFS protein mol2 file to a PyMol readable mol2 file.

        Parameters
        ----------
        klifs_metadata_loader
        """

        mol2_klifs_to_pymol_converter = Mol2KlifsToPymolConverter()
        mol2_klifs_to_pymol_converter.from_metadata(
            klifs_metadata_loader.data_essential,
            self.path_klifs_download
        )

        return mol2_klifs_to_pymol_converter

    def _convert_mol2_to_pdb(self, klifs_metadata_loader):
        """
        Convert protein mol2 file to pdb file.

        Parameters
        ----------
        klifs_metadata_loader
        """

        mol2_to_pdb_converter = Mol2ToPdbConverter()
        mol2_to_pdb_converter.from_klifs_metadata(
            klifs_metadata_loader.data_essential,
            self.path_klifs_download
        )

        return mol2_to_pdb_converter

    def _filter_klifs_metadata(self, klifs_metadata_loader):
        """

        Parameters
        ----------
        klifs_metadata_loader
        """

        klifs_metadata_filter = KlifsMetadataFilter()
        klifs_metadata_filter.from_klifs_metadata(
            klifs_metadata_loader.data_essential,
            self.path_klifs_download
        )

        return klifs_metadata_filter


class Encoding:

    def __init__(self):

        self.path_results = None

    def execute(self, klifs_metadata_filter, path_results):

        # Check if input parameters
        if not self.path_results.exists():
            raise FileNotFoundError(f'Directory not found: {self.path_results}')
        if isinstance(klifs_metadata_filter, KlifsMetadataFilter):
            raise ValueError(f'TBA')  # TODO write error message, what makes sense?

        # Set as class attributes
        self.path_results = Path(path_results)

        # Generate fingerprints
        logger.info(f'***FingerprintGenerator')
        fingerprint_generator = FingerprintGenerator()
        fingerprint_generator.from_metadata(
            klifs_metadata_filter.data_essential,
            self.path_results
        )

        # Save class object to file
        with open(self.path_results / 'encoding_fingerprint_generator.p', 'wb') as f:
            pickle.dump(fingerprint_generator, f)


class Similarity:

    def __init__(self):

        self.path_results = None

    def execute(self, fingerprint_generator, distance_measures, feature_weighting_schemes, path_results):

        # Check if input parameters
        if not self.path_results.exists():
            raise FileNotFoundError(f'Directory not found: {self.path_results}')
        if isinstance(fingerprint_generator, FingerprintGenerator):
            raise ValueError(f'TBA')  # TODO write error message, what makes sense?

        # Set as class attributes
        self.path_results = Path(path_results)

        # All against all fingerprint comparison
        logger.info(f'***FeatureDistancesGenerator')

        for distance_measure_name, distance_measure in distance_measures.items():

            # Generate feature distances (FeatureDistancesGenerator)
            logger.info(f'***FeatureDistancesGenerator: {distance_measure_name}')

            feature_distances_generator = FeatureDistancesGenerator()
            feature_distances_generator.from_fingerprint_generator(fingerprint_generator)

            # Save class object to file
            with open(self.path_results / f'similarity_feature_distances_{distance_measure_name}.p', 'wb') as f:
                pickle.dump(feature_distances_generator, f)

            for feature_weights_name, feature_weights in feature_weighting_schemes.items():
                # Generate fingerprint distance (FingerprintDistanceGenerator)
                logger.info(f'***FingerprintDistanceGenerator: {distance_measure_name}')

                fingerprint_distance_generator = FingerprintDistanceGenerator()
                fingerprint_distance_generator.from_feature_distances_generator(
                    feature_distances_generator,
                    feature_weights
                )

                # Save class object to file
                with open(
                    self.path_results / f'similarity_fingerprint_distance_{distance_measure}_{feature_weights_name}.p', 'wb'
                ) as f:
                    pickle.dump(feature_distances_generator, f)


def main(
    path_klifs_overview,
    path_klifs_export,
    path_klifs_download,
    path_results,
    distance_measures,
    feature_weighting_schemes
):

    # Preprocess KLIFS data
    preprocessing = Preprocessing()
    klifs_metadata_filter = preprocessing.execute(
        path_klifs_overview,
        path_klifs_export,
        path_klifs_download,
        path_results
    )

    # Encode KLIFS pockets
    encoding = Encoding()
    fingerprint_generator = encoding.execute(
        klifs_metadata_filter,
        path_results
    )

    # Calculate all-against-all similarity
    similarity = Similarity()
    similarity.execute(
        fingerprint_generator,
        distance_measures,
        feature_weighting_schemes,
        path_results
    )


if __name__ == "__main__":
    pass
