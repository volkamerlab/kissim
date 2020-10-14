"""
kissim.encoding.api TODO
"""

import datetime
import logging
from multiprocessing import cpu_count, Pool
import numpy as np
import pandas as pd
from scipy.special import cbrt
from scipy.stats.stats import moment

from .features import SideChainOrientationFeature, SpatialFeatures, PhysicoChemicalFeatures
from ..definitions import DISTANCE_CUTOFFS, MOMENT_CUTOFFS
from ..auxiliary import KlifsMoleculeLoader, PdbChainLoader

logger = logging.getLogger(__name__)


class FingerprintGenerator:
    """
    Generate fingerprints for multiple molecules. Uses parallel computing of fingerprint pairs.

    Attributes
    ----------
    data : dict of kissim.encoding.Fingerprint
        Fingerprints for multiple molecules.
    path_klifs_download : pathlib.Path or str
        Path to directory of KLIFS dataset files.
    """

    def __init__(self):

        self.data = None
        self.path_klifs_download = None

    def from_metadata(self, klifs_metadata, path_klifs_download):
        """
        Generate fingerprints for multiple molecules described in KLIFS metadata.

        Parameters
        ----------
        klifs_metadata : pandas.DataFrame
            Metadata (columns) for KLIFS molecules (rows).
        """

        start = datetime.datetime.now()

        logger.info(f"ENCODING: FingerprintGenerator")

        # Set path to KLIFS download
        self.path_klifs_download = path_klifs_download

        # Number of CPUs on machine
        num_cores = cpu_count() - 1
        logger.info(f"Number of cores used: {num_cores}")

        # Create pool with `num_processes` processes
        pool = Pool(processes=num_cores)

        # Get KLIFS entries as list
        entry_list = [j for i, j in klifs_metadata.iterrows()]

        # Apply function to each chunk in list
        fingerprints_list = pool.map(self._get_fingerprint, entry_list)

        # Close and join pool
        pool.close()
        pool.join()

        logger.info(f"Number of fingerprints: {len(fingerprints_list)}")

        # Transform to dict
        self.data = {
            i.molecule_code: i
            for i in fingerprints_list
            if i is not None  # Removes emtpy fingerprints
        }

        end = datetime.datetime.now()

        logger.info(f"Start of fingerprint generation: {start}")
        logger.info(f"End of fingerprint generation: {end}")

    def _get_fingerprint(self, klifs_metadata_entry):
        """
        Get fingerprint.

        Parameters
        ----------
        klifs_metadata_entry : pandas.Series
            KLIFS metadata describing a pocket entry in the KLIFS dataset.

        Returns
        -------
        kissim.similarity.Fingerprint
            Fingerprint
        """

        try:

            fingerprint = Fingerprint()
            fingerprint.from_metadata_entry(klifs_metadata_entry, self.path_klifs_download)

            return fingerprint

        except Exception as e:

            logger.info(f"Molecule with empty fingerprint: {klifs_metadata_entry.filepath}")
            logger.error(e)

            return None


class SideChainOrientationGenerator:
    """
    Generate side chain orientations for multiple molecules. Uses parallel computing of
    fingerprint pairs.

    Attributes
    ----------
    data : dict of kissim.encoding.SideChainOrientationFeature
        Fingerprints for multiple molecules.
    path_klifs_download : pathlib.Path or str
        Path to directory of KLIFS dataset files.
    """

    def __init__(self):

        self.data = None
        self.path_klifs_download = None

    def from_metadata(self, klifs_metadata, path_klifs_download):
        """
        Generate side chain orientation features for multiple molecules described in
        KLIFS metadata.

        Parameters
        ----------
        klifs_metadata : pandas.DataFrame
            Metadata (columns) for KLIFS molecules (rows).
        path_klifs_download : pathlib.Path or str
            Path to directory of KLIFS dataset files.
        """

        # Get start time of script
        start = datetime.datetime.now()

        logger.info(f"Calculate side chain orientations...")

        # Set path to KLIFS download
        self.path_klifs_download = path_klifs_download

        # Number of CPUs on machine
        num_cores = cpu_count() - 1
        logger.info(f"Number of cores used: {num_cores}")

        # Create pool with `num_processes` processes
        pool = Pool(processes=num_cores)

        # Get KLIFS entries as list
        entry_list = [j for i, j in klifs_metadata.iterrows()]

        # Apply function to each chunk in list
        fingerprints_list = pool.map(self._get_sco, entry_list)

        # Close and join pool
        pool.close()
        pool.join()

        logger.info(f"Number of fingerprints: {len(fingerprints_list)}")

        # Transform to dict
        self.data = {i.molecule_code: i for i in fingerprints_list}

        # Get end time of script
        end = datetime.datetime.now()

        logger.info(start)
        logger.info(end)

    def _get_sco(self, klifs_metadata_entry):
        """
        Get side chain orientation.

        Parameters
        ----------
        klifs_metadata_entry : pandas.Series
            KLIFS metadata describing a pocket entry in the KLIFS dataset.

        Returns
        -------
        kissim.similarity.SideChainOrientationFeature
            Side chain orientation.
        """

        try:

            klifs_molecule_loader = KlifsMoleculeLoader()
            klifs_molecule_loader.from_metadata_entry(
                klifs_metadata_entry, self.path_klifs_download
            )
            molecule = klifs_molecule_loader.molecule

            pdb_chain_loader = PdbChainLoader()
            pdb_chain_loader.from_metadata_entry(klifs_metadata_entry, self.path_klifs_download)
            chain = pdb_chain_loader.chain

            feature = SideChainOrientationFeature()
            feature.from_molecule(molecule, chain)

            return feature

        except Exception as e:

            logger.info(f"Molecule with empty fingerprint: {klifs_metadata_entry.filepath}")
            logger.error(e)

            return None


class Fingerprint:
    """
    Kinase pocket is defined by 85 pre-aligned residues in KLIFS, which are described each with
    (i) 8 physicochemical and
    (ii) 4 distance features as well as
    (iii) the first three moments of aforementioned feature distance distributions.
    Fingerprints can consist of all or a subset of these three feature types.


    Attributes
    ----------
    molecule_code : str
        Molecule code as defined by KLIFS in mol2 file.
    fingerprint : dict of pandas.DataFrame
        Fingerprint, consisting of physicochemical, distance and moment features.
    fingerprint_normalized : dict of pandas.DataFrame
        Normalized fingerprint, consisting of physicochemical, distance and moment features.

    Notes
    -----
    PHYSICOCHEMICAL features (85 x 8 matrix = 680 bits):

    - Size
    - Pharmacophoric features:
      Hydrogen bond donor, hydrogen bond acceptor, aromatic, aliphatic and charge feature
    - Side chain orientation
    - Half sphere exposure

    SPATIAL features:

    - DISTANCE of each residue to 4 reference points (85 x 4 matrix = 340 bits):
      - Binding site centroid
      - Hinge region
      - DFG region
      - Front pocket
    - MOMENTS for distance distributions for the 4 reference points (4 x 3 matrix = 12 bits):
      - Moment 1: Mean
      - Moment 2: Standard deviation
      - Moment 3: Skewness (cube root)

    The terminology used for the feature hierarchy is the following:
    Feature category, e.g. spatial or physicochemical
    - Feature type, e.g. distance or physicochemical
      - Feature, e.g. distance to centroid or size
    """

    def __init__(self):

        self.molecule_code = None

        self.fingerprint = {"physicochemical": None, "distances": None, "moments": None}
        self.fingerprint_normalized = {"physicochemical": None, "distances": None, "moments": None}

        self.features_verbose = {
            "reference_points": None,
            "side_chain_orientation": None,
            "exposure": None,
        }

    @property
    def physicochemical(self):
        return self.fingerprint["physicochemical"]

    @property
    def distances(self):
        return self.fingerprint["distances"]

    @property
    def moments(self):
        return self.fingerprint["moments"]

    @property
    def physicochemical_distances(self):
        return self._get_fingerprint("physicochemical_distances", normalized=False)

    @property
    def physicochemical_moments(self):
        return self._get_fingerprint("physicochemical_moments", normalized=False)

    @property
    def physicochemical_normalized(self):
        return self.fingerprint_normalized["physicochemical"]

    @property
    def distances_normalized(self):
        return self.fingerprint_normalized["distances"]

    @property
    def moments_normalized(self):
        return self.fingerprint_normalized["moments"]

    @property
    def physicochemical_distances_normalized(self):
        return self._get_fingerprint("physicochemical_distances", normalized=True)

    @property
    def physicochemical_moments_normalized(self):
        return self._get_fingerprint("physicochemical_moments", normalized=True)

    def from_metadata_entry(self, klifs_metadata_entry, path_klifs_download):
        """
        Get kinase fingerprint from KLIFS metadata entry.

        Parameters
        ----------
        klifs_metadata_entry : pandas.Series
            KLIFS metadata describing a pocket entry in the KLIFS dataset.
        path_klifs_download : pathlib.Path or str
            Path to directory of KLIFS dataset files.
        """

        klifs_molecule_loader = KlifsMoleculeLoader()
        klifs_molecule_loader.from_metadata_entry(klifs_metadata_entry, path_klifs_download)
        molecule = klifs_molecule_loader.molecule

        pdb_chain_loader = PdbChainLoader()
        pdb_chain_loader.from_metadata_entry(klifs_metadata_entry, path_klifs_download)
        chain = pdb_chain_loader.chain

        self.from_molecule(molecule, chain)

    def from_molecule(self, molecule, chain):
        """
        Get kinase fingerprint from molecule.

        Parameters
        ----------
        molecule : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
            Content of mol2 or pdb file as BioPandas object.
        chain : Bio.PDB.Chain.Chain
            Chain from PDB file.
        """

        self.molecule_code = molecule.code

        physicochemical_features = PhysicoChemicalFeatures()
        physicochemical_features.from_molecule(molecule, chain)

        spatial_features = SpatialFeatures()
        spatial_features.from_molecule(molecule)

        self.fingerprint["physicochemical"] = physicochemical_features.features
        self.fingerprint["distances"] = spatial_features.features
        self.fingerprint["moments"] = self._calc_moments(spatial_features.features)

        self.fingerprint_normalized["physicochemical"] = self._normalize_physicochemical_bits()
        self.fingerprint_normalized["distances"] = self._normalize_distances_bits()
        self.fingerprint_normalized["moments"] = self._normalize_moments_bits()

        # Add verbose feature details
        self.features_verbose["reference_points"] = spatial_features.reference_points
        self.features_verbose["exposure"] = physicochemical_features.features_verbose["exposure"]
        self.features_verbose[
            "side_chain_orientation"
        ] = physicochemical_features.features_verbose["side_chain_orientation"]

    def _get_fingerprint(self, fingerprint_type, normalized=True):
        """
        Get fingerprint containing both physicochemical and spatial bits
        (available types: distances or moments).

        Parameters
        ----------
        fingerprint_type : str
            Type of fingerprint, i.e. fingerprint with
            physicochemical and either distances or moments bits
            (physicochemical + distances or physicochemical + moments).
        normalized : bool
            Normalized or non-normalized form of fingerprint (default: normalized).

        Returns
        -------
        dict of pandas.DataFrames
            Fingerprint containing physicochemical and spatial bits.
        """

        fingerprint_types = "physicochemical_distances physicochemical_moments".split()

        if fingerprint_type == "physicochemical_distances":

            if normalized:
                return {
                    "physicochemical": self.physicochemical_normalized,
                    "distances": self.distances_normalized,
                }
            else:
                return {"physicochemical": self.physicochemical, "distances": self.distances}

        elif fingerprint_type == "physicochemical_moments":

            if normalized:
                return {
                    "physicochemical": self.physicochemical_normalized,
                    "moments": self.moments_normalized,
                }
            else:
                return {"physicochemical": self.physicochemical, "moments": self.moments}
        else:
            raise ValueError(
                f'Fingerprint type unknown. Please choose from {", ".join(fingerprint_types)}.'
            )

    def _normalize_physicochemical_bits(self):
        """
        Normalize physicochemical bits.

        Returns
        -------
        pandas.DataFrame
            8 physicochemical features (columns) for 85 residues (rows).
        """

        if self.physicochemical is not None:

            # Make a copy of DataFrame
            normalized = self.physicochemical.copy()

            # Normalize size
            normalized["size"] = normalized["size"].apply(lambda x: self._normalize(x, 1.0, 3.0))

            # Normalize pharmacophoric features: HBD, HBA and charge
            normalized["hbd"] = normalized["hbd"].apply(lambda x: self._normalize(x, 0.0, 3.0))
            normalized["hba"] = normalized["hba"].apply(lambda x: self._normalize(x, 0.0, 2.0))
            normalized["charge"] = normalized["charge"].apply(
                lambda x: self._normalize(x, -1.0, 1.0)
            )

            # No normalization needed for aromatic and aliphatic features which are already 0 or 1

            # Normalize side chain orientation
            normalized["sco"] = normalized["sco"].apply(lambda x: self._normalize(x, 0.0, 2.0))

            # No normalization needed for exposure feature which is already between 0 and 1

            return normalized

        else:
            return None

    def _normalize_distances_bits(self):
        """
        Normalize distances bits.

        Returns
        -------
        pandas.DataFrame
            4 distance features (columns) for 85 residues (rows).
        """

        if self.distances is not None:

            # Make a copy of DataFrame
            normalized = self.distances.copy()

            # Normalize using cutoffs defined for each reference point
            normalized["distance_to_centroid"] = normalized["distance_to_centroid"].apply(
                lambda x: self._normalize(
                    x,
                    DISTANCE_CUTOFFS["distance_to_centroid"][0],
                    DISTANCE_CUTOFFS["distance_to_centroid"][1],
                )
            )
            normalized["distance_to_hinge_region"] = normalized["distance_to_hinge_region"].apply(
                lambda x: self._normalize(
                    x,
                    DISTANCE_CUTOFFS["distance_to_hinge_region"][0],
                    DISTANCE_CUTOFFS["distance_to_hinge_region"][1],
                )
            )
            normalized["distance_to_dfg_region"] = normalized["distance_to_dfg_region"].apply(
                lambda x: self._normalize(
                    x,
                    DISTANCE_CUTOFFS["distance_to_dfg_region"][0],
                    DISTANCE_CUTOFFS["distance_to_dfg_region"][1],
                )
            )
            normalized["distance_to_front_pocket"] = normalized["distance_to_front_pocket"].apply(
                lambda x: self._normalize(
                    x,
                    DISTANCE_CUTOFFS["distance_to_front_pocket"][0],
                    DISTANCE_CUTOFFS["distance_to_front_pocket"][1],
                )
            )

            return normalized

        else:
            return None

    def _normalize_moments_bits(self):
        """
        Normalize moments bits.

        Returns
        -------
        pandas.DataFrame
            3 moment features (columns) for 4 distance distributions residues (rows).
        """

        if self.moments is not None:

            # Make a copy of DataFrame
            normalized = self.moments.copy()

            # Normalize using cutoffs defined for each moment
            normalized["moment1"] = normalized["moment1"].apply(
                lambda x: self._normalize(
                    x, MOMENT_CUTOFFS["moment1"][0], MOMENT_CUTOFFS["moment1"][1]
                )
            )
            normalized["moment2"] = normalized["moment2"].apply(
                lambda x: self._normalize(
                    x, MOMENT_CUTOFFS["moment2"][0], MOMENT_CUTOFFS["moment2"][1]
                )
            )
            normalized["moment3"] = normalized["moment3"].apply(
                lambda x: self._normalize(
                    x, MOMENT_CUTOFFS["moment3"][0], MOMENT_CUTOFFS["moment3"][1]
                )
            )

            return normalized

        else:
            return None

    @staticmethod
    def _normalize(value, minimum, maximum):
        """
        Normalize a value using minimum-maximum normalization.
        Values equal or lower / greater than the minimum / maximum value are set to 0.0 / 1.0.

        Parameters
        ----------
        value : float or int
            Value to be normalized.
        minimum : float or int
            Minimum value for normalization, values equal/greater than this minimum are set to 0.0.
        maximum : float or int
            Maximum value for normalization, values equal/greater than this maximum are set to 1.0.

        Returns
        -------
        float
            Normalized value.
        """

        if minimum < value < maximum:
            return (value - minimum) / float(maximum - minimum)
        elif value <= minimum:
            return 0.0
        elif value >= maximum:
            return 1.0
        elif np.isnan(value):
            return np.nan
        else:
            raise ValueError(f"Unexpected value to be normalized: {value}")

    @staticmethod
    def _calc_moments(distances):
        """
        Calculate first, second, and third moment (mean, standard deviation, and skewness)
        for a distance distribution.

        Parameters
        ----------
        distances : pandas.DataFrame
            Distance distribution, i.e. distances (rows) from reference point (columns)
            to all representatives/points.

        Returns
        -------
        pandas.DataFrame
            First, second, and third moment (column) of distance distribution (row).
        """

        # Get first, second, and third moment (mean, standard deviation, and skewness)
        # for a distance distribution
        # Second and third moment: delta degrees of freedom = 0 (divisor N)
        if len(distances) > 0:
            m1 = distances.mean()
            m2 = distances.std(ddof=0)
            m3 = pd.Series(
                cbrt(moment(distances, moment=3, nan_policy="omit")),
                index=distances.columns.tolist(),
            )
        else:
            raise ValueError(f"No data available to calculate moments.")

        # Store all moments in DataFrame
        moments = pd.concat([m1, m2, m3], axis=1)
        moments.columns = ["moment1", "moment2", "moment3"]

        return moments
