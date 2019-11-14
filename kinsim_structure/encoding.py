"""
encoding.py

Subpocket-based structural fingerprint for kinase pocket comparison.

Handles the primary functions for the structural kinase fingerprint encoding.
"""

import datetime
import logging
from multiprocessing import cpu_count, Pool

from Bio.PDB import HSExposureCA, HSExposureCB, Vector
from Bio.PDB.Chain import Chain
from Bio.PDB import calc_angle
import nglview as nv
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import cbrt
from scipy.stats.stats import moment

from kinsim_structure.auxiliary import KlifsMoleculeLoader, PdbChainLoader
from kinsim_structure.auxiliary import center_of_mass, split_klifs_code

logger = logging.getLogger(__name__)

FEATURE_NAMES = {
    'physicochemical': 'size hbd hba charge aromatic aliphatic sco exposure'.split(),
    'distances': 'distance_to_centroid distance_to_hinge_region distance_to_dfg_region distance_to_front_pocket'.split(),
    'moments': 'moment1 moment2 moment3'.split()
}

SITEALIGN_FEATURES = pd.read_csv(Path(__file__).parent / 'data' / 'sitealign_features.csv', index_col=0)

MODIFIED_RESIDUE_CONVERSION = {
    'CAF': 'CYS',
    'CME': 'CYS',
    'CSS': 'CYS',
    'OCY': 'CYS',
    'KCX': 'LYS',
    'MSE': 'MET',
    'PHD': 'ASP',
    'PTR': 'TYR'
}

EXPOSURE_RADIUS = 12.0

N_HEAVY_ATOMS = {
    'GLY': 0,
    'ALA': 1,
    'CYS': 2,
    'SER': 2,
    'PRO': 3,
    'THR': 3,
    'VAL': 3,
    'ASN': 4,
    'ASP': 4,
    'ILE': 4,
    'LEU': 4,
    'MET': 4,
    'GLN': 5,
    'GLU': 5,
    'LYS': 5,
    'HIS': 6,
    'ARG': 7,
    'PHE': 7,
    'TYR': 8,
    'TRP': 10
}

N_HEAVY_ATOMS_CUTOFF = {  # Number of heavy atoms needed for side chain centroid calculation (>75% coverage)
    'GLY': 0,
    'ALA': 1,
    'CYS': 2,
    'SER': 2,
    'PRO': 3,
    'THR': 3,
    'VAL': 3,
    'ASN': 3,
    'ASP': 3,
    'ILE': 3,
    'LEU': 3,
    'MET': 3,
    'GLN': 4,
    'GLU': 4,
    'LYS': 4,
    'HIS': 5,
    'ARG': 6,
    'PHE': 6,
    'TYR': 6,
    'TRP': 8
}

ANCHOR_RESIDUES = {
    'hinge_region': [16, 47, 80],
    'dfg_region': [19, 24, 81],
    'front_pocket': [6, 48, 75]
}  # Are the same as in Eva's implementation

DISTANCE_CUTOFFS = {  # 99% percentile of all distances
    'distance_to_centroid': (3.05, 21.38),
    'distance_to_hinge_region': (4.10, 23.07),
    'distance_to_dfg_region': (4.62, 26.69),
    'distance_to_front_pocket': (5.46, 23.55)
}

MOMENT_CUTOFFS = {  # 99% percentile of all moments
    'moment1': (11.68, 14.14),
    'moment2': (3.29, 5.29),
    'moment3': (-1.47, 4.66)
}

# KLIFS IDs for hinge/DFG region (taken from KLIFS website)
HINGE_KLIFS_IDS = [46, 47, 48]
DFG_KLIFS_IDS = [81, 82, 83]


class FingerprintGenerator:
    """
    Generate fingerprints for multiple molecules. Uses parallel computing of fingerprint pairs.

    Attributes
    ----------
    data : dict of kinsim_structure.encoding.Fingerprint
        Fingerprints for multiple molecules.
    """

    def __init__(self):

        self.data = None

    def from_metadata(self, klifs_metadata):
        """
        Generate fingerprints for multiple molecules described in KLIFS metadata.

        Parameters
        ----------
        klifs_metadata : pandas.DataFrame
            Metadata (columns) for KLIFS molecules (rows).
        """

        # Get start time of script
        start = datetime.datetime.now()
        print(start)

        logger.info(f'Calculate fingerprints...')

        # Number of CPUs on machine
        num_cores = cpu_count() - 1
        logger.info(f'Number of cores used: {num_cores}')

        # Create pool with `num_processes` processes
        pool = Pool(processes=num_cores)

        # Get KLIFS entries as list
        entry_list = [j for i, j in klifs_metadata.iterrows()]

        # Apply function to each chunk in list
        fingerprints_list = pool.map(self._get_fingerprint, entry_list)

        # Close and join pool
        pool.close()
        pool.join()

        logger.info(f'Number of fingerprints: {len(fingerprints_list)}')

        # Transform to dict
        self.data = {
            i.molecule_code: i for i in fingerprints_list
        }

        # Get end time of script
        end = datetime.datetime.now()
        print(end)

        logger.info(start)
        logger.info(end)

    @staticmethod
    def _get_fingerprint(klifs_metadata_entry):
        """
        Get fingerprint.

        Parameters
        ----------
        klifs_metadata_entry : pandas.Series
            KLIFS metadata describing a pocket entry in the KLIFS dataset.

        Returns
        -------
        kinsim_structure.similarity.Fingerprint
            Fingerprint
        """

        try:

            fingerprint = Fingerprint()
            fingerprint.from_metadata_entry(klifs_metadata_entry)

            return fingerprint

        except Exception as e:

            logger.info(f'Molecule with empty fingerprint: {klifs_metadata_entry.molecule_code}')
            logger.error(e)

            return None


class SideChainOrientationGenerator:
    """
    Generate side chain orientations for multiple molecules. Uses parallel computing of fingerprint pairs.

    Attributes
    ----------
    data : dict of kinsim_structure.encoding.SideChainOrientationFeature
        Fingerprints for multiple molecules.
    """

    def __init__(self):

        self.data = None

    def from_metadata(self, klifs_metadata):
        """
        Generate side chain orientation features for multiple molecules described in KLIFS metadata.

        Parameters
        ----------
        klifs_metadata : pandas.DataFrame
            Metadata (columns) for KLIFS molecules (rows).
        """

        # Get start time of script
        start = datetime.datetime.now()
        print(start)

        logger.info(f'Calculate fingerprints...')

        # Number of CPUs on machine
        num_cores = cpu_count() - 1
        logger.info(f'Number of cores used: {num_cores}')

        # Create pool with `num_processes` processes
        pool = Pool(processes=num_cores)

        # Get KLIFS entries as list
        entry_list = [j for i, j in klifs_metadata.iterrows()]

        # Apply function to each chunk in list
        fingerprints_list = pool.map(self._get_sco, entry_list)

        # Close and join pool
        pool.close()
        pool.join()

        logger.info(f'Number of fingerprints: {len(fingerprints_list)}')

        # Transform to dict
        self.data = {
            i.molecule_code: i for i in fingerprints_list
        }

        # Get end time of script
        end = datetime.datetime.now()
        print(end)

        logger.info(start)
        logger.info(end)

    @staticmethod
    def _get_sco(klifs_metadata_entry):
        """
        Get side chain orientation.

        Parameters
        ----------
        klifs_metadata_entry : pandas.Series
            KLIFS metadata describing a pocket entry in the KLIFS dataset.

        Returns
        -------
        kinsim_structure.similarity.SideChainOrientationFeature
            Side chain orientation.
        """

        try:

            klifs_molecule_loader = KlifsMoleculeLoader(klifs_metadata_entry=klifs_metadata_entry)
            molecule = klifs_molecule_loader.molecule

            pdb_chain_loader = PdbChainLoader(klifs_metadata_entry)
            chain = pdb_chain_loader.chain

            feature = SideChainOrientationFeature()
            feature.from_molecule(molecule, chain)

            return feature

        except Exception as e:

            logger.info(f'Molecule with empty fingerprint: {klifs_metadata_entry.molecule_code}')
            logger.error(e)

            return None


class Fingerprint:
    """
    Kinase pocket is defined by 85 pre-aligned residues in KLIFS, which are described each with (i) 8 physicochemical
    and (ii) 4 distance features as well as (iii) the first three moments of aforementioned feature distance
    distributions. Fingerprints can consist of all or a subset of these three feature types.


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
    - Pharmacophoric features: Hydrogen bond donor, hydrogen bond acceptor, aromatic, aliphatic and charge feature
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

        self.fingerprint = {
            'physicochemical': None,
            'distances': None,
            'moments': None
        }
        self.fingerprint_normalized = {
            'physicochemical': None,
            'distances': None,
            'moments': None
        }

    @property
    def physicochemical(self):
        return self.fingerprint['physicochemical']

    @property
    def distances(self):
        return self.fingerprint['distances']

    @property
    def moments(self):
        return self.fingerprint['moments']

    @property
    def physicochemical_distances(self):
        return self._get_fingerprint('physicochemical_distances', normalized=False)

    @property
    def physicochemical_moments(self):
        return self._get_fingerprint('physicochemical_moments', normalized=False)

    @property
    def physicochemical_normalized(self):
        return self.fingerprint_normalized['physicochemical']

    @property
    def distances_normalized(self):
        return self.fingerprint_normalized['distances']

    @property
    def moments_normalized(self):
        return self.fingerprint_normalized['moments']

    @property
    def physicochemical_distances_normalized(self):
        return self._get_fingerprint('physicochemical_distances', normalized=True)

    @property
    def physicochemical_moments_normalized(self):
        return self._get_fingerprint('physicochemical_moments', normalized=True)

    def from_metadata_entry(self, klifs_metadata_entry):
        """
        Get kinase fingerprint from KLIFS metadata entry.

        Parameters
        ----------
        klifs_metadata_entry : pandas.Series
            KLIFS metadata describing a pocket entry in the KLIFS dataset.
        """

        klifs_molecule_loader = KlifsMoleculeLoader(klifs_metadata_entry=klifs_metadata_entry)
        molecule = klifs_molecule_loader.molecule

        pdb_chain_loader = PdbChainLoader(klifs_metadata_entry)
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

        self.fingerprint['physicochemical'] = physicochemical_features.features
        self.fingerprint['distances'] = spatial_features.features
        self.fingerprint['moments'] = self._calc_moments(spatial_features.features)

        self.fingerprint_normalized['physicochemical'] = self._normalize_physicochemical_bits()
        self.fingerprint_normalized['distances'] = self._normalize_distances_bits()
        self.fingerprint_normalized['moments'] = self._normalize_moments_bits()

    def _get_fingerprint(self, fingerprint_type, normalized=True):
        """
        Get fingerprint containing both physicochemical and spatial bits (available types: distances or moments).

        Parameters
        ----------
        fingerprint_type : str
            Type of fingerprint, i.e. fingerprint with physicochemical and either distances or moments bits
            (physicochemical + distances or physicochemical + moments).
        normalized : bool
            Normalized or non-normalized form of fingerprint (default: normalized).

        Returns
        -------
        dict of pandas.DataFrames
            Fingerprint containing physicochemical and spatial bits.
        """

        fingerprint_types = 'physicochemical_distances physicochemical_moments'.split()

        if fingerprint_type == 'physicochemical_distances':

            if normalized:
                return {
                    'physicochemical': self.physicochemical_normalized,
                    'distances': self.distances_normalized
                }
            else:
                return {
                    'physicochemical': self.physicochemical,
                    'distances': self.distances
                }

        elif fingerprint_type == 'physicochemical_moments':

            if normalized:
                return {
                    'physicochemical': self.physicochemical_normalized,
                    'moments': self.moments_normalized
                }
            else:
                return {
                    'physicochemical': self.physicochemical,
                    'moments': self.moments
                }
        else:
            raise ValueError(f'Fingerprint type unknown. Please choose from {", ".join(fingerprint_types)}.')

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
            normalized['size'] = normalized['size'].apply(lambda x: self._normalize(x, 1.0, 3.0))

            # Normalize pharmacophoric features: HBD, HBA and charge
            normalized['hbd'] = normalized['hbd'].apply(lambda x: self._normalize(x, 0.0, 3.0))
            normalized['hba'] = normalized['hba'].apply(lambda x: self._normalize(x, 0.0, 2.0))
            normalized['charge'] = normalized['charge'].apply(lambda x: self._normalize(x, -1.0, 1.0))

            # No normalization needed for aromatic and aliphatic features which are already 0 or 1

            # Normalize side chain orientation
            normalized['sco'] = normalized['sco'].apply(lambda x: self._normalize(x, 0.0, 2.0))

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
            normalized['distance_to_centroid'] = normalized['distance_to_centroid'].apply(
                lambda x: self._normalize(
                    x,
                    DISTANCE_CUTOFFS['distance_to_centroid'][0],
                    DISTANCE_CUTOFFS['distance_to_centroid'][1]
                )
            )
            normalized['distance_to_hinge_region'] = normalized['distance_to_hinge_region'].apply(
                lambda x: self._normalize(
                    x,
                    DISTANCE_CUTOFFS['distance_to_hinge_region'][0],
                    DISTANCE_CUTOFFS['distance_to_hinge_region'][1]
                )
            )
            normalized['distance_to_dfg_region'] = normalized['distance_to_dfg_region'].apply(
                lambda x: self._normalize(
                    x,
                    DISTANCE_CUTOFFS['distance_to_dfg_region'][0],
                    DISTANCE_CUTOFFS['distance_to_dfg_region'][1]
                )
            )
            normalized['distance_to_front_pocket'] = normalized['distance_to_front_pocket'].apply(
                lambda x: self._normalize(
                    x,
                    DISTANCE_CUTOFFS['distance_to_front_pocket'][0],
                    DISTANCE_CUTOFFS['distance_to_front_pocket'][1]
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
            normalized['moment1'] = normalized['moment1'].apply(
                lambda x: self._normalize(
                    x,
                    MOMENT_CUTOFFS['moment1'][0],
                    MOMENT_CUTOFFS['moment1'][1]
                )
            )
            normalized['moment2'] = normalized['moment2'].apply(
                lambda x: self._normalize(
                    x,
                    MOMENT_CUTOFFS['moment2'][0],
                    MOMENT_CUTOFFS['moment2'][1]
                )
            )
            normalized['moment3'] = normalized['moment3'].apply(
                lambda x: self._normalize(
                    x,
                    MOMENT_CUTOFFS['moment3'][0],
                    MOMENT_CUTOFFS['moment3'][1]
                )
            )

            return normalized

        else:
            return None

    @staticmethod
    def _normalize(value, minimum, maximum):
        """
        Normalize a value using minimum-maximum normalization. Values equal or lower / greater than the minimum /
        maximum value are set to 0.0 / 1.0.

        Parameters
        ----------
        value : float or int
            Value to be normalized.
        minimum : float or int
            Minimum value for normalization, values equal or greater than this minimum are set to 0.0.
        maximum : float or int
            Maximum value for normalization, values equal or greater than this maximum are set to 1.0.

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
            raise ValueError(f'Unexpected value to be normalized: {value}')

    @staticmethod
    def _calc_moments(distances):
        """
        Calculate first, second, and third moment (mean, standard deviation, and skewness) for a distance distribution.

        Parameters
        ----------
        distances : pandas.DataFrame
            Distance distribution, i.e. distances (rows) from reference point (columns) to all representatives/points.
        Returns
        -------
        pandas.DataFrame
            First, second, and third moment (column) of distance distribution (row).
        """

        # Get first, second, and third moment (mean, standard deviation, and skewness) for a distance distribution
        # Second and third moment: delta degrees of freedom = 0 (divisor N)
        if len(distances) > 0:
            m1 = distances.mean()
            m2 = distances.std(ddof=0)
            m3 = pd.Series(
                cbrt(
                    moment(
                        distances,
                        moment=3,
                        nan_policy='omit'
                    )
                ),
                index=distances.columns.tolist()
            )
        else:
            raise ValueError(f'No data available to calculate moments.')

        # Store all moments in DataFrame
        moments = pd.concat([m1, m2, m3], axis=1)
        moments.columns = ['moment1', 'moment2', 'moment3']

        return moments


class PhysicoChemicalFeatures:
    """
    Physicochemical features for each residue in the KLIFS-defined kinase binding site of 85 pre-aligned residues.

    Physicochemical properties:
    - Size
    - Pharmacophoric features: Hydrogen bond donor, hydrogen bond acceptor, aromatic, aliphatic and charge feature
    - Side chain orientation
    - Half sphere exposure

    Attributes
    ----------
    features : pandas.DataFrame
        8 features (columns) for 85 residues (rows).
    """

    def __init__(self):

        self.features = None

    def from_molecule(self, molecule, chain):
        """
        Get physicochemical properties for each residue of a molecule.

        Parameters
        ----------
        molecule : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
            Content of mol2 or pdb file as BioPandas object.
        chain : Bio.PDB.Chain.Chain
            Chain from PDB file.
        """

        pharmacophore_size = PharmacophoreSizeFeatures()
        pharmacophore_size.from_molecule(molecule)

        side_chain_orientation = SideChainOrientationFeature()
        side_chain_orientation.from_molecule(molecule, chain)

        exposure = ExposureFeature()
        exposure.from_molecule(molecule, chain, radius=12.0)

        # Concatenate all physicochemical features
        physicochemical_features = pd.concat(
            [
                pharmacophore_size.features,
                side_chain_orientation.features,
                exposure.features
            ],
            axis=1
        )

        # Bring all fingerprints to same dimensions (i.e. add currently missing residues in DataFrame)
        empty_df = pd.DataFrame([], index=range(1, 86))
        physicochemical_features = pd.concat([empty_df, physicochemical_features], axis=1)

        # Set all None to nan
        physicochemical_features.fillna(value=pd.np.nan, inplace=True)

        self.features = physicochemical_features


class SpatialFeatures:
    """
    Spatial features for each residue in the KLIFS-defined kinase binding site of 85 pre-aligned residues.

    Spatial properties:
    Distance of each residue to 4 reference points:
    - Binding site centroid
    - Hinge region
    - DFG region
    - Front pocket

    Attributes
    ----------
    reference_points : pandas.DataFrame
        Coordiantes (rows) for 4 reference points (columns).
    features : pandas.DataFrame
        4 features (columns) for 85 residues (rows).
    """

    def __init__(self):

        self.reference_points = None
        self.features = None

    def from_molecule(self, molecule):
        """
        Get spatial properties for each residue of a molecule.

        Parameters
        ----------
        molecule : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
            Content of mol2 or pdb file as BioPandas object.
        """

        # Get reference points
        self.reference_points = self.get_reference_points(molecule)

        # Get all residues' CA atoms in molecule (set KLIFS position as index)
        residues_ca = molecule.df[molecule.df.atom_name == 'CA']['klifs_id x y z'.split()]
        residues_ca.set_index('klifs_id', drop=True, inplace=True)

        distances = {}

        for name, coord in self.reference_points.items():

            # If any reference points coordinate is None, set also distance to None

            if coord.isna().any():
                distances[f'distance_to_{name}'] = None
            else:
                distance = (residues_ca - coord).transpose().apply(lambda x: np.linalg.norm(x))
                distance.rename(name, inplace=True)
                distances[f'distance_to_{name}'] = np.round(distance, 2)

        spatial_features = pd.DataFrame.from_dict(distances)

        # Bring all fingerprints to same dimensions (i.e. add currently missing residues in DataFrame)
        empty_df = pd.DataFrame([], index=range(1, 86))
        spatial_features = pd.concat([empty_df, spatial_features], axis=1)

        # Set all None to nan
        spatial_features.fillna(value=pd.np.nan, inplace=True)

        self.features = spatial_features

    def get_reference_points(self, molecule):
        """
        Get reference points of a molecule, i.e. the binding site centroid, hinge region, DFG region and front pocket.

        Parameters
        ----------
        molecule : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
            Content of mol2 or pdb file as BioPandas object.

        Returns
        -------
        pandas.DataFrame
            Coordiantes (rows) for 4 reference points (columns).
        """

        reference_points = dict()

        # Calculate centroid-based reference point:
        # Calculate mean of all CA atoms
        reference_points['centroid'] = molecule.df[molecule.df.atom_name == 'CA']['x y z'.split()].mean()

        # Calculate anchor-based reference points:
        # Get anchor atoms for each anchor-based reference point
        anchors = self.get_anchor_atoms(molecule)

        for reference_point_name, anchor_atoms in anchors.items():

            # If any anchor atom None, set also reference point coordinates to None
            if anchor_atoms.isna().values.any():
                reference_points[reference_point_name] = [None, None, None]
            else:
                reference_points[reference_point_name] = anchor_atoms.mean()

        return pd.DataFrame.from_dict(reference_points)

    @staticmethod
    def get_anchor_atoms(molecule):
        """
        For each anchor-based reference points (i.e. hinge region, DFG region and front pocket)
        get the three anchor (i.e. CA) atoms.

        Parameters
        ----------
        molecule : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
            Content of mol2 or pdb file as BioPandas object.

        Returns
        -------
        dict of pandas.DataFrames
            Coordinates (x, y, z) of the three anchor atoms (rows=anchor residue KLIFS ID x columns=coordinates) for
            each of the anchor-based reference points.
        """

        anchors = dict()

        # Calculate anchor-based reference points
        # Process each reference point: Collect anchor residue atoms and calculate their mean
        for reference_point_name, anchor_klifs_ids in ANCHOR_RESIDUES.items():

            anchor_atoms = []

            # Process each anchor residue: Get anchor atom
            for anchor_klifs_id in anchor_klifs_ids:

                # Select anchor atom, i.e. CA atom of KLIFS ID (anchor residue)
                anchor_atom = molecule.df[
                    (molecule.df.klifs_id == anchor_klifs_id) &
                    (molecule.df.atom_name == 'CA')
                ]

                # If this anchor atom exists, append to anchor atoms list
                if len(anchor_atom) == 1:
                    anchor_atom.set_index('klifs_id', inplace=True)
                    anchor_atom.index.name = None
                    anchor_atoms.append(anchor_atom[['x', 'y', 'z']])

                # If this anchor atom does not exist, do workarounds
                elif len(anchor_atom) == 0:

                    # Do residues (and there CA atoms) exist next to anchor residue?
                    atom_before = molecule.df[
                        (molecule.df.klifs_id == anchor_klifs_id - 1) &
                        (molecule.df.atom_name == 'CA')
                        ]
                    atom_after = molecule.df[
                        (molecule.df.klifs_id == anchor_klifs_id + 1) &
                        (molecule.df.atom_name == 'CA')
                        ]
                    atom_before.set_index('klifs_id', inplace=True, drop=False)
                    atom_after.set_index('klifs_id', inplace=True, drop=False)

                    # If both neighboring CA atoms exist, get their mean as alternative anchor atom
                    if len(atom_before) == 1 and len(atom_after) == 1:
                        anchor_atom_alternative = pd.concat([atom_before, atom_after])[['x', 'y', 'z']].mean()
                        anchor_atom_alternative = pd.DataFrame({anchor_klifs_id: anchor_atom_alternative}).transpose()
                        anchor_atoms.append(anchor_atom_alternative)

                    elif len(atom_before) == 1 and len(atom_after) == 0:
                        atom_before.set_index('klifs_id', inplace=True)
                        anchor_atoms.append(atom_before[['x', 'y', 'z']])

                    elif len(atom_after) == 1 and len(atom_before) == 0:
                        atom_after.set_index('klifs_id', inplace=True)
                        anchor_atoms.append(atom_after[['x', 'y', 'z']])

                    else:
                        atom_missing = pd.DataFrame.from_dict(
                            {anchor_klifs_id: [None, None, None]},
                            orient='index',
                            columns='x y z'.split()
                        )
                        anchor_atoms.append(atom_missing)

                # If there are several anchor atoms, something's wrong...
                else:
                    raise ValueError(f'Too many anchor atoms for'
                                     f'{molecule.code}, {reference_point_name}, {anchor_klifs_id}: '
                                     f'{len(anchor_atom)} (one atom allowed).')

            anchors[reference_point_name] = pd.concat(anchor_atoms)

        return anchors

    @staticmethod
    def save_cgo_refpoints(klifs_metadata_entry, output_path):
        """
        Save CGO PyMol file showing a kinase with anchor residues, reference points and highlighted hinge and DFG
        region.

        Parameters
        ----------
        klifs_metadata_entry : pandas.Series
            KLIFS metadata describing a pocket entry in the KLIFS dataset.
        output_path : str or pathlib.Path
            Path to directory where data file should be saved.
        """

        output_path = Path(output_path)

        # PyMol sphere colors (for reference points)
        sphere_colors = {
            'centroid': [1.0, 0.65, 0.0],  # orange
            'hinge_region': [1.0, 0.0, 1.0],  # magenta
            'dfg_region': [0.25, 0.41, 0.88],  # skyblue
            'front_pocket': [0.0, 1.0, 0.0]  # green
        }

        # Load molecule from KLIFS metadata entry
        klifs_molecule_loader = KlifsMoleculeLoader(klifs_metadata_entry=klifs_metadata_entry)
        molecule = klifs_molecule_loader.molecule

        # Path to molecule file
        mol2_path = klifs_molecule_loader.file_from_metadata_entry(klifs_metadata_entry)

        # Output path
        Path(output_path).mkdir(parents=True, exist_ok=True)

        # Mol2 residue IDs for hinge/DFG region
        hinge_mol2_ids = molecule.df[molecule.df.klifs_id.isin(HINGE_KLIFS_IDS)].res_id.unique()
        dfg_mol2_ids = molecule.df[molecule.df.klifs_id.isin(DFG_KLIFS_IDS)].res_id.unique()

        # Get reference points and anchor atoms (coordinates)
        space = SpatialFeatures()
        space.from_molecule(molecule)
        ref_points = space.reference_points.transpose()
        anchor_atoms = space.get_anchor_atoms(molecule)

        # Drop missing reference points and anchor atoms
        ref_points.dropna(axis=0, how='any', inplace=True)
        for ref_point_name, anchor_atoms_per_ref_point in anchor_atoms.items():
            anchor_atoms_per_ref_point.dropna(axis=0, how='any', inplace=True)

        # Collect all text lines to be written to file
        lines = []

        # Set descriptive PyMol object name for reference points
        obj_name = f'refpoints_{molecule.code[6:]}'

        # Imports
        lines.append('from pymol import *')
        lines.append('import os')
        lines.append('from pymol.cgo import *\n')

        # Load pocket structure
        lines.append(f'cmd.load("{mol2_path}", "pocket_{molecule.code[6:]}")\n')
        lines.append(f'cmd.show("cartoon", "pocket_{molecule.code[6:]}")')
        lines.append(f'cmd.hide("lines", "pocket_{molecule.code[6:]}")')
        lines.append(f'cmd.color("gray", "pocket_{molecule.code[6:]}")\n')
        lines.append(f'cmd.set("cartoon_transparency", 0.5, "pocket_{molecule.code[6:]}")')
        lines.append(f'cmd.set("opaque_background", "off")\n')

        # Color hinge and DFG region
        lines.append(f'cmd.set_color("hinge_color", {sphere_colors["hinge_region"]})')
        lines.append(f'cmd.set_color("dfg_color", {sphere_colors["dfg_region"]})')
        lines.append(f'cmd.color("hinge_color", "pocket_{molecule.code[6:]} and resi {"+".join([str(i) for i in hinge_mol2_ids])}")')
        lines.append(f'cmd.color("dfg_color", "pocket_{molecule.code[6:]} and resi {"+".join([str(i) for i in dfg_mol2_ids])}")\n')

        # Add spheres, i.e. reference points and anchor atoms
        lines.append(f'obj_{obj_name} = [\n')  # Variable cannot start with digit, thus add prefix obj_

        # Reference points
        for ref_point_name, ref_point in ref_points.iterrows():

            # Set and write sphere color to file
            lines.append(
                f'\tCOLOR, '
                f'{str(sphere_colors[ref_point_name][0])}, '
                f'{str(sphere_colors[ref_point_name][1])}, '
                f'{str(sphere_colors[ref_point_name][2])},'
            )

            # Write reference point coordinates and size to file
            lines.append(
                f'\tSPHERE, '
                f'{str(ref_point["x"])}, '
                f'{str(ref_point["y"])}, '
                f'{str(ref_point["z"])}, '
                f'{str(1)},'
            )

            # Write anchor atom coordinates and size to file
            if ref_point_name != 'centroid':
                for anchor_atom_index, anchor_atom in anchor_atoms[ref_point_name].iterrows():
                    lines.append(
                        f'\tSPHERE, '
                        f'{str(anchor_atom["x"])}, '
                        f'{str(anchor_atom["y"])}, '
                        f'{str(anchor_atom["z"])}, '
                        f'{str(0.5)},'
                    )

        # Write command to file that will load the reference points as PyMol object
        lines.append(f']\n')

        # Add KLIFS IDs to CA atoms as labels

        for res_id, klifs_id in zip(molecule.df.res_id.unique(), molecule.df.klifs_id.unique()):
            lines.append(
                f'cmd.label(selection="pocket_{molecule.code[6:]} and name CA and resi {res_id}", expression="\'{klifs_id}\'")'

            )

        lines.append(f'\ncmd.load_cgo(obj_{obj_name}, "{obj_name}")')

        with open(output_path / f'refpoints_{molecule.code[6:]}.py', 'w') as f:
            f.write('\n'.join(lines))

        # In PyMol enter the following to save png
        # PyMOL > ray 900, 900
        # PyMOL > save refpoints.png


class SideChainOrientationFeature:
    """
    Side chain orientation for each residue in the KLIFS-defined kinase binding site of 85 pre-aligned residues.
    Side chain orientation of a residue is defined by the vertex angle formed by (i) the residue's CA atom,
    (ii) the residue's side chain centroid, and (iii) the pocket centroid (calculated based on its CA atoms), whereby
    the CA atom forms the vertex.

    Attributes
    ----------
    molecule_code : str
        KLIFS code.
    features : pandas.DataFrame
        1 feature, i.e. side chain orientation, (column) for 85 residues (rows).
    features_verbose : pandas.DataFrame
        Feature, Ca, Cb, and centroid vectors as well as metadata information (columns) for 85 residues (row).
    vector_pocket_centroid : Bio.PDB.Vector.Vector
        Vector to pocket centroid.
    """

    def __init__(self):

        self.molecule_code = None
        self.features = None
        self.features_verbose = None
        self.vector_pocket_centroid = None  # Necessary to not calculate pocket centroid for each residue again

    def from_molecule(self, molecule, chain):
        """
        Get side chain orientation for each residue in a molecule (pocket).
        Side chain orientation of a residue is defined by the vertex angle formed by (i) the residue's CA atom,
        (ii) the residue's side chain centroid, and (iii) the pocket centroid (calculated based on its CA atoms),
        whereby the CA atom forms the vertex.

        Parameters
        ----------
        molecule : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
            Content of mol2 or pdb file as BioPandas object.
        chain : Bio.PDB.Chain.Chain
            Chain from PDB file.
        """

        self.molecule_code = molecule.code

        # Get pocket residues
        pocket_residues = self._get_pocket_residues(molecule, chain)

        # Get vectors (for each residue CA atoms, side chain centroid, pocket centroid)
        pocket_vectors = self._get_pocket_vectors(pocket_residues, chain)

        # Get vertex angles (for each residue, vertex angle between aforementioned points)
        vertex_angles = self._get_vertex_angles(pocket_vectors)

        # Transform vertex angles into categories
        categories = self._get_categories(vertex_angles)

        # Store categories
        self.features = categories
        # Store categories, vertex angles plus vectors and metadata
        self.features_verbose = pd.concat([pocket_vectors, vertex_angles, categories], axis=1)

    @staticmethod
    def _get_pocket_residues(molecule, chain):
        """
        Get KLIFS pocket residues from PDB structural data: Bio.PDB.Residue.Residue plus metadata, i.e. KLIFS residue
        ID, PDB residue ID, and residue name for all pocket residues.

        Parameters
        ----------
        molecule : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
            Content of mol2 or pdb file as BioPandas object.
        chain : Bio.PDB.Chain.Chain
            Chain from PDB file.

        Returns
        -------
        pandas.DataFrame
            Pocket residues: Bio.PDB.Residue.Residue plus metadata, i.e. KLIFS residue ID, PDB residue ID, and residue
            name (columns) for all pocket residues (rows).
        """

        # Get KLIFS pocket metadata, e.g. PDB residue IDs from mol2 file (DataFrame)
        pocket_residues = pd.DataFrame(
            molecule.df.groupby('klifs_id res_id res_name'.split()).groups.keys(),
            columns='klifs_id res_id res_name'.split()

        )
        pocket_residues.set_index('klifs_id', drop=False, inplace=True)

        # Select residues from chain based on PDB residue IDs and add to DataFrame
        pocket_residues_list = []

        for residue_id in pocket_residues.res_id:

            try:  # Standard amino acids
                pocket_residue = chain[residue_id]

            except KeyError:  # Non-standard amino acid
                pocket_residue = [i for i in chain.get_list() if i.get_id()[1] == residue_id][0]

            pocket_residues_list.append(pocket_residue)

        pocket_residues['pocket_residues'] = pocket_residues_list

        return pocket_residues

    def _get_pocket_vectors(self, pocket_residues, chain):
        """
        Get vectors to CA, residue side chain centroid, and pocket centroid.

        Parameters
        ----------
        pocket_residues : pandas.DataFrame
            Pocket residues: Bio.PDB.Residue.Residue plus metadata, i.e. KLIFS residue ID, PDB residue ID, and residue
            name (columns) for all pocket residues (rows).
        chain : Bio.PDB.Chain.Chain
            Chain from PDB file.

        Returns
        -------
        pandas.DataFrame
            Vectors to CA, residue side chain centroid, and pocket centroid for each residue of a molecule, alongside
            with metadata on KLIFS residue ID, PDB residue ID, and residue name.
        """

        # Save here values per residue
        data = []

        # Calculate pocket centroid
        if not self.vector_pocket_centroid:
            self.vector_pocket_centroid = self._get_pocket_centroid(pocket_residues)

        # Calculate CA atom and side chain centroid
        for residue in pocket_residues.pocket_residues:

            vector_ca = self._get_ca(residue)
            vector_side_chain_centroid = self._get_side_chain_centroid(residue, chain)

            data.append([vector_ca, vector_side_chain_centroid, self.vector_pocket_centroid])

        data = pd.DataFrame(
            data,
            index=pocket_residues.klifs_id,
            columns='ca side_chain_centroid pocket_centroid'.split()
        )

        metadata = pocket_residues['klifs_id res_id res_name'.split()]

        if len(metadata) != len(data):
            raise ValueError(f'DataFrames to be concatenated must be of same length: '
                             f'Metadata has {len(metadata)} rows, CA/CB/centroid data has {len(data)} rows.')

        return pd.concat([metadata, data], axis=1)

    @staticmethod
    def _get_vertex_angles(pocket_vectors):
        """
        Get vertex angles for residues' side chain orientations to the molecule (pocket) centroid.
        Side chain orientation of a residue is defined by the vertex_angle formed by (i) the residue's CB atom,
        (ii) the residue's side chain centroid, and (iii) the pocket centroid (calculated based on its CA atoms),
        whereby the CA atom forms the vertex.

        Parameters
        ----------
        pocket_vectors : pandas.DataFrame
            Vectors to CA, residue side chain centroid, and pocket centroid for each residue of a molecule, alongside
            with metadata on KLIFS residue ID, PDB residue ID, and residue name (columns) for 85 pocket residues.

        Returns
        -------
        pandas.DataFrame
            Vertex angles (column) for up to 85 residues (rows).
        """

        vertex_angles = []

        for index, row in pocket_vectors.iterrows():

            # If all three vectors available, calculate vertex_angle - otherwise set vertex_angle to None

            if row.ca and row.side_chain_centroid and row.pocket_centroid:
                # Calculate vertex vertex_angle: CA atom is vertex
                vertex_angle = np.degrees(
                    calc_angle(
                        row.side_chain_centroid, row.ca, row.pocket_centroid
                    )
                )
                vertex_angles.append(vertex_angle.round(2))
            else:
                vertex_angles.append(None)

        # Cast to DataFrame
        vertex_angles = pd.DataFrame(
            vertex_angles,
            index=pocket_vectors.klifs_id,
            columns=['vertex_angle']
        )

        return vertex_angles

    def _get_categories(self, vertex_angles):
        """
        Get side chain orientation category for pocket residues based on their side chain orientation vertex angles.
        The side chain orientation towards the pocket is described with the following three categories:
        Inwards (0.0), intermediate (1.0), and outwards (2.0).

        Parameters
        ----------
        vertex_angles : pandas.DataFrame
            Vertex angles (column) for up to 85 residues (rows).

        Returns
        -------
        pandas.DataFrame
            Side chain orientation categories (column) for up to 85 residues (rows).
        """

        if 'vertex_angle' not in vertex_angles.columns:
            raise ValueError('Input DataFrame needs column with name "vertex_angle".')

        categories = [
            self._get_category_from_vertex_angle(vertex_angle) for vertex_angle in vertex_angles.vertex_angle
        ]

        # Cast from Series to DataFrame and set column name for feature
        categories = pd.DataFrame(
            categories,
            index=vertex_angles.index,
            columns=['sco']
        )

        return categories

    def _get_category_from_vertex_angle(self, vertex_angle):
        """
        Transform a given vertex angle into a category value, which defines the side chain orientation towards the
        pocket: Inwards (category 0.0), intermediate (category 1.0), and outwards (category 2.0).

        Parameters
        ----------
        vertex_angle : float
            Vertex angle between a residue's CA atom (vertex), side chain centroid and pocket centroid. Ranges between
            0.0 and 180.0.

        Returns
        -------
        float
            Side chain orientation towards pocket: Inwards (category 0.0), intermediate (category 1.0), and outwards
            (category 2.0).
        """

        if 0.0 <= vertex_angle <= 45.0:  # Inwards
            return 0.0
        elif 45.0 < vertex_angle <= 90.0:  # Intermediate
            return 1.0
        elif 90.0 < vertex_angle <= 180.0:  # Outwards
            return 2.0
        elif np.isnan(vertex_angle):
            return np.nan
        else:
            raise ValueError(f'Molecule {self.molecule_code}: Unknown vertex angle {vertex_angle}. '
                             f'Only values between 0.0 and 180.0 allowed.')

    def _get_ca(self, residue):
        """
        Get residue's CA atom.

        Parameters
        ----------
        residue : Bio.PDB.Residue.Residue
            Residue.

        Returns
        -------
        Bio.PDB.vectors.Vector or None
            Residue's CA vector.
        """

        atom_names = [atoms.fullname for atoms in residue.get_atoms()]

        # Set CA atom

        if 'CA' in atom_names:
            vector_ca = residue['CA'].get_vector()
        else:
            logger.info(f'{self.molecule_code}: SCO: CA atom: Missing in {residue}.')
            vector_ca = None

        return vector_ca

    def _get_side_chain_centroid(self, residue, chain):
        """
        Get residue's side chain centroid.

        Parameters
        ----------
        residue : Bio.PDB.Residue.Residue
            Residue.
        chain : Bio.PDB.Chain.Chain
            Chain from PDB file.

        Returns
        -------
        Bio.PDB.vectors.Vector or None
            Residue's side chain centroid.
        """

        # Select only atoms that are
        # - not part of the backbone
        # - not oxygen atoms (OXT) on the terminal carboxyl group
        # - not H atoms

        selected_atoms = [
            atom for atom in residue.get_atoms() if
            (atom.fullname not in 'N CA C O OXT'.split()) & (not atom.get_id().startswith('H'))
        ]

        n_atoms = len(selected_atoms)

        # Set side chain centroid
        exception = None

        if residue.id[0] == ' ':  # Standard residues

            n_atoms_cutoff = N_HEAVY_ATOMS_CUTOFF[residue.get_resname()]

            if residue.get_resname() == 'GLY':  # GLY residue

                side_chain_centroid = self._get_pcb_from_residue(residue, chain)

                if side_chain_centroid is None:
                    exception = 'GLY - None'

            elif residue.get_resname() == 'ALA':  # ALA residue

                try:
                    side_chain_centroid = residue['CB'].get_vector()

                except KeyError:
                    side_chain_centroid = self._get_pcb_from_residue(residue, chain)

                    if side_chain_centroid is not None:
                        exception = 'ALA - pCB atom'
                    else:
                        exception = 'ALA - None'

            elif n_atoms >= n_atoms_cutoff:  # Other standard residues with enough side chain atoms

                side_chain_centroid = Vector(center_of_mass(selected_atoms, geometric=True))

            else:  # Other standard residues with too few side chain atoms

                try:
                    side_chain_centroid = residue['CB'].get_vector()
                    exception = f'Standard residue - CB atom, only {n_atoms}/{n_atoms_cutoff} residues'

                except KeyError:
                    side_chain_centroid = self._get_pcb_from_residue(residue, chain)

                    if side_chain_centroid is not None:
                        exception = f'Standard residue - pCB atom, only {n_atoms}/{n_atoms_cutoff} residues'
                    else:
                        exception = f'Standard residue - None, only {n_atoms}/{n_atoms_cutoff} residues'

        else:  # Non-standard residues

            if n_atoms > 0:
                side_chain_centroid = Vector(center_of_mass(selected_atoms, geometric=True))
                exception = f'Non-standard residue - centroid of {n_atoms} atoms'
            else:
                side_chain_centroid = None
                exception = 'Non-standard residue - None'

        if exception:
            logger.info(f'{self.molecule_code}: SCO: Side chain centroid for '
                        f'residue {residue.get_resname()}, {residue.id} with {n_atoms} atoms is: '
                        f'{exception}.')

        return side_chain_centroid

    def _get_pocket_centroid(self, pocket_residues):
        """
        Get centroid of pocket CA atoms.

        Parameters
        ----------
        pocket_residues : pandas.DataFrame
            Pocket residues: Bio.PDB.Residue.Residue plus metadata, i.e. KLIFS residue ID, PDB residue ID, and residue
            name (columns) for all pocket residues (rows).

        Returns
        -------
        Bio.PDB.vectors.Vector or None
            Pocket centroid.
        """

        # Initialize list for all CA atoms in pocket
        ca_vectors = []

        # Log missing CA atoms
        ca_atoms_missing = []

        for residue in pocket_residues.pocket_residues:
            try:
                ca_vectors.append(residue['CA'])
            except KeyError:
                ca_atoms_missing.append(residue)

        if len(ca_atoms_missing) > 0:
            logger.info(f'{self.molecule_code}: SCO: Pocket centroid: '
                        f'{len(ca_atoms_missing)} missing CA atom(s): {ca_atoms_missing}')

        try:
            return Vector(center_of_mass(ca_vectors, geometric=True))
        except ValueError:
            logger.info(f'{self.molecule_code}: SCO: Pocket centroid: '
                        f'Cannot be calculated. {len(ca_vectors)} CA atoms available.')
            return None

    @staticmethod
    def _get_pcb_from_gly(residue):
        """
        Get pseudo-CB atom coordinate for GLY residue.

        Parameters
        ----------
        residue : Bio.PDB.Residue.Residue
            Residue.

        Returns
        -------
        Bio.PDB.vectors.Vector or None
            Pseudo-CB atom vector for GLY centered at CA atom (= pseudo-CB atom coordinate).
        """

        if residue.get_resname() == 'GLY':

            # Get pseudo-CB for GLY (vector centered at origin)
            chain = Chain(id='X')  # Set up chain instance
            pcb = HSExposureCB(chain)._get_gly_cb_vector(residue)

            if pcb is None:
                return None

            else:
                # Center pseudo-CB vector at CA atom to get pseudo-CB coordinate
                ca = residue['CA'].get_vector()
                ca_pcb = ca + pcb
                return ca_pcb

        else:
            raise ValueError(f'Residue must be GLY, but is {residue.get_resname()}.')

    def _get_pcb_from_residue(self, residue, chain):
        """
        Get pseudo-CB atom coordinate for non-GLY residue.

        Parameters
        ----------
        residue : Bio.PDB.Residue.Residue
            Residue.
        chain : Bio.PDB.Chain.Chain
            Chain from PDB file.

        Returns
        -------
        Bio.PDB.vectors.Vector or None
            Pseudo-CB atom vector for residue centered at CA atom (= pseudo-CB atom coordinate).
        """

        if residue.get_resname() == 'GLY':
            return self._get_pcb_from_gly(residue)

        else:

            # Get residue before and after input residue (if not available return None)
            try:
                # Non-standard residues will throw KeyError here but I am ok with not considering those cases, since
                # hetero residues are not always enumerated correctly
                # (sometimes non-standard residues are named e.g. "_180" in PDB files)
                residue_before = chain[residue.id[1] - 1]
                residue_after = chain[residue.id[1] + 1]

            except KeyError:  # If residue before or after do not exist
                return None

            # Get pseudo-CB for non-GLY residue
            pcb = HSExposureCA(Chain(id='X'))._get_cb(residue_before, residue, residue_after)

            if pcb is None:  # If one or more of the three residues have no CA
                return None

            else:
                # Center pseudo-CB vector at CA atom to get pseudo-CB coordinate
                ca = residue['CA'].get_vector()
                ca_pcb = ca + pcb[0]
                return ca_pcb

    def save_as_cgo(self, output_path):
        """
        Save CA atom, side chain centroid and pocket centroid as spheres and label CA atom with side chain orientation
        vertex angle value to PyMol cgo file.

        Parameters
        ----------
        output_path : pathlib.Path or str
            Path to output directory.
        """

        # Get molecule and molecule code
        code = split_klifs_code(self.molecule_code)

        # Get pocket residues
        pocket_residues_ids = list(self.features_verbose.res_id)

        # List contains lines for python script
        lines = [f'from pymol import *', f'import os', f'from pymol.cgo import *\n']

        # Fetch PDB, remove solvent, remove unnecessary chain(s) and residues
        lines.append(f'cmd.fetch("{code["pdb_id"]}")')
        lines.append(f'cmd.remove("solvent")')
        if code["chain"]:
            lines.append(f'cmd.remove("{code["pdb_id"]} and not chain {code["chain"]}")')
        lines.append(f'cmd.remove("all and not (resi {"+".join([str(i) for i in pocket_residues_ids])})")')
        lines.append(f'')

        # Set sphere color and size
        sphere_colors = {
            'ca': [0.0, 1.0, 0.0],  # Green
            'side_chain_centroid': [1.0, 0.0, 0.0],  # Red
            'pocket_centroid': [0.0, 0.0, 1.0],  # Blue
        }
        sphere_size = {
            'ca': str(0.2),
            'side_chain_centroid': str(0.2),
            'pocket_centroid': str(1)
        }

        # Collect all PyMol objects here (in order to group them after loading them to PyMol)
        obj_names = []
        obj_angle_names = []

        for index, row in self.features_verbose.iterrows():

            # Set PyMol object name: residue ID
            obj_name = f'{row.res_id}'
            obj_names.append(obj_name)

            if not np.isnan(row.sco):

                # Add angle to CA atom in the form of a label
                obj_angle_name = f'angle_{row.res_id}'
                obj_angle_names.append(obj_angle_name)

                lines.append(
                    f'cmd.pseudoatom(object="angle_{row.res_id}", '
                    f'pos=[{str(row.ca[0])}, {str(row.ca[1])}, {str(row.ca[2])}], '
                    f'label={str(round(row.sco, 1))})'
                )

            vectors = {
                'ca': row.ca,
                'side_chain_centroid': row.side_chain_centroid,
                'pocket_centroid': row.pocket_centroid
            }

            # Write all spheres for current residue in cgo format
            lines.append(f'obj_{obj_name} = [')  # Variable cannot start with digit, thus add prefix obj_

            # For each reference point, write sphere color, coordinates and size to file
            for key, vector in vectors.items():

                if vector:
                    # Set sphere color
                    sphere_color = list(sphere_colors[key])

                    # Write sphere a) color and b) coordinates and size to file
                    lines.extend(
                        [
                            f'\tCOLOR, {str(sphere_color[0])}, {str(sphere_color[1])}, {str(sphere_color[2])},',
                            f'\tSPHERE, {str(vector[0])}, {str(vector[1])}, {str(vector[2])}, {sphere_size[key]},'
                        ]
                    )

            # Load the spheres as PyMol object
            lines.extend(
                [
                    f']',
                    f'cmd.load_cgo(obj_{obj_name}, "{obj_name}")',
                    ''
                ]

            )
        # Group all objects to one group
        lines.append(f'cmd.group("{self.molecule_code.replace("/", "_")}", "{" ".join(obj_names + obj_angle_names)}")')

        cgo_path = Path(output_path) / f'side_chain_orientation_{self.molecule_code.split("/")[1]}.py'
        with open(cgo_path, 'w') as f:
            f.write('\n'.join(lines))

        # In PyMol enter the following to save png
        # PyMOL > ray 900, 900
        # PyMOL > save refpoints.png

    def show_in_nglviewer(self):

        # Get molecule and molecule code
        code = split_klifs_code(self.molecule_code)

        pdb_id = code['pdb_id']
        chain = code['chain']

        viewer = nv.show_pdbid(pdb_id, default=False)
        viewer.clear()
        viewer.add_cartoon(selection=f':{chain}', assembly='AU')
        viewer.center(selection=f':{chain}')

        # Representation parameters
        sphere_radius = {
            'ca': 0.3,
            'side_chain_centroid': 0.3,
            'pocket_centroid': 1
        }

        colors = {
            'ca': [0, 1, 0],
            'side_chain_centroid': [1, 0, 0],
            'pocket_centroid': [0, 0, 1]
        }

        # Show side chain orientation feature per residue
        for index, row in self.features_verbose.iterrows():

            res_id = row.res_id

            viewer.add_representation(repr_type='line', selection=f'{res_id}:{chain}')
            viewer.add_label(selection=f'{res_id}:{chain}.CA')  # TODO: Add angles as label here

            if row.ca:
                ca = list(row.ca.get_array())
                viewer.shape.add_sphere(ca, colors['ca'], sphere_radius['ca'])

            if row.side_chain_centroid:
                side_chain_centroid = list(row.side_chain_centroid.get_array())
                viewer.shape.add_sphere(side_chain_centroid, colors['side_chain_centroid'], sphere_radius['side_chain_centroid'])

            if row.pocket_centroid:
                pocket_centroid = list(row.pocket_centroid.get_array())
                viewer.shape.add_sphere(pocket_centroid, colors['pocket_centroid'], sphere_radius['pocket_centroid'])

        viewer.gui_style = 'ngl'

        return viewer


class ExposureFeature:
    """
    Exposure for each residue in the KLIFS-defined kinase binding site of 85 pre-aligned residues.
    Exposure of a residue describes the ratio of CA atoms in the upper sphere half around the CA-CB vector
    divided by the all CA atoms (given a sphere radius).

    Attributes
    ----------
    features : pandas.DataFrame
        1 feature (columns) for 85 residues (rows).

    References
    ----------
    Hamelryck, "An Amino Acid Has Two Sides: ANew 2D Measure Provides a Different View of Solvent Exposure",
    PROTEINS: Structure, Function, and Bioinformatics 59:38–48 (2005).
    """

    def __init__(self):

        self.features = None
        self.features_verbose = None

    def from_molecule(self, molecule, chain, radius=12.0):
        """
        Get exposure for each residue of a molecule.

        Parameters
        ----------
        molecule : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
            Content of mol2 or pdb file as BioPandas object.
        chain : Bio.PDB.Chain.Chain
            Chain from PDB file.
        radius : float
            Sphere radius to be used for half sphere exposure calculation.
        """

        # Get exposure data for all molecule's residues calculated based on HSExposureCA and HSExposureCB
        exposures_molecule = self.get_molecule_exposures(chain, radius)

        # Get residues IDs belonging to KLIFS binding site
        klifs_res_ids = molecule.df.groupby(by=['res_id', 'klifs_id'], sort=False).groups.keys()
        klifs_res_ids = pd.DataFrame(klifs_res_ids, columns=['res_id', 'klifs_id'])
        klifs_res_ids.set_index('res_id', inplace=True, drop=False)

        # Keep only KLIFS residues
        # i.e. remove non-KLIFS residues and add KLIFS residues that were skipped in exposure calculation
        exposures = klifs_res_ids.join(exposures_molecule, how='left')

        # Set index (from residue IDs) to KLIFS IDs
        exposures.set_index('klifs_id', inplace=True, drop=True)

        # Add column with CB exposure values AND CA exposure values if CB exposure values are missing
        exposures['exposure'] = exposures.apply(
            lambda row: row.ca_exposure if np.isnan(row.cb_exposure) else row.cb_exposure,
            axis=1
        )

        self.features = pd.DataFrame(
            exposures.exposure,
            index=exposures.exposure.index,
            columns=['exposure']
        )
        self.features_verbose = exposures

    def get_molecule_exposures(self, chain, radius=12.0):
        """
        Get half sphere exposure calculation based on CB and CA atoms for full molecule.

        Parameters
        ----------
        chain : Bio.PDB.Chain.Chain
            Chain from PDB file.
        radius : float
            Sphere radius to be used for half sphere exposure calculation.

        Returns
        -------
        pandas.DataFrame
            Half sphere exposure data for both HSExposureCA and HSExposureCB calculation (columns for both methods
            each: up, down, angle CB-CA-pCB, and exposure ratio) for each molecule residue (index: residue ID).
        """

        # Calculate exposure values
        exposures_cb = self.get_molecule_exposure_by_method(chain, radius, method='HSExposureCB')
        exposures_ca = self.get_molecule_exposure_by_method(chain, radius, method='HSExposureCA')

        # Join both exposures calculations
        exposures_both = exposures_ca.join(exposures_cb, how='outer')

        return exposures_both

    @staticmethod
    def get_molecule_exposure_by_method(chain, radius=12.0, method='HSExposureCB'):
        """
        Get exposure values for a given Half Sphere Exposure method, i.e. HSExposureCA or HSExposureCB.

        Parameters
        ----------
        chain : Bio.PDB.Chain.Chain
            Chain from PDB file.
        radius : float
            Sphere radius to be used for half sphere exposure calculation.
        method : str
            Half sphere exposure method name: HSExposureCA or HSExposureCB.

        Returns
        -------
        pandas.DataFrame
            Half sphere exposure data (columns: up, down, angle CB-CA-pCB, and exposure ratio) for each molecule
            residue (index: residue ID).
        """

        methods = 'HSExposureCB HSExposureCA'.split()

        # Calculate exposure values
        if method == methods[0]:
            exposures = HSExposureCB(chain, radius)
        elif method == methods[1]:
            exposures = HSExposureCA(chain, radius)
        else:
            raise ValueError(f'Method {method} unknown. Please choose from: {", ".join(methods)}')

        # Define column names
        up = f'{method[-2:].lower()}_up'
        down = f'{method[-2:].lower()}_down'
        angle = f'{method[-2:].lower()}_angle_CB-CA-pCB'
        exposure = f'{method[-2:].lower()}_exposure'

        # Transform into DataFrame
        exposures = pd.DataFrame(
            exposures.property_dict,
            index=[up, down, angle],
            dtype=float
        ).transpose()
        exposures.index = [i[1][1] for i in exposures.index]

        # Calculate exposure value: up / (up + down)
        exposures[exposure] = exposures[up] / (exposures[up] + exposures[down])

        return exposures


class PharmacophoreSizeFeatures:
    """
    Pharmacophore and size features for each residue in the KLIFS-defined kinase binding site of 85 pre-aligned
    residues, as described by SiteAlign (Schalon et al. Proteins. 2008).

    Pharmacophoric features include hydrogen bond donor, hydrogen bond acceptor, aromatic, aliphatic and charge feature.

    Attributes
    ----------
    features : pandas.DataFrame
        6 features (columns) for 85 residues (rows).

    References
    ----------
    Schalon et al., "A simple and fuzzy method to align and compare druggable ligand‐binding sites",
    Proteins, 2008.
    """

    def __init__(self):

        self.molecule_code = None
        self.features = None

    def from_molecule(self, molecule):
        """
        Get pharmacophoric and size features for each residues of a molecule.

        Parameters
        ----------
        molecule : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
            Content of mol2 or pdb file as BioPandas object.

        Returns
        -------
        pandas.DataFrame
            Pharmacophoric and size features (columns) for each residue = KLIFS position (rows).
        """

        self.molecule_code = molecule.code

        feature_matrix = []

        for feature_name in SITEALIGN_FEATURES.columns:

            # Select from DataFrame first row per KLIFS position (index) and residue name
            residues = molecule.df.groupby(by='klifs_id', sort=False).first()['res_name']

            # Get feature values for each KLIFS position
            features = residues.apply(lambda residue: self.from_residue(residue, feature_name))
            features.rename(feature_name, inplace=True)

            feature_matrix.append(features)

        features = pd.concat(feature_matrix, axis=1)

        self.features = features

    def from_residue(self, residue_name, feature_name):
        """
        Get feature value for residue's size and pharmacophoric features (i.e. number of hydrogen bond donor,
        hydrogen bond acceptors, charge features, aromatic features or aliphatic features)
        (according to SiteAlign feature encoding).

        Parameters
        ----------
        residue_name : str
            Three-letter code for residue.
        feature_name : str
            Feature name.

        Returns
        -------
        int
            Residue's size value according to SiteAlign feature encoding.
        """

        if feature_name not in SITEALIGN_FEATURES.columns:
            raise KeyError(f'Feature {feature_name} does not exist. '
                           f'Please choose from: {", ".join(SITEALIGN_FEATURES.columns)}')

        try:

            feature_value = SITEALIGN_FEATURES.loc[residue_name, feature_name]

        except KeyError:

            if residue_name in MODIFIED_RESIDUE_CONVERSION.keys():

                logger.info(f'{self.molecule_code}, {feature_name} feature: '
                            f'Non-standard amino acid {residue_name} is processed as '
                            f'{MODIFIED_RESIDUE_CONVERSION[residue_name]}.')

                residue_name = MODIFIED_RESIDUE_CONVERSION[residue_name]
                feature_value = SITEALIGN_FEATURES.loc[residue_name, feature_name]

            else:

                logger.info(f'{self.molecule_code}, {feature_name} feature: '
                            f'Non-standard amino acid {residue_name} is set to None.')

                feature_value = None

        return feature_value
