"""
encoding.py

Subpocket-based structural fingerprint for kinase pocket comparison.

Handles the primary functions for the structural kinase fingerprint encoding.
"""

import logging

from Bio.PDB import HSExposureCA, HSExposureCB, Selection, Vector
from Bio.PDB import calc_angle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import cbrt
from scipy.stats.stats import moment

from kinsim_structure.auxiliary import KlifsMoleculeLoader, PdbChainLoader
from kinsim_structure.auxiliary import center_of_mass

logger = logging.getLogger(__name__)


FEATURE_NAMES = [
    'size',
    'hbd',
    'hba',
    'charge',
    'aromatic',
    'aliphatic',
    'sco',
    'exposure',
    'distance_to_centroid',
    'distance_to_hinge_region',
    'distance_to_dfg_region',
    'distance_to_front_pocket'
]

ANCHOR_RESIDUES = {
    'hinge_region': [16, 47, 80],
    'dfg_region': [19, 24, 81],
    'front_pocket': [6, 48, 75]
}

FEATURE_LOOKUP = {
    'size': {
        1: 'ALA CYS GLY PRO SER THR VAL'.split(),
        2: 'ASN ASP GLN GLU HIS ILE LEU LYS MET'.split(),
        3: 'ARG PHE TRP TYR'.split()
    },
    'hbd': {
        0: 'ALA ASP GLU GLY ILE LEU MET PHE PRO VAL'.split(),
        1: 'ASN CYS GLN HIS LYS SER THR TRP TYR'.split(),
        3: 'ARG'.split()
    },  # Note: it is correct that 2 is missing!
    'hba': {
        0: 'ALA ARG CYS GLY ILE LEU LYS MET PHE PRO TRP VAL'.split(),
        1: 'ASN GLN HIS SER THR TYR'.split(),
        2: 'ASP GLU'.split()
    },
    'charge': {
        -1: 'ASP GLU'.split(),
        0: 'ALA ASN CYS GLN GLY HIS ILE LEU MET PHE PRO SER TRP TYR VAL'.split(),
        1: 'ARG LYS THR'.split()
    },
    'aromatic': {
        0: 'ALA ARG ASN ASP CYS GLN GLU GLY ILE LEU LYS MET PRO SER THR VAL'.split(),
        1: 'HIS PHE TRP TYR'.split()
    },
    'aliphatic': {
        0: 'ARG ASN ASP GLN GLU GLY HIS LYS PHE SER TRP TYR'.split(),
        1: 'ALA CYS ILE LEU MET PRO THR VAL'.split()
    }
}

STANDARD_AA = 'ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split()

MODIFIED_AA_CONVERSION = {
    'CAF': 'CYS',
    'CME': 'CYS',
    'CSS': 'CYS',
    'OCY': 'CYS',
    'KCX': 'LYS',
    'MSE': 'MET',
    'PHD': 'ASP',
    'PTR': 'TYR'
}

MEDIAN_SIDE_CHAIN_ORIENTATION = pd.read_csv(
    Path(__file__).parent / 'data' / 'side_chain_orientation_mean_median.csv',
    index_col='residue_name'
)['sco_median']

EXPOSURE_RADIUS = 13.0


class Fingerprint:
    """
    Kinase fingerprint with 8 physicochemical and 4 spatial features for each residue in the KLIFS-defined
    kinase binding site of 85 pre-aligned residues.

    Physicochemical features:
    - Size
    - Pharmacophoric features: Hydrogen bond donor, hydrogen bond acceptor, aromatic, aliphatic and charge feature
    - Side chain orientation
    - Half sphere exposure

    Spatial features:
    Distance of each residue to 4 reference points:
    - Binding site centroid
    - Hinge region
    - DFG region
    - Front pocket

    Two fingerprint types are offered:
    - Fingeprint type 1:
      8 physicochemical and 4 spatial (distance) features (columns) for 85 residues (rows)
      = 1020 bit fingerprint.
    - Fingerprint type 2 consisting of two parts:
      (i)  8 physicochemical features (columns) for 85 residues (rows)
           = 680 bit fingerprint
      (ii) 12 spatial features, i.e. first three moments for each of the 4 distance distributions over 85 residues
           = 12 bit fingerprint

    Attributes
    ----------
    molecule_code : str
        Molecule code as defined by KLIFS in mol2 file.
    fingerprint_type1 : pandas.DataFrame
        Fingerprint type 1.
    fingerprint_type2 : dict of pandas.DataFrame
        Fingerprint type 2.
    """

    def __init__(self):

        self.molecule_code = None
        self.fingerprint_type1 = None
        self.fingerprint_type2 = None
        self.fingerprint_type1_normalized = None
        self.fingerprint_type1_normalized = None

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

        self.fingerprint_type1 = self._get_fingerprint_type1(physicochemical_features, spatial_features)
        self.fingerprint_type2 = self._get_fingerprint_type2()
        self.fingerprint_type1_normalized = self._normalize_fingerprint_type1()
        self.fingerprint_type2_normalized = self._normalize_fingerprint_type2()

    @staticmethod
    def _get_fingerprint_type1(physicochemical_features, spatial_features):

        features = pd.concat(
            [
                physicochemical_features.features[FEATURE_NAMES[:8]],
                spatial_features.features[FEATURE_NAMES[8:]]
            ],
            axis=1
        ).copy()

        # Bring all fingerprints to same dimensions (i.e. add currently missing residues in DataFrame)
        empty_df = pd.DataFrame([], index=range(1, 86))
        features = pd.concat([empty_df, features], axis=1)

        # Set all None to nan
        features.fillna(value=pd.np.nan, inplace=True)

        return features

    def _get_fingerprint_type2(self):

        if self.fingerprint_type1 is not None:

            physicochemical_bits = self.fingerprint_type1[FEATURE_NAMES[:8]]
            spatial_bits = self.fingerprint_type1[FEATURE_NAMES[8:]]

            moments = self._calc_moments(spatial_bits)

            return {
                'physchem': physicochemical_bits,
                'moments': moments
            }

    def _normalize_fingerprint_type1(self):

        if self.fingerprint_type1 is not None:

            normalized_physchem = self._normalize_physicochemical_bits()
            normalized_distances = self._normalize_distances_bits()

            normalized = pd.concat(
                [normalized_physchem, normalized_distances],
                axis=1
            )

            return normalized

        else:
            return None

    def _normalize_fingerprint_type2(self):

        if self.fingerprint_type2 is not None:

            normalized_physchem = self.fingerprint_type1_normalized[FEATURE_NAMES[:8]]
            normalized_moments = self.fingerprint_type2['moments']  # TODO no normalization is done here, change this?

            normalized = {
                'physchem': normalized_physchem,
                'moments': normalized_moments
            }

            return normalized

        else:
            return None

    def _normalize_physicochemical_bits(self):

        # Make a copy of DataFrame
        normalized = self.fingerprint_type1[FEATURE_NAMES[:8]].copy()

        # Normalize size
        normalized['size'] = normalized['size'].apply(lambda x: (x - 1) / 2.0)

        # Normalize pharmacophoric features
        normalized['hbd'] = normalized['hbd'].apply(lambda x: x / 3.0)
        normalized['hba'] = normalized['hba'].apply(lambda x: x / 2.0)
        normalized['charge'] = normalized['charge'].apply(lambda x: (x + 1) / 2.0)
        normalized['aromatic'] = normalized['hba'].apply(lambda x: x / 1.0)
        normalized['aliphatic'] = normalized['hba'].apply(lambda x: x / 1.0)

        # Normalize side chain orientation
        normalized['sco'] = normalized['sco'].apply(lambda x: x / 180.0)

        # Normalize exposure
        normalized['exposure'] = normalized['exposure'].apply(lambda x: x / 1)

        return normalized

    def _normalize_distances_bits(self):

        # Make a copy of DataFrame
        normalized = self.fingerprint_type1[FEATURE_NAMES[8:]].copy()

        # Normalize distances
        distance_normalizer = 35.0

        normalized['distance_to_centroid'] = normalized['distance_to_centroid'].apply(
            lambda x: x / distance_normalizer if x <= distance_normalizer or np.isnan(x) else 1.0
        )
        normalized['distance_to_hinge_region'] = normalized['distance_to_hinge_region'].apply(
            lambda x: x / distance_normalizer if x <= distance_normalizer or np.isnan(x) else 1.0
        )
        normalized['distance_to_dfg_region'] = normalized['distance_to_dfg_region'].apply(
            lambda x: x / distance_normalizer if x <= distance_normalizer or np.isnan(x) else 1.0
        )
        normalized['distance_to_front_pocket'] = normalized['distance_to_front_pocket'].apply(
            lambda x: x / distance_normalizer if x <= distance_normalizer or np.isnan(x) else 1.0
        )

        if not (normalized.iloc[:, 1:12].fillna(0) <= 1).any().any():
            raise ValueError(f'Normalization failed for {self.molecule_code}: Values greater 1!')

        return normalized

    def _normalize_moments_bits(self):

        return self.fingerprint_type2['moments']

    @staticmethod
    def _calc_moments(distances):
        """
        Calculate first, second, and third moment (mean, standard deviation, and skewness) for a distance distribution.
        Parameters
        ----------
        distances : pandas.DataFrame
            Distance distribution, i.e. distances from reference point to all representatives (points)
        Returns
        -------
        pandas.DataFrame
            First, second, and third moment of distance distribution.
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
            # In case there is only one data point.
            # However, this should not be possible due to restrictions in get_shape function.
            logger.info(f'Only one data point available for moment calculation, thus write None to moments.')
            m1, m2, m3 = pd.np.nan, pd.np.nan, pd.np.nan

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
        6 features (columns) for 85 residues (rows).
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
        exposure.from_molecule(molecule, chain)

        # Concatenate all physicochemical features
        physicochemical_features = pd.concat(
            [
                pharmacophore_size.features,
                side_chain_orientation.features,
                exposure.features
            ],
            axis=1
        )

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

        self.features = pd.DataFrame.from_dict(distances)

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

        # KLIFS IDs for hinge/DFG region (taken from KLIFS website)
        hinge_klifs_ids = [46, 47, 48]
        dfg_klifs_ids = [81, 82, 83]

        # Load molecule from KLIFS metadata entry
        klifs_molecule_loader = KlifsMoleculeLoader(klifs_metadata_entry=klifs_metadata_entry)
        molecule = klifs_molecule_loader.molecule

        # Path to molecule file
        mol2_path = klifs_molecule_loader.file_from_metadata_entry(klifs_metadata_entry)

        # Output path
        Path(output_path).mkdir(parents=True, exist_ok=True)

        # Mol2 residue IDs for hinge/DFG region
        hinge_mol2_ids = molecule.df[molecule.df.klifs_id.isin(hinge_klifs_ids)].res_id.unique()
        dfg_mol2_ids = molecule.df[molecule.df.klifs_id.isin(dfg_klifs_ids)].res_id.unique()

        # Get reference points and anchor atoms (coordinates)
        space = SpatialFeatures()
        space.from_molecule(molecule)
        ref_points = space.reference_points.transpose()
        anchor_atoms = space.get_anchor_atoms(molecule)

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
        for ref_point_index, ref_point in ref_points.iterrows():

            # Set and write sphere color to file
            sphere_color = list()
            lines.append(
                f'\tCOLOR, '
                f'{str(sphere_colors[ref_point_index][0])}, '
                f'{str(sphere_colors[ref_point_index][1])}, '
                f'{str(sphere_colors[ref_point_index][2])},'
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
            if ref_point_index != 'centroid':
                for anchor_atom_index, anchor_atom in anchor_atoms[ref_point_index].iterrows():
                    lines.append(
                        f'\tSPHERE, '
                        f'{str(anchor_atom["x"])}, '
                        f'{str(anchor_atom["y"])}, '
                        f'{str(anchor_atom["z"])}, '
                        f'{str(0.5)},'
                    )

        # Write command to file that will load the reference points as PyMol object
        lines.append(f']\ncmd.load_cgo(obj_{obj_name}, "{obj_name}")')

        with open(output_path / f'refpoints_{molecule.code[6:]}.py', 'w') as f:
            f.write('\n'.join(lines))

        # In PyMol enter the following to save png
        # PyMOL > ray 900, 900
        # PyMOL > save refpoints.png


class SideChainOrientationFeature:
    """
    Side chain orientations for each residue in the KLIFS-defined kinase binding site of 85 pre-aligned residues, as
    described by SiteAlign (Schalon et al. Proteins. 2008).
    Side chain orientation of a residue is defined by the angle between the molecule's CB-CA and CB-centroid vectors.

    Attributes
    ----------
    features : pandas.DataFrame
        1 feature (columns) for 85 residues (rows).

    References
    ----------
    Schalon et al., "A simple and fuzzy method to align and compare druggable ligand‐binding sites",
    Proteins, 2008.
    """

    def __init__(self):

        self.features = None

    def from_molecule(self, molecule, chain, verbose=False, fill_missing=False):
        """
        Get side chain orientation for each residue of a molecule.

        Parameters
        ----------
        molecule : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
            Content of mol2 or pdb file as BioPandas object.
        chain : Bio.PDB.Chain.Chain
            Chain from PDB file.
        verbose : bool
            Either return angles only (default) or angles plus CA, CB and centroid points as well as metadata.
        fill_missing : bool
            Fill missing values with median value of respective amino acid angle distribution.
        """

        # Calculate/get CA, CB and centroid points
        ca_cb_com_vectors = self._get_ca_cb_com_vectors(molecule, chain)

        # Save here angle values per residue
        side_chain_orientation = []

        for index, row in ca_cb_com_vectors.iterrows():

            if row.ca and row.cb:
                angle = np.degrees(calc_angle(row.ca, row.cb, row.com))
                side_chain_orientation.append(angle.round(2))

            # If Ca and Cb are missing for angle calculation...
            else:
                # ... set median value to residue and GLY to 0
                if fill_missing:
                    angle = MEDIAN_SIDE_CHAIN_ORIENTATION[row.residue_name]

                # ... set None value to residue
                else:
                    angle = None
                side_chain_orientation.append(angle)

        # Either return angles only or angles plus CA, CB and centroid points as well as metadata
        if not verbose:
            side_chain_orientation = pd.DataFrame(
                side_chain_orientation,
                index=ca_cb_com_vectors.klifs_id,
                columns=['sco']
            )
            self.features = side_chain_orientation
        else:
            side_chain_orientation = pd.DataFrame(
                side_chain_orientation,
                index=ca_cb_com_vectors.index,
                columns=['sco']
            )
            self.features = pd.concat([ca_cb_com_vectors, side_chain_orientation], axis=1)

    @staticmethod
    def _get_ca_cb_com_vectors(molecule, chain):
        """
        Get CA, CB and centroid points for each residue of a molecule.

        Parameters
        ----------
        molecule : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
            Content of mol2 or pdb file as BioPandas object.
        chain : Bio.PDB.Chain.Chain
            Chain from PDB file.

        Returns
        -------
        pandas.DataFrame
            CA, CB and centroid points for each residue of a molecule.
        """

        # Get KLIFS residues in PDB file based on KLIFS mol2 file
        # Data type: list of Bio.PDB.Residue.Residue
        residues = Selection.unfold_entities(entity_list=chain, target_level='R')

        # Get residue IDs of binding site from mol2 file
        residue_ids = [int(i) for i in molecule.df.res_id.unique()]

        # Select KLIFS residues
        klifs_residues = [residue for residue in residues if residue.get_full_id()[3][1] in residue_ids]

        # Save here values per residue
        data = []
        metadata = pd.DataFrame(
            list(molecule.df.groupby(by=['klifs_id', 'res_id', 'res_name'], sort=False).groups.keys()),
            columns='klifs_id residue_id residue_name'.split()
        )

        for residue in klifs_residues:

            atom_names = [atoms.fullname for atoms in residue.get_atoms()]

            # Set CA atom
            if 'CA' in atom_names:
                vector_ca = residue['CA'].get_vector()
            else:
                vector_ca = None

            # Set CB atom
            if 'CB' in atom_names:
                vector_cb = residue['CB'].get_vector()
            else:
                vector_cb = None

            # Set centroid
            vector_com = Vector(center_of_mass(residue, geometric=True))

            data.append([vector_ca, vector_cb, vector_com])

        data = pd.DataFrame(
            data,
            columns='ca cb com'.split()
        )

        if len(metadata) != len(data):
            raise ValueError(f'DataFrames to be concatenated must be of same length: '
                             f'Metadata has {len(metadata)} rows, CA/CB/centroid data has {len(data)} rows.')

        return pd.concat([metadata, data], axis=1)


class ExposureFeature:
    """
    Exposure for each residue in the KLIFS-defined kinase binding site of 85 pre-aligned residues.
    Exposure of a residue describes the ratio of CA atoms in the upper sphere half around the CA-CB vector
    divided by the all CA atoms (given a sphere radius).

    Attributes
    ----------
    features : pandas.DataFrame
        1 features (columns) for 85 residues (rows).
    """

    def __init__(self):

        self.features = None

    def from_molecule(self, molecule, chain, verbose=False):
        """
        Get exposure for each residue of a molecule.

        Parameters
        ----------
        molecule : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
            Content of mol2 or pdb file as BioPandas object.
        chain : Bio.PDB.Chain.Chain
            Chain from PDB file.
        verbose : bool
            Either return exposure only (default) or additional info on HSExposureCA and HSExposureCB values.
        """

        # Calculate exposure values
        exposures_cb = self.get_exposure_by_method(chain, method='HSExposureCB')
        exposures_ca = self.get_exposure_by_method(chain, method='HSExposureCA')

        # Join both exposures calculations
        exposures_both = exposures_ca.join(exposures_cb, how='outer')

        # Get residues IDs belonging to KLIFS binding site
        klifs_res_ids = molecule.df.groupby(by=['res_id', 'klifs_id'], sort=False).groups.keys()
        klifs_res_ids = pd.DataFrame(klifs_res_ids, columns=['res_id', 'klifs_id'])
        klifs_res_ids.set_index('res_id', inplace=True, drop=False)

        # Keep only KLIFS residues
        # i.e. remove non-KLIFS residues and add KLIFS residues that were skipped in exposure calculation
        exposures = klifs_res_ids.join(exposures_both, how='left')

        # Set index (from residue IDs) to KLIFS IDs
        exposures.set_index('klifs_id', inplace=True, drop=True)

        # Add column with CB exposure values AND CA exposure values if CB exposure values are missing
        exposures['exposure'] = exposures.apply(
            lambda row: row.ca_exposure if np.isnan(row.cb_exposure) else row.cb_exposure,
            axis=1
        )

        if not verbose:
            self.features = pd.DataFrame(
                exposures.exposure,
                index=exposures.exposure.index,
                columns=['exposure']
            )
        else:
            self.features = exposures

    @staticmethod
    def get_exposure_by_method(chain, method='HSExposureCB'):
        """
        Get exposure values for a given Half Sphere Exposure method, i.e. HSExposureCA or HSExposureCB.

        Parameters
        ----------
        chain : Bio.PDB.Chain.Chain
            Chain from PDB file.
        method : str
            Half sphere exposure method name: HSExposureCA or HSExposureCB.

        References
        ----------
        Read on HSExposure module here: https://biopython.org/DIST/docs/api/Bio.PDB.HSExposure-pysrc.html
        """

        methods = 'HSExposureCB HSExposureCA'.split()

        # Calculate exposure values
        if method == methods[0]:
            exposures = HSExposureCB(chain, EXPOSURE_RADIUS)
        elif method == methods[1]:
            exposures = HSExposureCA(chain, EXPOSURE_RADIUS)
        else:
            raise ValueError(f'Method {method} unknown. Please choose from: {", ".join(methods)}')

        # Define column names
        up = f'{method[-2:].lower()}_up'
        down = f'{method[-2:].lower()}_down'
        angle = f'{method[-2:].lower()}_angle_Ca-Cb_Ca-pCb'
        exposure = f'{method[-2:].lower()}_exposure'

        # Transform into DataFrame
        exposures = pd.DataFrame(
            exposures.property_dict,
            index=[up, down, angle]
        ).transpose()
        exposures.index = [i[1][1] for i in exposures.index]

        # Check that exposures are floats (important for subsequent division)
        if (exposures[up].dtype != 'float64') | (exposures[down].dtype != 'float64'):
            raise TypeError(f'Wrong dtype, float64 needed!')

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

        feature_types = 'size hbd hba charge aromatic aliphatic'.split()

        feature_matrix = []

        for feature_type in feature_types:

            # Select from DataFrame first row per KLIFS position (index) and residue name
            residues = molecule.df.groupby(by='klifs_id', sort=False).first()['res_name']

            # Report non-standard residues in molecule
            non_standard_residues = set(residues) - set(STANDARD_AA)
            if len(non_standard_residues) > 0:
                logger.info(f'Non-standard amino acid in {molecule.code}: {non_standard_residues}')

            features = residues.apply(lambda residue: self.from_residue(residue, feature_type))
            features.rename(feature_type, inplace=True)

            feature_matrix.append(features)

        features = pd.concat(feature_matrix, axis=1)

        self.features = features

    @staticmethod
    def from_residue(residue, feature_type):
        """
        Get feature value for residue's size and pharmacophoric features (i.e. number of hydrogen bond donor,
        hydrogen bond acceptors, charge features, aromatic features or aliphatic features)
        (according to SiteAlign feature encoding).

        Parameters
        ----------
        residue : str
            Three-letter code for residue.
        feature_type : str
            Feature type name.

        Returns
        -------
        int
            Residue's size value according to SiteAlign feature encoding.
        """

        if feature_type not in FEATURE_LOOKUP.keys():
            raise KeyError(f'Feature {feature_type} does not exist. '
                           f'Please choose from: {", ".join(FEATURE_LOOKUP.keys())}')

        # Manual addition of modified residue(s)
        if residue in MODIFIED_AA_CONVERSION.keys():
            residue = MODIFIED_AA_CONVERSION[residue]

        # Start with a feature of None
        result = None

        # If residue name is listed in the feature lookup, assign respective feature
        for feature, residues in FEATURE_LOOKUP[feature_type].items():

            if residue in residues:
                result = feature

        return result
