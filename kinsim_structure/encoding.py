"""
encoding.py

Subpocket-based structural fingerprint for kinase pocket comparison.

Handles the primary functions for the structural kinase fingerprint.
"""

import logging
import re

from Bio.PDB import HSExposureCA, HSExposureCB, Selection, Vector
from Bio.PDB import calc_angle
import numpy as np
import pandas as pd

from kinsim_structure.auxiliary import get_klifs_residues_mol2topdb, center_of_mass

logger = logging.getLogger(__name__)


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
    },
    'hba': {
        0: 'ALA ARG CYS GLY ILE LEU LYS MET PHE PRO TRP VAL'.split(),
        1: 'ASN GLN HIS SER THR TYR'.split(),
        2: 'ASP GLU'.split()
    },
    'charge': {
        0: 'ALA ASN CYS GLN GLY HIS ILE LEU MET PHE PRO SER TRP TYR VAL'.split(),
        1: 'ARG LYS THR'.split(),
        -1: 'ASP GLU'.split()
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


class Fingerprint:

    def __init__(self, molecule):

        self.features = self.from_molecule(molecule)

    @staticmethod
    def from_molecule(molecule):

        physicochemical_features = PhysicoChemicalFeatures(molecule)
        spatial_features = SpatialFeatures(molecule)

        features = pd.concat(
            [
                physicochemical_features.features,
                spatial_features.features
            ],
            axis=1
        )

        # Bring all Fingerprints to same dimensions (i.e. add currently missing residues in DataFrame)
        empty_df = pd.DataFrame([], index=range(1, 86))
        features = pd.concat([empty_df, features], axis=1)

        return features


class PhysicoChemicalFeatures:

    def __init__(self, molecule):

        self.features = self.from_molecule(molecule)

    @staticmethod
    def from_molecule(molecule):

        pharmacophore_size = PharmacophoreSizeFeatures(molecule)
        side_chain_orientation = SideChainOrientationFeature(molecule)

        # Concatenate all physicochemical features
        physicochemical_features = pd.concat(
            [
                pharmacophore_size.features,
                side_chain_orientation.features
            ],
            axis=1
        )

        return physicochemical_features


class SpatialFeatures:

    def __init__(self, molecule):

        self.reference_points = self.get_reference_points(molecule)
        self.features = self.from_molecule(molecule)

    def from_molecule(self, molecule):
        """

        Parameters
        ----------
        molecule : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
            Content of mol2 or pdb file as BioPandas object.

        Returns
        -------

        """

        # Get all residues' CA atoms in molecule (set KLIFS position as index)
        residues = molecule.df[molecule.df.atom_name == 'CA']['klifs_id x y z'.split()]
        residues.set_index('klifs_id', drop=True, inplace=True)

        distances = {}

        for name, coord in self.reference_points.items():

            distance = (residues - coord).transpose().apply(lambda x: np.linalg.norm(x))
            distance.rename(name, inplace=True)
            distances[f'distance_to_{name}'] = np.round(distance, 2)

        return pd.DataFrame.from_dict(distances)

    @staticmethod
    def get_reference_points(molecule):
        """

        Parameters
        ----------
        molecule

        Returns
        -------

        """

        reference_points = dict()

        # Calculate centroid-based reference point
        reference_points['centroid'] = molecule.df['x y z'.split()].mean()

        # Calculate anchor-based reference points
        for anchor_name, anchor_klifs_id in ANCHOR_RESIDUES.items():

            # Select all anchor residues' CA atoms
            molecule_ca = molecule.df[
                (molecule.df.klifs_id.isin(anchor_klifs_id)) &
                (molecule.df.atom_name == 'CA')
            ]

            reference_points[anchor_name] = molecule_ca[['x', 'y', 'z']].mean()

        return pd.DataFrame.from_dict(reference_points)


class SideChainOrientationFeature:

    def __init__(self, molecule):

        self.features = self.from_molecule(molecule)

    def from_molecule(self, molecule, ca_cb_com_vectors=None):
        """
        Calculate side chain orientation for each residue of a molecule.

        Parameters
        ----------
        molecule : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
            Content of mol2 or pdb file as BioPandas object.
        ca_cb_com_vectors : pandas.DataFrame
            CA, CB and centroid points for each residue of a molecule.

        Returns
        -------
        list of float
            Side chain orientation (angle) for each residue of a molecule.
        """

        # If only molecule (and CA, CB and centroid points) given, calculate these points
        if ca_cb_com_vectors is None:
            ca_cb_com_vectors = self._get_ca_cb_com_vectors(molecule)

        # Save here angle values per residue
        side_chain_orientation = []

        for index, row in ca_cb_com_vectors.iterrows():

            if row.ca and row.cb:
                angle = np.degrees(calc_angle(row.ca, row.cb, row.com))
                side_chain_orientation.append(round(angle, 2))
            else:
                angle = None
                side_chain_orientation.append(angle)

        return pd.DataFrame(side_chain_orientation, index=ca_cb_com_vectors.klifs_id, columns=['sco'])

    @staticmethod
    def _get_ca_cb_com_vectors(molecule):
        """
        Get CA, CB and centroid points for each residue of a molecule.

        Parameters
        ----------
        molecule : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
            Content of mol2 or pdb file as BioPandas object.

        Returns
        -------
        pandas.DataFrame
            CA, CB and centroid points for each residue of a molecule.
        """

        # Get KLIFS residues in PDB file based on KLIFS mol2 file
        klifs_residues = get_klifs_residues_mol2topdb(molecule)

        # Save here values per residue
        data = []
        metadata = pd.DataFrame(
            list(molecule.df.groupby(by=['klifs_id', 'res_id', 'res_name']).groups.keys()),
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

            # Only needed for verbose function call
            data.append([vector_ca, vector_cb, vector_com])

        data = pd.DataFrame(
            data,
            columns='ca cb com'.split()
        )

        if len(metadata) != len(data):
            raise ValueError(f'DataFrames to be concatenated must be of same length')

        return pd.concat([metadata, data], axis=1)


class ExposureFeature:
    pass


class PharmacophoreSizeFeatures:

    def __init__(self, molecule):

        self.features = self.from_molecule(molecule)

    def from_molecule(self, molecule):
        """
        Get pharmacophoric and size features for all residues of a molecule.

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
            residues = molecule.df.groupby(by='klifs_id').first()['res_name']
            features = residues.apply(lambda residue: self.from_residue(residue, feature_type))
            features.rename(feature_type, inplace=True)

            feature_matrix.append(features)

        features = pd.concat(feature_matrix, axis=1)

        return features

    def from_residue(self, residue, feature_type):
        """
        Get feature value for residue's size and pharmacophoric features (i.e. number of hydrogen bond donor,
        hydrogen bond acceptors, charge features, aromatic features or aliphatic features )
        (according to SiteAlign feature encoding).

        Parameters
        ----------
        residue : str
            Three-letter code for residue.
        feature_type : str


        Returns
        -------
        int
            Residue's size value according to SiteAlign feature encoding.

        References
        ----------
        [1]_ Schalon et al., "A simple and fuzzy method to align and compare druggable ligand‐binding sites",
        Proteins, 2008.
        """

        if feature_type not in FEATURE_LOOKUP.keys():
            raise KeyError(f'Feature {feature_type} does not exist. '
                           f'Please choose from: {", ".join(FEATURE_LOOKUP.keys())}')

        result = None

        for feature, residues in FEATURE_LOOKUP[feature_type].items():

            if residue in residues:
                result = feature

        return result


def exposure(mol, kette):
    """Input: MOL2 file of a kinase binding site, PDB code as string
    Calculates half sphere exposure for binding site residues.
    Output: Numpy array with exposure values for all binding site residues (dtype = float)"""
    # write residue IDs of binding site from MOL2 file into list (in the correct order, non redundant)
    res_ids = []
    for name in mol.df.subst_name:
        if name in res_ids:
            continue
        else:
            res_ids.append(name)
    # it happened, that the MOL2 data frame did not only contatin residues but also ligand information
    res_nos = []
    tag_or_lig = []
    for x in res_ids:
        # check if ID id amino acid and not ligand
        if re.match(r'^[A-Z]{3}', x) and not '_' in x:
            res_nos.append(int(x[3:]))
        else:
            tag_or_lig.append(res_ids.index(x))
    # align binding site residue IDs from MOL2 file to the key format of HSE output dictionary
    hse_keys = [(kette.id, (' ', ident,' ')) for ident in res_nos]
    # get protein residue IDs from PDB file and select the binding site residues based on the IDs from MOL2 file
    res_list = Selection.unfold_entities(kette, 'R')
    keys = [res for res in res_list if res.get_full_id()[3][1] in res_nos]
    # compute HSEa and HSEb for the chain
    RADIUS = 13.0
    hse_cb = HSExposureCB(kette, RADIUS)
    hse_ca = HSExposureCA(kette, RADIUS)

    hse_vector = []
    # Case: there is no HSE value computed for the first n residue(s)
    # solution: take value of first residue that HSE could be calculated for and assign it to the missing residues
    # no averaging possible in this case
    i = 0
    if hse_keys[0][1] not in [k[1] for k in hse_cb.keys()] and hse_keys[0][1] not in [k[1] for k in hse_ca.keys()]:
        while hse_keys[i][1] not in [k[1] for k in hse_cb.keys()] and hse_keys[i][1] not in [k[1] for k in hse_ca.keys()]:
            ident = hse_keys[i][1][1]
            rest = [res for res in keys if res.id[1] == ident][0]
            logging.warning('no hse for first residue ('+str(rest.resname)+str(ident)+') of binding site sequence, compensate by next available hse')
            i += 1

    if i!=0:
        key = hse_keys[i]
        ident = key[1][1]
        rest = [res for res in keys if res.id[1] == ident][0]
        pred = rest.resname
        pred_id = ident
        logging.warning('missing hse compensated by value of '+str(pred)+str(pred_id))
        # if there is no hse value in HSEb dict, take value from HSEa dict, else residue is missing
        if key[1] in [k[1] for k in hse_cb.keys()]:
            v = hse_cb[key][0]/float((hse_cb[key][0]+hse_cb[key][1]))
        else:
            v = hse_ca[key][0]/float((hse_ca[key][0]+hse_ca[key][1]))
        hse_vector = [v for c in range(0,i+1)]
        i += 1
        pred_value = v
    # iterate over binding site residue keys and extract HSE value from dictionary

    missing = False
    m = 0
    m_name = []
    for j in range(i, len(hse_keys)):
        # get residue key, residue id and residue entity
        key = hse_keys[j]
        ident = key[1][1]
        rest = [res for res in keys if res.id[1] == ident][0]
        # check dictionaries for residue key, if residue in keys, remember residue ID in case following residue
        # is missing in dict and a dummy needs to be computed
        if key[1] in [k[1] for k in hse_cb.keys()]:
            v = hse_cb[key][0]/float((hse_cb[key][0]+hse_cb[key][1]))
            # if there are missing residues between predecessor and current key, assign average of last and current
            # value to these
            if missing == True:
                dummy = np.average([pred_value, v])
                for n in range(m):
                    hse_vector.append(dummy)
                    n += 1
                logging.info('assigned average hse of '+pred+str(pred_id)+' and '+str(rest.resname)+str(ident)+' to missing residues '+','.join(m_name))
                m = 0
                m_name = []
                missing = False
            hse_vector.append(v)
            pred_value = v
            pred = rest.resname
            pred_id = ident
        elif key[1] in [k[1] for k in hse_ca.keys()]:
            v = hse_ca[key][0]/float((hse_ca[key][0]+hse_ca[key][1]))
            # if there are missing resiude between predecessor and current key, assign average of last and current
            # value to these
            if missing == True:
                dummy = np.average([pred_value, v])
                for n in range(m):
                    hse_vector.append(dummy)
                    n += 1
                logging.info('assigned average hse of '+pred+str(pred_id)+' and '+str(rest.resname)+str(ident)+' to missing residues '+','.join(m_name))
                m = 0
                m_name = []
                missing = False
            hse_vector.append(v)
            pred_value = v
            pred = rest.resname
            pred_id = ident
        else:
            # if residue is no dict key, raise counter
            logging.warning('no hse available for current residue '+str(rest.resname)+str(ident))
            missing = True
            m += 1
        # for missing residues at the end of the binding site sequence, append value of last available residue
    while len(hse_vector) < len(hse_keys):
        hse_vector.append(pred_value)
    for elem in tag_or_lig:
        hse_vector.insert(elem,0.0)
    return np.array(hse_vector, dtype=float)
