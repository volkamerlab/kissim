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


def get_ca_cb_com_vectors(molecule):
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
    ca = []
    cb = []
    com = []

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
        ca.append(vector_ca)
        cb.append(vector_cb)
        com.append(vector_com)

    return pd.DataFrame([ca, cb, com], index=['ca', 'cb', 'com']).transpose()


def get_side_chain_orientation(molecule):
    """
    Input: MOL2 file, PDB code as string, chain as char/string
    Calculates side chain orientation for every binding site residue
    Output: side chain orientation values for the binding site (numpy array, dtype = float)
    """

    # Calculate CA, CB and centroid point per residue
    angle_points = get_ca_cb_com_vectors(molecule)

    # Save here angle values per residue
    side_chain_orientation = []

    for index, row in angle_points.iterrows():

        if row.ca and row.cb:
            angle = np.degrees(calc_angle(row.ca, row.cb, row.com))
            side_chain_orientation.append(round(angle, 2))
        else:
            angle = None
            side_chain_orientation.append(angle)

    return side_chain_orientation


def get_feature_size(residue):
    """
    Get feature value for residue's size according to SiteAlign feature encoding.

    Parameters
    ----------
    residue : str
        Three-letter code for residue.

    Returns
    -------
    int
        Residue's size value according to SiteAlign feature encoding.

    References
    ----------
    [1]_ Schalon et al., "A simple and fuzzy method to align and compare druggable ligand‐binding sites",
    Proteins, 2008.
    """

    residue_sizes = {
        1: 'ALA CYS GLY PRO SER THR VAL'.split(),
        2: 'ASN ASP GLN GLU HIS ILE LEU LYS MET'.split(),
        3: 'ARG PHE TRP TYR'.split()
    }

    result = None

    for feature, residues in residue_sizes.items():

        if residue in residues:
            result = feature

    return result


def get_feature_hbd(residue):
    """
    Get feature value for residue's hydrogen bond donor according to SiteAlign feature encoding.

    Parameters
    ----------
    residue : str
        Three-letter code for residue.

    Returns
    -------
    int
        Residue's hydrogen bond donor value according to SiteAlign feature encoding.

    References
    ----------
    [1]_ Schalon et al., "A simple and fuzzy method to align and compare druggable ligand‐binding sites",
    Proteins, 2008.
    """

    residue_sizes = {
        0: 'ALA ASP GLU GLY ILE LEU MET PHE PRO VAL'.split(),
        1: 'ASN CYS GLN HIS LYS SER THR TRP TYR'.split(),
        3: 'ARG'.split()
    }

    result = None

    for feature, residues in residue_sizes.items():

        if residue in residues:
            result = feature

    return result


def get_feature_hba(residue):
    """
    Get feature value for residue's size according to SiteAlign feature encoding.

    Parameters
    ----------
    residue : str
        Three-letter code for residue.

    Returns
    -------
    int
        Residue's H-bond acceptor value according to SiteAlign feature encoding.

    References
    ----------
    [1]_ Schalon et al., "A simple and fuzzy method to align and compare druggable ligand‐binding sites",
    Proteins, 2008.
    """

    residue_sizes = {
        0: 'ALA ARG CYS GLY ILE LEU LYS MET PHE PRO TRP VAL'.split(),
        1: 'ASN GLN HIS SER THR TYR'.split(),
        2: 'ASP GLU'.split()
    }

    result = None

    for feature, residues in residue_sizes.items():

        if residue in residues:
            result = feature

    return result


def get_feature_charge(residue):
    """
    Get feature value for residue's charge according to SiteAlign feature encoding.

    Parameters
    ----------
    residue : str
        Three-letter code for residue.

    Returns
    -------
    int
        Residue's charge value according to SiteAlign feature encoding.

    References
    ----------
    [1]_ Schalon et al., "A simple and fuzzy method to align and compare druggable ligand‐binding sites",
    Proteins, 2008.
    """

    residue_sizes = {
        0: 'ALA ASN CYS GLN GLY HIS ILE LEU MET PHE PRO SER TRP TYR VAL'.split(),
        1: 'ARG LYS THR'.split(),
        -1: 'ASP GLU'.split()
    }

    result = None

    for feature, residues in residue_sizes.items():

        if residue in residues:
            result = feature

    return result


def get_feature_aromatic(residue):
    """
    Get feature value for residue's aromatic feature according to SiteAlign feature encoding.

    Parameters
    ----------
    residue : str
        Three-letter code for residue.

    Returns
    -------
    int
        Residue's aromatic feature value according to SiteAlign feature encoding.

    References
    ----------
    [1]_ Schalon et al., "A simple and fuzzy method to align and compare druggable ligand‐binding sites",
    Proteins, 2008.
    """

    residue_sizes = {
        0: 'ALA ARG ASN ASP CYS GLN GLU GLY ILE LEU LYS MET PRO SER THR VAL'.split(),
        1: 'HIS PHE TRP TYR'.split()
    }

    result = None

    for feature, residues in residue_sizes.items():

        if residue in residues:
            result = feature

    return result


def get_feature_aliphatic(residue):
    """
    Get feature value for residue's aliphatic feature according to SiteAlign feature encoding.

    Parameters
    ----------
    residue : str
        Three-letter code for residue.

    Returns
    -------
    int
        Residue's aliphatic feature value according to SiteAlign feature encoding.

    References
    ----------
    [1]_ Schalon et al., "A simple and fuzzy method to align and compare druggable ligand‐binding sites",
    Proteins, 2008.
    """

    residue_sizes = {
        0: 'ARG ASN ASP GLN GLU GLY HIS LYS PHE SER TRP TYR'.split(),
        1: 'ALA CYS ILE LEU MET PRO THR VAL'.split()
    }

    result = None

    for feature, residues in residue_sizes.items():

        if residue in residues:
            result = feature

    return result
