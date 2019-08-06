"""
analysis.py

Subpocket-based structural fingerprint for kinase pocket comparison.

Handles the primary functions for the structural kinase fingerprint analysis.
"""

from collections import defaultdict
import logging

import pandas as pd

from kinsim_structure.auxiliary import MoleculeLoader, get_mol2paths_from_metadata
from kinsim_structure.encoding import get_ca_cb_com_vectors, get_side_chain_orientation

logger = logging.getLogger(__name__)


def calculate_gap_rate(klifs_metadata):
    """
    Calculate gap rate at every KLIFS MSA position across the filtered kinase data set

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

    return pd.concat([pd.Series(range(1, 86), name='klifs_position'),
                      pd.Series(gaps, name='gaps'),
                      pd.Series(gap_rate, name='gap_rate')],
                     axis=1)


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

    # Input parameters
    mol2_paths = get_mol2paths_from_metadata(klifs_metadata)
    missing_paths = []

    # 20 standard amino acids
    standard_aminoacids = 'ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split()

    # Will contain per structure (key) list of non-standard amino acids (values)
    non_standard_aminoacids = {}

    for mol2_path in mol2_paths:

        if not mol2_path.exists():
            print(f'Path does not exist: {mol2_path}')
            missing_paths.append(mol2_path)
            continue

        molecule_loader = MoleculeLoader(mol2_path)
        molecule = molecule_loader.molecules[0]

        # Get deduplicated amino acid names
        res_names = set(molecule.df.res_name)

        # Retain only non-standard amino acids
        non_standard_aminoacids_per_file = [res_name for res_name in res_names if res_name not in standard_aminoacids]

        # If structure contains non-standard amino acids, add to dict
        if non_standard_aminoacids_per_file:
            non_standard_aminoacids[molecule.code] = non_standard_aminoacids_per_file

    return non_standard_aminoacids


def get_missing_ca_cb_stats(klifs_metadata):
    """
    """

    mol2_paths = get_mol2paths_from_metadata(klifs_metadata)
    missing_paths = []

    stats = []

    for mol2_path in mol2_paths:

        if not mol2_path.exists():
            print(f'Path does not exist: {mol2_path}')
            missing_paths.append(mol2_path)
            continue

        # Load data from file
        molecule_loader = MoleculeLoader(mol2_path)
        molecule = molecule_loader.molecules[0]

        points = get_ca_cb_com_vectors(molecule)
        points['klifs_code'] = molecule.code

        stats.append(points)

    return pd.concat(stats)


def get_side_chain_orientation_stats(klifs_metadata):
    """
    Collect all side chain orientations in KLIFS dataset, ordered by residue name (e.g. ALA).

    Parameters
    ----------
    klifs_metadata : pandas.DataFrame
        KLIFS metadata describing every pocket entry in the KLIFS dataset.

    Returns
    -------
    dict
        For each residue name, list observed angles plus metadata, i.e. molecule and residue code.
    """

    mol2_paths = get_mol2paths_from_metadata(klifs_metadata)
    missing_paths = []

    stats = defaultdict(pd.DataFrame)

    for mol2_path in mol2_paths:

        if not mol2_path.exists():
            print(f'Path does not exist: {mol2_path}')
            missing_paths.append(mol2_path)
            continue

        # Load data from file
        molecule_loader = MoleculeLoader(mol2_path)
        molecule = molecule_loader.molecules[0]

        # Get residue code (ALA50), residue name (ALA) and side chain orientations (angles)
        subst_names = molecule.df.groupby(by='subst_name')['res_id'].first().index  # ALA50
        residue_names = molecule.df.groupby(by='subst_name')['res_name'].first().values  # ALA
        angles = get_side_chain_orientation(molecule)  # Angles in degrees

        # For each residue save residue code, residue name and angle as list
        for subst_name, residue, angle in zip(subst_names, residue_names, angles):
            stats[residue].append([molecule.code, subst_name, angle])

    # Transform dict values from list of lists to pandas.DataFrame
    for key, value in stats.items():
        stats[key] = pd.DataFrame(value, columns=['molecule', 'residue', 'angle'])

    print(f'Missing paths: {", ".join(missing_paths)}')

    return stats