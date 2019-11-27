"""
Unit and regression test for kinsim_structure.encoding.SideChainOrientationFeature class.
"""

import numpy as np
from pathlib import Path
import pytest

from Bio.PDB import Vector
import pandas as pd

from kinsim_structure.auxiliary import KlifsMoleculeLoader, PdbChainLoader
from kinsim_structure.encoding import SideChainOrientationFeature

PATH_TEST_DATA = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data'


@pytest.mark.parametrize('path_mol2, path_pdb, chain_id, res_id_mean, n_pocket_atoms', [
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ABL1/2g2i_chainA/pocket.mol2',
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ABL1/2g2i_chainA/protein_pymol.pdb',
        'A',
        315.95,
        659
    )
])
def test_get_pocket_residues(path_mol2, path_pdb, chain_id, res_id_mean, n_pocket_atoms):
    """
    Test the mean of the pocket's PDB residue IDs and the number of pocket atoms.

    Parameters
    ----------
    path_mol2 : pathlib.Path
        Path to mol2 file.
    path_pdb : pathlib.Path
        Path to cif file.
    chain_id : str
        Chain ID.
    res_id_mean : float
        Mean of pocket's PDB residue IDs.
    n_pocket_atoms : int
        Number of pocket atoms.
    """

    path_klifs_metadata = PATH_TEST_DATA / 'klifs_metadata.csv'

    klifs_molecule_loader = KlifsMoleculeLoader()
    klifs_molecule_loader.from_file(path_mol2, path_klifs_metadata)
    pdb_chain_loader = PdbChainLoader()
    pdb_chain_loader.from_file(path_pdb, chain_id)

    feature = SideChainOrientationFeature()
    pocket_residues = feature._get_pocket_residues(klifs_molecule_loader.molecule, pdb_chain_loader.chain)

    # Get and test the mean of pocket PDB residue IDs and the number of pocket atoms
    res_id_mean_calculated = pocket_residues.res_id.mean()

    pocket_atoms = []
    for residue in pocket_residues.pocket_residues:
        for atom in residue:
            if not atom.get_name().startswith('H'):  # Count only non-hydrogen atoms
                pocket_atoms.append(atom.get_name())
    n_pocket_atoms_calculated = len(pocket_atoms)

    assert np.isclose(res_id_mean_calculated, res_id_mean, rtol=1e-03)
    assert n_pocket_atoms_calculated == n_pocket_atoms


@pytest.mark.parametrize('path_pdb, chain_id, residue_id, ca', [
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ALK/2yjr_altA_chainA/protein_pymol.pdb',
        'A',
        1272,
        [5.78, 18.76, 31.15]
    ),  # Residue has CA
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ALK/2yjr_altA_chainA/protein_pymol.pdb',
        'A',
        1273,
        None
    )  # Residue has no CA
])
def test_get_ca(path_pdb, chain_id, residue_id, ca):
    """
    Test if CA atom is retrieved correctly from a residue (test if-else cases).

    Parameters
    ----------
    path_pdb : pathlib.Path
        Path to cif file.
    chain_id : str
        Chain ID.
    residue_id : int
        Residue ID.
    ca : list of int or None
        3D coordinates of CA atom.
    """

    pdb_chain_loader = PdbChainLoader()
    pdb_chain_loader.from_file(path_pdb, chain_id)

    chain = pdb_chain_loader.chain
    residue = chain[residue_id]

    feature = SideChainOrientationFeature()
    ca_calculated = feature._get_ca(residue)

    if ca_calculated and ca:
        # Check only x coordinate
        assert np.isclose(list(ca_calculated)[0], ca[0], rtol=1e-03)
        assert isinstance(ca_calculated, Vector)
    else:
        assert ca_calculated == ca


@pytest.mark.parametrize('path_pdb, chain_id, residue_id', [
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ADCK3/5i35_chainA/protein_pymol.pdb',
        'A',
        337
    ),  # ALA
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ADCK3/5i35_chainA/protein_pymol.pdb',
        'A',
        357
    ),  # Non-standard residue
])
def test_get_pcb_from_gly_valueerror(path_pdb, chain_id, residue_id):
    """
    Test exceptions in pseudo-CB calculation for GLY.

    Parameters
    ----------
    path_pdb : pathlib.Path
        Path to cif file.
    chain_id : str
        Chain ID.
    residue_id : int
        Residue ID.
    """

    pdb_chain_loader = PdbChainLoader()
    pdb_chain_loader.from_file(path_pdb, chain_id)

    chain = pdb_chain_loader.chain
    try:
        residue = chain[residue_id]
    except KeyError:
        # For non-standard residue MSE indexing did not work, thus use this workaround
        residue = [i for i in chain.get_list() if i.get_id()[1] == residue_id][0]

    with pytest.raises(ValueError):
        feature = SideChainOrientationFeature()
        feature._get_pcb_from_gly(residue)


@pytest.mark.parametrize('path_pdb, chain_id, residue_id, ca_pcb', [
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ADCK3/5i35_chainA/protein_pymol.pdb',
        'A',
        272,
        np.array([12.22, 8.37, 31.38])
    ),  # GLY
])
def test_get_pcb_from_gly(path_pdb, chain_id, residue_id, ca_pcb):
    """
    Test pseudo-CB calculation for GLY.

    Parameters
    ----------
    path_pdb : pathlib.Path
        Path to cif file.
    chain_id : str
        Chain ID.
    residue_id : int
        Residue ID.
    ca_pcb : numpy.array
        Pseudo-CB atom coordinates.
    """

    pdb_chain_loader = PdbChainLoader()
    pdb_chain_loader.from_file(path_pdb, chain_id)

    chain = pdb_chain_loader.chain
    try:
        residue = chain[residue_id]
    except KeyError:
        # For non-standard residue MSE indexing did not work, thus use this workaround
        residue = [i for i in chain.get_list() if i.get_id()[1] == residue_id][0]

    feature = SideChainOrientationFeature()
    ca_pcb_calculated = feature._get_pcb_from_gly(residue)

    assert np.isclose(ca_pcb_calculated.get_array().mean(), ca_pcb.mean(), rtol=1e-04)


@pytest.mark.parametrize('path_pdb, chain_id, residue_id, ca_pcb', [
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ADCK3/5i35_chainA/protein_pymol.pdb',
        'A',
        272,
        np.array([12.22, 8.37, 31.38])
    ),  # GLY
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ADCK3/5i35_chainA/protein_pymol.pdb',
        'A',
        337,
        np.array([4.89, 12.19, 43.60])
    ),  # Residue with +- residue
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/CHK1/4jik_chainA/protein_pymol.pdb',
        'A',
        19,
        None
    ),  # Residue without + residue
])
def test_get_pcb_from_residue(path_pdb, chain_id, residue_id, ca_pcb):
    """
    Test pseudo-CB calculation for a residue.

    Parameters
    ----------
    path_pdb : pathlib.Path
        Path to cif file.
    chain_id : str
        Chain ID.
    residue_id : int
        Residue ID.
    ca_pcb : numpy.array
        Pseudo-CB atom coordinates.
    """

    pdb_chain_loader = PdbChainLoader()
    pdb_chain_loader.from_file(path_pdb, chain_id)

    chain = pdb_chain_loader.chain
    try:
        residue = chain[residue_id]
    except KeyError:
        # For non-standard residue MSE indexing did not work, thus use this workaround
        residue = [i for i in chain.get_list() if i.get_id()[1] == residue_id][0]

    feature = SideChainOrientationFeature()
    feature.chain = chain  # Set chain value artificially
    ca_pcb_calculated = feature._get_pcb_from_residue(residue, chain)

    if ca_pcb is None:
        assert ca_pcb_calculated is None
    else:
        assert np.isclose(ca_pcb_calculated.get_array().mean(), ca_pcb.mean(), rtol=1e-04)


@pytest.mark.parametrize('path_pdb, chain_id, residue_id, side_chain_centroid', [
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ADCK3/5i35_chainA/protein_pymol.pdb',
        'A',
        272,
        np.array([12.22, 8.37, 31.38])
    ),  # GLY with pCB
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/CHK1/3ot8_altA_chainA/protein_pymol.pdb',
        'A',
        18,
        None
    ),  # GLY without pCB (missing C and CA)
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ADCK3/5i35_chainA/protein_pymol.pdb',
        'A',
        337,
        np.array([4.73, 12.85, 43.35])
    ),  # ALA with CB
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/CHK1/3nlb_chainA/protein_pymol.pdb',
        'A',
        19,
        np.array([5.47, 13.78, 32.29])
    ),  # ALA with pCB
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/CHK1/4jik_chainA/protein_pymol.pdb',
        'A',
        19,
        None
    ),  # ALA without pCB (missing CA)
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ADCK3/5i35_chainA/protein_pymol.pdb',
        'A',
        336,
        np.array([4.48, 15.79, 46.66])
    ),  # Standard residue (side chain) with enough atoms
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/5l4q_altA_chainA/protein_pymol.pdb',
        'A',
        130,
        np.array([-5.11, 20.31, 49.99])
    ),  # Standard residue with too few atoms but CB atom
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/BTK/4yhf_altA_chainB/protein_pymol.pdb',
        'B',
        412,
        np.array([3.42, 12.33, 35.24])
    ),  # Standard residue with too few atoms, no CB atom, but pCB atom
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/CHK1/4jik_chainA/protein_pymol.pdb',
        'A',
        51,
        None
    ),  # Standard residue with too few atoms and no CB and pCB atom
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ADCK3/5i35_chainA/protein_pymol.pdb',
        'A',
        357,
        np.array([12.94, 22.55, 44.96])
    ),  # Non-standard residue with enough atoms (>0)
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/5l4q_altA_chainA/protein_pymol.pdb',
        'A',
        57,
        np.array([10.44, 12.84, 31.17])
    ),  # Side chain containing H atoms
    # ('some.pdb', 'X', 0, None),  # Non-standard residue side chain with no atoms
])
def test_get_side_chain_centroid(path_pdb, chain_id, residue_id, side_chain_centroid):
    """
    Test if side chain centroid is retrieved correctly from a residue (test if-else cases).

    Parameters
    ----------
    path_pdb : pathlib.Path
        Path to cif file.
    chain_id : str
        Chain ID.
    residue_id : int
        Residue ID.
    side_chain_centroid : list of int or None
        3D coordinates of CA atom.
    """

    pdb_chain_loader = PdbChainLoader()
    pdb_chain_loader.from_file(path_pdb, chain_id)

    chain = pdb_chain_loader.chain
    try:
        residue = chain[residue_id]
    except KeyError:
        # For non-standard residue MSE indexing did not work, thus use this workaround
        residue = [i for i in chain.get_list() if i.get_id()[1] == residue_id][0]

    feature = SideChainOrientationFeature()
    side_chain_centroid_calculated = feature._get_side_chain_centroid(residue, chain)
    print(side_chain_centroid_calculated)

    if side_chain_centroid is not None:
        # Check only x coordinate
        assert np.isclose(side_chain_centroid_calculated.get_array().mean(), side_chain_centroid.mean(), rtol=1e-03)
        assert isinstance(side_chain_centroid_calculated, Vector)
    else:
        assert side_chain_centroid_calculated == side_chain_centroid


@pytest.mark.parametrize('path_mol2, path_pdb, chain_id, pocket_centroid', [
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ABL1/2g2i_chainA/pocket.mol2',
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ABL1/2g2i_chainA/protein_pymol.pdb',
        'A',
        np.array([0.99, 21.06, 36.70])
    )
])
def test_get_pocket_centroid(path_mol2, path_pdb, chain_id, pocket_centroid):
    """
    Test pocket centroid calculation.

    Parameters
    ----------
    path_mol2 : pathlib.Path
        Path to mol2 file.
    path_pdb : pathlib.Path
        Path to cif file.
    chain_id : str
        Chain ID.
    pocket_centroid : list of float
        Pocket centroid coordinates.
    """

    path_klifs_metadata = PATH_TEST_DATA / 'klifs_metadata.csv'

    klifs_molecule_loader = KlifsMoleculeLoader()
    klifs_molecule_loader.from_file(path_mol2, path_klifs_metadata)
    pdb_chain_loader = PdbChainLoader()
    pdb_chain_loader.from_file(path_pdb, chain_id)

    feature = SideChainOrientationFeature()
    pocket_residues = feature._get_pocket_residues(klifs_molecule_loader.molecule, pdb_chain_loader.chain)
    pocket_centroid_calculated = feature._get_pocket_centroid(pocket_residues)

    if pocket_centroid is not None:
        assert np.isclose(pocket_centroid_calculated.get_array().mean(), pocket_centroid.mean(), rtol=1e-03)
        assert isinstance(pocket_centroid_calculated, Vector)
    else:
        assert pocket_centroid_calculated == pocket_centroid


@pytest.mark.parametrize('path_mol2, path_pdb, chain_id, n_vectors', [
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ABL1/2g2i_chainA/pocket.mol2',
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ABL1/2g2i_chainA/protein_pymol.pdb',
        'A',
        82
    )
])
def test_get_pocket_vectors(path_mol2, path_pdb, chain_id, n_vectors):
    """
    Test if returned DataFrame for pocket vectors contains correct column names and correct number of vectors
    (= number of pocket residues).

    Parameters
    ----------
    path_mol2 : pathlib.Path
        Path to mol2 file.
    path_pdb : pathlib.Path
        Path to cif file.
    chain_id : str
        Chain ID.
    n_vectors : int
        Number of vectors (= number of pocket residues)
    """

    path_klifs_metadata = PATH_TEST_DATA / 'klifs_metadata.csv'

    klifs_molecule_loader = KlifsMoleculeLoader()
    klifs_molecule_loader.from_file(path_mol2, path_klifs_metadata)
    pdb_chain_loader = PdbChainLoader()
    pdb_chain_loader.from_file(path_pdb, chain_id)

    feature = SideChainOrientationFeature()
    pocket_residues = feature._get_pocket_residues(klifs_molecule_loader.molecule, pdb_chain_loader.chain)
    pocket_vectors = feature._get_pocket_vectors(pocket_residues, pdb_chain_loader.chain)

    # Test if DataFrame contains correct columns
    pocket_vectors_columns = ['klifs_id', 'res_id', 'res_name', 'ca', 'side_chain_centroid', 'pocket_centroid']
    assert list(pocket_vectors.columns) == pocket_vectors_columns
    assert len(pocket_vectors) == n_vectors


@pytest.mark.parametrize('path_mol2, path_pdb, chain_id, angles_mean', [
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ABL1/2g2i_chainA/pocket.mol2', 
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ABL1/2g2i_chainA/protein_pymol.pdb', 
        'A', 
        95.07
    )
])
def test_get_vertex_angles(path_mol2, path_pdb, chain_id, angles_mean):
    """
    Test if vertex angles are calculated correctly (check mean angle), and if returned DataFrame contains correct column
    name.

    Parameters
    ----------
    path_mol2 : pathlib.Path
        Path to mol2 file.
    path_pdb : pathlib.Path
        Path to cif file.
    chain_id : str
        Chain ID.
    angles_mean : float
        Mean of non-None angles.
    """

    path_klifs_metadata = PATH_TEST_DATA / 'klifs_metadata.csv'

    klifs_molecule_loader = KlifsMoleculeLoader()
    klifs_molecule_loader.from_file(path_mol2, path_klifs_metadata)
    pdb_chain_loader = PdbChainLoader()
    pdb_chain_loader.from_file(path_pdb, chain_id)

    feature = SideChainOrientationFeature()
    pocket_residues = feature._get_pocket_residues(klifs_molecule_loader.molecule, pdb_chain_loader.chain)
    pocket_vectors = feature._get_pocket_vectors(pocket_residues, pdb_chain_loader.chain)
    angles_calculated = feature._get_vertex_angles(pocket_vectors)

    assert list(angles_calculated.columns) == ['vertex_angle']

    # Calculate and test mean of all angles (excluding NaN values)
    angles_mean_calculated = angles_calculated.vertex_angle.mean()
    assert np.isclose(angles_mean_calculated, angles_mean, rtol=1e-03)


@pytest.mark.parametrize('vertex_angles', [
    pd.DataFrame([0.0]*85, index=range(1, 86), columns=['sco'])  # Wrong column
])
def test_get_categories_valueerror(vertex_angles):
    """
    Test if exception are raised.

    Parameters
    ----------
    vertex_angles : pandas.DataFrame
        Vertex angles (column) for up to 85 residues (rows).
    """

    with pytest.raises(ValueError):
        feature = SideChainOrientationFeature()
        feature._get_categories(vertex_angles)


@pytest.mark.parametrize('vertex_angles, categories', [
    (
        pd.DataFrame([0.0]*85, index=range(1, 86), columns=['vertex_angle']),
        pd.DataFrame([0.0]*85, index=range(1, 86), columns=['sco']),
    )
])
def test_get_categories(vertex_angles, categories):
    """
    Test transformation of vertex angles to categories (for side chain orientation).

    Parameters
    ----------
    vertex_angles : pandas.DataFrame
        Vertex angles (column) for up to 85 residues (rows).
    categories : pandas.DataFrame
        Side chain orientation categories (column) for up to 85 residues (rows).
    """

    feature = SideChainOrientationFeature()
    categories_calculated = feature._get_categories(vertex_angles)

    assert categories_calculated.equals(categories)


@pytest.mark.parametrize('vertex_angle, category', [
    (0.0, 0.0),
    (1.0, 0.0),
    (45.0, 0.0),
    (46.0, 1.0),
    (90.0, 1.0),
    (91.0, 2.0),
    (180.0, 2.0),
    (np.nan, np.nan)
])
def test_get_category_from_vertex_angle(vertex_angle, category):
    """
    Test tranformation of vertex angle to category (for side chain orientation).

    Parameters
    ----------
    vertex_angle : float
        Vertex angle between a residue's CA atom (vertex), side chain centroid and pocket centroid. Ranges between
        0.0 and 180.0.
    category : float
        Side chain orientation towards pocket: Inwards (category 0.0), intermediate (category 1.0), and outwards
        (category 2.0).
    """

    feature = SideChainOrientationFeature()
    category_calculated = feature._get_category_from_vertex_angle(vertex_angle)

    if not np.isnan(vertex_angle):
        assert isinstance(category_calculated, float)
        assert category_calculated == category
    else:
        assert np.isnan(category_calculated)


@pytest.mark.parametrize('path_mol2, path_pdb, chain_id', [
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ABL1/2g2i_chainA/pocket.mol2', 
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ABL1/2g2i_chainA/protein_pymol.pdb', 
        'A'
    )
])
def test_from_molecule(path_mol2, path_pdb, chain_id):
    """
    Test if SideChainOrientation attributes features and features_verbose contain the correct column names.
    Values are tested already in other test_sidechainorientation_* unit tests.

    Parameters
    ----------
    path_mol2 : pathlib.Path
        Path to mol2 file.
    path_pdb : pathlib.Path
        Path to cif file.
    chain_id : str
        Chain ID.
    """

    path_klifs_metadata = PATH_TEST_DATA / 'klifs_metadata.csv'

    klifs_molecule_loader = KlifsMoleculeLoader()
    klifs_molecule_loader.from_file(path_mol2, path_klifs_metadata)
    pdb_chain_loader = PdbChainLoader()
    pdb_chain_loader.from_file(path_pdb, chain_id)

    feature = SideChainOrientationFeature()
    feature.from_molecule(klifs_molecule_loader.molecule, pdb_chain_loader.chain)

    # Check column names
    features_columns = ['sco']
    features_verbose_columns = 'klifs_id res_id res_name ca side_chain_centroid pocket_centroid vertex_angle sco'.split()

    # Test column names
    assert list(feature.features.columns) == features_columns
    assert list(feature.features_verbose.columns) == features_verbose_columns
