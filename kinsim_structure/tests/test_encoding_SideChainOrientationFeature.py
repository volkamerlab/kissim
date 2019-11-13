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


@pytest.mark.parametrize('mol2_filename, pdb_filename, chain_id, res_id_mean, n_pocket_atoms', [
    ('ABL1/2g2i_chainA/pocket.mol2', '2g2i.cif', 'A', 315.95, 659)
])
def test_get_pocket_residues(mol2_filename, pdb_filename, chain_id, res_id_mean, n_pocket_atoms):
    """
    Test the mean of the pocket's PDB residue IDs and the number of pocket atoms.

    Parameters
    ----------
    mol2_filename : str
        Path to mol2 file.
    pdb_filename : str
        Path to cif file.
    chain_id : str
        Chain ID.
    res_id_mean : float
        Mean of pocket's PDB residue IDs.
    n_pocket_atoms : int
        Number of pocket atoms.
    """

    mol2_path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / mol2_filename
    pdb_path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / pdb_filename

    klifs_molecule_loader = KlifsMoleculeLoader(mol2_path=mol2_path)
    pdb_chain_loader = PdbChainLoader(pdb_path=pdb_path, chain_id=chain_id)

    feature = SideChainOrientationFeature()
    pocket_residues = feature._get_pocket_residues(klifs_molecule_loader.molecule, pdb_chain_loader.chain)

    # Get and test the mean of pocket PDB residue IDs and the number of pocket atoms
    res_id_mean_calculated = pocket_residues.res_id.mean()
    n_pocket_atoms_calculated = sum([len(residue) for residue in pocket_residues.pocket_residues])

    assert np.isclose(res_id_mean_calculated, res_id_mean, rtol=1e-03)
    assert n_pocket_atoms_calculated == n_pocket_atoms


@pytest.mark.parametrize('pdb_filename, chain_id, residue_id, ca', [
    ('2yjr.cif', 'A', 1272, [41.08, 39.79, 10.72]),  # Residue has CA
    ('2yjr.cif', 'A', 1273, None)  # Residue has no CA
])
def test_get_ca(pdb_filename, chain_id, residue_id, ca):
    """
    Test if CA atom is retrieved correctly from a residue (test if-else cases).

    Parameters
    ----------
    pdb_filename : str
        Path to cif file.
    chain_id : str
        Chain ID.
    residue_id : int
        Residue ID.
    ca : list of int or None
        3D coordinates of CA atom.
    """

    pdb_path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / pdb_filename

    pdb_chain_loader = PdbChainLoader(pdb_path=pdb_path, chain_id=chain_id)

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


@pytest.mark.parametrize('pdb_filename, chain_id, residue_id', [
    ('5i35.cif', 'A', 337),  # ALA
    ('5i35.cif', 'A', 357),  # Non-standard residue
])
def test_get_pcb_from_gly_valueerror(pdb_filename, chain_id, residue_id):
    """
    Test exceptions in pseudo-CB calculation for GLY.

    Parameters
    ----------
    pdb_filename : str
        Path to cif file.
    chain_id : str
        Chain ID.
    residue_id : int
        Residue ID.
    """

    pdb_path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / pdb_filename

    pdb_chain_loader = PdbChainLoader(pdb_path=pdb_path, chain_id=chain_id)

    chain = pdb_chain_loader.chain
    try:
        residue = chain[residue_id]
    except KeyError:
        # For non-standard residue MSE indexing did not work, thus use this workaround
        residue = [i for i in chain.get_list() if i.get_id()[1] == residue_id][0]

    with pytest.raises(ValueError):
        feature = SideChainOrientationFeature()
        feature._get_pcb_from_gly(residue)


@pytest.mark.parametrize('pdb_filename, chain_id, residue_id, ca_pcb', [
    ('5i35.cif', 'A', 272, np.array([52.24, 30.20, 32.25])),  # GLY
])
def test_get_pcb_from_gly(pdb_filename, chain_id, residue_id, ca_pcb):
    """
    Test pseudo-CB calculation for GLY.

    Parameters
    ----------
    pdb_filename : str
        Path to cif file.
    chain_id : str
        Chain ID.
    residue_id : int
        Residue ID.
    ca_pcb : numpy.array
        Pseudo-CB atom coordinates.
    """

    pdb_path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / pdb_filename

    pdb_chain_loader = PdbChainLoader(pdb_path=pdb_path, chain_id=chain_id)

    chain = pdb_chain_loader.chain
    try:
        residue = chain[residue_id]
    except KeyError:
        # For non-standard residue MSE indexing did not work, thus use this workaround
        residue = [i for i in chain.get_list() if i.get_id()[1] == residue_id][0]

    feature = SideChainOrientationFeature()
    ca_pcb_calculated = feature._get_pcb_from_gly(residue)

    assert np.isclose(ca_pcb_calculated.get_array().mean(), ca_pcb.mean(), rtol=1e-04)


@pytest.mark.parametrize('pdb_filename, chain_id, residue_id, ca_pcb', [
    ('5i35.cif', 'A', 272, np.array([52.24, 30.20, 32.25])),  # GLY
    ('5i35.cif', 'A', 337, np.array([64.06, 27.13, 23.97])),  # Residue is ALA with CB
    ('4jik.cif', 'A', 19, None),  # Residue is ALA without CB
])
def test_get_pcb_from_residue(pdb_filename, chain_id, residue_id, ca_pcb):
    """
    Test pseudo-CB calculation for a residue.

    Parameters
    ----------
    pdb_filename : str
        Path to cif file.
    chain_id : str
        Chain ID.
    residue_id : int
        Residue ID.
    ca_pcb : numpy.array
        Pseudo-CB atom coordinates.
    """

    pdb_path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / pdb_filename

    pdb_chain_loader = PdbChainLoader(pdb_path=pdb_path, chain_id=chain_id)

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


@pytest.mark.parametrize('pdb_filename, chain_id, residue_id, side_chain_centroid', [
    ('5i35.cif', 'A', 272, [52.24, 30.20, 32.25]),  # GLY with pCB
    ('3ot8.cif', 'A', 18, None),  # GLY without pCB (missing C and CA)
    ('5i35.cif', 'A', 337, [63.66, 26.90, 23.41]),  # ALA with CB
    ('3nlb.cif', 'A', 19, [9.77, -10.47, 10.16]),  # ALA with pCB
    ('4jik.cif', 'A', 19, None),  # ALA without pCB (missing CA)
    ('5i35.cif', 'A', 336, [65.77, 23.74, 21.13]),  # Standard residue (side chain) with enough atoms
    ('5l4q.cif', 'A', 130, [-15.11, -1.78, -18.79]),  # Standard residue with too few atoms but CB atom
    ('4yhf.cif', 'B', 421, [-2.95, -2.36, -14.49]),  # Standard residue with too few atoms, no CB atom, but pCB atom
    ('4jik.cif', 'A', 51, None),  # Standard residue with too few atoms and no CB and pCB atom
    ('5i35.cif', 'A', 357, [59.72, 14.73, 22.72]),  # Non-standard residue with enough atoms (>0)
    #('xxxx.cif', 'X', 0, None),  # Non-standard residue side chain with no atoms
    ('5l4q.cif', 'A', 57, [-27.53, 0.05, -41.01]),  # Side chain containing H atoms
])
def test_get_side_chain_centroid(pdb_filename, chain_id, residue_id, side_chain_centroid):
    """
    Test if side chain centroid is retrieved correctly from a residue (test if-else cases).

    Parameters
    ----------
    pdb_filename : str
        Path to cif file.
    chain_id : str
        Chain ID.
    residue_id : int
        Residue ID.
    side_chain_centroid : list of int or None
        3D coordinates of CA atom.
    """

    pdb_path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / pdb_filename

    pdb_chain_loader = PdbChainLoader(pdb_path=pdb_path, chain_id=chain_id)

    chain = pdb_chain_loader.chain
    try:
        residue = chain[residue_id]
    except KeyError:
        # For non-standard residue MSE indexing did not work, thus use this workaround
        residue = [i for i in chain.get_list() if i.get_id()[1] == residue_id][0]

    feature = SideChainOrientationFeature()
    side_chain_centroid_calculated = feature._get_side_chain_centroid(residue, chain)
    print(side_chain_centroid_calculated)

    if side_chain_centroid_calculated and side_chain_centroid:
        # Check only x coordinate
        assert np.isclose(list(side_chain_centroid_calculated)[0], side_chain_centroid[0], rtol=1e-03)
        assert isinstance(side_chain_centroid_calculated, Vector)
    else:
        assert side_chain_centroid_calculated == side_chain_centroid


@pytest.mark.parametrize('mol2_filename, pdb_filename, chain_id, pocket_centroid', [
    ('ABL1/2g2i_chainA/pocket.mol2', '2g2i.cif', 'A', [-16.21, -32.25, -12.14])
])
def test_get_pocket_centroid(mol2_filename, pdb_filename, chain_id, pocket_centroid):
    """
    Test pocket centroid calculation.

    Parameters
    ----------
    mol2_filename : str
        Path to mol2 file.
    pdb_filename : str
        Path to cif file.
    chain_id : str
        Chain ID.
    pocket_centroid : list of float
        Pocket centroid coordinates.
    """

    mol2_path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / mol2_filename
    pdb_path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / pdb_filename

    klifs_molecule_loader = KlifsMoleculeLoader(mol2_path=mol2_path)
    pdb_chain_loader = PdbChainLoader(pdb_path=pdb_path, chain_id=chain_id)

    feature = SideChainOrientationFeature()
    pocket_residues = feature._get_pocket_residues(klifs_molecule_loader.molecule, pdb_chain_loader.chain)
    pocket_centroid_calculated = feature._get_pocket_centroid(pocket_residues)

    if pocket_centroid_calculated and pocket_centroid:
        # Check only x coordinate
        assert np.isclose(list(pocket_centroid_calculated)[0], pocket_centroid[0], rtol=1e-03)
        assert isinstance(pocket_centroid_calculated, Vector)
    else:
        assert pocket_centroid_calculated == pocket_centroid


@pytest.mark.parametrize('mol2_filename, pdb_filename, chain_id, x_mean', [
    (
        'ABL1/2g2i_chainA/pocket.mol2',
        '2g2i.cif',
        'A',
        {'ca': -16.2129, 'side_chain_centroid': -16.2073, 'pocket_centroid': -16.2129}
    )
])
def test_get_pocket_vectors(mol2_filename, pdb_filename, chain_id, x_mean):
    """
    Test if pocket vectors are calculated correctly (check mean x coordinates of all CA atoms, side chain centroids,
    and pocket centroid in the pocket), and if returned DataFrame contains correct column names.

    Parameters
    ----------
    mol2_filename : str
        Path to mol2 file.
    pdb_filename : str
        Path to cif file.
    chain_id : str
        Chain ID.
    x_mean : dict
        X coordinates of all CA atoms, side chain centroids, and pocket centroid in the pocket.
    """

    mol2_path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / mol2_filename
    pdb_path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / pdb_filename

    klifs_molecule_loader = KlifsMoleculeLoader(mol2_path=mol2_path)
    pdb_chain_loader = PdbChainLoader(pdb_path=pdb_path, chain_id=chain_id)

    feature = SideChainOrientationFeature()
    pocket_residues = feature._get_pocket_residues(klifs_molecule_loader.molecule, pdb_chain_loader.chain)
    pocket_vectors = feature._get_pocket_vectors(pocket_residues, pdb_chain_loader.chain)

    # Test if DataFrame contains correct columns
    pocket_vectors_columns = ['klifs_id', 'res_id', 'res_name', 'ca', 'side_chain_centroid', 'pocket_centroid']
    assert list(pocket_vectors.columns) == pocket_vectors_columns

    # Calculate x coordinate mean values for all three vector lists
    x_mean_calculated = {
        'ca': pocket_vectors.ca.dropna().apply(lambda x: x.get_array()[0]).mean(),
        'side_chain_centroid': pocket_vectors.side_chain_centroid.dropna().apply(lambda x: x.get_array()[0]).mean(),
        'pocket_centroid': pocket_vectors.pocket_centroid.dropna().apply(lambda x: x.get_array()[0]).mean()
    }

    # Test mean x coordinate of CA atoms
    assert np.isclose(x_mean_calculated['ca'], x_mean['ca'], rtol=1e-03)

    # Test mean x coordinate of side chain centroid
    assert np.isclose(x_mean_calculated['side_chain_centroid'], x_mean['side_chain_centroid'], rtol=1e-06)

    # Test mean x coordinate of pocket centroid
    assert np.isclose(x_mean_calculated['pocket_centroid'], x_mean['pocket_centroid'], rtol=1e-03)


@pytest.mark.parametrize('mol2_filename, pdb_filename, chain_id, angles_mean', [
    ('ABL1/2g2i_chainA/pocket.mol2', '2g2i.cif', 'A', 95.07)
])
def test_get_vertex_angles(mol2_filename, pdb_filename, chain_id, angles_mean):
    """
    Test if vertex angles are calculated correctly (check mean angle), and if returned DataFrame contains correct column
    name.

    Parameters
    ----------
    mol2_filename : str
        Path to mol2 file.
    pdb_filename : str
        Path to cif file.
    chain_id : str
        Chain ID.
    angles_mean : float
        Mean of non-None angles.
    """

    mol2_path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / mol2_filename
    pdb_path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / pdb_filename

    klifs_molecule_loader = KlifsMoleculeLoader(mol2_path=mol2_path)
    pdb_chain_loader = PdbChainLoader(pdb_path=pdb_path, chain_id=chain_id)

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


@pytest.mark.parametrize('mol2_filename, pdb_filename, chain_id', [
    ('ABL1/2g2i_chainA/pocket.mol2', '2g2i.cif', 'A')
])
def test_from_molecule(mol2_filename, pdb_filename, chain_id):
    """
    Test if SideChainOrientation attributes features and features_verbose contain the correct column names.
    Values are tested already in other test_sidechainorientation_* unit tests.

    Parameters
    ----------
    mol2_filename : str
        Path to mol2 file.
    pdb_filename : str
        Path to cif file.
    chain_id : str
        Chain ID.
    """

    mol2_path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / mol2_filename
    pdb_path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / pdb_filename

    klifs_molecule_loader = KlifsMoleculeLoader(mol2_path=mol2_path)
    pdb_chain_loader = PdbChainLoader(pdb_path=pdb_path, chain_id=chain_id)

    feature = SideChainOrientationFeature()
    feature.from_molecule(klifs_molecule_loader.molecule, pdb_chain_loader.chain)

    # Check column names
    features_columns = ['sco']
    features_verbose_columns = 'klifs_id res_id res_name ca side_chain_centroid pocket_centroid vertex_angle sco'.split()

    # Test column names
    assert list(feature.features.columns) == features_columns
    assert list(feature.features_verbose.columns) == features_verbose_columns
