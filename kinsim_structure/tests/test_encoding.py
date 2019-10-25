"""
Unit and regression test for kinsim_structure.encoding classes and methods.
"""

import math
import numpy as np
import pandas as pd
from pathlib import Path
import pytest

from Bio.PDB import Vector

from kinsim_structure.encoding import Fingerprint, FEATURE_NAMES
from kinsim_structure.auxiliary import KlifsMoleculeLoader, PdbChainLoader
from kinsim_structure.encoding import PhysicoChemicalFeatures, SpatialFeatures
from kinsim_structure.encoding import PharmacophoreSizeFeatures, SideChainOrientationFeature
from kinsim_structure.encoding import ObsoleteSideChainAngleFeature


@pytest.mark.parametrize('fingerprint_type1, normalized_fingerprint_type1', [
    (
        pd.DataFrame(
            [
                [3, 3, 2, 1, 1, 1, 180, 1, 35, 35, 35, 35]
            ],
            columns=FEATURE_NAMES
        ),
        pd.DataFrame(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            ],
            columns=FEATURE_NAMES
        )
    ),
    (
        pd.DataFrame(
            [
                [np.nan]*12
            ],
            columns=FEATURE_NAMES
        ),
        pd.DataFrame(
            [
                [np.nan]*12
            ],
            columns=FEATURE_NAMES
        )
    ),
    (
            pd.DataFrame(
                [
                    [3, 3, 2, 1, 1, 1, 180, 1, 36, 36, 36, 36]
                ],
                columns=FEATURE_NAMES
            ),
            pd.DataFrame(
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                ],
                columns=FEATURE_NAMES
            )
    ),
])
def test_normalize_fingerprint_type1(fingerprint_type1, normalized_fingerprint_type1):
    """
    Test normalization of fingerprint type 1 (physicochemical and distance bits).

    Parameters
    ----------
    fingerprint : pandas.DataFrame
        Fingerprint.
    normalized_fingerprint : pandas.DataFrame
        Normalized fingerprint.
    """

    # Set fingerprint
    fp = Fingerprint()
    fp.molecule_code = 'molecule'
    fp.fingerprint_type1 = fingerprint_type1

    assert fp._normalize_fingerprint_type1().equals(normalized_fingerprint_type1)


@pytest.mark.parametrize('fingerprint_type1, fingerprint_type2', [
    (
        pd.DataFrame(
            [
                [3, 3, 2, 1, 1, 1, 180, 1, 35, 35, 35, 35],
                [3, 3, 2, 1, 1, 1, 180, 1, 35, 35, 35, 35],
                [3, 3, 2, 1, 1, 1, 180, 1, 35, 35, 35, 35],
                [3, 3, 2, 1, 1, 1, 180, 1, 35, 35, 35, 35],
                [3, 3, 2, 1, 1, 1, 180, 1, 35, 35, 35, 35],

            ],
            columns=FEATURE_NAMES
        ),
        {
                'physchem': pd.DataFrame(
                    [
                        [3, 3, 2, 1, 1, 1, 180, 1],
                        [3, 3, 2, 1, 1, 1, 180, 1],
                        [3, 3, 2, 1, 1, 1, 180, 1],
                        [3, 3, 2, 1, 1, 1, 180, 1],
                        [3, 3, 2, 1, 1, 1, 180, 1]
                    ],
                    columns=FEATURE_NAMES[:8]
                ),
                'moments': pd.DataFrame(
                    [
                        [35.0, 35.0, 35.0, 35.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0]
                    ],
                    columns=FEATURE_NAMES[8:],
                    index='moment1 moment2 moment3'.split()
                ).transpose()
        }
    )
])
def test_get_fingerprint_type2(fingerprint_type1, fingerprint_type2):
    """
    Test fingerprint type 2 generation (physicochemical and distance moments bits).

    Parameters
    ----------
    fingerprint_type1 : pandas.DataFrame
        Fingerprint type 1.
    fingerprint_type2 : dict of pandas.DataFrame
        Fingerprint type 2.
    """

    # Set fingerprint
    fp = Fingerprint()
    fp.molecule_code = 'molecule'
    fp.fingerprint_type1 = fingerprint_type1

    assert fp._get_fingerprint_type2()['physchem'].equals(fingerprint_type2['physchem'])
    assert fp._get_fingerprint_type2()['moments'].equals(fingerprint_type2['moments'])


@pytest.mark.parametrize('fingerprint_type2, normalized_fingerprint_type2', [
    (
        {
                'physchem': pd.DataFrame(
                    [
                        [3, 3, 2, 1, 1, 1, 180, 1],
                        [3, 3, 2, 1, 1, 1, 180, 1],
                        [3, 3, 2, 1, 1, 1, 180, 1],
                        [3, 3, 2, 1, 1, 1, 180, 1],
                        [3, 3, 2, 1, 1, 1, 180, 1]
                    ],
                    columns=FEATURE_NAMES[:8]
                ),
                'moments': pd.DataFrame(
                    [
                        [35.0, 35.0, 35.0, 35.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0]
                    ],
                    columns=FEATURE_NAMES[8:],
                    index='moment1 moment2 moment3'.split()
                ).transpose()
        },
        {
                'physchem': pd.DataFrame(
                    [
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                    ],
                    columns=FEATURE_NAMES[:8]
                ),
                'moments': pd.DataFrame(
                    [
                        [35.0, 35.0, 35.0, 35.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0]
                    ],
                    columns=FEATURE_NAMES[8:],
                    index='moment1 moment2 moment3'.split()
                ).transpose()
        }
    )
])
def test_normalize_fingerprint_type2(fingerprint_type2, normalized_fingerprint_type2):
    pass


@pytest.mark.parametrize('mol2_filename, pdb_filename, chain_id', [
    ('ABL1/2g2i_chainA/pocket.mol2', '2g2i.cif', 'A')
])
def test_physicochemicalfeatures_from_molecule(mol2_filename, pdb_filename, chain_id):
    """
    Test length (85 rows for 85 KLIFS residues) and columns names of features DataFrame.
    Values are tested already in respective feature unit test.

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

    physicochemicalfeatures = PhysicoChemicalFeatures()
    physicochemicalfeatures.from_molecule(klifs_molecule_loader.molecule, pdb_chain_loader.chain)
    features = physicochemicalfeatures.features

    physiochemicalfeatures_columns = 'size hbd hba charge aromatic aliphatic sco exposure'.split()
    assert list(features.columns) == physiochemicalfeatures_columns
    assert len(features) == 85


@pytest.mark.parametrize('mol2_filename', [
    'ABL1/2g2i_chainA/pocket.mol2'
])
def test_spatialfeatures_from_molecule(mol2_filename):
    """
    Test length (85 rows for 85 KLIFS residues) and columns names of features DataFrame.
    Values are tested already in respective feature unit test.

    Parameters
    ----------
    mol2_filename : str
        Path to mol2 file.
    """
    mol2_path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / mol2_filename

    klifs_molecule_loader = KlifsMoleculeLoader(mol2_path=mol2_path)

    spatialfeatures = SpatialFeatures()
    spatialfeatures.from_molecule(klifs_molecule_loader.molecule)
    features = spatialfeatures.features

    spatialfeatures_columns = [
        'distance_to_centroid',
        'distance_to_hinge_region',
        'distance_to_dfg_region',
        'distance_to_front_pocket'
    ]
    assert list(features.columns) == spatialfeatures_columns
    assert len(features) == 85


@pytest.mark.parametrize('filename, residue, feature_type, feature', [
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'ALA', 'size', 1),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'ASN', 'size', 2),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'ARG', 'size', 3),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'PTR', 'size', 3),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'MSE', 'size', 2),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'XXX', 'size', None),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'ALA', 'hbd', 0),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'ASN', 'hbd', 1),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'ARG', 'hbd', 3),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'XXX', 'hbd', None),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'ALA', 'hba', 0),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'ASN', 'hba', 1),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'ASP', 'hba', 2),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'XXX', 'hba', None),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'ALA', 'charge', 0),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'ARG', 'charge', 1),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'ASP', 'charge', -1),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'XXX', 'charge', None),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'ALA', 'aromatic', 0),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'HIS', 'aromatic', 1),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'XXX', 'aromatic', None),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'ARG', 'aliphatic', 0),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'ALA', 'aliphatic', 1),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'XXX', 'aliphatic', None)

])
def test_pharmacophoresizefeature_from_residue(filename, residue, feature_type, feature):
    """
    Test function for retrieval of residue's size and pharmacophoric features (i.e. number of hydrogen bond donor,
    hydrogen bond acceptors, charge features, aromatic features or aliphatic features )

    Parameters
    ----------
    filename : str
        Path to file originating from test data folder.
    residue : str
        Three-letter code for residue.
    feature_type : str
        Feature type name.
    feature : int or None
        Feature value.
    """

    # Load molecule
    mol2_path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / filename

    klifs_molecule_loader = KlifsMoleculeLoader(mol2_path=mol2_path)
    molecule = klifs_molecule_loader.molecule

    # Get pharmacophore and size features
    pharmacophore_size_feature = PharmacophoreSizeFeatures()
    pharmacophore_size_feature.from_molecule(molecule)

    # Call feature from residue function
    assert pharmacophore_size_feature.from_residue(residue, feature_type) == feature


@pytest.mark.parametrize('mol2_filename, pdb_filename, chain_id, res_id_mean, n_pocket_atoms', [
    ('ABL1/2g2i_chainA/pocket.mol2', '2g2i.cif', 'A', 315.95, 659)
])
def test_sidechainorientation_get_pocket_residues(mol2_filename, pdb_filename, chain_id, res_id_mean, n_pocket_atoms):
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
def test_sidechainorientation_get_ca(pdb_filename, chain_id, residue_id, ca):
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


@pytest.mark.parametrize('pdb_filename, chain_id, residue_id, side_chain_centroid', [
    ('5i35.cif', 'A', 336, [65.77, 23.74, 21.13]),  # Residue has enough side chain atoms for centroid calculation
    ('5i35.cif', 'A', 337, None),  # Residue has <= 1 side chain atoms
    ('5i35.cif', 'A', 357, [59.72, 14.73, 22.72]),  # Non-standard amino acid
    ('5l4q.cif', 'A', 57, [-27.53, 0.05, -41.01]),  # Side chain containing H atoms
    ('5l4q.cif', 'A', 130, None)  # Side chain with too many missing residues
])
def test_sidechainorientation_get_side_chain_centroid(pdb_filename, chain_id, residue_id, side_chain_centroid):
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
    side_chain_centroid_calculated = feature._get_side_chain_centroid(residue)
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
def test_sidechainorientation_get_pocket_centroid(mol2_filename, pdb_filename, chain_id, pocket_centroid):
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
        {'ca': -16.21, 'side_chain_centroid': -16.18, 'pocket_centroid': -16.21}
    )
])
def test_sidechainorientation_get_pocket_vectors(mol2_filename, pdb_filename, chain_id, x_mean):
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
    pocket_vectors = feature._get_pocket_vectors(pocket_residues)

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
    assert np.isclose(x_mean_calculated['side_chain_centroid'], x_mean['side_chain_centroid'], rtol=1e-03)

    # Test mean x coordinate of pocket centroid
    assert np.isclose(x_mean_calculated['pocket_centroid'], x_mean['pocket_centroid'], rtol=1e-03)


@pytest.mark.parametrize('mol2_filename, pdb_filename, chain_id, angles_mean', [
    ('ABL1/2g2i_chainA/pocket.mol2', '2g2i.cif', 'A', 93.75)
])
def test_sidechainorientation_get_vertex_angles(mol2_filename, pdb_filename, chain_id, angles_mean):
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
    pocket_vectors = feature._get_pocket_vectors(pocket_residues)
    angles_calculated = feature._get_vertex_angles(pocket_vectors)

    assert list(angles_calculated.columns) == ['sco']

    # Calculate and test mean of all angles (excluding NaN values)
    angles_mean_calculated = angles_calculated.sco.mean()
    assert np.isclose(angles_mean_calculated, angles_mean, rtol=1e-03)


@pytest.mark.parametrize('mol2_filename, pdb_filename, chain_id', [
    ('ABL1/2g2i_chainA/pocket.mol2', '2g2i.cif', 'A')
])
def test_sidechainorientation_from_molecule(mol2_filename, pdb_filename, chain_id):
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
    features_verbose_columns = ['klifs_id', 'res_id', 'res_name', 'ca', 'side_chain_centroid', 'pocket_centroid', 'sco']

    # Test column names
    assert list(feature.features.columns) == features_columns
    assert list(feature.features_verbose.columns) == features_verbose_columns


@pytest.mark.parametrize('mol2_filename, pdb_filename, chain_id, sca', [
    (
        'ABL1/2g2i_chainA/pocket.mol2',
        '2g2i.cif',
        'A',
        pd.DataFrame(
            [[1, 'HIS', 110.55], [4, 'GLY', 180.00], [15, 'ALA', 180.00]],
            columns='klifs_id residue_name sca'.split()
        )
    )
])
def test_side_chain_angle_feature(mol2_filename, pdb_filename, chain_id, sca):
    """
    Test if side chain angles are assigned correctly (also for special cases, i.e. GLY and ALA).

    mol2_filename : str
        Path to mol2 file.
    pdb_filename : str
        Path to cif file.
    chain_id : str
        Chain ID.
    sca : pandas.DataFrame
        Side chain angles, KLIFS IDs and residue names (columns) of selected residues (rows).
    """

    mol2_path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / mol2_filename
    pdb_path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / pdb_filename

    klifs_molecule_loader = KlifsMoleculeLoader(mol2_path=mol2_path)
    pdb_chain_loader = PdbChainLoader(pdb_path=pdb_path, chain_id=chain_id)

    sca_feature = ObsoleteSideChainAngleFeature()
    sca_feature.from_molecule(klifs_molecule_loader.molecule, pdb_chain_loader.chain)
    sca_calculated = sca_feature.features_verbose

    for index, row in sca.iterrows():
        assert sca_calculated[sca_calculated.klifs_id == row.klifs_id].sca.values[0] == row.sca


@pytest.mark.parametrize('filename, reference_point_name, anchor_residue_klifs_ids, x_coordinate', [
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'hinge_region', [16, 47, 80], 6.25545),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'dfg_region', [20, 23, 81], 11.6846),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'front_pocket', [6, 48, 75], float('nan'))
])
def test_get_anchor_atoms(filename, reference_point_name, anchor_residue_klifs_ids, x_coordinate):
    """
    Test function that calculates the anchor atoms for different scenarios (missing anchor residues, missing neighbors)

    Parameters
    ----------
    filename : str
        Path to file originating from test data folder.
    reference_point_name : str
        Reference point name, e.g. 'hinge_region'.
    anchor_residue_klifs_ids : list of int
        List of KLIFS IDs that are used to calculate a given reference point.
    x_coordinate: float
        X coordinate of first anchor atom.
    """

    mol2_path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / filename

    klifs_molecule_loader = KlifsMoleculeLoader(mol2_path=mol2_path)
    molecule = klifs_molecule_loader.molecule

    # Delete residues

    # Case: Missing anchor residue but neighboring residues available
    molecule.df.drop(molecule.df[molecule.df.klifs_id == 16].index, inplace=True)

    # Case: Missing anchor residue but neighboring residues available
    molecule.df.drop(molecule.df[molecule.df.klifs_id.isin([18, 19])].index, inplace=True)

    # Case: Missing anchor residue but neighboring residues available
    molecule.df.drop(molecule.df[molecule.df.klifs_id.isin([24, 25])].index, inplace=True)

    # Case: Missing anchor and neighboring residues
    molecule.df.drop(molecule.df[molecule.df.klifs_id.isin([5, 6, 7])].index, inplace=True)

    # Get spatial features
    spatial_features = SpatialFeatures()
    spatial_features.from_molecule(molecule)

    # Get anchor atoms
    anchors = spatial_features.get_anchor_atoms(molecule)

    assert list(anchors[reference_point_name].index) == anchor_residue_klifs_ids

    # Ugly workaround to test NaN values in anchors
    if math.isnan(x_coordinate):
        assert math.isnan(anchors[reference_point_name].loc[anchor_residue_klifs_ids[0], 'x'])
    else:
        assert anchors[reference_point_name].loc[anchor_residue_klifs_ids[0], 'x'] == x_coordinate


@pytest.mark.parametrize('filename, x_coordinate', [
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 1.02664)
])
def test_get_reference_points(filename, x_coordinate):
    """
    Test calculation of reference point "centroid" of a pocket.

    Parameters
    ----------
    filename : str
        Path to file originating from test data folder.
    x_coordinate: float
        X coordinate of the centroid.
    """

    mol2_path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / filename

    klifs_molecule_loader = KlifsMoleculeLoader(mol2_path=mol2_path)
    molecule = klifs_molecule_loader.molecule

    # Get spatial features
    spatial_features = SpatialFeatures()
    spatial_features.from_molecule(molecule)

    # Get reference points
    reference_points = spatial_features.get_reference_points(molecule)
    print(reference_points.centroid.x)

    assert np.isclose(reference_points.centroid.x, x_coordinate, rtol=1e-04)
