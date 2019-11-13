"""
Unit and regression test for kinsim_structure.similarity.AllAgainstAllComparison methods.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kinsim_structure.auxiliary import KlifsMoleculeLoader, PdbChainLoader
from kinsim_structure.encoding import Fingerprint, FingerprintGenerator
from kinsim_structure.similarity import FeatureDistances, FingerprintDistance, FeatureDistancesGenerator, FingerprintDistanceGenerator


def generate_fingerprints_from_files(mol2_filenames, pdb_filenames, chain_ids):
    """
    Helper function: Generate multiple fingerprints from files.

    Parameters
    ----------
    mol2_filenames : list of str
        Paths to multiple mol2 files.
    pdb_filenames : list of str
        Paths to multiple cif files.
    chain_ids : list of str
        Multiple chain IDs.

    Returns
    -------
    list of kinsim_structure.encoding.Fingerprint
        List of fingerprints.
    """

    fingerprints = []

    for mol2_filename, pdb_filename, chain_id in zip(mol2_filenames, pdb_filenames, chain_ids):
        mol2_path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / mol2_filename
        pdb_path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / pdb_filename

        klifs_molecule_loader = KlifsMoleculeLoader(mol2_path=mol2_path)
        pdb_chain_loader = PdbChainLoader(pdb_path=pdb_path, chain_id=chain_id)

        fingerprint = Fingerprint()
        fingerprint.from_molecule(klifs_molecule_loader.molecule, pdb_chain_loader.chain)

        fingerprints.append(fingerprint)

    return fingerprints


@pytest.mark.parametrize('mol2_filenames, pdb_filenames, chain_ids', [
    (
        ['ABL1/2g2i_chainA/pocket.mol2', 'AAK1/4wsq_altA_chainB/pocket.mol2'],
        ['2g2i.cif', '4wsq.cif'],
        ['A', 'B']
    )

])
def test_get_fingerprint_distance(mol2_filenames, pdb_filenames, chain_ids):
    """
    Test if return type is FingerprintDistance class instance.

    Parameters
    ----------
    mol2_filenames : list of str
        Paths to two mol2 files.
    pdb_filenames : list of str
        Paths to two cif files.
    chain_ids : list of str
        Two chain IDs.
    """

    # Fingerprints
    fingerprints = generate_fingerprints_from_files(mol2_filenames, pdb_filenames, chain_ids)

    # FeatureDistances
    feature_distances = FeatureDistances()
    feature_distances.from_fingerprints(fingerprints[0], fingerprints[1])

    # Test
    fingerprint_distance_generator = FingerprintDistanceGenerator()
    fingerprint_distance_calculated = fingerprint_distance_generator._get_fingerprint_distance(
        feature_distances
    )

    assert isinstance(fingerprint_distance_calculated, FingerprintDistance)


@pytest.mark.parametrize('mol2_filenames, pdb_filenames, chain_ids', [
    (
        ['ABL1/2g2i_chainA/pocket.mol2', 'AAK1/4wsq_altA_chainB/pocket.mol2', 'AAK1/4wsq_altA_chainB/pocket.mol2'],
        ['2g2i.cif', '4wsq.cif', '4wsq.cif'],
        ['A', 'B', 'B']
    )

])
def test_get_fingerprint_distance_from_list(mol2_filenames, pdb_filenames, chain_ids):
    """
    Test if return type is instance of list of FeatureDistance class instances.

    Parameters
    ----------
    mol2_filenames : list of str
        Paths to multiple mol2 files.
    pdb_filenames : list of str
        Paths to multiple cif files.
    chain_ids : list of str
        Multiple chain IDs.
    """

    # Fingerprints
    fingerprints = generate_fingerprints_from_files(mol2_filenames, pdb_filenames, chain_ids)

    # FingerprintGenerator
    fingerprint_generator = FingerprintGenerator()
    fingerprint_generator.data = {i.molecule_code: i for i in fingerprints}

    # FeatureDistancesGenerator
    feature_distances_generator = FeatureDistancesGenerator()
    feature_distances_generator.from_fingerprint_generator(fingerprint_generator)
    feature_distances_list = list(feature_distances_generator.data.values())

    # Test
    fingerprint_distance_generator = FingerprintDistanceGenerator()
    fingerprint_distance_list = fingerprint_distance_generator._get_fingerprint_distance_from_list(
        fingerprint_distance_generator._get_fingerprint_distance, feature_distances_list
    )

    assert isinstance(fingerprint_distance_list, list)

    for i in fingerprint_distance_list:
        assert isinstance(i, FingerprintDistance)


@pytest.mark.parametrize(
    'mol2_filenames, pdb_filenames, chain_ids, distance_measure, feature_weights, molecule_codes, kinase_names', [
        (
            [
                'ABL1/2g2i_chainA/pocket.mol2',
                'AAK1/4wsq_altA_chainB/pocket.mol2',
                'AAK1/4wsq_altA_chainB/pocket.mol2'
            ],
            [
                '2g2i.cif',
                '4wsq.cif',
                '4wsq.cif'
            ],
            [
                'A',
                'B',
                'B'
            ],
            'scaled_euclidean',
            None,
            ['HUMAN/AAK1_4wsq_altA_chainB', 'HUMAN/ABL1_2g2i_chainA'],
            ['AAK1', 'ABL1']
        )
    ]
)
def test_from_feature_distances_generator(
        mol2_filenames, pdb_filenames, chain_ids, distance_measure, feature_weights, molecule_codes, kinase_names
):
    """
    Test FingerprintDistanceGenerator class attributes.

    Parameters
    ----------
    mol2_filenames : list of str
        Paths to multiple mol2 files.
    pdb_filenames : list of str
        Paths to multiple cif files.
    chain_ids : list of str
        Multiple chain IDs.
    distance_measure : str
        Type of distance measure, defaults to Euclidean distance.
    feature_weights : dict of float or None
        Feature weights of the following form:
        (i) None
            Default feature weights: All features equally distributed to 1/15 (15 feature in total).
        (ii) By feature type
            Feature types to be set are: physicochemical, distances, and moments.
        (iii) By feature:
            Features to be set are: size, hbd, hba, charge, aromatic, aliphatic, sco, exposure,
            distance_to_centroid, distance_to_hinge_region, distance_to_dfg_region, distance_to_front_pocket,
            moment1, moment2, and moment3.
        For (ii) and (iii): All floats must sum up to 1.0.
    molecule_codes : list of str
        List of molecule codes associated with input fingerprints.
    kinase_names : list of str
        List of kinase names associated with input fingerprints.
    """

    # Fingerprints
    fingerprints = generate_fingerprints_from_files(mol2_filenames, pdb_filenames, chain_ids)

    # FingerprintGenerator
    fingerprint_generator = FingerprintGenerator()
    fingerprint_generator.data = {i.molecule_code: i for i in fingerprints}

    # FeatureDistancesGenerator
    feature_distances_generator = FeatureDistancesGenerator()
    feature_distances_generator.from_fingerprint_generator(fingerprint_generator)

    # FingerprintDistanceGenerator
    fingerprint_distance_generator = FingerprintDistanceGenerator()
    fingerprint_distance_generator.from_feature_distances_generator(feature_distances_generator)

    # Test attributes
    assert fingerprint_distance_generator.distance_measure == distance_measure
    assert fingerprint_distance_generator.feature_weights == feature_weights
    assert fingerprint_distance_generator.molecule_codes == molecule_codes
    assert fingerprint_distance_generator.kinase_names == kinase_names

    assert isinstance(fingerprint_distance_generator.data, pd.DataFrame)
    assert list(fingerprint_distance_generator.data.columns) == 'molecule_code_1 molecule_code_2 distance coverage'.split()


@pytest.mark.parametrize('molecule_codes, data, fill, structure_distance_matrix', [
    (
        'a b c'.split(),
        pd.DataFrame(
            [['a', 'b', 0.5, 1.0], ['a', 'c', 0.75, 1.0], ['b', 'c', 1.0, 1.0]],
            columns='molecule_code_1 molecule_code_2 distance coverage'.split()
        ),
        False,
        pd.DataFrame(
            [[0.0, 0.5, 0.75], [np.nan, 0.0, 1.0], [np.nan, np.nan, 0.0]],
            columns='a b c'.split(),
            index='a b c'.split()
        )
    ),
    (
        'a b c'.split(),
        pd.DataFrame(
            [['a', 'b', 0.5, 1.0], ['a', 'c', 0.75, 1.0], ['b', 'c', 1.0, 1.0]],
            columns='molecule_code_1 molecule_code_2 distance coverage'.split()
        ),
        True,
        pd.DataFrame(
            [[0.0, 0.5, 0.75], [0.5, 0.0, 1.0], [0.75, 1.0, 0.0]],
            columns='a b c'.split(),
            index='a b c'.split()
        )
    )
])
def test_get_structure_distance_matrix(molecule_codes, data, fill, structure_distance_matrix):

    # Set dummy FingerprintDistanceGenerator class attributes
    fingerprint_distance_generator = FingerprintDistanceGenerator()
    fingerprint_distance_generator.molecule_codes = molecule_codes
    fingerprint_distance_generator.data = data

    # Test generation of structure distance matrix
    structure_distance_matrix_calculated = fingerprint_distance_generator.get_structure_distance_matrix(fill)

    assert structure_distance_matrix_calculated.equals(structure_distance_matrix)


@pytest.mark.parametrize('molecule_codes, kinase_names, data, by, fill, structure_distance_matrix', [
    (
        'HUMAN/kinase1_pdb1 HUMAN/kinase1_pdb2 HUMAN/kinase2_pdb1'.split(),
        'kinase1 kinase2'.split(),
        pd.DataFrame(
            [
                ['HUMAN/kinase1_pdb1', 'HUMAN/kinase1_pdb2', 0.5, 1.0],
                ['HUMAN/kinase1_pdb1', 'HUMAN/kinase2_pdb1', 0.75, 1.0],
                ['HUMAN/kinase1_pdb2', 'HUMAN/kinase2_pdb1', 1.0, 1.0]
            ],
            columns='molecule_code_1 molecule_code_2 distance coverage'.split()
        ),
        'minimum',
        False,
        pd.DataFrame(
            [[0.5, 0.75], [np.nan, 0.0]],
            columns='kinase1 kinase2'.split(),
            index='kinase1 kinase2'.split()
        )
    ),  # Minimum
    (
        'HUMAN/kinase1_pdb1 HUMAN/kinase1_pdb2 HUMAN/kinase2_pdb1'.split(),
        'kinase1 kinase2'.split(),
        pd.DataFrame(
            [
                ['HUMAN/kinase1_pdb1', 'HUMAN/kinase1_pdb2', 0.5, 1.0],
                ['HUMAN/kinase1_pdb1', 'HUMAN/kinase2_pdb1', 0.75, 1.0],
                ['HUMAN/kinase1_pdb2', 'HUMAN/kinase2_pdb1', 1.0, 1.0]
            ],
            columns='molecule_code_1 molecule_code_2 distance coverage'.split()
        ),
        'minimum',
        True,
        pd.DataFrame(
            [[0.5, 0.75], [0.75, 0.0]],
            columns='kinase1 kinase2'.split(),
            index='kinase1 kinase2'.split()
            )
    ),  # Fill=True
    (
        'HUMAN/kinase1_pdb1 HUMAN/kinase1_pdb2 HUMAN/kinase2_pdb1'.split(),
        'kinase1 kinase2'.split(),
        pd.DataFrame(
            [
                ['HUMAN/kinase1_pdb1', 'HUMAN/kinase1_pdb2', 0.5, 1.0],
                ['HUMAN/kinase1_pdb1', 'HUMAN/kinase2_pdb1', 0.75, 1.0],
                ['HUMAN/kinase1_pdb2', 'HUMAN/kinase2_pdb1', 1.0, 1.0]
            ],
            columns='molecule_code_1 molecule_code_2 distance coverage'.split()
        ),
        'maximum',
        False,
        pd.DataFrame(
            [[0.5, 1.0], [np.nan, 0.0]],
            columns='kinase1 kinase2'.split(),
            index='kinase1 kinase2'.split()
        )
    ),  # Maximum
    (
        'HUMAN/kinase1_pdb1 HUMAN/kinase1_pdb2 HUMAN/kinase2_pdb1'.split(),
        'kinase1 kinase2'.split(),
        pd.DataFrame(
            [
                ['HUMAN/kinase1_pdb1', 'HUMAN/kinase1_pdb2', 0.5, 1.0],
                ['HUMAN/kinase1_pdb1', 'HUMAN/kinase2_pdb1', 0.75, 1.0],
                ['HUMAN/kinase1_pdb2', 'HUMAN/kinase2_pdb1', 1.0, 1.0]
            ],
            columns='molecule_code_1 molecule_code_2 distance coverage'.split()
        ),
        'mean',
        False,
        pd.DataFrame(
            [[0.5, 0.875], [np.nan, 0.0]],
            columns='kinase1 kinase2'.split(),
            index='kinase1 kinase2'.split()
        )
    ),  # Minimum
])
def test_get_kinase_distance_matrix(molecule_codes, kinase_names, data, by, fill, structure_distance_matrix):

    # Set dummy FingerprintDistanceGenerator class attributes
    fingerprint_distance_generator = FingerprintDistanceGenerator()
    fingerprint_distance_generator.molecule_codes = molecule_codes
    fingerprint_distance_generator.kinase_names = kinase_names
    fingerprint_distance_generator.data = data

    # Test generation of structure distance matrix
    structure_distance_matrix_calculated = fingerprint_distance_generator.get_kinase_distance_matrix(
        by,
        fill
    )

    assert structure_distance_matrix_calculated.equals(structure_distance_matrix)
