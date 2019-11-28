"""
Unit and regression test for kinsim_structure.similarity.AllAgainstAllComparison methods.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kinsim_structure.auxiliary import KlifsMoleculeLoader, PdbChainLoader
from kinsim_structure.encoding import Fingerprint, FingerprintGenerator
from kinsim_structure.similarity import FeatureDistances, FingerprintDistance, \
    FeatureDistancesGenerator, FingerprintDistanceGenerator

PATH_TEST_DATA = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data'


def generate_fingerprints_from_files(path_klifs_metadata, paths_mol2, paths_pdb, chain_ids):
    """
    Helper function: Generate multiple fingerprints from files.

    Parameters
    ----------
    path_klifs_metadata : pathlib.Path
        Path to unfiltered KLIFS metadata.
    paths_mol2 : list of pathlib.Path
        Paths to multiple mol2 files.
    paths_pdb : list of pathlib.Path
        Paths to multiple cif files.
    chain_ids : list of str
        Multiple chain IDs.

    Returns
    -------
    list of kinsim_structure.encoding.Fingerprint
        List of fingerprints.
    """

    fingerprints = []

    for path_mol2, path_pdb, chain_id in zip(paths_mol2, paths_pdb, chain_ids):

        klifs_molecule_loader = KlifsMoleculeLoader()
        klifs_molecule_loader.from_file(path_mol2, path_klifs_metadata)
        pdb_chain_loader = PdbChainLoader()
        pdb_chain_loader.from_file(path_pdb, chain_id)

        fingerprint = Fingerprint()
        fingerprint.from_molecule(klifs_molecule_loader.molecule, pdb_chain_loader.chain)

        fingerprints.append(fingerprint)

    return fingerprints


@pytest.mark.parametrize('path_klifs_metadata, path_mol2s, path_pdbs, chain_ids', [
    (
        PATH_TEST_DATA / 'klifs_metadata.csv',
        [
            PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ABL1/2g2i_chainA/pocket.mol2',
            PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/pocket.mol2'
        ],
        [
            PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ABL1/2g2i_chainA/protein_pymol.pdb',
            PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/protein_pymol.pdb'
        ],
        [
            'A',
            'B'
        ]
    )

])
def test_get_fingerprint_distance(path_klifs_metadata, path_mol2s, path_pdbs, chain_ids):
    """
    Test if return type is FingerprintDistance class instance.

    Parameters
    ----------
    path_klifs_metadata : pathlib.Path
        Path to unfiltered KLIFS metadata.
    path_mol2s : list of str
        Paths to two mol2 files.
    path_pdbs : list of str
        Paths to two cif files.
    chain_ids : list of str
        Two chain IDs.
    """

    # Fingerprints
    fingerprints = generate_fingerprints_from_files(path_klifs_metadata, path_mol2s, path_pdbs, chain_ids)

    # FeatureDistances
    feature_distances = FeatureDistances()
    feature_distances.from_fingerprints(fingerprints[0], fingerprints[1])

    # FingerprintDistanceGenerator
    fingerprint_distance_generator = FingerprintDistanceGenerator()
    fingerprint_distance_calculated = fingerprint_distance_generator._get_fingerprint_distance(
        feature_distances
    )

    assert isinstance(fingerprint_distance_calculated, FingerprintDistance)


@pytest.mark.parametrize('path_klifs_metadata, path_mol2s, path_pdbs, chain_ids', [
    (

        PATH_TEST_DATA / 'klifs_metadata.csv',
        [
            PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ABL1/2g2i_chainA/pocket.mol2',
            PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/pocket.mol2',
            PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/pocket.mol2'
        ],
        [
            PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ABL1/2g2i_chainA/protein_pymol.pdb',
            PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/protein_pymol.pdb',
            PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/protein_pymol.pdb'
        ],
        [
            'A',
            'B',
            'B'
        ]
    )

])
def test_get_fingerprint_distance_from_list(path_klifs_metadata, path_mol2s, path_pdbs, chain_ids):
    """
    Test if return type is instance of list of FeatureDistance class instances.

    Parameters
    ----------
    path_klifs_metadata : pathlib.Path
        Path to unfiltered KLIFS metadata.
    path_mol2s : list of str
        Paths to multiple mol2 files.
    path_pdbs : list of str
        Paths to multiple cif files.
    chain_ids : list of str
        Multiple chain IDs.
    """

    # Fingerprints
    fingerprints = generate_fingerprints_from_files(path_klifs_metadata, path_mol2s, path_pdbs, chain_ids)

    # FingerprintGenerator
    fingerprint_generator = FingerprintGenerator()
    fingerprint_generator.data = {i.molecule_code: i for i in fingerprints}

    # FeatureDistancesGenerator
    feature_distances_generator = FeatureDistancesGenerator()
    feature_distances_generator.from_fingerprint_generator(fingerprint_generator)
    feature_distances_list = list(feature_distances_generator.data.values())

    # FingerprintDistanceGenerator
    fingerprint_distance_generator = FingerprintDistanceGenerator()
    fingerprint_distance_list = fingerprint_distance_generator._get_fingerprint_distance_from_list(
        fingerprint_distance_generator._get_fingerprint_distance, feature_distances_list
    )

    assert isinstance(fingerprint_distance_list, list)

    for i in fingerprint_distance_list:
        assert isinstance(i, FingerprintDistance)


@pytest.mark.parametrize(
    'path_klifs_metadata, path_mol2s, path_pdbs, chain_ids, distance_measure, feature_weights, molecule_codes, kinase_names', [
        (
            PATH_TEST_DATA / 'klifs_metadata.csv',
            [
                PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ABL1/2g2i_chainA/pocket.mol2',
                PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/pocket.mol2',
                PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/pocket.mol2'
            ],
            [
                PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ABL1/2g2i_chainA/protein_pymol.pdb',
                PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/protein_pymol.pdb',
                PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/protein_pymol.pdb'
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
        path_klifs_metadata, path_mol2s, path_pdbs, chain_ids, distance_measure, feature_weights, molecule_codes, kinase_names
):
    """
    Test FingerprintDistanceGenerator class attributes.

    Parameters
    ----------
    path_klifs_metadata : pathlib.Path
        Path to unfiltered KLIFS metadata.
    path_mol2s : list of str
        Paths to multiple mol2 files.
    path_pdbs : list of str
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
    fingerprints = generate_fingerprints_from_files(path_klifs_metadata, path_mol2s, path_pdbs, chain_ids)

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
