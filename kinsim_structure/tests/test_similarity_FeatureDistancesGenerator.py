"""
Unit and regression test for kinsim_structure.similarity.AllAgainstAllComparison methods.
"""

from pathlib import Path

import pytest

from kinsim_structure.encoding import Fingerprint, FingerprintGenerator
from kinsim_structure.auxiliary import KlifsMoleculeLoader, PdbChainLoader
from kinsim_structure.similarity import FeatureDistances, FeatureDistancesGenerator


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

    # Fingerprints
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


@pytest.mark.parametrize('fingerprints, empty_fingerprints', [
    ({'a': Fingerprint(), 'b': None}, {'a': Fingerprint()}),
    ({'a': Fingerprint()}, {'a': Fingerprint()})
])
def test_remove_empty_fingerprints(fingerprints, empty_fingerprints):
    """
    Test removal of empty fingerprints (None) from fingerprints dictionary.

    Parameters
    ----------
    fingerprints : dict of (kinsim_structure.encoding.Fingerprint or None)
        Dictionary of fingerprints: Keys are molecule codes and values are fingerprint data.
    empty_fingerprints : dict of kinsim_structure.encoding.Fingerprint
        Dictionary of non-empty fingerprints: Keys are molecule codes and values are fingerprint data.
    """

    generator = FeatureDistancesGenerator()
    empty_fingerprints_calculated = generator._remove_empty_fingerprints(fingerprints)

    assert empty_fingerprints_calculated.keys() == empty_fingerprints.keys()


@pytest.mark.parametrize('fingerprints, pairs', [
    ({'a': Fingerprint(), 'b': Fingerprint(), 'c': Fingerprint()}, [['a', 'b'], ['a', 'c'], ['b', 'c']])
])
def test_get_fingerprint_pairs(fingerprints, pairs):
    """
    Test calculation of all fingerprint pair combinations from fingerprints dictionary.

    Parameters
    ----------
    fingerprints : dict of (kinsim_structure.encoding.Fingerprint or None)
        Dictionary of fingerprints: Keys are molecule codes and values are fingerprint data.
    pairs : list of list of str
        List of molecule code pairs (list).
    """

    generator = FeatureDistancesGenerator()
    pairs_calculated = generator._get_fingerprint_pairs(fingerprints)

    for pair_calculated, pair in zip(pairs_calculated, pairs):
        assert pair_calculated == pair


@pytest.mark.parametrize('mol2_filenames, pdb_filenames, chain_ids', [
    (
        ['ABL1/2g2i_chainA/pocket.mol2', 'AAK1/4wsq_altA_chainB/pocket.mol2'],
        ['2g2i.cif', '4wsq.cif'],
        ['A', 'B']
    )

])
def test_get_feature_distances(mol2_filenames, pdb_filenames, chain_ids):
    """
    Test if return type is instance of FeatureDistance class.

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

    # Fingerprint dictionary and pair names
    pair = [i.molecule_code for i in fingerprints]
    fingerprints = {i.molecule_code: i for i in fingerprints}

    # Test feature distance calculation
    generator = FeatureDistancesGenerator()
    feature_distances_calculated = generator._get_feature_distances(pair, fingerprints)

    assert isinstance(feature_distances_calculated, FeatureDistances)


@pytest.mark.parametrize('mol2_filenames, pdb_filenames, chain_ids', [
    (
        ['ABL1/2g2i_chainA/pocket.mol2', 'AAK1/4wsq_altA_chainB/pocket.mol2', 'AAK1/4wsq_altA_chainB/pocket.mol2'],
        ['2g2i.cif', '4wsq.cif', '4wsq.cif'],
        ['A', 'B', 'B']
    )

])
def test_get_feature_distances_from_list(mol2_filenames, pdb_filenames, chain_ids):
    """
    Test if return type is instance of list of FeatureDistance class.

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

    # Fingerprint dictionary and pair names
    fingerprints = {i.molecule_code: i for i in fingerprints}

    # Test bulk feature distance calculation
    generator = FeatureDistancesGenerator()
    pairs = generator._get_fingerprint_pairs(fingerprints)

    feature_distances_list = generator._get_feature_distances_from_list(
        generator._get_feature_distances, pairs, fingerprints
    )

    assert isinstance(feature_distances_list, list)

    for i in feature_distances_list:
        assert isinstance(i, FeatureDistances)


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
            ['HUMAN/ABL1_2g2i_chainA', 'HUMAN/AAK1_4wsq_altA_chainB'],
            ['AAK1', 'ABL1']
        )
    ]
)
def test_from_fingerprints(
        mol2_filenames, pdb_filenames, chain_ids, distance_measure, feature_weights, molecule_codes, kinase_names
):
    """
    Test FeatureDistancesGenerator class attributes.

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
    """

    # Fingerprints
    fingerprints = generate_fingerprints_from_files(mol2_filenames, pdb_filenames, chain_ids)

    # Fingerprint dictionary and pair names
    fingerprint_generator = FingerprintGenerator()
    fingerprint_generator.data = {i.molecule_code: i for i in fingerprints}

    # Test FeatureDistancesGenerator class attributes
    feature_distances_generator = FeatureDistancesGenerator()
    feature_distances_generator.from_fingerprint_generator(fingerprint_generator)

    # Test attributes
    assert feature_distances_generator.distance_measure == distance_measure
    assert isinstance(feature_distances_generator.data, dict)

    # Test example value from dictionary
    example_key = list(feature_distances_generator.data.keys())[0]
    assert isinstance(feature_distances_generator.data[example_key], FeatureDistances)
