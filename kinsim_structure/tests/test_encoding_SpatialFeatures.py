"""
Unit and regression test for kinsim_structure.encoding.SpatialFeatures class.
"""

import math
import numpy as np
from pathlib import Path
import pytest

from kinsim_structure.auxiliary import KlifsMoleculeLoader
from kinsim_structure.encoding import SpatialFeatures

PATH_TEST_DATA = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data'


@pytest.mark.parametrize('path_klifs_metadata, path_mol2', [
    (
        PATH_TEST_DATA / 'klifs_metadata.csv',
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ABL1/2g2i_chainA/pocket.mol2'
    )
])
def test_from_molecule(path_klifs_metadata, path_mol2):
    """
    Test length (85 rows for 85 KLIFS residues) and columns names of features DataFrame.
    Values are tested already in respective feature unit test.

    Parameters
    ----------
    path_klifs_metadata : pathlib.Path
        Path to unfiltered KLIFS metadata.
    path_mol2 : pathlib.Path
        Path to mol2 file.
    """

    klifs_molecule_loader = KlifsMoleculeLoader()
    klifs_molecule_loader.from_file(path_mol2, path_klifs_metadata)

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


@pytest.mark.parametrize('path_klifs_metadata, path_mol2, reference_point_name, anchor_residue_klifs_ids, x_coordinate', [
    (
        PATH_TEST_DATA / 'klifs_metadata.csv',
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/pocket.mol2',
        'hinge_region',
        [16, 47, 80],
        6.25545
    ),
    (
        PATH_TEST_DATA / 'klifs_metadata.csv',
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/pocket.mol2',
        'dfg_region',
        [20, 23, 81],
        11.6846
    ),
    (
        PATH_TEST_DATA / 'klifs_metadata.csv',
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/pocket.mol2',
        'front_pocket',
        [6, 48, 75],
        float('nan')
    )
])
def test_get_anchor_atoms(path_klifs_metadata, path_mol2, reference_point_name, anchor_residue_klifs_ids, x_coordinate):
    """
    Test function that calculates the anchor atoms for different scenarios (missing anchor residues, missing neighbors)

    Parameters
    ----------
    path_klifs_metadata : pathlib.Path
        Path to unfiltered KLIFS metadata.
    path_mol2 : pathlib.Path
        Path to file originating from test data folder.
    reference_point_name : str
        Reference point name, e.g. 'hinge_region'.
    anchor_residue_klifs_ids : list of int
        List of KLIFS IDs that are used to calculate a given reference point.
    x_coordinate: float
        X coordinate of first anchor atom.
    """

    klifs_molecule_loader = KlifsMoleculeLoader()
    klifs_molecule_loader.from_file(path_mol2, path_klifs_metadata)
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


@pytest.mark.parametrize('path_klifs_metadata, path_mol2, x_coordinate', [
    (
        PATH_TEST_DATA / 'klifs_metadata.csv',
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/pocket.mol2',
        1.02664
    )
])
def test_get_reference_points(path_klifs_metadata, path_mol2, x_coordinate):
    """
    Test calculation of reference point "centroid" of a pocket.

    Parameters
    ----------
    path_klifs_metadata : pathlib.Path
        Path to unfiltered KLIFS metadata.
    path_mol2 : pathlib.Path
        Path to file originating from test data folder.
    x_coordinate: float
        X coordinate of the centroid.
    """

    klifs_molecule_loader = KlifsMoleculeLoader()
    klifs_molecule_loader.from_file(path_mol2, path_klifs_metadata)
    molecule = klifs_molecule_loader.molecule

    # Get spatial features
    spatial_features = SpatialFeatures()
    spatial_features.from_molecule(molecule)

    # Get reference points
    reference_points = spatial_features.get_reference_points(molecule)
    print(reference_points.centroid.x)

    assert np.isclose(reference_points.centroid.x, x_coordinate, rtol=1e-04)
