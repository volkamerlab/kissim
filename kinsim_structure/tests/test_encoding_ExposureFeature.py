"""
Unit and regression test for kinsim_structure.encoding.ExposureFeature class.
"""

from pathlib import Path
import pytest

import numpy as np

from kinsim_structure.auxiliary import KlifsMoleculeLoader, PdbChainLoader
from kinsim_structure.encoding import ExposureFeature

PATH_TEST_DATA = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data'


@pytest.mark.parametrize('path_pdb, chain_id, radius, method, n_residues, up_mean, down_mean, index_mean', [
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/protein_pymol.pdb',
        'B',
        12.0,
        'HSExposureCA',
        308,
        12.3636,
        16.4708,
        187.5
    ),
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/protein_pymol.pdb',
        'B',
        12.0,
        'HSExposureCB',
        310,
        13.1355,
        15.5806,
        187.5
    )

])
def test_get_exposure_by_method(path_pdb, chain_id, radius, method, n_residues, up_mean, down_mean, index_mean):
    """
    Test half sphere exposure and exposure ratio calculation as well as the result format.

    Parameters
    ----------
    path_pdb : str
        Path to cif file.
    chain_id : str
        Chain ID.
    radius : float
        Sphere radius to be used for half sphere exposure calculation.
    method : str
        Half sphere exposure method name: HSExposureCA or HSExposureCB.
    n_residues : int
        Number of residues in exposure calculation result.
    up_mean : float
        Mean of all exposure up values.
    down_mean : float
        Mean of all exposure down values.
    index_mean : float
        Mean of all residue IDs.
    """

    # Load pdb file
    pdb_chain_loader = PdbChainLoader()
    pdb_chain_loader.from_file(path_pdb, chain_id)

    # Get exposure
    feature = ExposureFeature()
    exposures_calculated = feature.get_molecule_exposure_by_method(pdb_chain_loader.chain, radius, method)

    # Get method prefix, i.e. ca or cb
    prefix = method[-2:].lower()

    # Test DataFrame length
    assert len(exposures_calculated) == n_residues

    # Test column names
    columns = [f'{prefix}_{i}' for i in ['up', 'down', 'angle_CB-CA-pCB', 'exposure']]
    assert list(exposures_calculated.columns) == columns

    # Test exposure up values (mean)
    up_mean_calculated = exposures_calculated[f'{prefix}_up'].mean()
    assert np.isclose(up_mean_calculated, up_mean, rtol=1e-05)

    # Test exposure down values (mean)
    down_mean_calculated = exposures_calculated[f'{prefix}_down'].mean()
    assert np.isclose(down_mean_calculated, down_mean, rtol=1e-05)

    # Test residue IDs (mean)
    index_mean_calculated = np.array(exposures_calculated.index).mean()
    assert np.isclose(index_mean_calculated, index_mean, rtol=1e-05)

    # Test for example residue the exposure ratio calculation
    example_residue = exposures_calculated.iloc[0]
    ratio = example_residue[f'{prefix}_exposure']
    ratio_calculated = example_residue[f'{prefix}_up'] / (example_residue[f'{prefix}_up'] + example_residue[f'{prefix}_down'])
    assert np.isclose(ratio_calculated, ratio, rtol=1e-04)


@pytest.mark.parametrize('path_pdb, chain_id, radius, n_residues, missing_exposure', [
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/AAK1/4wsq_altA_chainB/protein_pymol.pdb',
        'B',
        12.0,
        310,
        {'ca': [33, 342], 'cb': []}
    ),
])
def test_get_molecule_exposures(path_pdb, chain_id, radius, n_residues, missing_exposure):
    """
    Test join of HSExposureCA and HSExposureCB data.

    Parameters
    ----------
    path_pdb : str
        Path to cif file.
    chain_id : str
        Chain ID.
    radius : float
        Sphere radius to be used for half sphere exposure calculation.
    n_residues : int
        Number of residues in exposure calculation result.
    missing_exposure : dict of list of int
        Residue IDs with missing exposures for HSExposureCA and HSExposureCB calculation.
    """

    # Load pdb file
    pdb_chain_loader = PdbChainLoader()
    pdb_chain_loader.from_file(path_pdb, chain_id)

    # Get exposure
    feature = ExposureFeature()
    exposures_calculated = feature.get_molecule_exposures(pdb_chain_loader.chain, radius)

    # Test DataFrame length
    assert len(exposures_calculated) == n_residues

    # Test column names
    column_names_ca = ['ca_up', 'ca_down', 'ca_angle_CB-CA-pCB', 'ca_exposure']
    column_names_cb = ['cb_up', 'cb_down', 'cb_angle_CB-CA-pCB', 'cb_exposure']
    column_names = column_names_ca + column_names_cb
    assert list(exposures_calculated.columns) == column_names

    # Test missing residues in HSExposureCA and HSExposureCB calculation
    missing_residues_calculated = dict()

    missing_residues_calculated['ca'] = list(exposures_calculated[exposures_calculated.ca_up.isna()].index)
    assert missing_residues_calculated['ca'] == missing_exposure['ca']

    missing_residues_calculated['cb'] = list(exposures_calculated[exposures_calculated.cb_up.isna()].index)
    assert missing_residues_calculated['cb'] == missing_exposure['cb']


@pytest.mark.parametrize('path_mol2, path_pdb, chain_id, radius, n_residues, missing_exposure', [
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ABL1/2g2i_chainA/pocket.mol2',
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ABL1/2g2i_chainA/protein_pymol.pdb',
        'A',
        12.0,
        82,
        {'ca': [5, 6], 'cb': []}
    ),
    (
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/CHK1/3nlb_chainA/pocket.mol2',
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/CHK1/3nlb_chainA/protein_pymol.pdb',
        'A',
        12.0,
        85,
        {'ca': [], 'cb': [7]}
    )
])
def test_from_molecule(path_mol2, path_pdb, chain_id, radius, n_residues, missing_exposure):
    """
    Test KLIFS ID subset of molecule exposure values and correct selection of HSExposureCB and HSExposureCA values as
    final exposure value (use CB, but if not available use CA).

    Parameters
    ----------
    path_mol2 : pathlib.Path
        Path to mol2 file.
    path_pdb : pathlib.Path
        Path to cif file.
    chain_id : str
        Chain ID.
    radius : float
        Sphere radius to be used for half sphere exposure calculation.
    n_residues : int
        Number of pocket residues in exposure calculation result.
    missing_exposure : dict of list of int
        Residue IDs with missing exposures for HSExposureCA and HSExposureCB calculation.
    """

    path_klifs_metadata = PATH_TEST_DATA / 'klifs_metadata.csv'

    # Load pdb and mol2 files
    klifs_molecule_loader = KlifsMoleculeLoader()
    klifs_molecule_loader.from_file(path_mol2, path_klifs_metadata)
    pdb_chain_loader = PdbChainLoader()
    pdb_chain_loader.from_file(path_pdb, chain_id)

    # Get exposure
    exposure_feature = ExposureFeature()
    exposure_feature.from_molecule(klifs_molecule_loader.molecule, pdb_chain_loader.chain, radius)

    # Test number of pocket residues in exposure calculation
    assert len(exposure_feature.features) == n_residues
    assert len(exposure_feature.features_verbose) == n_residues

    # Test column names in class attribute "features"
    column_names = ['exposure']
    assert list(exposure_feature.features.columns) == column_names

    # Test column names in class attribute "features_verbose"
    column_names_ca = ['ca_up', 'ca_down', 'ca_angle_CB-CA-pCB', 'ca_exposure']
    column_names_cb = ['cb_up', 'cb_down', 'cb_angle_CB-CA-pCB', 'cb_exposure']
    column_names = ['res_id'] + column_names_ca + column_names_cb + ['exposure']
    assert list(exposure_feature.features_verbose.columns) == column_names

    # Test for residues with missing exposures
    exposures_calculated = exposure_feature.features_verbose
    missing_residues_calculated = dict()
    missing_residues_calculated['ca'] = list(exposures_calculated[exposures_calculated.ca_up.isna()].index)
    missing_residues_calculated['cb'] = list(exposures_calculated[exposures_calculated.cb_up.isna()].index)
    assert missing_residues_calculated['ca'] == missing_exposure['ca']
    assert missing_residues_calculated['cb'] == missing_exposure['cb']

    # Test resulting exposure (HSExposureCB values, unless they are missing, then set HSExposureCA values)
    for index, row in exposures_calculated.iterrows():

        if index in missing_exposure['cb']:
            assert exposures_calculated.loc[index].exposure == exposures_calculated.loc[index].ca_exposure
        else:
            assert exposures_calculated.loc[index].exposure == exposures_calculated.loc[index].cb_exposure
