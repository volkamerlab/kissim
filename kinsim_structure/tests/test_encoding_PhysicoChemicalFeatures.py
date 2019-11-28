"""
Unit and regression test for kinsim_structure.encoding.PhysicoChemicalFeatures class.
"""

from pathlib import Path
import pytest

from kinsim_structure.auxiliary import KlifsMoleculeLoader, PdbChainLoader
from kinsim_structure.encoding import PhysicoChemicalFeatures

PATH_TEST_DATA = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data'


@pytest.mark.parametrize('path_klifs_metadata, path_mol2, path_pdb, chain_id', [
    (
        PATH_TEST_DATA / 'klifs_metadata.csv',
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ABL1/2g2i_chainA/pocket.mol2', 
        PATH_TEST_DATA / 'KLIFS_download' / 'HUMAN/ABL1/2g2i_chainA/protein_pymol.pdb', 
        'A'
    )
])
def test_from_molecule(path_klifs_metadata, path_mol2, path_pdb, chain_id):
    """
    Test length (85 rows for 85 KLIFS residues) and columns names of features DataFrame.
    Values are tested already in respective feature unit test.

    Parameters
    ----------
    path_klifs_metadata : pathlib.Path
        Path to unfiltered KLIFS metadata.
    path_mol2 : pathlib.Path
        Path to mol2 file.
    path_pdb : pathlib.Path
        Path to cif file.
    chain_id : str
        Chain ID.
    """

    klifs_molecule_loader = KlifsMoleculeLoader()
    klifs_molecule_loader.from_file(path_mol2, path_klifs_metadata)
    pdb_chain_loader = PdbChainLoader()
    pdb_chain_loader.from_file(path_pdb, chain_id)

    physicochemicalfeatures = PhysicoChemicalFeatures()
    physicochemicalfeatures.from_molecule(klifs_molecule_loader.molecule, pdb_chain_loader.chain)
    features = physicochemicalfeatures.features

    physiochemicalfeatures_columns = 'size hbd hba charge aromatic aliphatic sco exposure'.split()
    assert list(features.columns) == physiochemicalfeatures_columns
    assert len(features) == 85
