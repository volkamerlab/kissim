"""
Unit and regression test for kinsim_structure.encoding.PhysicoChemicalFeatures class.
"""

from pathlib import Path
import pytest

from kinsim_structure.auxiliary import KlifsMoleculeLoader, PdbChainLoader
from kinsim_structure.encoding import PhysicoChemicalFeatures

PATH_TEST_DATA = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data'


@pytest.mark.parametrize('mol2_filename, pdb_filename, chain_id', [
    ('HUMAN/ABL1/2g2i_chainA/pocket.mol2', 'HUMAN/ABL1/2g2i_chainA/protein_pymol.pdb', 'A')
])
def test_from_molecule(mol2_filename, pdb_filename, chain_id):
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

    path_klifs_metadata = PATH_TEST_DATA / 'klifs_metadata.csv'
    path_mol2 = PATH_TEST_DATA / 'KLIFS_download' / mol2_filename
    path_pdb = PATH_TEST_DATA / 'KLIFS_download' / pdb_filename

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
