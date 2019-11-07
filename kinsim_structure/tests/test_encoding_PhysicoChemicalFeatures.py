"""
Unit and regression test for kinsim_structure.encoding.PhysicoChemicalFeatures class.
"""

from pathlib import Path
import pytest

from kinsim_structure.auxiliary import KlifsMoleculeLoader, PdbChainLoader
from kinsim_structure.encoding import PhysicoChemicalFeatures


@pytest.mark.parametrize('mol2_filename, pdb_filename, chain_id', [
    ('ABL1/2g2i_chainA/pocket.mol2', '2g2i.cif', 'A')
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
