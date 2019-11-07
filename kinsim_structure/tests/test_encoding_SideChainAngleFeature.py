"""
Unit and regression test for kinsim_structure.encoding.SideChainAngleFeature class.
"""

import pandas as pd
from pathlib import Path
import pytest

from kinsim_structure.auxiliary import KlifsMoleculeLoader, PdbChainLoader
from kinsim_structure.encoding import ObsoleteSideChainAngleFeature


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
def test_from_molecule(mol2_filename, pdb_filename, chain_id, sca):
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
