"""
Unit and regression test for kinsim_structure.encoding.PharmacophoreSizeFeatures class.
"""

from pathlib import Path
import pytest

from kinsim_structure.auxiliary import KlifsMoleculeLoader
from kinsim_structure.encoding import PharmacophoreSizeFeatures


@pytest.mark.parametrize('residue_name, feature_name, feature', [
    ('ALA', 'size', 1),  # Size
    ('ASN', 'size', 2),
    ('ARG', 'size', 3),
    ('PTR', 'size', 3),  # Converted non-standard
    ('MSE', 'size', 2),  # Converted non-standard
    ('XXX', 'size', None),  # Non-convertable non-standard
    ('ALA', 'hbd', 0),
    ('ASN', 'hbd', 1),
    ('ARG', 'hbd', 3),
    ('XXX', 'hbd', None),
    ('ALA', 'hba', 0),
    ('ASN', 'hba', 1),
    ('ASP', 'hba', 2),
    ('XXX', 'hba', None),
    ('ALA', 'charge', 0),
    ('ARG', 'charge', 1),
    ('ASP', 'charge', -1),
    ('XXX', 'charge', None),
    ('ALA', 'aromatic', 0),
    ('HIS', 'aromatic', 1),
    ('XXX', 'aromatic', None),
    ('ARG', 'aliphatic', 0),
    ('ALA', 'aliphatic', 1),
    ('XXX', 'aliphatic', None)

])
def test_from_residue(residue_name, feature_name, feature):
    """
    Test function for retrieval of residue's size and pharmacophoric features (i.e. number of hydrogen bond donor,
    hydrogen bond acceptors, charge features, aromatic features or aliphatic features )

    Parameters
    ----------
    residue_name : str
        Three-letter code for residue.
    feature_name : str
        Feature type name.
    feature : int or None
        Feature value.
    """

    pharmacophore_size_feature = PharmacophoreSizeFeatures()

    # Call feature from residue function
    feature_calculated = pharmacophore_size_feature.from_residue(residue_name, feature_name)

    if feature_calculated:  # If not None
        assert isinstance(feature_calculated, float)

    assert feature_calculated == feature


@pytest.mark.parametrize('filename, molecule_code, shape', [
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'HUMAN/AAK1_4wsq_altA_chainB', (85, 6)),
    ('ABL1/2g2i_chainA/pocket.mol2', 'HUMAN/ABL1_2g2i_chainA', (82, 6))  # Contains not full KLIFS positions

])
def test_pharmacophoresizefeatures_from_residue(filename, molecule_code, shape):
    """
    Test PharmacohpreSizeFeatures class attributes.

    Parameters
    ----------
    filename : str
        Path to file originating from test data folder.
    molecule_code : str
        Molecule code as defined by KLIFS in mol2 file.
    """

    # Load molecule
    mol2_path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / filename

    klifs_molecule_loader = KlifsMoleculeLoader(mol2_path=mol2_path)
    molecule = klifs_molecule_loader.molecule

    # Get pharmacophore and size features
    pharmacophore_size_feature = PharmacophoreSizeFeatures()
    pharmacophore_size_feature.from_molecule(molecule)

    assert pharmacophore_size_feature.molecule_code == molecule_code
    assert pharmacophore_size_feature.features.shape == shape
