
import math
from pathlib import Path
import pytest

from kinsim_structure.auxiliary import KlifsMoleculeLoader
from kinsim_structure.encoding import PharmacophoreSizeFeatures, SpatialFeatures


@pytest.mark.parametrize('filename, residue, feature_type, feature', [
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'ALA', 'size', 1),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'ASN', 'size', 2),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'ARG', 'size', 3),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'PTR', 'size', 3),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'XXX', 'size', None),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'ALA', 'hbd', 0),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'ASN', 'hbd', 1),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'ARG', 'hbd', 3),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'XXX', 'hbd', None),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'ALA', 'hba', 0),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'ASN', 'hba', 1),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'ASP', 'hba', 2),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'XXX', 'hba', None),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'ALA', 'charge', 0),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'ARG', 'charge', 1),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'ASP', 'charge', -1),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'XXX', 'charge', None),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'ALA', 'aromatic', 0),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'HIS', 'aromatic', 1),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'XXX', 'aromatic', None),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'ARG', 'aliphatic', 0),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'ALA', 'aliphatic', 1),
    ('AAK1/4wsq_altA_chainB/pocket.mol2', 'XXX', 'aliphatic', None)

])
def test_feature_from_residue(filename, residue, feature_type, feature):
    """
    Test function for retrieval of residue's size and pharmacophoric features (i.e. number of hydrogen bond donor,
    hydrogen bond acceptors, charge features, aromatic features or aliphatic features )

    Parameters
    ----------
    filename : str
        Path to file originating from test data folder.
    residue : str
        Three-letter code for residue.
    feature_type : str
        Feature type name.
    feature : int or None
        Feature value.
    """

    # Load molecule
    mol2_path = '/home/dominique/Documents/projects/kinsim_structure/kinsim_structure/tests/data/AAK1/4wsq_altA_chainB/pocket.mol2'
    #mol2_path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / filename

    klifs_molecule_loader = KlifsMoleculeLoader(mol2_path=mol2_path)
    molecule = klifs_molecule_loader.molecule

    # Get pharmacophore and size features
    pharmacophore_size_feature = PharmacophoreSizeFeatures()
    pharmacophore_size_feature.from_molecule(molecule)

    # Call feature from residue function
    assert pharmacophore_size_feature.from_residue(residue, feature_type) == feature


@pytest.mark.parametrize('filename, reference_point_name, anchor_residue_klifs_ids, x_coordinate', [
    ('hinge_region', [16, 47, 80], 6.25545),
    ('dfg_region', [20, 23, 81], 11.6846),
    ('front_pocket', [6, 48, 75], float('nan'))
])
def test_get_anchor_atoms(filename, reference_point_name, anchor_residue_klifs_ids, x_coordinate):
    """
    Test function that calculates the anchor atoms for different scenarios (missing anchor residues, missing neighbors)

    Parameters
    ----------
    filename : str
        Path to file originating from test data folder.
    reference_point_name : str
        Reference point name, e.g. 'hinge_region'.
    anchor_residue_klifs_ids : list of int
        List of KLIFS IDs that are used to calculate a given reference point.
    x_coordinate: float
        X coordinate of first anchor atom.
    """

    mol2_path = '/home/dominique/Documents/projects/kinsim_structure/kinsim_structure/tests/data/AAK1/4wsq_altA_chainB/pocket.mol2'
    # mol2_path = Path(__name__).parent / 'kinsim_structure' / 'tests' / 'data' / filename

    klifs_molecule_loader = KlifsMoleculeLoader(mol2_path=mol2_path)
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
