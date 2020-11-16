"""
Unit and regression test for kissim.encoding class methods.
"""

import math
from pathlib import Path

from Bio.PDB import Vector
import numpy as np
import pandas as pd
import pytest

from kissim.auxiliary import KlifsMoleculeLoader, PdbChainLoader
from kissim.encoding.schema import FEATURE_NAMES
from kissim.encoding.definitions import DISTANCE_CUTOFFS, MOMENT_CUTOFFS
from kissim.encoding.features import (
    PharmacophoreSizeFeatures,
    ExposureFeature,
    SideChainOrientationFeature,
    PhysicoChemicalFeatures,
    SpatialFeatures,
)
from kissim.encoding.api import Fingerprint

PATH_TEST_DATA = Path(__name__).parent / "kissim" / "tests" / "data"


class TestsPhysicoChemicalFeatures:
    """
    Test PhysicoChemicalFeatures class methods.
    """

    @pytest.mark.parametrize(
        "path_klifs_metadata, path_mol2, path_pdb, chain_id",
        [
            (
                PATH_TEST_DATA / "klifs_metadata.csv",
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ABL1/2g2i_chainA/pocket.mol2",
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ABL1/2g2i_chainA/protein_pymol.pdb",
                "A",
            )
        ],
    )
    def test_from_molecule(self, path_klifs_metadata, path_mol2, path_pdb, chain_id):
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
        physicochemicalfeatures.from_molecule(
            klifs_molecule_loader.molecule, pdb_chain_loader.chain
        )
        features = physicochemicalfeatures.features

        physiochemicalfeatures_columns = (
            "size hbd hba charge aromatic aliphatic sco exposure".split()
        )
        assert list(features.columns) == physiochemicalfeatures_columns
        assert len(features) == 85


class TestsSpatialFeatures:
    """
    Test SpatialFeatures class methods.
    """

    @pytest.mark.parametrize(
        "path_klifs_metadata, path_mol2",
        [
            (
                PATH_TEST_DATA / "klifs_metadata.csv",
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ABL1/2g2i_chainA/pocket.mol2",
            )
        ],
    )
    def test_from_molecule(self, path_klifs_metadata, path_mol2):
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
            "distance_to_centroid",
            "distance_to_hinge_region",
            "distance_to_dfg_region",
            "distance_to_front_pocket",
        ]
        assert list(features.columns) == spatialfeatures_columns
        assert len(features) == 85

    @pytest.mark.parametrize(
        "path_klifs_metadata, path_mol2, reference_point_name, anchor_residue_klifs_ids, x_coordinate",
        [
            (
                PATH_TEST_DATA / "klifs_metadata.csv",
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/AAK1/4wsq_altA_chainB/pocket.mol2",
                "hinge_region",
                [16, 47, 80],
                6.25545,
            ),
            (
                PATH_TEST_DATA / "klifs_metadata.csv",
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/AAK1/4wsq_altA_chainB/pocket.mol2",
                "dfg_region",
                [20, 23, 81],
                11.6846,
            ),
            (
                PATH_TEST_DATA / "klifs_metadata.csv",
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/AAK1/4wsq_altA_chainB/pocket.mol2",
                "front_pocket",
                [6, 48, 75],
                float("nan"),
            ),
        ],
    )
    def test_get_anchor_atoms(
        self,
        path_klifs_metadata,
        path_mol2,
        reference_point_name,
        anchor_residue_klifs_ids,
        x_coordinate,
    ):
        """
        Test function that calculates the anchor atoms for different scenarios
        (missing anchor residues, missing neighbors)

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
            assert math.isnan(anchors[reference_point_name].loc[anchor_residue_klifs_ids[0], "x"])
        else:
            assert (
                anchors[reference_point_name].loc[anchor_residue_klifs_ids[0], "x"] == x_coordinate
            )

    @pytest.mark.parametrize(
        "path_klifs_metadata, path_mol2, x_coordinate",
        [
            (
                PATH_TEST_DATA / "klifs_metadata.csv",
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/AAK1/4wsq_altA_chainB/pocket.mol2",
                1.02664,
            )
        ],
    )
    def test_get_reference_points(self, path_klifs_metadata, path_mol2, x_coordinate):
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


class TestsFingerprint:
    """
    Test Fingerprint class methods.
    """

    @pytest.mark.parametrize(
        "path_klifs_metadata, path_mol2, path_pdb, chain_id",
        [
            (
                PATH_TEST_DATA / "klifs_metadata.csv",
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ABL1/2g2i_chainA/pocket.mol2",
                PATH_TEST_DATA / "KLIFS_download" / "HUMAN/ABL1/2g2i_chainA/protein_pymol.pdb",
                "A",
            )
        ],
    )
    def test_from_molecule(self, path_klifs_metadata, path_mol2, path_pdb, chain_id):
        """
        Test if Fingerprint class attributes (accessed via property function) have correct
        DataFrame shape, column and index names.

        Parameters
        ----------
        path_klifs_metadata : pathlib.Path
            Path to unfiltered KLIFS metadata.
        path_mol2 : pathlib.Path
            Path to mol2 file.
        path_pdb : pathlib.Path
            Path to cif file.
        chain_id : pathlib.Path
            Chain ID.
        """

        klifs_molecule_loader = KlifsMoleculeLoader()
        klifs_molecule_loader.from_file(path_mol2, path_klifs_metadata)
        pdb_chain_loader = PdbChainLoader()
        pdb_chain_loader.from_file(path_pdb, chain_id)

        fingerprint = Fingerprint()
        fingerprint.from_molecule(klifs_molecule_loader.molecule, pdb_chain_loader.chain)

        klifs_ids = list(range(1, 86))

        # Non-normalized
        assert fingerprint.physicochemical.shape == (85, 8)
        assert list(fingerprint.physicochemical.index) == klifs_ids
        assert list(fingerprint.physicochemical.columns) == FEATURE_NAMES["physicochemical"]

        assert fingerprint.distances.shape == (85, 4)
        assert list(fingerprint.distances.index) == klifs_ids
        assert list(fingerprint.distances.columns) == FEATURE_NAMES["distances"]

        assert fingerprint.moments.shape == (4, 3)
        assert list(fingerprint.moments.index) == FEATURE_NAMES["distances"]
        assert list(fingerprint.moments.columns) == FEATURE_NAMES["moments"]

        assert (
            list(fingerprint.physicochemical_distances.keys())
            == "physicochemical distances".split()
        )
        assert (
            list(fingerprint.physicochemical_moments.keys()) == "physicochemical moments".split()
        )

        # Normalized
        assert fingerprint.physicochemical_normalized.shape == (85, 8)
        assert list(fingerprint.physicochemical_normalized.index) == klifs_ids
        assert (
            list(fingerprint.physicochemical_normalized.columns)
            == FEATURE_NAMES["physicochemical"]
        )

        assert fingerprint.distances.shape == (85, 4)
        assert list(fingerprint.distances_normalized.index) == klifs_ids
        assert list(fingerprint.distances_normalized.columns) == FEATURE_NAMES["distances"]

        assert fingerprint.moments.shape == (4, 3)
        assert list(fingerprint.moments_normalized.index) == FEATURE_NAMES["distances"]
        assert list(fingerprint.moments_normalized.columns) == FEATURE_NAMES["moments"]

        assert (
            list(fingerprint.physicochemical_distances_normalized.keys())
            == "physicochemical distances".split()
        )
        assert (
            list(fingerprint.physicochemical_moments_normalized.keys())
            == "physicochemical moments".split()
        )

    @pytest.mark.parametrize(
        "physicochemical, physicochemical_normalized",
        [
            (
                pd.DataFrame(
                    [[3.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                    columns=FEATURE_NAMES["physicochemical"],
                ),
                pd.DataFrame(
                    [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0]],
                    columns=FEATURE_NAMES["physicochemical"],
                ),
            ),
            (
                pd.DataFrame(
                    [[2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.8]],
                    columns=FEATURE_NAMES["physicochemical"],
                ),
                pd.DataFrame(
                    [[0.5, 0.3333, 0.5, 0.5, 0.0, 0.0, 1.0, 0.8]],
                    columns=FEATURE_NAMES["physicochemical"],
                ),
            ),
            (
                pd.DataFrame([[np.nan] * 8], columns=FEATURE_NAMES["physicochemical"]),
                pd.DataFrame([[np.nan] * 8], columns=FEATURE_NAMES["physicochemical"]),
            ),
        ],
    )
    def test_normalize_physicochemical_bits(self, physicochemical, physicochemical_normalized):
        """
        Test normalization of physicochemical bits.

        Parameters
        ----------
        physicochemical : pandas.DataFrame
            Physicochemical bits.
        physicochemical_normalized : pandas.DataFrame
            Normalized physicochemical bits.
        """

        fingerprint = Fingerprint()
        fingerprint.molecule_code = "molecule"
        fingerprint.fingerprint["physicochemical"] = physicochemical

        physicochemical_normalized_calculated = fingerprint._normalize_physicochemical_bits()

        for feature in FEATURE_NAMES["physicochemical"]:

            if np.isnan(physicochemical.iloc[0, 0]):
                assert np.isnan(physicochemical_normalized_calculated[feature][0]) and np.isnan(
                    physicochemical_normalized[feature][0]
                )
            else:
                assert np.isclose(
                    physicochemical_normalized_calculated[feature][0],
                    physicochemical_normalized[feature][0],
                    rtol=1e-03,
                )

    @pytest.mark.parametrize(
        "distances, distances_normalized",
        [
            (
                pd.DataFrame([[1, 1, 1, 1]], columns=FEATURE_NAMES["distances"]),
                pd.DataFrame([[0.0, 0.0, 0.0, 0.0]], columns=FEATURE_NAMES["distances"]),
            ),
            (
                pd.DataFrame(
                    [
                        [
                            DISTANCE_CUTOFFS["distance_to_centroid"][0],
                            DISTANCE_CUTOFFS["distance_to_hinge_region"][0],
                            DISTANCE_CUTOFFS["distance_to_dfg_region"][0],
                            DISTANCE_CUTOFFS["distance_to_front_pocket"][0],
                        ]
                    ],
                    columns=FEATURE_NAMES["distances"],
                ),
                pd.DataFrame([[0.0, 0.0, 0.0, 0.0]], columns=FEATURE_NAMES["distances"]),
            ),
            (
                pd.DataFrame([[10, 10, 10, 10]], columns=FEATURE_NAMES["distances"]),
                pd.DataFrame(
                    [[0.3792, 0.3110, 0.2438, 0.2510]], columns=FEATURE_NAMES["distances"]
                ),
            ),
            (
                pd.DataFrame(
                    [
                        [
                            DISTANCE_CUTOFFS["distance_to_centroid"][1],
                            DISTANCE_CUTOFFS["distance_to_hinge_region"][1],
                            DISTANCE_CUTOFFS["distance_to_dfg_region"][1],
                            DISTANCE_CUTOFFS["distance_to_front_pocket"][1],
                        ]
                    ],
                    columns=FEATURE_NAMES["distances"],
                ),
                pd.DataFrame([[1.0, 1.0, 1.0, 1.0]], columns=FEATURE_NAMES["distances"]),
            ),
            (
                pd.DataFrame([[30, 30, 30, 30]], columns=FEATURE_NAMES["distances"]),
                pd.DataFrame([[1.0, 1.0, 1.0, 1.0]], columns=FEATURE_NAMES["distances"]),
            ),
            (
                pd.DataFrame([[np.nan] * 4], columns=FEATURE_NAMES["distances"]),
                pd.DataFrame([[np.nan] * 4], columns=FEATURE_NAMES["distances"]),
            ),
        ],
    )
    def test_normalize_distances_bits(self, distances, distances_normalized):
        """
        Test normalization of distances bits.

        Parameters
        ----------
        distances : pandas.DataFrame
            Distances.
        distances_normalized : pandas.DataFrame
            Normalized distances.
        """

        fingerprint = Fingerprint()
        fingerprint.molecule_code = "molecule"
        fingerprint.fingerprint["distances"] = distances

        distances_normalized_calculated = fingerprint._normalize_distances_bits()

        for feature in FEATURE_NAMES["distances"]:

            if np.isnan(distances.iloc[0, 0]):
                assert np.isnan(distances_normalized_calculated[feature][0]) and np.isnan(
                    distances_normalized[feature][0]
                )
            else:
                assert np.isclose(
                    distances_normalized_calculated[feature][0],
                    distances_normalized[feature][0],
                    rtol=1e-03,
                )

    @pytest.mark.parametrize(
        "moments, moments_normalized",
        [
            (
                pd.DataFrame([[11, 3, -2]], columns=FEATURE_NAMES["moments"]),
                pd.DataFrame([[0.0, 0.0, 0.0]], columns=FEATURE_NAMES["moments"]),
            ),
            (
                pd.DataFrame(
                    [
                        [
                            MOMENT_CUTOFFS["moment1"][0],
                            MOMENT_CUTOFFS["moment2"][0],
                            MOMENT_CUTOFFS["moment3"][0],
                        ]
                    ],
                    columns=FEATURE_NAMES["moments"],
                ),
                pd.DataFrame([[0.0, 0.0, 0.0]], columns=FEATURE_NAMES["moments"]),
            ),
            (
                pd.DataFrame([[12, 4, 1]], columns=FEATURE_NAMES["moments"]),
                pd.DataFrame([[0.1301, 0.355, 0.4030]], columns=FEATURE_NAMES["moments"]),
            ),
            (
                pd.DataFrame(
                    [
                        [
                            MOMENT_CUTOFFS["moment1"][1],
                            MOMENT_CUTOFFS["moment2"][1],
                            MOMENT_CUTOFFS["moment3"][1],
                        ]
                    ],
                    columns=FEATURE_NAMES["moments"],
                ),
                pd.DataFrame([[1.0, 1.0, 1.0]], columns=FEATURE_NAMES["moments"]),
            ),
            (
                pd.DataFrame([[15, 6, 5]], columns=FEATURE_NAMES["moments"]),
                pd.DataFrame([[1.0, 1.0, 1.0]], columns=FEATURE_NAMES["moments"]),
            ),
            (
                pd.DataFrame([[np.nan] * 3], columns=FEATURE_NAMES["moments"]),
                pd.DataFrame([[np.nan] * 3], columns=FEATURE_NAMES["moments"]),
            ),
        ],
    )
    def test_normalize_moments_bits(self, moments, moments_normalized):

        fingerprint = Fingerprint()
        fingerprint.molecule_code = "molecule"
        fingerprint.fingerprint["moments"] = moments

        moments_normalized_calculated = fingerprint._normalize_moments_bits()

        for feature in FEATURE_NAMES["moments"]:

            if np.isnan(moments.iloc[0, 0]):
                assert np.isnan(moments_normalized_calculated[feature][0]) and np.isnan(
                    moments_normalized[feature][0]
                )
            else:
                assert np.isclose(
                    moments_normalized_calculated[feature][0],
                    moments_normalized[feature][0],
                    rtol=1e-03,
                )

    @pytest.mark.parametrize(
        "distances, moments",
        [
            (
                pd.DataFrame([[1, 1], [4, 4], [4, 4]]),
                pd.DataFrame(
                    [[3.00, 1.41, -1.26], [3.00, 1.41, -1.26]],
                    columns="moment1 moment2 moment3".split(),
                ),
            ),
            (
                pd.DataFrame([[1, 2]]),
                pd.DataFrame(
                    [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], columns="moment1 moment2 moment3".split()
                ),
            ),
        ],
    )
    def test_calc_moments(self, distances, moments):

        fingerprint = Fingerprint()
        moments_calculated = fingerprint._calc_moments(distances)

        print(len(distances))
        print(moments_calculated)

        assert np.isclose(moments_calculated.moment1[0], moments.moment1[0], rtol=1e-02)
        assert np.isclose(moments_calculated.moment2[0], moments.moment2[0], rtol=1e-02)
        assert np.isclose(moments_calculated.moment3[0], moments.moment3[0], rtol=1e-02)

    @pytest.mark.parametrize(
        "value, minimum, maximum, value_normalized",
        [
            (1.00, 2.00, 3.00, 0.00),
            (2.00, 2.00, 3.00, 0.00),
            (2.50, 2.00, 3.00, 0.50),
            (3.00, 2.00, 3.00, 1.00),
            (4.00, 2.00, 3.00, 1.00),
            (np.nan, 2.00, 3.00, np.nan),
        ],
    )
    def test_normalize(self, value, minimum, maximum, value_normalized):
        """
        Test value normalization.

        Parameters
        ----------
        value : float or int
                Value to be normalized.
        minimum : float or int
            Minimum value for normalization, values equal or greater than this minimum are set
            to 0.0.
        maximum : float or int
            Maximum value for normalization, values equal or greater than this maximum are set
            to 1.0.
        value_normalized : float
            Normalized value.
        """

        fingerprint = Fingerprint()
        value_normalized_calculated = fingerprint._normalize(value, minimum, maximum)

        if np.isnan(value):
            assert np.isnan(value_normalized_calculated) and np.isnan(value_normalized)
        else:
            assert np.isclose(value_normalized_calculated, value_normalized, rtol=1e-06)
