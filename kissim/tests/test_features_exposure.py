"""
Unit and regression test for kissim.encoding.features.exposure class methods.
"""

import pytest

import pandas as pd
from opencadd.databases.klifs import setup_remote

from kissim.io import PocketBioPython
from kissim.encoding.features import SolventExposureFeature, SolventExposureFeatureKlifs

REMOTE = setup_remote()


class TestsSolventExposureFeature:
    """
    Test SolventExposureFeature class methods.
    """

    @pytest.mark.parametrize(
        "structure_id, remote",
        [(12347, REMOTE)],
    )
    def test_from_structure_klifs_id(self, structure_id, remote):
        """
        Test if SolventExposureFeature can be set from KLIFS ID.
        """
        feature = SolventExposureFeatureKlifs.from_structure_klifs_id(structure_id)
        assert isinstance(feature, SolventExposureFeatureKlifs)
        # Test class attributes
        assert isinstance(feature._residue_ids, list)
        for residue_id in feature._residue_ids:
            assert isinstance(residue_id, int)
        assert isinstance(feature._ratio_ca, list)
        for ratio, ratio_ca, ratio_cb in zip(feature._ratio, feature._ratio_ca, feature._ratio_cb):
            assert isinstance(ratio, float)
            assert isinstance(ratio_ca, float)
            assert isinstance(ratio_cb, float)

    @pytest.mark.parametrize(
        "structure_id, remote, values_mean",
        [(12347, REMOTE, 0.552123)],
    )
    def test_values(self, structure_id, remote, values_mean):
        """
        Test class property: side chain orientation values.
        The mean refers to the mean of non-NaN values.
        """
        feature = SolventExposureFeatureKlifs.from_structure_klifs_id(structure_id)
        assert isinstance(feature.values, list)
        values_mean_calculated = pd.Series(feature.values).dropna().mean()
        assert values_mean == pytest.approx(values_mean_calculated)

    @pytest.mark.parametrize(
        "structure_id, remote",
        [(12347, REMOTE)],
    )
    def test_details(self, structure_id, remote):
        """
        Test class property: side chain orientation details.
        """
        feature = SolventExposureFeatureKlifs.from_structure_klifs_id(structure_id)
        assert isinstance(feature.details, pd.DataFrame)
        assert feature.details.columns.to_list() == [
            "exposure.ratio",
            "exposure.ratio_ca",
            "exposure.ratio_cb",
        ]
        assert feature.details.index.to_list() == feature._residue_ids

    @pytest.mark.parametrize(
        "structure_id, remote, radius, method, n_residues, up_mean, down_mean",
        [
            (3833, REMOTE, 12.0, "HSExposureCA", 85, 13.505882, 17.235294),
            (3833, REMOTE, 12.0, "HSExposureCB", 85, 14.470588, 16.270588),
        ],
    )
    def test_get_exposure_by_method(
        self, structure_id, remote, radius, method, n_residues, up_mean, down_mean
    ):
        """
        Test half sphere exposure and exposure ratio calculation as well as the result format.

        Parameters
        ----------
        structure_id : int
            KLIFS structure ID.
        remote : None or opencadd.databases.klifs.session.Session
            Remote KLIFS session. If None, generate new remote session.
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
        """

        # Calculate exposure
        pocket = PocketBioPython.from_structure_klifs_id(structure_id, remote)
        exposures_calculated = SolventExposureFeature._get_exposures_by_method(
            pocket, radius, method
        )

        # Test DataFrame length
        assert len(exposures_calculated) == n_residues

        # Test column names
        prefix = method[-2:].lower()
        columns = [f"{prefix}.{i}" for i in ["up", "down", "angle_cb_ca_pcb", "exposure"]]
        assert list(exposures_calculated.columns) == columns

        # Test exposure up values (mean)
        up_mean_calculated = exposures_calculated[f"{prefix}.up"].mean()
        assert up_mean == pytest.approx(up_mean_calculated)

        # Test exposure down values (mean)
        down_mean_calculated = exposures_calculated[f"{prefix}.down"].mean()
        assert down_mean == pytest.approx(down_mean_calculated)

        # Test for example residue the exposure ratio calculation
        example_residue = exposures_calculated.iloc[0]
        ratio = example_residue[f"{prefix}.exposure"]
        ratio_calculated = example_residue[f"{prefix}.down"] / (
            example_residue[f"{prefix}.up"] + example_residue[f"{prefix}.down"]
        )
        assert ratio == pytest.approx(ratio_calculated)

    @pytest.mark.parametrize(
        "structure_id, remote, radius, n_residues, missing_exposure",
        [(12347, REMOTE, 12.0, 78, {"ca": [463, 468, 595], "cb": []})],
    )
    def test_get_exposures(self, structure_id, remote, radius, n_residues, missing_exposure):
        """
        Test join of HSExposureCA and HSExposureCB data.

        Parameters
        ----------
        structure_id : int
            KLIFS structure ID.
        remote : None or opencadd.databases.klifs.session.Session
            Remote KLIFS session. If None, generate new remote session.
        radius : float
            Sphere radius to be used for half sphere exposure calculation.
        n_residues : int
            Number of residues in exposure calculation result.
        missing_exposure : dict of list of int
            Residue IDs with missing exposures for HSExposureCA and HSExposureCB calculation.
        """

        # Get exposure
        pocket = PocketBioPython.from_structure_klifs_id(structure_id, remote)
        feature = SolventExposureFeature()
        exposures_calculated = feature._get_exposures(pocket, radius)

        # Test DataFrame length
        assert len(exposures_calculated) == n_residues

        # Test column names
        column_names_ca = ["ca.up", "ca.down", "ca.angle_cb_ca_pcb", "ca.exposure"]
        column_names_cb = ["cb.up", "cb.down", "cb.angle_cb_ca_pcb", "cb.exposure"]
        column_names = column_names_ca + column_names_cb + ["exposure"]
        assert list(exposures_calculated.columns) == column_names

        # Test missing residues in HSExposureCA and HSExposureCB calculation
        assert (
            missing_exposure["ca"]
            == exposures_calculated[exposures_calculated["ca.exposure"].isna()].index.to_list()
        )
        assert (
            missing_exposure["cb"]
            == exposures_calculated[exposures_calculated["cb.exposure"].isna()].index.to_list()
        )
