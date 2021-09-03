"""
Unit and regression test for kissim.encoding.FingerprintGeneratorNormalized.
"""

import pytest
import numpy as np

from kissim.encoding import FingerprintGeneratorNormalized


class TestFingerprintGeneratorNormalized:
    """
    Test normalized fingerprints class.
    """

    @pytest.mark.parametrize(
        "method, fine_grained, structure_klifs_id, fingerprint_values_array_sum, fingerprint_normalized_values_array_sum",
        [
            ("min_max", True, 109, 5108.226235, 409.057707),
            ("min_max", False, 109, 5108.226235, 398.950979),
        ],
    )
    def test_from_fingerprint_generator(
        self,
        fingerprint_generator,
        method,
        fine_grained,
        structure_klifs_id,
        fingerprint_values_array_sum,
        fingerprint_normalized_values_array_sum,
    ):
        """
        Test for the first fingerprint in the template fingerprints if the sum of unnormalized and
        normalized fingerprint values is correct.
        """

        fingerprint_generator_normalized = (
            FingerprintGeneratorNormalized.from_fingerprint_generator(
                fingerprint_generator, method, fine_grained
            )
        )

        fingerprint = fingerprint_generator.data[structure_klifs_id]
        fingerprint_normalized = fingerprint_generator_normalized.data[structure_klifs_id]

        fingerprint_values_array_sum_calculated = np.nansum(fingerprint.values_array())
        assert (
            pytest.approx(fingerprint_values_array_sum_calculated, abs=1e-6)
            == fingerprint_values_array_sum
        )

        fingerprint_normalized_values_array_sum_calculated = np.nansum(
            fingerprint_normalized.values_array()
        )
        assert (
            pytest.approx(fingerprint_normalized_values_array_sum_calculated, abs=1e-6)
            == fingerprint_normalized_values_array_sum
        )
