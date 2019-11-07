"""
Unit and regression test for kinsim_structure.encoding.Fingerprint class.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pytest

from kinsim_structure.encoding import Fingerprint, DISTANCE_CUTOFFS, MOMENT_CUTOFFS, FEATURE_NAMES
from kinsim_structure.auxiliary import KlifsMoleculeLoader, PdbChainLoader


@pytest.mark.parametrize('mol2_filename, pdb_filename, chain_id', [
    ('ABL1/2g2i_chainA/pocket.mol2', '2g2i.cif', 'A')
])
def test_from_molecule(mol2_filename, pdb_filename, chain_id):
    """
    Test if Fingerprint class attributes (accessed via property function) have correct DataFrame shape, column and
    index names.

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

    fingerprint = Fingerprint()
    fingerprint.from_molecule(klifs_molecule_loader.molecule, pdb_chain_loader.chain)

    klifs_ids = list(range(1, 86))

    # Non-normalized
    assert fingerprint.physicochemical.shape == (85, 8)
    assert list(fingerprint.physicochemical.index) == klifs_ids
    assert list(fingerprint.physicochemical.columns) == FEATURE_NAMES['physicochemical']

    assert fingerprint.distances.shape == (85, 4)
    assert list(fingerprint.distances.index) == klifs_ids
    assert list(fingerprint.distances.columns) == FEATURE_NAMES['distances']

    assert fingerprint.moments.shape == (4, 3)
    assert list(fingerprint.moments.index) == FEATURE_NAMES['distances']
    assert list(fingerprint.moments.columns) == FEATURE_NAMES['moments']

    assert list(fingerprint.physicochemical_distances.keys()) == 'physicochemical distances'.split()
    assert list(fingerprint.physicochemical_moments.keys()) == 'physicochemical moments'.split()

    # Normalized
    assert fingerprint.physicochemical_normalized.shape == (85, 8)
    assert list(fingerprint.physicochemical_normalized.index) == klifs_ids
    assert list(fingerprint.physicochemical_normalized.columns) == FEATURE_NAMES['physicochemical']

    assert fingerprint.distances.shape == (85, 4)
    assert list(fingerprint.distances_normalized.index) == klifs_ids
    assert list(fingerprint.distances_normalized.columns) == FEATURE_NAMES['distances']

    assert fingerprint.moments.shape == (4, 3)
    assert list(fingerprint.moments_normalized.index) == FEATURE_NAMES['distances']
    assert list(fingerprint.moments_normalized.columns) == FEATURE_NAMES['moments']

    assert list(fingerprint.physicochemical_distances_normalized.keys()) == 'physicochemical distances'.split()
    assert list(fingerprint.physicochemical_moments_normalized.keys()) == 'physicochemical moments'.split()


@pytest.mark.parametrize('physicochemical, physicochemical_normalized', [
    (
        pd.DataFrame(
            [
                [3, 3, 2, 1, 1, 1, 180, 1]
            ],
            columns=FEATURE_NAMES['physicochemical']
        ),
        pd.DataFrame(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            ],
            columns=FEATURE_NAMES['physicochemical']
        )
    ),
    (
        pd.DataFrame(
            [
                [2, 1, 1, 0, 0, 0, 90, 0.8]
            ],
            columns=FEATURE_NAMES['physicochemical']
        ),
        pd.DataFrame(
            [
                [0.5, 0.3333, 0.5, 0.5, 0.0, 0.0, 0.5, 0.8]
            ],
            columns=FEATURE_NAMES['physicochemical']
        )
    ),
    (
        pd.DataFrame(
            [
                [np.nan] * 8
            ],
            columns=FEATURE_NAMES['physicochemical']
        ),
        pd.DataFrame(
            [
                [np.nan] * 8
            ],
            columns=FEATURE_NAMES['physicochemical']
        )
    )
])
def test_normalize_physicochemical_bits(physicochemical, physicochemical_normalized):
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
    fingerprint.molecule_code = 'molecule'
    fingerprint.fingerprint['physicochemical'] = physicochemical

    physicochemical_normalized_calculated = fingerprint._normalize_physicochemical_bits()

    for feature in FEATURE_NAMES['physicochemical']:

        if np.isnan(physicochemical.iloc[0, 0]):
            assert np.isnan(
                physicochemical_normalized_calculated[feature][0]
            ) and np.isnan(
                physicochemical_normalized[feature][0]
            )
        else:
            assert np.isclose(
                physicochemical_normalized_calculated[feature][0],
                physicochemical_normalized[feature][0],
                rtol=1e-03
            )


@pytest.mark.parametrize('distances, distances_normalized', [
    (
        pd.DataFrame(
            [
                [1, 1, 1, 1]
            ],
            columns=FEATURE_NAMES['distances']
        ),
        pd.DataFrame(
            [
                [0.0, 0.0, 0.0, 0.0]
            ],
            columns=FEATURE_NAMES['distances']
        )
    ),
    (
        pd.DataFrame(
            [
                [
                    DISTANCE_CUTOFFS['distance_to_centroid'][0],
                    DISTANCE_CUTOFFS['distance_to_hinge_region'][0],
                    DISTANCE_CUTOFFS['distance_to_dfg_region'][0],
                    DISTANCE_CUTOFFS['distance_to_front_pocket'][0]
                ]
            ],
            columns=FEATURE_NAMES['distances']
        ),
        pd.DataFrame(
            [
                [0.0, 0.0, 0.0, 0.0]
            ],
            columns=FEATURE_NAMES['distances']
        )
    ),
    (
        pd.DataFrame(
            [
                [10, 10, 10, 10]
            ],
            columns=FEATURE_NAMES['distances']
        ),
        pd.DataFrame(
            [
                [0.3792, 0.3110, 0.2438, 0.2510]
            ],
            columns=FEATURE_NAMES['distances']
        )
    ),
    (
        pd.DataFrame(
            [
                [
                    DISTANCE_CUTOFFS['distance_to_centroid'][1],
                    DISTANCE_CUTOFFS['distance_to_hinge_region'][1],
                    DISTANCE_CUTOFFS['distance_to_dfg_region'][1],
                    DISTANCE_CUTOFFS['distance_to_front_pocket'][1]
                ]
            ],
            columns=FEATURE_NAMES['distances']
        ),
        pd.DataFrame(
            [
                [1.0, 1.0, 1.0, 1.0]
            ],
            columns=FEATURE_NAMES['distances']
        )
    ),
    (
        pd.DataFrame(
            [
                [30, 30, 30, 30]
            ],
            columns=FEATURE_NAMES['distances']
        ),
        pd.DataFrame(
            [
                [1.0, 1.0, 1.0, 1.0]
            ],
            columns=FEATURE_NAMES['distances']
        )
    ),
    (
        pd.DataFrame(
            [
                [np.nan] * 4
            ],
            columns=FEATURE_NAMES['distances']
        ),
        pd.DataFrame(
            [
                [np.nan] * 4
            ],
            columns=FEATURE_NAMES['distances']
        )
    )
])
def test_normalize_distances_bits(distances, distances_normalized):
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
    fingerprint.molecule_code = 'molecule'
    fingerprint.fingerprint['distances'] = distances

    distances_normalized_calculated = fingerprint._normalize_distances_bits()

    for feature in FEATURE_NAMES['distances']:

        if np.isnan(distances.iloc[0, 0]):
            assert np.isnan(
                distances_normalized_calculated[feature][0]
            ) and np.isnan(
                distances_normalized[feature][0]
            )
        else:
            assert np.isclose(
                distances_normalized_calculated[feature][0],
                distances_normalized[feature][0],
                rtol=1e-03
            )


@pytest.mark.parametrize('moments, moments_normalized', [
    (
        pd.DataFrame(
            [
                [11, 3, -2]
            ],
            columns=FEATURE_NAMES['moments']
        ),
        pd.DataFrame(
            [
                [0.0, 0.0, 0.0]
            ],
            columns=FEATURE_NAMES['moments']
        )
    ),
    (
        pd.DataFrame(
            [
                [
                    MOMENT_CUTOFFS['moment1'][0],
                    MOMENT_CUTOFFS['moment2'][0],
                    MOMENT_CUTOFFS['moment3'][0]
                ]
            ],
            columns=FEATURE_NAMES['moments']
        ),
        pd.DataFrame(
            [
                [0.0, 0.0, 0.0]
            ],
            columns=FEATURE_NAMES['moments']
        )
    ),
    (
        pd.DataFrame(
            [
                [12, 4, 1]
            ],
            columns=FEATURE_NAMES['moments']
        ),
        pd.DataFrame(
            [
                [0.1301, 0.355, 0.4030]
            ],
            columns=FEATURE_NAMES['moments']
        )
    ),
    (
        pd.DataFrame(
            [
                [
                    MOMENT_CUTOFFS['moment1'][1],
                    MOMENT_CUTOFFS['moment2'][1],
                    MOMENT_CUTOFFS['moment3'][1]
                ]
            ],
            columns=FEATURE_NAMES['moments']
        ),
        pd.DataFrame(
            [
                [1.0, 1.0, 1.0]
            ],
            columns=FEATURE_NAMES['moments']
        )
    ),
    (
        pd.DataFrame(
            [
                [15, 6, 5]
            ],
            columns=FEATURE_NAMES['moments']
        ),
        pd.DataFrame(
            [
                [1.0, 1.0, 1.0]
            ],
            columns=FEATURE_NAMES['moments']
        )
    ),
    (
        pd.DataFrame(
            [
                [np.nan] * 3
            ],
            columns=FEATURE_NAMES['moments']
        ),
        pd.DataFrame(
            [
                [np.nan] * 3
            ],
            columns=FEATURE_NAMES['moments']
        )
    )
])
def test_normalize_moments_bits(moments, moments_normalized):

    fingerprint = Fingerprint()
    fingerprint.molecule_code = 'molecule'
    fingerprint.fingerprint['moments'] = moments

    moments_normalized_calculated = fingerprint._normalize_moments_bits()

    for feature in FEATURE_NAMES['moments']:

        if np.isnan(moments.iloc[0, 0]):
            assert np.isnan(
                moments_normalized_calculated[feature][0]
            ) and np.isnan(
                moments_normalized[feature][0]
            )
        else:
            assert np.isclose(
                moments_normalized_calculated[feature][0],
                moments_normalized[feature][0],
                rtol=1e-03
            )


@pytest.mark.parametrize('distances, moments', [
    (
        pd.DataFrame([[1, 1], [4, 4], [4, 4]]),
        pd.DataFrame([[3.00, 1.41, -1.26], [3.00, 1.41, -1.26]], columns='moment1 moment2 moment3'.split())
    ),
    (
        pd.DataFrame([[1, 2]]),
        pd.DataFrame([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], columns='moment1 moment2 moment3'.split())
    )
])
def test_calc_moments(distances, moments):

    fingerprint = Fingerprint()
    moments_calculated = fingerprint._calc_moments(distances)

    print(len(distances))
    print(moments_calculated)

    assert np.isclose(moments_calculated.moment1[0], moments.moment1[0], rtol=1e-02)
    assert np.isclose(moments_calculated.moment2[0], moments.moment2[0], rtol=1e-02)
    assert np.isclose(moments_calculated.moment3[0], moments.moment3[0], rtol=1e-02)


@pytest.mark.parametrize('value, minimum, maximum, value_normalized', [
    (1.00, 2.00, 3.00, 0.00),
    (2.00, 2.00, 3.00, 0.00),
    (2.50, 2.00, 3.00, 0.50),
    (3.00, 2.00, 3.00, 1.00),
    (4.00, 2.00, 3.00, 1.00),
    (np.nan, 2.00, 3.00, np.nan)

])
def test_normalize(value, minimum, maximum, value_normalized):
    """
    Test value normalization.

    Parameters
    ----------
    value : float or int
            Value to be normalized.
    minimum : float or int
        Minimum value for normalization, values equal or greater than this minimum are set to 0.0.
    maximum : float or int
        Maximum value for normalization, values equal or greater than this maximum are set to 1.0.
    value_normalized : float
        Normalized value.
    """

    fingerprint = Fingerprint()
    value_normalized_calculated = fingerprint._normalize(value, minimum, maximum)

    if np.isnan(value):
        assert np.isnan(value_normalized_calculated) and np.isnan(value_normalized)
    else:
        assert np.isclose(value_normalized_calculated, value_normalized, rtol=1e-06)
