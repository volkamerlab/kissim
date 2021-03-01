"""
kissim.schema

Defines basic schema.
"""

FEATURE_NAMES = ["physicochemical", "spatial"]
FEATURE_NAMES_PHYSICOCHEMICAL_DICT = [
    "size",
    "hbd",
    "hba",
    "charge",
    "aromatic",
    "aliphatic",
    "sco",
    "sco.vertex_angle",
    "exposure",
    "exposure.ratio",
]
FEATURE_NAMES_PHYSICOCHEMICAL = [
    "size",
    "hbd",
    "hba",
    "charge",
    "aromatic",
    "aliphatic",
    "sco",
    "exposure",
]
FEATURE_NAMES_SPATIAL = ["distances", "moments"]
FEATURE_NAMES_SPATIAL_DICT = ["distances", "moments", "subpocket_centers"]
FEATURE_NAMES_DISTANCES_AND_MOMENTS = [
    "hinge_region",
    "dfg_region",
    "front_pocket",
    "center",
]

DISTANCES_FEATURE_NAMES = {
    "physicochemical": [
        "size",
        "hbd",
        "hba",
        "charge",
        "aromatic",
        "aliphatic",
        "sco",
        "exposure",
    ],
    "distances": [
        "distance_to_centroid",
        "distance_to_hinge_region",
        "distance_to_dfg_region",
        "distance_to_front_pocket",
    ],
    "moments": ["moment1", "moment2", "moment3"],
}
