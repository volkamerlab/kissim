"""
kissim.encoding.definitions 

Handles kissim-specific definitions.
"""

import pandas as pd

SIDE_CHAIN_REPRESENTATIVE = {
    "ALA": "CB",
    "ARG": "CG",
    "ASN": "CG",
    "ASP": "CG",
    "CYS": "SG",
    "GLN": "CD",
    "GLU": "CD",
    "HIS": "CE1",
    "ILE": "CD1",
    "LEU": "CG",
    "LYS": "NZ",
    "MET": "CE",
    "PHE": "CZ",
    "PRO": "CB",
    "SER": "OG",
    "THR": "CB",
    "TRP": "CE2",
    "TYR": "OH",
    "VAL": "CB",
}

SITEALIGN_FEATURES = pd.DataFrame.from_dict(
    {
        "ALA": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "ARG": [3.0, 3.0, 0.0, 1.0, 0.0, 0.0],
        "ASN": [2.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        "ASP": [2.0, 0.0, 2.0, -1.0, 0.0, 0.0],
        "CYS": [1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "GLN": [2.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        "GLU": [2.0, 0.0, 2.0, -1.0, 0.0, 0.0],
        "GLY": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "HIS": [2.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        "ILE": [2.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "LEU": [2.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "LYS": [2.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        "MET": [2.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "PHE": [3.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "PRO": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "SER": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        "THR": [1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
        "TRP": [3.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        "TYR": [3.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        "VAL": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    },
    orient="index",
    columns=["size", "hbd", "hba", "charge", "aromatic", "aliphatic"],
)

EXPOSURE_RADIUS = 12.0

ANCHOR_RESIDUES = {
    "hinge_region": [16, 47, 80],
    "dfg_region": [19, 24, 81],
    "front_pocket": [10, 48, 72],
}

SUBPOCKETS = {
    "anchor_residue.klifs_ids": list(ANCHOR_RESIDUES.values()),
    "subpocket.name": list(ANCHOR_RESIDUES.keys()),
    "subpocket.color": ["magenta", "cornflowerblue", "green"],
}

# TODO update
DISTANCE_CUTOFFS = {  # 99% percentile of all distances
    "center": (3.05, 21.38),
    "hinge_region": (4.10, 23.07),
    "dfg_region": (4.62, 26.69),
    "front_pocket": (5.46, 23.55),
}

# TODO update
MOMENT_CUTOFFS = {  # 99% percentile of all moments
    1: (11.68, 14.14),
    2: (3.29, 5.29),
    3: (-1.47, 4.66),
}
