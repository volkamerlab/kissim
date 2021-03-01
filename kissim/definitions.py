"""
kissim.encoding.definitions 

Handles definitions.
"""

import pandas as pd

STANDARD_AMINO_ACIDS = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]

NON_STANDARD_AMINO_ACID_CONVERSION = {
    "CAF": "CYS",
    "CME": "CYS",
    "CSS": "CYS",
    "OCY": "CYS",
    "KCX": "LYS",
    "MSE": "MET",
    "PHD": "ASP",
    "PTR": "TYR",
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

# Cutoffs defined in this notebook:
# https://github.com/volkamerlab/kissim_app/blob/master/notebooks/fingerprints/spatial_feature_cutoffs.ipynb
DISTANCE_CUTOFFS = {
    "hinge_region": (2.0, 31.0),
    "dfg_region": (0.0, 34.0),
    "front_pocket": (1.0, 33.0),
    "center": (1.0, 29.0),
}
MOMENT_CUTOFFS = {1: (11.0, 17.0), 2: (2.0, 7.0), 3: (-3.0, 7.0)}
