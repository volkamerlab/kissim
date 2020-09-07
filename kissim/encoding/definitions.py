"""
kissim.encoding.definitions TODO
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

MODIFIED_RESIDUE_CONVERSION = {
    "CAF": "CYS",
    "CME": "CYS",
    "CSS": "CYS",
    "OCY": "CYS",
    "KCX": "LYS",
    "MSE": "MET",
    "PHD": "ASP",
    "PTR": "TYR",
}

EXPOSURE_RADIUS = 12.0

N_HEAVY_ATOMS = {
    "GLY": 0,
    "ALA": 1,
    "CYS": 2,
    "SER": 2,
    "PRO": 3,
    "THR": 3,
    "VAL": 3,
    "ASN": 4,
    "ASP": 4,
    "ILE": 4,
    "LEU": 4,
    "MET": 4,
    "GLN": 5,
    "GLU": 5,
    "LYS": 5,
    "HIS": 6,
    "ARG": 7,
    "PHE": 7,
    "TYR": 8,
    "TRP": 10,
}

N_HEAVY_ATOMS_CUTOFF = (
    {  # Number of heavy atoms needed for side chain centroid calculation (>75% coverage)
        "GLY": 0,
        "ALA": 1,
        "CYS": 2,
        "SER": 2,
        "PRO": 3,
        "THR": 3,
        "VAL": 3,
        "ASN": 3,
        "ASP": 3,
        "ILE": 3,
        "LEU": 3,
        "MET": 3,
        "GLN": 4,
        "GLU": 4,
        "LYS": 4,
        "HIS": 5,
        "ARG": 6,
        "PHE": 6,
        "TYR": 6,
        "TRP": 8,
    }
)

ANCHOR_RESIDUES = {
    "hinge_region": [16, 47, 80],
    "dfg_region": [19, 24, 81],
    "front_pocket": [6, 48, 75],
}  # Are the same as in Eva's implementation

DISTANCE_CUTOFFS = {  # 99% percentile of all distances
    "distance_to_centroid": (3.05, 21.38),
    "distance_to_hinge_region": (4.10, 23.07),
    "distance_to_dfg_region": (4.62, 26.69),
    "distance_to_front_pocket": (5.46, 23.55),
}

MOMENT_CUTOFFS = {  # 99% percentile of all moments
    "moment1": (11.68, 14.14),
    "moment2": (3.29, 5.29),
    "moment3": (-1.47, 4.66),
}

# KLIFS IDs for hinge/DFG region (taken from KLIFS website)
HINGE_KLIFS_IDS = [46, 47, 48]
DFG_KLIFS_IDS = [81, 82, 83]
