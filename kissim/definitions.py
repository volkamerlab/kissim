"""
kissim.encoding.definitions 

Handles definitions.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import json

PATH_DATA = Path(__file__).parent / "data"

# Standard amino acids
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

# Non-standard amino acids their conversion to parent standard amino acids
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

# Site Align features for standard amino acids
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
        "THR": [1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        "TRP": [3.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        "TYR": [3.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        "VAL": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    },
    orient="index",
    columns=["size", "hbd", "hba", "charge", "aromatic", "aliphatic"],
)

# Side chain representative atom for standard amino acids
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
# Cutoffs that split side chain angle into categories inwards, intermediate, and outwards
SIDE_CHAIN_ANGLE_CUTOFFS = [45.0, 90.0]

# Sphere radius used to calculate solvent exposure
EXPOSURE_RADIUS = 12.0
# Cutoffs that split exposure ratio into categories low, intermediate, and high solvent exposure
EXPOSURE_RATIO_CUTOFFS = [0.45, 0.55]

# Define subpockets based on anchor residues
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

# Summary of possible discrete feature values
# Note: HBA only contains 0.0, 1.0 and 3.0 but we want to include the missing 2.0 as well to
# emphasize on the larger step between the first and last two categories
DISCRETE_FEATURE_VALUES = {
    "size": SITEALIGN_FEATURES["size"].sort_values().unique().tolist(),
    "hbd": np.arange(SITEALIGN_FEATURES["hbd"].min(), SITEALIGN_FEATURES["hbd"].max() + 1),
    "hba": SITEALIGN_FEATURES["hba"].sort_values().unique().tolist(),
    "charge": SITEALIGN_FEATURES["charge"].sort_values().unique().tolist(),
    "aromatic": SITEALIGN_FEATURES["aromatic"].sort_values().unique().tolist(),
    "aliphatic": SITEALIGN_FEATURES["aliphatic"].sort_values().unique().tolist(),
    "sco": [1.0, 2.0, 3.0],
    "exposure": [1.0, 2.0, 3.0],
}

# Metadata for features (useful for plotting!)
FEATURE_METADATA = {
    "size": ("size", ["small", "intermediate", "large"]),
    "hbd": ("number of HBDs", ["0", "1", "2", "3"]),
    "hba": ("number of HBAs", ["0", "1", "2"]),
    "charge": ("charge", ["negative", "neutral", "positive"]),
    "aromatic": ("aromatic?", ["no", "yes"]),
    "aliphatic": ("aliphatic?", ["no", "yes"]),
    "sco": (
        "side chain orientation w.r.t. pocket center",
        ["inwards", "intermediate", "outwards"],
    ),
    "exposure": ("solvent exposure", ["low", "intermediate", "high"]),
    "hinge_region": (r"distance to hinge region center [$\AA$]", None),
    "dfg_region": (r"distance to DFG region center [$\AA$]", None),
    "front_pocket": (r"distance to front pocket center [$\AA$]", None),
    "center": (r"distance to pocket center [$\AA$]", None),
}

# Distance and moment cutoffs used for fingerprint normalization
# Cutoffs defined in this notebook:
# https://github.com/volkamerlab/kissim_app/blob/master/notebooks/004_fingerprints/002_spatial_feature_cutoffs.ipynb
DISTANCE_CUTOFFS = {}
MOMENT_CUTOFFS = {}
for how in ["fine", "coarse"]:

    DISTANCE_CUTOFFS[how] = pd.read_csv(
        PATH_DATA / f"min_max_distances_{how}.csv", index_col=[0, 1]
    )
    MOMENT_CUTOFFS[how] = pd.read_csv(PATH_DATA / f"min_max_moments_{how}.csv", index_col=[0, 1])

# KLIFS pocket residue subsets by DFG conformation
with open(PATH_DATA / "klifs_pocket_residue_subset.json") as f:
    KLIFS_POCKET_RESIDUE_SUBSET = json.load(f)
