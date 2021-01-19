"""
kissim.schema

Defines basic schema.
"""

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
