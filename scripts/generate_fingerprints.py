"""
This script generates all fingerprints for a given dataset.
"""

import datetime
from pathlib import Path
import pickle

import pandas as pd

from kinsim_structure.auxiliary import KlifsMoleculeLoader, PdbChainLoader
from kinsim_structure.encoding import Fingerprint

# Load IO paths
path_to_data = Path('/') / 'home' / 'dominique' / 'Documents' / 'data' / 'kinsim' / '20190724_full'
path_to_kinsim = Path('/') / 'home' / 'dominique' / 'Documents' / 'projects' / 'kinsim_structure'
path_to_results = path_to_kinsim / 'results'

metadata_path = path_to_data / 'preprocessed' / 'klifs_metadata_preprocessed.csv'

# Load metadata
klifs_metadata = pd.read_csv(metadata_path)

# Initialize lists to save results from looping
fingerprints = []
error_entries = []

# Get start time
start = datetime.datetime.now()

for index, klifs_metadata_entry in klifs_metadata.iterrows():

    print(f'{index + 1} / {klifs_metadata.shape[0]}')

    try:
        klifs_molecule_loader = KlifsMoleculeLoader(klifs_metadata_entry=klifs_metadata_entry)
        pdb_chain_loader = PdbChainLoader(klifs_metadata_entry=klifs_metadata_entry)

        molecule = klifs_molecule_loader.molecule
        chain = pdb_chain_loader.chain

        fp = Fingerprint()
        fp.from_molecule(molecule, chain)

        fingerprint = fp.features
        fingerprint['metadata_index'] = index
        fingerprint['molecule_code'] = molecule.code

        fingerprints.append(fingerprint)

    except:
        error_entries.append(klifs_metadata_entry)

fingerprints = pd.concat(fingerprints)

# Save fingerprints
fingerprints.to_csv(path_to_results / 'fingerprints.csv')

# Save problematic entries
with open(path_to_results / 'fingerprints_error_entries.p', 'wb') as f:
    pickle.dump(error_entries, f)

# Get start time
end = datetime.datetime.now()

print(start)
print(end)
