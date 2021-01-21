"""
Unit and regression test for kissim's cli modules.

# ENCODE
# Test input options
kissim encode
kissim encode -i 12347 -o "kissim/tests/data/fingerprints.json"
kissim encode -i 12347 109 -o "kissim/tests/data/fingerprints.json"
kissim encode -i "kissim/tests/data/structure_klifs_ids.txt" -o "kissim/tests/data/fingerprints.json"

# Test output options
kissim encode -i 12347 109

# Test number of cores
kissim encode -i 12347 109 -o "kissim/tests/data/fingerprints.json" -c 4

# Test local KLIFS session
kissim encode -i 12347 109 -o "kissim/tests/data/fingerprints.json" -l "kissim/tests/data/KLIFS_download"
kissim encode -i 109 110 -o "kissim/tests/data/fingerprints.json" -l "kissim/tests/data/KLIFS_download"


# COMPARE
kissim compare
kissim compare -i "kissim/tests/data/fingerprints.json" -o "kissim/tests/data/distances.csv" 
"""
