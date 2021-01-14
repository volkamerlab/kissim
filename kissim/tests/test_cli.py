"""
Unit and regression test for kissim's cli modules.

# Test input options
kissim-encoder
kissim-encoder -i 12347 -o "fps.json"
kissim-encoder -i 12347 109 -o "fps.json"
kissim-encoder -i "data/structure_klifs_ids.txt" -o "fps.json"

# Test output options
kissim-encoder -i 12347 109

# Test number of cores
kissim-encoder -i 12347 109 -o "fps.json" -c 4

# Test local KLIFS session
kissim-encoder -i 12347 109 -o "fps.json" -l "data/KLIFS_download"  # TODO error 12347
kissim-encoder -i 109 110 -o "fps.json" -l "data/KLIFS_download"
"""

