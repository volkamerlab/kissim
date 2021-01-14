"""
Unit and regression test for kissim's cli modules.

# Test input options
python cli.py
python cli.py -i 12347 -o "fps.json"
python cli.py -i 12347 109 -o "fps.json"
python cli.py -i "data/structure_klifs_ids.txt" -o "fps.json"

# Test output options
python cli.py -i 12347 109

# Test number of cores
python cli.py -i 12347 109 -o "fps.json" -c 4

# Test local KLIFS session
python cli.py -i 12347 109 -o "fps.json" -l "data/KLIFS_download"  # TODO error 12347
python cli.py -i 109 110 -o "fps.json" -l "data/KLIFS_download"
"""

