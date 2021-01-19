"""
Unit and regression test for kissim.io.PocketDataFrame class methods.

Note that class 
- methods (`from_structure_klifs_id`)
- attributes (`_residue_ids`, `residue_ixs`) and 
- properties (`residues`)
are tested in test_io.py.

Note also that PocketDataFrame is a subclass of opencadd's PocketKlifs class, 
which is tested thouroughly in opencadd:
- attributes (`data`, `center`)
- properties (`ca_atoms`)
"""
