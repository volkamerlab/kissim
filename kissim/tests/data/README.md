# Test data

This folder contains files used for unit testing.

## Folder structure

The test data's folder structure mimics the input data's folder structure needed to use the package (* only included if used for unit testing).

```
KLIFS_download/
  SPECIES/			# e.g. HUMAN
    KINASENAME/  		# e.g. EGFR
      pdbid\_[altX\_]chainX/  	# e.g. 3w33_altA_chainA
        pocket.mol2  		# KLIFS pocket data*
        protein.mol2  		# KLIFS protein data*
        protein_pymol.pdb  	# KLIFS protein data converted to pdb using PyMol*
  KLIFS_export.csv  		# KLIFS metadata part 1
  overview.csv  		# KLIFS metadata part 2
klifs_metadata.csv		# KLIFS metadata merged from part 1 and 2
```
