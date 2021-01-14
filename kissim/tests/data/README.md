# Test data

This folder contains files used for unit testing.
Test data refers to a KLIFS download on 2020-12-23 covering only structures for the human ABL2 kinase.

## Folder structure

The test data's folder structure mimics the input data's folder structure needed to use the package.

```
KLIFS_download/
  SPECIES/			# e.g. HUMAN
    KINASENAME/  		# e.g. EGFR
      pdbid\_[altX\_]chainX/  	# e.g. 3w33_altA_chainA
        pocket.pdb  		# KLIFS pocket data in pdb format
        complex.pdb  		# KLIFS complex data in pdb format
  KLIFS_export.csv  		# KLIFS metadata part 1
  overview.csv  		# KLIFS metadata part 2
  klifs_metadata.csv		# KLIFS metadata merged from part 1 and 2
```
