Structural kinase similarity (`kissim`)
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/volkamerlab/kissim/workflows/CI/badge.svg)](https://github.com/volkamerlab/kissim/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/volkamerlab/kissim/branch/master/graph/badge.svg)](https://codecov.io/gh/volkamerlab/kissim/branch/master)

> ⚠ This project is work-in-progress. The API is not final.

**Subpocket-based structural fingerprint for kinase pocket comparison** 

The `kissim` packages offers a novel fingerprinting strategy designed specifically for kinase pockets, 
allowing for similarity studies across the structurally covered kinome. 
The kinase fingerprint is based on the [KLIFS](klifs.net/) pocket alignment, 
which defines 85 pocket residues for all kinase structures. 
This enables a residue-by-residue comparison without a computationally expensive alignment step. 
The pocket fingerprint consists of 85 concatenated residue fingerprints, 
each encoding a residue’s spatial and physicochemical properties. 
The spatial properties describe the residue’s position in relation to the kinase pocket centroid and 
important kinase subpockets, i.e. the hinge region, the DFG region, and the front pocket. 
The physicochemical properties encompass for each residue its pocket exposure, side chain angle, 
size and pharmacophoric features. 
Pairwise comparison of all kinases and clustering reveals kinome-wide similarities.



### Copyright

Copyright (c) 2019, Volkamer Lab


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.5.
