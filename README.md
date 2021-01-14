Structural kinase similarity (`kissim`)
==============================
[//]: # (Badges)
[![GH Actions Status](https://github.com/volkamerlab/kissim/workflows/CI/badge.svg)](https://github.com/volkamerlab/kissim/actions?query=branch%3Amaster)
[![AppVeyor Build status](https://ci.appveyor.com/api/projects/status/volkamerlab/branch/master?svg=true)](https://ci.appveyor.com/project/volkamerlab/kissim/branch/master)
[![codecov](https://codecov.io/gh/volkamerlab/kissim/branch/master/graph/badge.svg)](https://codecov.io/gh/volkamerlab/kissim/branch/master)

**Subpocket-based structural fingerprint for kinase pocket comparison**

Kinases are important and frequently studied drug targets for cancer and inflammatory diseases. 
Due to the highly conserved structure of kinases, especially at the ATP binding site, 
the main challenge while developing kinase inhibitors is selectivity, 
which requires a comprehensive understanding of kinase similarity. [1]  

Our package `kissim` offers a novel fingerprinting strategy designed specifically for kinase pockets, 
allowing for similarity studies across the structurally covered kinome. 
The kinase fingerprint is based on the KLIFS [2] pocket alignment, 
which defines 85 pocket residues for all kinase structures. 
This enables a residue-by-residue comparison without a computationally expensive alignment step. 
The pocket fingerprint consists of 85 concatenated residue fingerprints, 
each encoding a residue’s spatial and physicochemical properties. 
The spatial properties describe the residue’s position in relation to the kinase pocket centroid and 
important kinase subpockets, i.e. the hinge region, the DFG region, and the front pocket. 
The physicochemical properties encompass for each residue its pocket exposure, side chain angle, 
size and pharmacophoric features. 
Pairwise comparison of all kinases and clustering reveals kinome-wide similarities.

The potential of our subpocket-based structural kinase comparison is demonstrated by 
comparing our structure-based clustering to the sequence-based Manning kinome tree [3], 
predicting retrospectively ligand-based on- and off-target, 
as well as assessing structure-based conservation of residue positions. 
We believe that our analysis of the structurally covered kinome can help researchers 
(i) to detect potential promiscuities and off-targets at an early stage of inhibitor design and 
(ii) to conduct structure-informed polypharmacology studies.

[1] Kooistra and Volkamer. Ann Rep Med Chem. 2017, [2] van Linden et al. J Med Chem. 2014, [3] Manning et al. Science. 2002



### Copyright

Copyright (c) 2019, Volkamer Lab


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.0.
