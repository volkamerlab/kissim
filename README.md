Structural kinase similarity (`kissim`)
==============================

[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/volkamerlab/kissim/workflows/CI/badge.svg)](https://github.com/volkamerlab/kissim/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/volkamerlab/kissim/branch/main/graph/badge.svg)](https://codecov.io/gh/volkamerlab/kissim)
[![Documentation Status](https://readthedocs.org/projects/kissim/badge/?version=latest)](https://kissim.readthedocs.io/en/latest/?badge=latest)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/kissim.svg)](https://anaconda.org/conda-forge/kissim)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Subpocket-based structural fingerprint for kinase pocket comparison** 

![Subpocket-based structural fingerprint for kinase pockets](docs/_static/kissim_toc.png)

## Description

The `kissim` package offers a novel fingerprinting strategy designed specifically for kinase pockets, 
allowing for similarity studies across the structurally covered kinome. 
The kinase fingerprint is based on the [KLIFS](klifs.net/) pocket alignment, 
which defines 85 pocket residues for all kinase structures. 
This enables a residue-by-residue comparison without a computationally expensive alignment step. 

The pocket fingerprint consists of 85 concatenated residue fingerprints, 
each encoding a residue’s spatial and physicochemical properties. 
The spatial properties describe the residue’s position in relation to the kinase pocket center and 
important kinase subpockets, i.e. the hinge region, the DFG region, and the front pocket. 
The physicochemical properties encompass for each residue its size and pharmacophoric features, solvent exposure and side chain orientation.

Take a look at the [repository `kissim_app`](https://github.com/volkamerlab/kissim_app) for pairwise comparison of all kinases to study kinome-wide similarities.

## Documentation

The `kissim` package documentation is available [here](https://kissim.readthedocs.io/), including [installation instructions](https://kissim.readthedocs.io/en/latest/installing.html).

## Contact

Please [open an issue](https://github.com/volkamerlab/kissim/issues) if you have questions or suggestions.

We are looking forward to hearing from you!

## License

This work is published under the [MIT license](https://github.com/volkamerlab/kissim/blob/master/LICENSE).

Copyright (c) 2019, Volkamer Lab

## Citation
Have you used `kissim` in your research or found the tool useful? We'd be very grateful if you cited it using the following:

```
@article{sydow_2022_jcim,
  author = {Sydow, Dominique and Aßmann, Eva and Kooistra, Albert J. and Rippmann, Friedrich and Volkamer, Andrea},
  title = {KiSSim: Predicting Off-Targets from Structural Similarities in the Kinome},
  journal = {Journal of Chemical Information and Modeling},
  volume = {62},
  number = {10},
  pages = {2600-2616},
  year = {2022},
  doi = {10.1021/acs.jcim.2c00050}
```


## Acknowledgements

### Funding

Volkamer Lab's projects are supported by several public funding sources
(for more info see our [webpage](https://volkamerlab.org/)).

### Collaborators

The `kissim` project is a collaboration between the Volkamer Lab (Dominique Sydow, Eva Aßmann and Andrea Volkamer), Albert Kooistra (University of Copenhagen) and Friedrich Rippmann (Merck).

### External resources

#### Databases

- [KLIFS](https://klifs.net/)

#### Python packages

- Cheminformatics and structural bioinformatics:
  [`opencadd`](https://opencadd.readthedocs.io/en/latest/),
  [`biopython`](https://biopython.org/),
  [`biopandas`](http://rasbt.github.io/biopandas/)
- Data science (PyData stack):
  [`numpy`](https://numpy.org/),
  [`pandas`](https://pandas.pydata.org/),
  [`scipy`](https://scipy.org/),
  [`jupyter`](https://jupyter.org/),
  [`ipywidgets`](https://ipywidgets.readthedocs.io)
- Data visualization:
  [`matplotlib`](https://matplotlib.org/), 
  [`seaborn`](https://seaborn.pydata.org/),
  [`nglview`](http://nglviewer.org/nglview/latest/)
- Continuous integration:
  [`pytest`](https://docs.pytest.org),
  [`nbval`](https://nbval.readthedocs.io)
- Documentation:
  [`sphinx`](https://www.sphinx-doc.org),
  [`nbsphinx`](https://nbsphinx.readthedocs.io)
- Code style:
  [`black-nb`](https://github.com/tomcatling/black-nb)

#### Repository

Project is based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.5.
