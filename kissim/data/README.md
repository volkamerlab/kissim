# Package Data

This directory contains (i) additional data needed for the package and (ii) non-code related additional information (such as data files, molecular structures,  etc.).

## Including package data

Modify your package's `setup.py` file and the `setup()` command. Include the 
[`package_data`](http://setuptools.readthedocs.io/en/latest/setuptools.html#basic-use) keyword and point it at the 
correct files.

## Manifest

* `side_chain_orientation_mean_median.dat`: Mean and median values for side chain orientation angles per amino acid.
