.. kissim documentation master file, created by
   sphinx-quickstart on Thu Mar 15 13:55:56 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the ``kissim`` documentation
=======================================

Subpocket-based structural fingerprint for kinase pocket comparison

.. raw:: html

   <p align="center">
   <img src="_static/kissim_toc.png" alt="Subpocket-based structural fingerprint for kinase pockets" width="600"/>
   <br>
   <font size="1">
   Subpocket-based structural fingerprint for kinase pockets
   </font>
   </p>

The ``kissim`` package offers a novel fingerprinting strategy designed specifically for kinase pockets, 
allowing for similarity studies across the structurally covered kinome. 
The kinase fingerprint is based on the `KLIFS <klifs.net/>`_ pocket alignment, 
which defines 85 pocket residues for all kinase structures. 
This enables a residue-by-residue comparison without a computationally expensive alignment step. 

The pocket fingerprint consists of 85 concatenated residue fingerprints, 
each encoding a residue’s spatial and physicochemical properties. 
The spatial properties describe the residue’s position in relation to the kinase pocket center and 
important kinase subpockets, i.e. the hinge region, the DFG region, and the front pocket. 
The physicochemical properties encompass for each residue its size and pharmacophoric features, solvent exposure and side chain orientation.

Take a look at the ``kissim_app`` `repository <https://github.com/volkamerlab/kissim_app>`_ for pairwise comparison of all kinases to study kinome-wide similarities.

.. toctree::
   :maxdepth: 1
   :caption: User guide

   installing
   tutorials/api
   tutorials/cli

.. toctree::
   :maxdepth: 1
   :caption: Explore package

   tutorials/encoding
   tutorials/comparison
   tutorials/io

.. toctree::
   :maxdepth: 1
   :caption: Developers
   
   api




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
