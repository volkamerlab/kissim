Installing
==========


.. note::

    We are assuming you have a working ``mamba`` installation in your computer. 
    If this is not the case, please refer to their `official documentation <https://mamba.readthedocs.io/en/latest/installation.html#mamba>`_. 

    If you installed ``mamba`` into an existing ``conda`` installation, also make sure that the ``conda-forge`` channel is configured by running ``conda config --add channels conda-forge``.


Install from the conda package
------------------------------

1. Create a new conda environment called ``kissim`` with the ``kissim`` package and all its dependencies installed::

    mamba create -n kissim kissim

2. Activate the new conda environment::

    conda activate kissim

3. Test that your installation works::

    kissim -h


Install from the latest development snapshot
--------------------------------------------

Install the latest development snapshot from the `GitHub repository's main branch <https://github.com/volkamerlab/kissim>`_.


1. Create a new conda environment called ``kissim``::

    mamba env create -f https://raw.githubusercontent.com/volkamerlab/kissim/main/devtools/conda-envs/test_env.yaml -n kissim

2. Activate the new conda environment::

    conda activate kissim

3. Install ``kissim`` package via pip::

    pip install https://github.com/volkamerlab/kissim/archive/main.tar.gz

4. Test that your installation works::

    kissim -h

Citation
--------
You can use the refence below to cite ``kissim``:

.. code-block::

    @article{sydow_2022_jcim,
        author = {Sydow, Dominique and Aßmann, Eva and Kooistra, Albert J. and Rippmann, Friedrich and Volkamer, Andrea},
        title = {KiSSim: Predicting Off-Targets from Structural Similarities in the Kinome},
        journal = {Journal of Chemical Information and Modeling},
        volume = {62},
        number = {10},
        pages = {2600-2616},
        year = {2022},
        doi = {10.1021/acs.jcim.2c00050}
    }