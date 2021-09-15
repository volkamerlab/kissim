Installing
==========


.. note::

    We are assuming you have a working ``mamba`` installation in your computer. 
    If this is not the case, please refer to their `official documentation <https://mamba.readthedocs.io/en/latest/installation.html#mamba>`_. 


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
