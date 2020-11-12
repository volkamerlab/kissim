"""
kissim.dataset.explore

Explores the KLIFS dataset with common plotting functionalities.
"""

import matplotlib.pyplot as plt
import pandas as pd
from opencadd.databases.klifs import setup_remote


def plot_important_categories(structures):
    """
    Plot important categories for the KLIFS dataset:
    - Species
    - DFG conformations
    - aC helix conformations

    Parameters
    ----------
    structures : pandas.DataFrame
        Structures DataFrame from opencadd.databases.klifs module.

    Returns
    -------
    numpy.array of matplotlib.pyplot.axis
        Plot axis array.
    """
    fig, axes = plt.subplots(1, 4, figsize=(15, 3), sharey=True)
    structures.groupby("species.klifs").size().plot(
        kind="bar", ax=axes[0], rot=0, title="Species", xlabel="", ylabel="Number of structures"
    )
    structures.groupby("structure.dfg").size().plot(
        kind="bar", ax=axes[1], rot=0, title="DFG conformations", xlabel=""
    )
    structures.groupby("structure.ac_helix").size().plot(
        kind="bar", ax=axes[2], rot=0, title="aC helix conformations", xlabel=""
    )
    structures_have_ligand = structures["ligand.expo_id"] != "-"
    structures_have_ligand.groupby(structures_have_ligand).size().plot(
        kind="bar", ax=axes[3], title="Structure has ligand?", xlabel=""
    )
    return axes


def plot_resolution_vs_qualityscore(structures):
    """
    Plot resolution versus the KLIFS quality score.

    Parameters
    ----------
    structures : pandas.DataFrame
        Structures DataFrame from opencadd.databases.klifs module.

    Returns
    -------
    matplotlib.pyplot.axis
        Plot axis.
    """
    ax = structures.plot(
        x="structure.resolution",
        y="structure.qualityscore",
        kind="scatter",
        figsize=(4, 4),
        title="Resolution vs. quality score",
        s=2,
        alpha=0.2,
    )
    ax.set_xlabel("Resolution in $\AA$")
    ax.set_ylabel("KLIFS quality score")
    return ax


def plot_number_of_structures_per_kinase_pdb_pair(structures):
    """
    Plot the number of structures that have x structures per kinase-PDB ID pair.

    Parameters
    ----------
    structures : pandas.DataFrame
        Structures DataFrame from opencadd.databases.klifs module.

    Returns
    -------
    matplotlib.pyplot.axis
        Plot axis.
    """
    kinase_pdb_pair_sizes = structures.groupby(["kinase.klifs_name", "structure.pdb_id"]).size()
    ax = kinase_pdb_pair_sizes.plot(
        kind="hist",
        title="Number of structures per kinase-PDB pair",
        bins=kinase_pdb_pair_sizes.max(),
    )
    ax.set_xlabel("Number of structures per kinase-PDB pair")
    ax.set_ylabel("Number of kinase-PDB pairs")
    return ax


def plot_number_of_structures_per_kinase(structures, top_n_kinases=30):
    """
    Plot the number of structures per kinase (for the top N kinases with the most structures).

    Parameters
    ----------
    structures : pandas.DataFrame
        Structures DataFrame from opencadd.databases.klifs module.
    top_n_kinases : int
        Plot only the top N kinases with the most structures.

    Returns
    -------
    matplotlib.pyplot.axis
        Plot axis.
    """
    plot_height = top_n_kinases / 5
    n_structures_per_kinase = structures.groupby("kinase.klifs_name").size()
    ax = (
        n_structures_per_kinase.sort_values(ascending=False)
        .head(top_n_kinases)
        .sort_values()
        .plot(
            kind="barh",
            figsize=(4, plot_height),
            title=f"Number of structures per kinase (top {top_n_kinases} kinases)",
            xlabel="KLIFS kinase name",
        )
    )
    return ax


def plot_number_of_kinases_per_kinase_group(structures, remote=None):
    """
    Plot the number of kinases per kinase group.

    Parameters
    ----------
    structures : pandas.DataFrame
        Structures DataFrame from opencadd.databases.klifs module.
    remote : None or opencadd.databases.klifs.session.Session
        Remote KLIFS session. If None, generate new remote session.

    Returns
    -------
    matplotlib.pyplot.axis
        Plot axis.
    """
    kinase_ids = structures["kinase.klifs_id"].to_list()
    # Get kinases by kinase KLIFS IDs
    if remote is None:
        remote = setup_remote()
    kinases = remote.kinases.by_kinase_klifs_id(kinase_ids)
    ax = (
        kinases.groupby("kinase.group")
        .size()
        .sort_values()
        .plot(
            kind="barh",
            figsize=(4, 3),
            title="Number of kinases per kinase group",
            xlabel="Kinase group",
        )
    )
    return ax


def plot_missing_residues(structures, remote=None, anchor_residues=None):
    """
    Plot number of missing residues for each binding site residue.

    Parameters
    ----------
    structures : pandas.DataFrame
        Structures DataFrame from opencadd.databases.klifs module.
    remote : None or opencadd.databases.klifs.session.Session
        Remote KLIFS session. If None, generate new remote session.
    anchor_residues : None or dict (str: list of int)
        Dictionary of anchor residues (values: list of residue KLIFS IDs) for one or more
        subpockets (keys: subpocket name). If not none, asterisks is placed over anchor residue
        bars.

    Returns
    -------
    matplotlib.pyplot.axis
        Plot axis.
    """
    # Get missing residues
    missing_residues = _get_missing_residues(structures)
    # Get KLIFS colors (remotely for example structure)
    klifs_colors = _get_klifs_residue_colors(remote)
    # Plot missing residues
    ax = missing_residues.plot(
        kind="bar",
        figsize=(20, 5),
        xlabel="KLIFS residue ID",
        ylabel="Number of structures",
        color=klifs_colors,
    )
    ax = _label_anchor_residues(ax, anchor_residues)

    return ax


def plot_missing_subpockets(structures, anchor_residues):
    """
    Plot number of missing subpockets.

    Parameters
    ----------
    structures : pandas.DataFrame
        Structures DataFrame from opencadd.databases.klifs module.
    anchor_residues : dict (str: list of int)
        Dictionary of anchor residues (values: list of residue KLIFS IDs) for one or more
        subpockets (keys: subpocket name).

    Returns
    -------
    matplotlib.pyplot.axis
        Plot axis.
    """
    # Get missing residues
    missing_residues = _get_missing_residues(structures)
    # Get missing subpockets
    missing_subpockets = pd.Series(
        {key: missing_residues[value].sum() for key, value in anchor_residues.items()}
    )
    # Plot missing subpockets
    ax = missing_subpockets.sort_values().plot(
        kind="barh", title="Number of structures with missing subpocket center"
    )
    return ax


def plot_modified_residues(structures, remote=None, anchor_residues=None):
    """
    Plot number of modified residues ("X") for each binding site residue.

    Parameters
    ----------
    structures : pandas.DataFrame
        Structures DataFrame from opencadd.databases.klifs module.
    remote : None or opencadd.databases.klifs.session.Session
        Remote KLIFS session. If None, generate new remote session.
    anchor_residues : None or dict (str: list of int)
        Dictionary of anchor residues (values: list of residue KLIFS IDs) for one or more
        subpockets (keys: subpocket name). If not none, asterisks is placed over anchor residue
        bars.

    Returns
    -------
    matplotlib.pyplot.axis
        Plot axis.
    """
    # Format pocket residues
    pockets = _get_pockets(structures)
    # Get KLIFS colors (remotely for example structure)
    klifs_colors = _get_klifs_residue_colors(remote)
    ax = (
        pockets.apply(lambda x: x == "X")
        .sum()
        .plot(kind="bar", figsize=(20, 5), xlabel="KLIFS residue ID", color=klifs_colors)
    )
    ax = _label_anchor_residues(ax, anchor_residues)
    return ax


def _get_pockets(structures):
    """
    Get pocket residue sequence for all structures.

    Parameters
    ----------
    structures : pandas.DataFrame
        Structures DataFrame from opencadd.databases.klifs module.

    Returns
    -------
    pandas.DataFrame
        Residue names as one-letter amino acid code for all residues 1-85 in the pocket (columns)
        for all input structures (rows).
    """
    pockets = pd.DataFrame(
        structures["structure.pocket"].apply(list).to_list(), columns=range(1, 86)
    )
    return pockets


def _get_missing_residues(structures):
    """
    Across all input structures, get number of missing residues per pocket residue position (1-85).

    Parameters
    ----------
    structures : pandas.DataFrame
        Structures DataFrame from opencadd.databases.klifs module.

    Returns
    -------
    pd.Series
        Number of missing residues (values) for all pocket residue positions 1-85 (index).
    """

    pockets = _get_pockets(structures)
    missing_residues = pockets.apply(lambda x: x == "_").sum()
    return missing_residues


def _get_klifs_residue_colors(remote=None):
    """
    Get KLIFS residue colors from example structure KLIFS ID (12347).

    Parameters
    ----------
    remote : None or opencadd.databases.klifs.session.Session
        Remote KLIFS session. If None, generate new remote session.

    Returns
    -------
    list of str
        KLIFS residue colors (matplotlib color names).
    """

    if remote is None:
        remote = setup_remote()
    klifs_colors = remote.pockets.by_structure_klifs_id(12347)["residue.klifs_color"]
    return klifs_colors.to_list()


def _label_anchor_residues(ax, anchor_residues):
    """
    Label bars in barplot that are listed as anchor residues.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        Plot axis for bar plot with 85 bars.
    anchor_residues : None or dict (str: list of int)
        Dictionary of anchor residues (values: list of residue KLIFS IDs) for one or more
        subpockets (keys: subpocket name). If not none, asterisks is placed over anchor residue
        bars.

    Returns
    -------
    matplotlib.pyplot.axis
        Plot axis.
    """
    if anchor_residues:
        anchor_residues_flat = [
            residue_id for residue_ids in anchor_residues.values() for residue_id in residue_ids
        ]
        for residue_id, patch in enumerate(ax.patches, 1):
            if residue_id in anchor_residues_flat:
                ax.annotate("*", (patch.get_x(), patch.get_height()), size=10)
    return ax
