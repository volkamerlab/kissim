"""
kissim.io.core

Defines a basic class for structural objects for this package.
"""

import pandas as pd
from Bio.PDB.HSExposure import HSExposureCB


class Base:
    """
    Class defining the base for structural objects for this package.

    Attributes
    ----------
    dataframe : pandas.DataFrame
        Structure as DataFrame.
    _biopython : Bio.PDB.Structure.Structure
        Structure as biopython Structure object.
    _hse : dict of int: list [int, int, float]
        Half-sphere exposure per residues as defined by biopython:
        [TODO, TODO, TODO]
    """

    def __init__(self, dataframe, biopython, hse):

        self.dataframe = dataframe
        self._biopython = biopython
        self._hse = hse

    @property
    def ca_atoms(self):
        """
        Get CA atoms.
        """

        ca_atoms = []
        for residue_pdb_id, dataframe in self.dataframe.groupby("residue.pdb_id", sort=False):
            ca_atom = self.ca_atom(residue_pdb_id)
            ca_atoms.append(ca_atom)
        ca_atoms = pd.concat(ca_atoms).reset_index(drop=True)

        return ca_atoms

    def ca_atom(self, residue_pdb_id):
        """
        Get the residue's CA atom.

        Parameters
        ----------
        residue : pandas.DataFrame
            Residue.

        Returns
        -------
        pandas.Series
            Ca atom.
        """

        ca_atom = self.dataframe[
            (self.dataframe["atom.name"] == "CA")
            & (self.dataframe["residue.pdb_id"] == residue_pdb_id)
        ]

        if len(ca_atom) == 1:
            return ca_atom
        else:
            return None

    @property
    def pcb_atoms(self, residue_pdb_id):
        """
        TODO
        """

        return self._hse

    def pcb_atom(self, residue, chain):
        """
        Get pseudo-CB atom coordinate for non-GLY residue.

        Parameters
        ----------
        residue : Bio.PDB.Residue.Residue
            Residue.
        chain : Bio.PDB.Chain.Chain
            Chain from PDB file.

        Returns
        -------
        Bio.PDB.vectors.Vector or None
            Pseudo-CB atom vector for residue centered at CA atom (= pseudo-CB atom coordinate).
        """

        if residue.get_resname() == "GLY":
            return self._get_pcb_from_gly(residue)

        else:

            # Get residue before and after input residue (if not available return None)
            try:
                # Non-standard residues will throw KeyError here but I am ok with not considering
                # those cases, since
                # hetero residues are not always enumerated correctly
                # (sometimes non-standard residues are named e.g. "_180" in PDB files)
                residue_before = chain[residue.id[1] - 1]
                residue_after = chain[residue.id[1] + 1]

            except KeyError:  # If residue before or after do not exist
                return None

            # Get pseudo-CB for non-GLY residue
            pcb = HSExposureCA(Chain(id="X"))._get_cb(residue_before, residue, residue_after)

            if pcb is None:  # If one or more of the three residues have no CA
                return None

            else:
                # Center pseudo-CB vector at CA atom to get pseudo-CB coordinate
                ca = residue["CA"].get_vector()
                ca_pcb = ca + pcb[0]
                return ca_pcb

    def pcb_atom_from_gly(self, residue_pdb_id):
        """
        Get pseudo-CB atom coordinate for GLY residue.

        Parameters
        ----------
        residue : pandas.DataFrame
            Residue.

        Returns
        -------
        pandas.Series
            pCb atom.
        """

        if self.name == "GLY":

            # Get pseudo-CB for GLY (vector centered at origin)
            pcb = HSExposureCB(chain).gly_pcb_atom(residue)

            if pcb is None:
                return None

            else:
                # Center pseudo-CB vector at CA atom to get pseudo-CB coordinate
                ca = residue["CA"].get_vector()
                ca_pcb = ca + pcb
                return ca_pcb

        else:
            raise ValueError(f"Residue must be GLY, but is {residue.get_resname()}.")
