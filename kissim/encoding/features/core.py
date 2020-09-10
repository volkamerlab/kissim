"""
kissim.encoding.features.core
"""

from opencadd.io import DataFrame
import pandas as pd

from ...io import BiopythonChain


class Structure:
    """
    TODO
    """

    def __init__(self):
        """
        TODO dataframe must be full protein.
        """

        self.dataframe = None
        self.chain = None
        self._hse_cb = None

    def from_file(self, filepath):
        """
        TODO
        """

        dataframe = DataFrame.from_file(filepath)
        chain = BiopythonChain.from_file(filepath)

        self.dataframe = dataframe
        self.chain = chain
        self.hse_cb = HSExposureCB(chain)

    @property
    def ca_atoms(self):
        """
        TODO
        """

        ca_atoms = []
        for residue_pdb_id, dataframe in self.dataframe.groupby("residue.pdb_id", sort=False):
            ca_atom = self.ca_atom(residue_pdb_id)
            ca_atoms.append(ca_atom)
        ca_atoms = pd.concat(ca_atoms)

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

        return self._hse_cb

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
