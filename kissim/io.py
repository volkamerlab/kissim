"""
kissim.io

Defines input/output classes used in kissim (TODO may be moved to opencadd.io?)
"""

from pathlib import Path

from Bio.PDB import Chain, Residue, Atom
from Bio.PDB.PDBExceptions import PDBConstructionException
import numpy as np
from opencadd.io import DataFrame

from .schema import STANDARD_AMINO_ACIDS


class BiopythonChain:
    """
    Parse structural data into a BioPython object.

    TODO Include Structure and Model objects?
    """

    @classmethod
    def from_file(cls, filepath):
        """
        Load structures as DataFrame from file.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Path to structure file: mol2 files only.
        """

        filepath = Path(filepath)
        format = filepath.suffix[1:]

        if format == "mol2":
            chain = Mol2ToBiopythonChain.from_file(filepath)
        else:
            raise ValueError(f"The {format} format is not suppored or invalid.")

        return chain


class Mol2ToBiopythonChain:
    """
    Parse structural data into a BioPython object.

    TODO Include Structure and Model objects?
    """

    @classmethod
    def from_file(cls, mol2_file):
        """
        Get Biopython Chain object from file.

        Parameters
        ----------
        mol2_file : str or pathlib.Path
            Path to mol2 file.
        """

        dataframe = DataFrame.from_file(mol2_file)
        mol2_to_bpy = cls()
        chain = mol2_to_bpy.from_dataframe(dataframe)

        return chain

    def from_dataframe(self, dataframe):
        """
        Get Biopython Chain object from dataframe.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Structural chain data.

        Returns
        -------
        Bio.PDB.Chain.Chain
            Structural chain data as biopython chain.
        """

        # Format the input DataFrame (clean up residue PDB ID, add insertion code)
        dataframe = self._format_dataframe(dataframe)

        # Get chain
        chain = self._chain(dataframe)

        return chain

    @staticmethod
    def _format_dataframe(dataframe):
        """
        Format residue PDB ID and insertion code.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Structural chain data.

        Returns
        -------
        dataframe : pandas.DataFrame
            Structural chain data (with updated residue PDB ID and insertion code).
        """

        residue_pdb_ids = []
        insertion_codes = []

        for atom_ix, atom in dataframe.iterrows():

            residue_pdb_id = atom["residue.pdb_id"]

            # Set sequence identifier and insertion code
            try:
                residue_pdb_id = int(residue_pdb_id)
                insertion_code = " "
            except ValueError:
                # Some residue IDs contain underscores
                # - 5YKS: Residues on the N-terminus of the first amino acid
                #         (HIS_12 in KLIFS; HIS and -12 in PDB)
                # - 2J5E: Mutated residue
                #         (CYO_797 in KLIFS; CYO and 797 in PDB)
                # Convert these underscores into a minus sign
                residue_pdb_id = residue_pdb_id.replace("_", "-")

                # Some residue IDs contain insertion codes: 3HLL (56A, 93B)
                # Remove letter from end of string
                try:
                    residue_pdb_id = int(residue_pdb_id)
                    insertion_code = " "
                except ValueError:
                    insertion_code = residue_pdb_id[-1]
                    residue_pdb_id = int(residue_pdb_id[:-1])

            residue_pdb_ids.append(residue_pdb_id)
            insertion_codes.append(insertion_code)

        dataframe["residue.pdb_id"] = residue_pdb_ids
        dataframe["residue.insertion"] = insertion_codes

        # Now check if some of the negative PDB IDs should be positive
        # which is the case when the residue PDB ID before is larger (nonesense)!
        for atom_ix, atom in dataframe.iterrows():

            if atom_ix != 0:
                residue_pdb_id = dataframe.loc[atom_ix, "residue.pdb_id"]
                residue_pdb_id_before = dataframe.loc[atom_ix - 1, "residue.pdb_id"]
                if residue_pdb_id_before > residue_pdb_id:
                    dataframe.loc[atom_ix, "residue.pdb_id"] = abs(residue_pdb_id)

        return dataframe

    def _chain(self, dataframe):
        """
        Get Biopython Chain object from DataFrame.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Structural chain data.

        Returns
        -------
        Bio.PDB.Chain.Chain
            Structural chain data as biopython chain.
        """

        chain = Chain.Chain(id="")
        for (residue_pdb_id, residue_name, residue_insertion), residue_df in dataframe.groupby(
            ["residue.pdb_id", "residue.name", "residue.insertion"], sort=False
        ):
            residue = self._residue(residue_name, residue_pdb_id, residue_insertion)
            for atom_index, atom_df in residue_df.iterrows():
                atom_name = atom_df["atom.name"]
                x = atom_df["atom.x"]
                y = atom_df["atom.y"]
                z = atom_df["atom.z"]
                atom = self._atom(atom_name, x, y, z)
                try:
                    residue.add(atom)
                except PDBConstructionException as e:
                    print(f"Warning: Atom was skipped. PDBConstructionException: {e}")
            try:
                chain.add(residue)
            except PDBConstructionException as e:
                print(f"Warning: Residue was skipped. PDBConstructionException: {e}")

        return chain

    def _residue(self, residue_name, residue_pdb_id, residue_insertion):
        """
        Get Biopython Residue object.

        Parameters
        ----------
        residue_name : str
            Residue name (3-letter code).
        residue_pdb_id : int
            Residue PDB ID.
        residue_insertion : str
            Residue insertion code.

        Returns
        -------
        Bio.PDB.Residue.Residue
            Structural residue data as Biopython residue.
        """

        # Set hetero atom flag
        # Check https://github.com/biopython/biopython/blob/master/Bio/PDB/PDBParser.py
        if residue_name in STANDARD_AMINO_ACIDS:
            hetero_flag = " "
        elif residue_name == "HOH" or residue_name == "WAT":
            hetero_flag = "W"
        else:
            hetero_flag = f"H_{residue_name}"

        residue = Residue.Residue(
            id=(hetero_flag, residue_pdb_id, residue_insertion), resname=residue_name, segid=""
        )

        return residue

    def _atom(self, name, x, y, z):
        """
        Get Biopython Atom object.

        Parameters
        ----------
        name : str
            Atom name.
        x: float
            Atom x coordinate.
        y: float
            Atom y coordinate.
        z: float
            Atom z coordinate.

        Returns
        -------
        Bio.PDB.Atom.Atom
            Structural atom data as Biopython atom.
        """

        atom = Atom.Atom(
            name=name,
            coord=np.array([x, y, z]),
            bfactor=0,  # Dummy value
            occupancy=0,  # Dummy value
            altloc="",  # Dummy value
            fullname="",  # Dummy value
            serial_number=0,  # Dummy value
            element="C",  # Dummy value TODO
        )

        return atom
