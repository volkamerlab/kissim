"""
kissim.io

Defines input/output classes used in kissim (TODO may be moved to opencadd.io?)
"""

from Bio.PDB import Chain, Residue, Atom
from Bio.PDB.PDBExceptions import PDBConstructionException
import numpy as np

from .schema import STANDARD_AMINO_ACIDS


class DataFrameToBiopythonChain:
    """
    Parse a structural data in DataFrame into a BioPython object.
    """

    def from_file(self, filepath):
        """
        Get Biopython Chain object from file.
        """
        raise NotImplementedError("Not implemented yet!")

    def _from_dataframe(self, dataframe):
        """
        Get Biopython Chain object from DataFrame.

        TODO Include Structure and Model objects?
        """

        # Format the input DataFrame (clean up residue PDB ID, add insertion code)
        dataframe = self._format_dataframe(dataframe)

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
                atom = self._atom(
                    atom_name, x, y, z
                )
                try:
                    residue.add(atom)
                except PDBConstructionException as e:
                    print(f"Warning: Atom was skipped. PDBConstructionException: {e}")
            try:
                chain.add(residue)
            except PDBConstructionException as e:
                print(f"Warning: Residue was skipped. PDBConstructionException: {e}")

        return chain

    @staticmethod
    def _format_dataframe(dataframe):
        """
        Format residue PDB ID and insertion code.

        TODO
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

    def _residue(self, residue_name, residue_pdb_id, residue_insertion):
        """
        Get Biopython Residue object.
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
