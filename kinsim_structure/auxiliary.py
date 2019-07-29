"""
auxiliary.py

Subpocket-based structural fingerprint for kinase pocket comparison.

Handles the helper functions.
"""

import logging
from pathlib import Path

from biopandas.mol2 import PandasMol2, split_multimol2
from biopandas.pdb import PandasPdb
import pandas as pd

logger = logging.getLogger(__name__)


class MoleculeLoader:
    """
    Class used to load molecule data from mol2 and pdb files in the form of unified BioPandas objects.

    Attributes
    ----------
    molecule_path : str or pathlib.Path
        Absolute path to a mol2 (can contain multiple entries) or pdb file.
    remove_solvent : bool
        Set True to remove solvent molecules (default: False).
    molecules : list of biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
        List of molecule data in the form of BioPandas objects.
    n_molecules : int
        Number of molecules loaded.

    Examples
    --------
    >>> from kinsim_structure.auxiliary import MoleculeLoader
    >>> molecule_path = '/path/to/pdb/or/mol2'
    >>> molecule_loader = MoleculeLoader()
    >>> molecule_loader.load_molecule(molecule_path, remove_solvent=True)

    >>> molecules = molecule_loader.molecules  # Contains one or multiple molecule objects
    >>> molecule1 = molecules[0].df  # Molecule data
    >>> molecule1_id = molecules[0].code  # Molecule id

    >>> molecules[0].df == molecule_loader.molecules[0]
    True
    """

    def __init__(self, molecule_path, remove_solvent=False):

        self.molecule_path = Path(molecule_path)
        self.remove_solvent = remove_solvent
        self.molecules = self.load_molecule()
        self.n_molecules = len(self.molecules)

    def load_molecule(self):
        """
        Load one or multiple molecules from pdb or mol2 file.

        Returns
        -------
        list of biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
            List of BioPandas objects containing metadata and structural data of molecule(s) in mol2 file.
        """

        if self.molecule_path.exists():
            logger.info(f'File to be loaded: {self.molecule_path}', extra={'molecule_id': 'all'})
        else:
            logger.error(f'File not found: {self.molecule_path}', extra={'molecule_id': 'all'})
            raise FileNotFoundError(f'File not found: {self.molecule_path}')

        # Load molecule data
        if self.molecule_path.suffix == '.pdb':
            molecules = self._load_pdb(self.remove_solvent)
        elif self.molecule_path.suffix == '.mol2':
            molecules = self._load_mol2(self.remove_solvent)
        else:
            raise IOError(f'Unsupported file format {self.molecule_path.suffix}, only pdb and mol2 are supported.')

        logger.info('File loaded.', extra={'molecule_id': 'all'})

        return molecules

    def _load_mol2(self, remove_solvent=False):
        """
        Load molecule data from a mol2 file, which can contain multiple entries.

        Parameters
        ----------
        remove_solvent : bool
            Set True to remove solvent molecules (default: False).

        Returns
        -------
        list of biopandas.mol2.pandas_mol2.PandasMol2
            List of BioPandas objects containing metadata and structural data of molecule(s) in mol2 file.
        """

        # In case of multiple entries in one mol2 file, include iteration step
        molecules = []

        for mol2 in split_multimol2(str(self.molecule_path)):  # biopandas not compatible with pathlib

            # Mol2 files can have 9 or 10 columns.
            try:  # Try 9 columns.
                molecule = PandasMol2().read_mol2_from_list(
                                                            mol2_code=mol2[0],
                                                            mol2_lines=mol2[1],
                                                            columns={0: ('atom_id', int),
                                                                     1: ('atom_name', str),
                                                                     2: ('x', float),
                                                                     3: ('y', float),
                                                                     4: ('z', float),
                                                                     5: ('atom_type', str),
                                                                     6: ('subst_id', str),
                                                                     7: ('subst_name', str),
                                                                     8: ('charge', float)}
                                                            )

            except AssertionError:  # If 9 columns did not work, try 10 columns.
                molecule = PandasMol2().read_mol2_from_list(
                                                            mol2_code=mol2[0],
                                                            mol2_lines=mol2[1],
                                                            columns={0: ('atom_id', int),
                                                                     1: ('atom_name', str),
                                                                     2: ('x', float),
                                                                     3: ('y', float),
                                                                     4: ('z', float),
                                                                     5: ('atom_type', str),
                                                                     6: ('subst_id', str),
                                                                     7: ('subst_name', str),
                                                                     8: ('charge', float),
                                                                     9: ('status_bit', str)}
                                                            )

            # Insert additional columns (split ASN22 to ASN and 22)
            res_id_list = []
            res_name_list = []

            for i, j in zip(molecule.df['subst_name'], molecule.df['atom_type']):
                if i[:2] == j.upper():
                    # These are ions such as CA or MG
                    res_id_list.append(i[2:])
                    res_name_list.append(i[:2])
                else:
                    # These are amino acid, linkers, compounds, ...
                    res_id_list.append(i[3:])
                    res_name_list.append(i[:3])

            molecule.df.insert(loc=2, column='res_id', value=res_id_list)
            molecule.df.insert(loc=2, column='res_name', value=res_name_list)

            # Select columns of interest
            molecule._df = molecule.df.loc[:, ['atom_id',
                                               'atom_name',
                                               'res_id',
                                               'res_name',
                                               'subst_name',
                                               'x',
                                               'y',
                                               'z',
                                               'charge']]

            # Remove solvent if parameter remove_solvent=True
            if remove_solvent:
                ix = molecule.df.index[molecule.df['res_name'] == 'HOH']
                molecule.df.drop(index=ix, inplace=True)

            molecules.append(molecule)

        return molecules

    def _load_pdb(self, remove_solvent=False):
        """
        Load molecule data from a pdb file, which can contain multiple entries.

        Parameters
        ----------
        remove_solvent : bool
            Set True to remove solvent molecules (default: False).

        Returns
        -------
        list of biopandas.pdb.pandas_pdb.PandasPdb
            List of BioPandas objects containing metadata and structural data of molecule(s) in pdb file.
        """

        molecule = PandasPdb().read_pdb(str(self.molecule_path))  # biopandas not compatible with pathlib

        # If object has no code, set string from file stem and its folder name
        # E.g. "/mydir/pdb/3w32.mol2" will generate the code "pdb_3w32".
        if not (molecule.code or molecule.code.strip()):
            molecule.code = f'{self.molecule_path.parts[-2]}_{self.molecule_path.stem}'

        # Get both ATOM and HETATM lines of PDB file
        molecule._df = pd.concat([molecule.df['ATOM'], molecule.df['HETATM']])

        # Select columns of interest
        molecule._df = molecule.df.loc[:, ['atom_number',
                                           'atom_name',
                                           'residue_number',
                                           'residue_name',
                                           'x_coord',
                                           'y_coord',
                                           'z_coord',
                                           'charge']]

        # Insert additional columns
        molecule.df.insert(loc=4,
                           column='subst_name',
                           value=[f'{i}{j}' for i, j in zip(molecule.df['residue_name'], molecule.df['residue_number'])])

        # Rename columns
        molecule.df.rename(index=str, inplace=True, columns={'atom_number': 'atom_id',
                                                             'residue_number': 'res_id',
                                                             'residue_name': 'res_name',
                                                             'x_coord': 'x',
                                                             'y_coord': 'y',
                                                             'z_coord': 'z'})

        # Remove solvent if parameter remove_solvent=True
        if remove_solvent:
            ix = molecule.df.index[molecule.df['res_name'] == 'HOH']
            molecule.df.drop(index=ix, inplace=True)

        # Cast to list only for homogeneity with reading mol2 files that can have multiple entries
        molecules = [molecule]

        return molecules
