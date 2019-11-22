"""
auxiliary.py

Subpocket-based structural fingerprint for kinase pocket comparison.

Handles the helper functions.
"""

import logging
from pathlib import Path

from biopandas.pdb import PandasPdb
from biopandas.mol2 import PandasMol2, split_multimol2
from Bio.PDB import MMCIFParser, Selection, Entity
import pandas as pd

logger = logging.getLogger(__name__)

AMINO_ACIDS = pd.DataFrame(
    [
        'ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET MSE PHE PRO SER THR TRP TYR VAL _'.split(),
        'A R N D C Q E G H I L K M X F P S T W Y V _'.split()
    ],
    index='aa_three aa_one'.split()).transpose()


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
            pass
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

            except (AttributeError, ValueError):  # If 9 columns did not work, try 10 columns.
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

            for subst_name, atom_type in zip(molecule.df['subst_name'], molecule.df['atom_type']):

                # Some subst_name entries in the KLIFs mol2 files contain underscores.
                # Examples
                # - 5YKS: Residues on the N-terminus of (before) the first amino acid, i.e. 3C protease cutting site
                # - 2J5E: Mutated residue (CYO_797)
                # Convert these underscores into a minus sign (so that it can still be cast to int)
                subst_name = subst_name.replace('_', '-')

                # Some subst_name entries in the KLIFS mol2 files (in accordance with the original PDB files)
                # contain a letter after the residue ID
                # Examples
                # - 3HLL: Irregular residue ID (56A, 93B)
                # Remove letter from end of string
                try:
                    int(subst_name[-1])
                except ValueError:
                    subst_name = subst_name[:-1]

                # These are elements such as CA or MG
                if subst_name[:2] == atom_type.upper():
                    res_id_list.append(int(subst_name[2:]))
                    res_name_list.append(subst_name[:2])

                # These are amino acid, linkers, compounds, ...
                else:
                    res_id_list.append(int(subst_name[3:]))
                    res_name_list.append(subst_name[:3])

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


class KlifsMoleculeLoader:
    """
    Load molecule depending on input type (mol2 file path or KLIFS metadata entry).

    Attributes
    ----------
    klifs_metadata_path : pathlib.Path
        KLIFS metadata file path.
    molecule : biopandas.mol2.pandas_mol2.PandasMol2
        BioPandas objects containing metadata and structural data of molecule(s) in mol2 file
        and KLIFS position ID retrieved from KLIFS metadata.

    Parameters
    ----------
    mol2_path : str or pathlib.Path
        Mol2 file path.
    klifs_metadata_entry : pandas.Series
        KLIFS metadata describing a pocket entry in the KLIFS dataset.
    """

    def __init__(self, path_klifs_download):

        self.path_klifs_download = path_klifs_download
        self.klifs_metadata_path = path_klifs_download / 'klifs_metadata_unfiltered.csv'
        self.molecule = None

    def from_file(self, mol2_path):
        """
        Get molecule including KLIFS position IDs from a mol2 file path.

        This molecule has the form of a biopandas object, containing (i) the molecule code and
        (i) the molecule data, i.e. pandas.DataFrame: atoms (rows) x properties (columns), including
        KLIFS position IDs from the KLIFS metadata as additional property (column).

        Parameters
        ----------
        mol2_path : pathlib.Path
            Mol2 file path.

        Returns
        -------
        biopandas.mol2.pandas_mol2.PandasMol2
            BioPandas objects containing metadata and structural data of molecule(s) in mol2 file
            and KLIFS position ID retrieved from KLIFS metadata.

        """

        # Cast path to pathlib.Path and check if it exists
        mol2_path = Path(mol2_path)
        if not mol2_path.exists():
            raise FileNotFoundError(f'File does not exist: {mol2_path}')

        # Get molecule's KLIFS metadata entry from mol2 file
        klifs_metadata_entry = self.metadata_entry_from_file(mol2_path)

        # Get molecule
        molecule = self.load_molecule(klifs_metadata_entry, mol2_path)

        self.molecule = molecule

    def from_metadata_entry(self, klifs_metadata_entry):
        """
        Get molecule including KLIFS position IDs from a KLIFS metadata entry.

        This molecule has the form of a biopandas object, containing (i) the molecule code and
        (i) the molecule data, i.e. pandas.DataFrame: atoms (rows) x properties (columns), including
        KLIFS position IDs from the KLIFS metadata as additional property (column).

        Parameters
        ----------
        klifs_metadata_entry : pandas.Series
            KLIFS metadata describing a pocket entry in the KLIFS dataset.

        Returns
        -------
        biopandas.mol2.pandas_mol2.PandasMol2
            BioPandas objects containing metadata and structural data of molecule(s) in mol2 file
            and KLIFS position ID retrieved from KLIFS metadata.
        """

        # Get molecule's mol2 file path from KLIFS metadata entry
        mol2_path = self.file_from_metadata_entry(klifs_metadata_entry)

        # Cast path to pathlib.Path and check if it exists
        mol2_path = Path(mol2_path)
        if not mol2_path.exists():
            raise FileNotFoundError(f'File does not exist: {mol2_path}')

        # Get molecule
        molecule = self.load_molecule(klifs_metadata_entry, mol2_path)

        self.molecule = molecule

    @staticmethod
    def load_molecule(klifs_metadata_entry, mol2_path):
        """
        Load molecule from mol2 file in the form of a biopandas object, containing (i) the molecule code and
        (i) the molecule data, i.e. pandas.DataFrame: atoms (rows) x properties (columns).
        Add KLIFS position IDs from the KLIFS metadata as additional property (column).

        Parameters
        ----------
        klifs_metadata_entry : pandas.Series
            KLIFS metadata describing a pocket entry in the KLIFS dataset.
        mol2_path : pathlib.Path
            Mol2 file path.

        Returns
        -------
        biopandas.mol2.pandas_mol2.PandasMol2
            BioPandas objects containing metadata and structural data of molecule(s) in mol2 file
            and KLIFS position ID retrieved from KLIFS metadata.
        """

        # Load molecule from mol2 file
        molecule_loader = MoleculeLoader(mol2_path)
        molecule = molecule_loader.molecules[0]

        # List of KLIFS positions (starting at 1) excluding gap positions
        klifs_ids = [index for index, residue in enumerate(klifs_metadata_entry.pocket, 1) if residue != '_']

        # Number of atoms per residue in molecule (mol2file)
        # Note: sort=False important otherwise negative residue IDs will be sorted to the top
        number_of_atoms_per_residue = molecule.df.groupby(by='res_id', sort=False).size()

        # Get KLIFS position IDs for each atom in molecule
        klifs_ids_per_atom = []

        for klifs_id, n in zip(klifs_ids, number_of_atoms_per_residue):
            klifs_ids_per_atom = klifs_ids_per_atom + [klifs_id] * n

        # Add column for KLIFS position IDs to molecule
        molecule.df['klifs_id'] = klifs_ids_per_atom

        return molecule

    def metadata_entry_from_file(self, mol2_path):
        """
        Get the KLIFS metadata entry linked to a mol2 file path.

        Parameters
        ----------
        mol2_path : pathlib.Path
            Mol2 file path.

        Returns
        -------
        pandas.Series
            KLIFS metadata describing a pocket entry in the KLIFS dataset.
        """

        # Load KLIFS metadata
        klifs_metadata = pd.read_csv(self.klifs_metadata_path)

        # Get metadata from mol2 file path: kinase, PDB ID, alternate model and chain:
        mol2_path = Path(mol2_path)

        # Get kinase
        kinase = list(mol2_path.parents)[1].stem  # e.g. 'AAK1'

        # Get structure ID
        structure_id = list(mol2_path.parents)[0].stem.split('_')  # e.g. ['4wsq', 'altA', 'chainA']

        # Get PDB ID
        pdb_id = structure_id[0]  # e.g. '4wsq'

        # Get alternate model
        alternate_model = [i[-1] for i in structure_id if 'alt' in i]

        if alternate_model:
            alternate_model = alternate_model[0]  # e.g. ['A']
        else:
            alternate_model = '-'  # ['-']

        # Get chain
        chain = [i[-1] for i in structure_id if 'chain' in i]

        if chain:
            chain = chain[0]  # e.g. ['A']
        else:
            chain = '-'  # ['-']

        klifs_metadata_entry = klifs_metadata[
            (klifs_metadata.kinase == kinase) &
            (klifs_metadata.pdb_id == pdb_id) &
            (klifs_metadata.alternate_model == alternate_model) &
            (klifs_metadata.chain == chain)
        ]

        if len(klifs_metadata_entry) != 1:
            raise ValueError(f'Unvalid number of entries ({len(klifs_metadata_entry)}) in metadata for file: {mol2_path}')

        # Squeeze casts one row DataFrame to Series
        klifs_metadata_entry = klifs_metadata_entry.squeeze()

        return klifs_metadata_entry

    def file_from_metadata_entry(self, klifs_metadata_entry):
        """
        Get the mol2 file path linked to an entry in the KLIFS metadata.

        Parameters
        ----------
        klifs_metadata_entry : pandas.Series
            KLIFS metadata describing a pocket entry in the KLIFS dataset.

        Returns
        -------
        pathlib.Path
            Mol2 file path.
        """

        # Depending on whether alternate model and chain ID is given build file path:
        mol2_path = self.path_klifs_download / klifs_metadata_entry.filepath / 'pocket.mol2'

        # If file does not exist, raise error
        if not mol2_path.exists():
            raise FileNotFoundError(f'File not found: {mol2_path}')

        return mol2_path


class PdbChainLoader:
    """
    Load chain from PDB file as Biopython object.

    Attributes
    ----------
    chain : Bio.PDB.Chain.Chain
        Chain from PDB file.

    Parameters
    ----------
    klifs_metadata_entry : pandas.Series
        KLIFS metadata describing a pocket entry in the KLIFS dataset.
    """

    def __init__(self, klifs_metadata_entry=None, pdb_path=None, chain_id=None):

        if (pdb_path is not None) and (chain_id is not None):
            self.chain = self.from_file(pdb_path, chain_id)
        elif klifs_metadata_entry is not None:
            self.chain = self.from_metadata_entry(klifs_metadata_entry)
        else:
            self.chain = None

    def from_metadata_entry(self, klifs_metadata_entry):

        pdb_id = klifs_metadata_entry.pdb_id
        chain_id = klifs_metadata_entry.chain

        pdb_path = PATH_TO_DATA / 'raw' / 'PDB_download' / f'{pdb_id}.cif'

        chain = self.from_file(pdb_path, chain_id)

        return chain

    @staticmethod
    def from_file(pdb_path, chain_id):

        pdb_path = Path(pdb_path)

        if not pdb_path.exists():
            raise IOError(f'PDB file does not exist: {pdb_path}')

        # Get structure
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(
            structure_id=pdb_path.stem,
            filename=pdb_path
        )

        # Get alternate model (always use first model)
        model = structure[0]

        # Get pdb chain as denoted in metadata
        chain = model[chain_id]

        return chain


def get_amino_acids_1to3(one_letter_amino_acid):
    """
    Get three letter code for a one letter code amino acid.

    Parameters
    ----------
    one_letter_amino_acid : str
        One letter code for an amino acid.

    Returns
    -------
    str
        Three letter code for an amino acid.
    """
    return AMINO_ACIDS[AMINO_ACIDS.aa_one == one_letter_amino_acid].aa_three.iloc[0]


def get_amino_acids_3to1(three_letter_amino_acid):
    """
    Get one letter code for a three letter code amino acid.

    Parameters
    ----------
    three_letter_amino_acid : str
        Three letter code for an amino acid.

    Returns
    -------
    str
        One letter code for an amino acid.
    """
    return AMINO_ACIDS[AMINO_ACIDS.aa_three == three_letter_amino_acid].aa_one.iloc[0]


def split_klifs_code(klifs_code):
    """
    Split KLIFS molecule code into its components, i.e. species name, kinase group name, PDB ID, alternate model ID,
    and chain ID.

    Parameters
    ----------
    klifs_code: str
        KLIFS molecule code with the following pattern: species_kinasegroup_pdbid_(altX)_(chainA),
        brackets indicate optional positions.

    Returns
    -------
    list of str
        KLIFS molecule code components: species name, kinase name, PDB ID, alternate model ID, and chain ID.
    """

    code = klifs_code.replace('/', '_').split('_')

    # Get species name
    species = code[0]

    # Get kinase name
    kinase = code[1]

    # Get PDB ID
    pdb_id = code[2]

    # Get alternate model ID
    alternate_model = [i for i in code if 'alt' in i]
    if alternate_model:
        alternate_model = alternate_model[0][-1]  # Get 'X' from 'altX'
    else:
        alternate_model = None

    # Get chain ID
    chain = [i for i in code if 'chain' in i]
    if chain:
        chain = chain[0][-1]
    else:
        chain = None

    return {'species': species, 'kinase': kinase, 'pdb_id': pdb_id, 'alternate_model': alternate_model, 'chain': chain}


def get_klifs_residues_mol2topdb(molecule):
    """
    Retrieve the KLIFS residues from a PDB file using the KLIFS mol2 file as a template.

    Parameters
    ----------
    molecule : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
        Content of mol2 or pdb file as BioPandas object.

    Returns
    -------
    list of Bio.PDB.Residue.Residue
        Residues in biopython format.
    """

    # Get molecule code
    code = split_klifs_code(molecule.code)

    # Get residue IDs of binding site from mol2 file
    residue_ids = [int(i) for i in molecule.df.res_id.unique()]

    # Load PDB file and get residues
    pdb_path = f'/home/dominique/Documents/data/kinsim/20190724_full/raw/PDB_download/{code["pdb_id"]}.cif'

    if not Path(pdb_path).exists():
        raise IOError(f'PDB file does not exist: {pdb_path}')

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(
        structure_id=code['pdb_id'],
        filename=pdb_path
    )
    model = structure[0]
    chain = model[code['chain']]
    residues = Selection.unfold_entities(entity_list=chain, target_level='R')

    # Select KLIFS residues
    klifs_residues = [residue for residue in residues if residue.get_full_id()[3][1] in residue_ids]

    return klifs_residues


def center_of_mass(entity, geometric=False):
    """
    Calculates gravitic [default] or geometric center of mass of an Entity.
    Geometric assumes all masses are equal (geometric=True).

    Parameters
    ----------
    entity : Bio.PDB.Entity.Entity or list of Bio.PDB.Atom.Atom
        Contains atoms for which the center of mass / centroid needs to be calculated:
        a) Basic container object for PDB heirachy. Structure, Model, Chain and Residue are subclasses of Entity.
        b) List of container objects for atoms.

    geometric : bool
        Geometric assumes all masses are equal (geometric=True). Defaults to False.

    Returns
    -------
    list of floats
        Gravitic [default] or geometric center of mass of an Entity.

    References
    ----------
    Copyright (C) 2010, Joao Rodrigues (anaryin@gmail.com)
    This code is part of the Biopython distribution and governed by its license.
    Please see the LICENSE file that should have been included as part of this package.
    """

    # Structure, Model, Chain, Residue
    if isinstance(entity, Entity.Entity):
        atom_list = entity.get_atoms()
    # List of Atoms
    elif hasattr(entity, '__iter__') and [x for x in entity if x.level == 'A']:
        atom_list = entity
    # Some other weirdo object
    else:
        raise ValueError(f'Center of Mass can only be calculated from the following objects:\n'
                         f'Structure, Model, Chain, Residue, list of Atoms.')

    masses = []
    positions = [[], [], []]  # [ [X1, X2, ..] , [Y1, Y2, ...] , [Z1, Z2, ...] ]

    for atom in atom_list:
        masses.append(atom.mass)

        for i, coord in enumerate(atom.coord.tolist()):
            positions[i].append(coord)

    # If there is a single atom with undefined mass complain loudly.
    if 'ukn' in set(masses) and not geometric:
        raise ValueError(f'Some atoms don\'t have an element assigned.\n'
                         f'Try adding them manually or calculate the geometrical center of mass instead.')

    if geometric:
        return [sum(coord_list)/len(masses) for coord_list in positions]
    else:
        w_pos = [[], [], []]
        for atom_index, atom_mass in enumerate(masses):
            w_pos[0].append(positions[0][atom_index]*atom_mass)
            w_pos[1].append(positions[1][atom_index]*atom_mass)
            w_pos[2].append(positions[2][atom_index]*atom_mass)

        return [sum(coord_list)/sum(masses) for coord_list in w_pos]


def get_aminoacids_by_molecularweight(path_to_results):
    """
    Get string of sorted standard amino acid names in the form of 'GLY ALA SER ...'.

    Returns
    -------
    str
        Sorted standard amino acid names (upper case three latter code), separated by space.

    References
    ----------
    Molecular weight is taken from:
    https://www.sigmaaldrich.com/life-science/metabolomics/learning-center/amino-acid-reference-chart.html
    """

    molecular_weight = pd.read_csv(path_to_results / 'amino_acids_molecular_weight.csv')
    molecular_weight.residue_name = molecular_weight.residue_name.str.upper()
    return ' '.join(list(molecular_weight.sort_values(['molecular_weight']).residue_name))


def get_klifs_regions():
    """

    Returns
    -------

    """

    klifs_regions_definitions = {
        'I': range(1, 3 + 1),  # β-sheet I
        'g.l': range(4, 9 + 1),  # G-rich loop
        'II': range(10, 13 + 1),  # β-sheet II
        'III': range(14, 19 + 1),  # β-sheet III
        'aC': range(20, 30 + 1),
        'b.l': range(31, 37 + 1),  # loop connecting aC-helix to IV
        'IV': range(38, 41 + 1),  # β-sheet IV
        'V': range(42, 44 + 1),  # β-sheet V
        'GK': range(45, 45 + 1),  # gatekeeper
        'hinge': range(46, 48 + 1),
        'linker': range(49, 52 + 1),  # loop connecting the hinge to aD-helix
        'aD': range(53, 59 + 1),
        'aE': range(60, 64 + 1),
        'VI': range(65, 67 + 1),  # β-sheet VI
        'c.I': range(68, 75 + 1),  # catalytic loop
        'VII': range(76, 78 + 1),  # β-sheet VII
        'VIII': range(79, 79 + 1),  # β-sheet VIII
        'x': range(80, 80 + 1),  # residue preceding DFG-motif
        'DFG': range(81, 83 + 1),  # DFG-motif
        'a.l': range(84, 85 + 1)  # activation loop
    }

    klifs_regions = []
    for key, value in klifs_regions_definitions.items():
        klifs_regions = klifs_regions + [[key, i] for i in value]
    klifs_regions = pd.DataFrame(klifs_regions, columns=['region_name', 'klifs_id'])

    klifs_regions.index = klifs_regions.klifs_id

    return klifs_regions
