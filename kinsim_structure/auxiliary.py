"""
auxiliary.py

Subpocket-based structural fingerprint for kinase pocket comparison.

Handles the helper functions.
"""

import logging
from pathlib import Path
import re

from biopandas.mol2 import PandasMol2
from Bio.PDB import MMCIFParser, Selection, Vector, Entity, calc_angle
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)


def read_mol2(path):
    """

    Parameters
    ----------
    path

    Returns
    -------

    """

    path = str(path)

    molecule = PandasMol2().read_mol2(path=path,
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
    return molecule


def format_klifs_code(klifs_code):
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
        KLIFS molecule code components: species name, kinase group name, PDB ID, alternate model ID, and chain ID.
    """

    code = klifs_code.replace('/', '_').split('_')

    # Get species name
    species = code[0]

    # Get kinase group name
    group = code[1]

    # Get PDB ID
    pdb_id = code[2]

    # Get alternate model ID
    alternate_id = [i for i in code if 'alt' in i]
    if alternate_id:
        alternate_id = alternate_id[0][-1]  # Get 'X' from 'altX'
    else:
        alternate_id = None

    # Get chain ID
    chain_id = [i for i in code if 'chain' in i]
    if chain_id:
        chain_id = chain_id[0][-1]
    else:
        chain_id = None

    return {'species': species, 'group': group, 'pdb_id': pdb_id, 'alternate_id': alternate_id, 'chain_id': chain_id}


def get_klifs_residues_from_pdb(molecule):
    """

    Parameters
    ----------
    molecule


    Returns
    -------

    """

    # Get molecule code
    code = format_klifs_code(molecule.code)

    # Get residue IDs of binding site from mol2 file
    subst_names = [subst_name for subst_name in molecule.df.subst_name]

    residue_ids = []

    for subst_name in subst_names:
        if re.match(r'^[A-Z]{3}', subst_name) and not '_' in subst_name:  # Check if ID is amino acid and not ligand
            residue_ids.append(int(subst_name[3:]))
        else:
            raise ValueError(f'{molecule.code}: {subst_name}')

    # Load PDB file and get residues
    pdb_path = f'/home/dominique/Documents/data/kinsim/20190724_full/raw/PDB_download/{code["pdb_id"]}.cif'
    if Path(pdb_path).exists():
        parser = MMCIFParser()
        structure = parser.get_structure(structure_id=code['pdb_id'],
                                         filename=pdb_path)
    model = structure[0]
    chain = model[code['chain_id']]
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
    entity : Bio.PDB.Entity.Entity
        Basic container object for PDB heirachy. Structure, Model, Chain and Residue are subclasses of Entity.

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


def save_cgo_side_chain_orientation(klifs_path, output_path):
    """

    Parameters
    ----------
    klifs_path
    output_path

    Returns
    -------

    """
    # Get molecule and molecule code
    molecule = read_mol2(klifs_path)
    code = format_klifs_code(molecule.code)

    # Get KLIFS residues
    klifs_residues = get_klifs_residues_from_pdb(molecule)
    klifs_residues_ids = [str(residue.get_full_id()[3][1]) for residue in klifs_residues]

    # List contains lines for python script
    lines = [
        f'from pymol import *',
        f'import os',
        f'from pymol.cgo import *\n'
    ]

    # Fetch PDB, remove solvent, remove unnecessary chain(s) and residues
    lines.append(f'cmd.fetch("{code["pdb_id"]}")')
    lines.append(f'cmd.remove("solvent")')
    if code["chain_id"]:
        lines.append(f'cmd.remove("{code["pdb_id"]} and not chain {code["chain_id"]}")')
    lines.append(f'cmd.remove("all and not (resi {"+".join(klifs_residues_ids)})")')
    lines.append(f'')

    # Set sphere color and size
    sphere_colors = sns.color_palette('hls', 3)
    sphere_size = str(0.2)

    # Collect all PyMol objects here (in order to group them after loading them to PyMol)
    obj_names = []
    obj_angle_names = []

    for residue in klifs_residues:

        # Get residue ID
        residue_id = residue.get_full_id()[3][1]

        # Set PyMol object name: residue ID
        obj_name = f'{residue_id}'
        obj_names.append(obj_name)

        atom_names = [atoms.fullname for atoms in residue.get_atoms()]

        # Set CA atom
        if 'CA' in atom_names:
            vector_ca = residue['CA'].get_vector()
        else:
            vector_ca = None

        # Set CB atom
        if 'CB' in atom_names:
            vector_cb = residue['CB'].get_vector()
        else:
            vector_cb = None

        # Set centroid
        vector_com = Vector(center_of_mass(residue, geometric=True))

        if 'CA' in atom_names and 'CB' in atom_names:
            angle = np.degrees(calc_angle(vector_ca, vector_cb, vector_com))

            # Add angle to CB atom in the form of a label
            obj_angle_name = f'angle_{residue_id}'
            obj_angle_names.append(obj_angle_name)

            lines.append(
                f'cmd.pseudoatom(object="angle_{residue_id}", '
                f'pos=[{str(vector_cb[0])}, {str(vector_cb[1])}, {str(vector_cb[2])}], '
                f'label={str(round(angle, 1))})'
            )

        vectors = [vector_ca, vector_cb, vector_com]

        # Write all spheres for current residue in cgo format
        lines.append(f'obj_{obj_name} = [')  # Variable cannot start with digit, thus add prefix obj_

        # For each reference point, write sphere color, coordinates and size to file
        for index, vector in enumerate(vectors):

            if vector:
                # Set sphere color
                sphere_color = list(sphere_colors[index])

                # Write sphere a) color and b) coordinates and size to file
                lines.extend(
                    [
                        f'\tCOLOR, {str(sphere_color[0])}, {str(sphere_color[1])}, {str(sphere_color[2])},',
                        f'\tSPHERE, {str(vector[0])}, {str(vector[1])}, {str(vector[2])}, {sphere_size},'
                    ]
                )

        # Load the spheres as PyMol object
        lines.extend(
            [
                f']',
                f'cmd.load_cgo(obj_{obj_name}, "{obj_name}")',
                ''
            ]

        )
    # Group all objects to one group
    lines.append(f'cmd.group("{"_".join(code.values())}", "{" ".join(obj_names + obj_angle_names)}")')

    cgo_path = Path(output_path) / f'side_chain_orientation_{molecule.code.split("/")[1]}.py'
    with open(cgo_path, 'w') as f:
        f.write('\n'.join(lines))
