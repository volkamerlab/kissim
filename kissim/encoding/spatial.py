"""
kissim.encoding.spatial TODO
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ...auxiliary import KlifsMoleculeLoader
from ...definitions import ANCHOR_RESIDUES, HINGE_KLIFS_IDS, DFG_KLIFS_IDS

logger = logging.getLogger(__name__)


class SpatialFeatures:
    """
    Spatial features for each residue in the KLIFS-defined kinase binding site
    of 85 pre-aligned residues.

    Spatial properties:
    Distance of each residue to 4 reference points:
    - Binding site centroid
    - Hinge region
    - DFG region
    - Front pocket

    Attributes
    ----------
    reference_points : pandas.DataFrame
        Coordiantes (rows) for 4 reference points (columns).
    features : pandas.DataFrame
        4 features (columns) for 85 residues (rows).
    """

    def __init__(self):

        self.reference_points = None
        self.features = None

    def from_molecule(self, molecule):
        """
        Get spatial properties for each residue of a molecule.

        Parameters
        ----------
        molecule : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
            Content of mol2 or pdb file as BioPandas object.
        """

        # Get reference points
        self.reference_points = self.get_reference_points(molecule)

        # Get all residues' CA atoms in molecule (set KLIFS position as index)
        residues_ca = molecule.df[molecule.df.atom_name == "CA"]["klifs_id x y z".split()]
        residues_ca.set_index("klifs_id", drop=True, inplace=True)

        distances = {}

        for name, coord in self.reference_points.items():

            # If any reference points coordinate is None, set also distance to None

            if coord.isna().any():
                distances[f"distance_to_{name}"] = None
            else:
                distance = (residues_ca - coord).transpose().apply(np.linalg.norm)
                distance.rename(name, inplace=True)
                distances[f"distance_to_{name}"] = np.round(distance, 2)

        spatial_features = pd.DataFrame.from_dict(distances)

        # Bring all fingerprints to same dimensions
        # (i.e. add currently missing residues in DataFrame)
        empty_df = pd.DataFrame([], index=range(1, 86))
        spatial_features = pd.concat([empty_df, spatial_features], axis=1)

        # Set all None to nan
        spatial_features.fillna(value=pd.np.nan, inplace=True)

        self.features = spatial_features

    def get_reference_points(self, molecule):
        """
        Get reference points of a molecule, i.e. the binding site centroid, hinge region,
        DFG region and front pocket.

        Parameters
        ----------
        molecule : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
            Content of mol2 or pdb file as BioPandas object.

        Returns
        -------
        pandas.DataFrame
            Coordiantes (rows) for 4 reference points (columns).
        """

        reference_points = dict()

        # Calculate centroid-based reference point:
        # Calculate mean of all CA atoms
        reference_points["centroid"] = molecule.df[molecule.df.atom_name == "CA"][
            "x y z".split()
        ].mean()

        # Calculate anchor-based reference points:
        # Get anchor atoms for each anchor-based reference point
        anchors = self.get_anchor_atoms(molecule)

        for reference_point_name, anchor_atoms in anchors.items():

            # If any anchor atom None, set also reference point coordinates to None
            if anchor_atoms.isna().values.any():
                reference_points[reference_point_name] = [None, None, None]
            else:
                reference_points[reference_point_name] = anchor_atoms.mean()

        return pd.DataFrame.from_dict(reference_points)

    @staticmethod
    def get_anchor_atoms(molecule):
        """
        For each anchor-based reference points (i.e. hinge region, DFG region and front pocket)
        get the three anchor (i.e. CA) atoms.

        Parameters
        ----------
        molecule : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
            Content of mol2 or pdb file as BioPandas object.

        Returns
        -------
        dict of pandas.DataFrames
            Coordinates (x, y, z) of the three anchor atoms
            (rows=anchor residue KLIFS ID x columns=coordinates) for
            each of the anchor-based reference points.
        """

        anchors = dict()

        # Calculate anchor-based reference points
        # Process each reference point: Collect anchor residue atoms and calculate their mean
        for reference_point_name, anchor_klifs_ids in ANCHOR_RESIDUES.items():

            anchor_atoms = []

            # Process each anchor residue: Get anchor atom
            for anchor_klifs_id in anchor_klifs_ids:

                # Select anchor atom, i.e. CA atom of KLIFS ID (anchor residue)
                anchor_atom = molecule.df[
                    (molecule.df.klifs_id == anchor_klifs_id) & (molecule.df.atom_name == "CA")
                ]

                # If this anchor atom exists, append to anchor atoms list
                if len(anchor_atom) == 1:
                    anchor_atom.set_index("klifs_id", inplace=True)
                    anchor_atom.index.name = None
                    anchor_atoms.append(anchor_atom[["x", "y", "z"]])

                # If this anchor atom does not exist, do workarounds
                elif len(anchor_atom) == 0:

                    # Do residues (and there CA atoms) exist next to anchor residue?
                    atom_before = molecule.df[
                        (molecule.df.klifs_id == anchor_klifs_id - 1)
                        & (molecule.df.atom_name == "CA")
                    ]
                    atom_after = molecule.df[
                        (molecule.df.klifs_id == anchor_klifs_id + 1)
                        & (molecule.df.atom_name == "CA")
                    ]
                    atom_before.set_index("klifs_id", inplace=True, drop=False)
                    atom_after.set_index("klifs_id", inplace=True, drop=False)

                    # If both neighboring CA atoms exist, get their mean as alternative anchor atom
                    if len(atom_before) == 1 and len(atom_after) == 1:
                        anchor_atom_alternative = pd.concat([atom_before, atom_after])[
                            ["x", "y", "z"]
                        ].mean()
                        anchor_atom_alternative = pd.DataFrame(
                            {anchor_klifs_id: anchor_atom_alternative}
                        ).transpose()
                        anchor_atoms.append(anchor_atom_alternative)

                    elif len(atom_before) == 1 and len(atom_after) == 0:
                        atom_before.set_index("klifs_id", inplace=True)
                        anchor_atoms.append(atom_before[["x", "y", "z"]])

                    elif len(atom_after) == 1 and len(atom_before) == 0:
                        atom_after.set_index("klifs_id", inplace=True)
                        anchor_atoms.append(atom_after[["x", "y", "z"]])

                    else:
                        atom_missing = pd.DataFrame.from_dict(
                            {anchor_klifs_id: [None, None, None]},
                            orient="index",
                            columns="x y z".split(),
                        )
                        anchor_atoms.append(atom_missing)

                # If there are several anchor atoms, something's wrong...
                else:
                    raise ValueError(
                        f"Too many anchor atoms for"
                        f"{molecule.code}, {reference_point_name}, {anchor_klifs_id}: "
                        f"{len(anchor_atom)} (one atom allowed)."
                    )

            anchors[reference_point_name] = pd.concat(anchor_atoms)

        return anchors

    @staticmethod
    def save_cgo_refpoints(klifs_metadata_entry, path_klifs_download, output_path):
        """
        Save CGO PyMol file showing a kinase with anchor residues, reference points and highlighted
        hinge and DFG region.

        Parameters
        ----------
        klifs_metadata_entry : pandas.Series
            KLIFS metadata describing a pocket entry in the KLIFS dataset.
        path_klifs_download : pathlib.Path or str
            Path to directory of KLIFS dataset files.
        output_path : str or pathlib.Path
            Path to directory where data file should be saved.
        """

        output_path = Path(output_path)

        # PyMol sphere colors (for reference points)
        sphere_colors = {
            "centroid": [1.0, 0.65, 0.0],  # orange
            "hinge_region": [1.0, 0.0, 1.0],  # magenta
            "dfg_region": [0.25, 0.41, 0.88],  # skyblue
            "front_pocket": [0.0, 1.0, 0.0],  # green
        }

        # Load molecule from KLIFS metadata entry
        klifs_molecule_loader = KlifsMoleculeLoader()
        klifs_molecule_loader.from_metadata_entry(klifs_metadata_entry, path_klifs_download)
        molecule = klifs_molecule_loader.molecule

        # Path to molecule file
        path_mol2 = klifs_molecule_loader._file_from_metadata_entry(
            klifs_metadata_entry, path_klifs_download
        )

        # Output path
        Path(output_path).mkdir(parents=True, exist_ok=True)

        # Mol2 residue IDs for hinge/DFG region
        hinge_mol2_ids = molecule.df[molecule.df.klifs_id.isin(HINGE_KLIFS_IDS)].res_id.unique()
        dfg_mol2_ids = molecule.df[molecule.df.klifs_id.isin(DFG_KLIFS_IDS)].res_id.unique()

        # Get reference points and anchor atoms (coordinates)
        space = SpatialFeatures()
        space.from_molecule(molecule)
        ref_points = space.reference_points.transpose()
        anchor_atoms = space.get_anchor_atoms(molecule)

        # Drop missing reference points and anchor atoms
        ref_points.dropna(axis=0, how="any", inplace=True)
        for ref_point_name, anchor_atoms_per_ref_point in anchor_atoms.items():
            anchor_atoms_per_ref_point.dropna(axis=0, how="any", inplace=True)

        # Collect all text lines to be written to file
        lines = []

        # Set descriptive PyMol object name for reference points
        obj_name = f"refpoints_{molecule.code[6:]}"

        # Imports
        lines.append("from pymol import *")
        lines.append("import os")
        lines.append("from pymol.cgo import *\n")

        # Load pocket structure
        lines.append(f'cmd.load("{path_mol2}", "pocket_{molecule.code[6:]}")\n')
        lines.append(f'cmd.show("cartoon", "pocket_{molecule.code[6:]}")')
        lines.append(f'cmd.hide("lines", "pocket_{molecule.code[6:]}")')
        lines.append(f'cmd.color("gray", "pocket_{molecule.code[6:]}")\n')
        lines.append(f'cmd.set("cartoon_transparency", 0.5, "pocket_{molecule.code[6:]}")')
        lines.append(f'cmd.set("opaque_background", "off")\n')

        # Color hinge and DFG region
        lines.append(f'cmd.set_color("hinge_color", {sphere_colors["hinge_region"]})')
        lines.append(f'cmd.set_color("dfg_color", {sphere_colors["dfg_region"]})')
        lines.append(
            f'cmd.color("hinge_color", "pocket_{molecule.code[6:]} and '
            f'resi {"+".join([str(i) for i in hinge_mol2_ids])}")'
        )
        lines.append(
            f'cmd.color("dfg_color", "pocket_{molecule.code[6:]} and '
            f'resi {"+".join([str(i) for i in dfg_mol2_ids])}")\n'
        )

        # Add spheres, i.e. reference points and anchor atoms
        lines.append(
            f"obj_{obj_name} = [\n"
        )  # Variable cannot start with digit, thus add prefix obj_

        # Reference points
        for ref_point_name, ref_point in ref_points.iterrows():

            # Set and write sphere color to file
            lines.append(
                f"\tCOLOR, "
                f"{str(sphere_colors[ref_point_name][0])}, "
                f"{str(sphere_colors[ref_point_name][1])}, "
                f"{str(sphere_colors[ref_point_name][2])},"
            )

            # Write reference point coordinates and size to file
            lines.append(
                f"\tSPHERE, "
                f'{str(ref_point["x"])}, '
                f'{str(ref_point["y"])}, '
                f'{str(ref_point["z"])}, '
                f"{str(1)},"
            )

            # Write anchor atom coordinates and size to file
            if ref_point_name != "centroid":
                for anchor_atom_index, anchor_atom in anchor_atoms[ref_point_name].iterrows():
                    lines.append(
                        f"\tSPHERE, "
                        f'{str(anchor_atom["x"])}, '
                        f'{str(anchor_atom["y"])}, '
                        f'{str(anchor_atom["z"])}, '
                        f"{str(0.5)},"
                    )

        # Write command to file that will load the reference points as PyMol object
        lines.append(f"]\n")

        # Add KLIFS IDs to CA atoms as labels

        for res_id, klifs_id in zip(molecule.df.res_id.unique(), molecule.df.klifs_id.unique()):
            lines.append(
                f'cmd.label(selection="pocket_{molecule.code[6:]} and '
                f'name CA and resi {res_id}", expression="\'{klifs_id}\'")'
            )

        lines.append(f'\ncmd.load_cgo(obj_{obj_name}, "{obj_name}")')

        with open(output_path / f"refpoints_{molecule.code[6:]}.py", "w") as f:
            f.write("\n".join(lines))

        # In PyMol enter the following to save png
        # PyMOL > ray 900, 900
        # PyMOL > save refpoints.png
