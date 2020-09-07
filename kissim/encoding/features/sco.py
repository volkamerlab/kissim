"""
kissim.encoding.features.sco TODO
"""

import logging

from Bio.PDB import calc_angle, HSExposureCA, HSExposureCB, Vector
from Bio.PDB.Chain import Chain
import numpy as np
import pandas as pd

from ..definitions import N_HEAVY_ATOMS_CUTOFF
from kissim.auxiliary import center_of_mass

logger = logging.getLogger(__name__)


class SideChainOrientationFeature:
    """
    Side chain orientation for each residue in the KLIFS-defined kinase binding site
    of 85 pre-aligned residues.
    Side chain orientation of a residue is defined by the vertex angle formed by
    (i) the residue's CA atom,
    (ii) the residue's side chain centroid, and
    (iii) the pocket centroid (calculated based on its CA atoms), whereby the CA atom forms the
    vertex.

    Attributes
    ----------
    molecule_code : str
        KLIFS code.
    features : pandas.DataFrame
        1 feature, i.e. side chain orientation, (column) for 85 residues (rows).
    features_verbose : pandas.DataFrame
        Feature, Ca, Cb, and centroid vectors as well as metadata information (columns)
        for 85 residues (row).
    vector_pocket_centroid : Bio.PDB.Vector.Vector
        Vector to pocket centroid.
    """

    def __init__(self):

        self.molecule_code = None
        self.features = None
        self.features_verbose = None
        self.vector_pocket_centroid = (
            None  # Necessary to not calculate pocket centroid for each residue again
        )

    def from_molecule(self, molecule, chain):
        """
        Get side chain orientation for each residue in a molecule (pocket).
        Side chain orientation of a residue is defined by the vertex angle formed by
        (i) the residue's CA atom,
        (ii) the residue's side chain centroid, and
        (iii) the pocket centroid (calculated based on its CA atoms), whereby the CA atom forms the
        vertex.

        Parameters
        ----------
        molecule : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
            Content of mol2 or pdb file as BioPandas object.
        chain : Bio.PDB.Chain.Chain
            Chain from PDB file.
        """

        self.molecule_code = molecule.code

        # Get pocket residues
        pocket_residues = self._get_pocket_residues(molecule, chain)

        # Get vectors (for each residue CA atoms, side chain centroid, pocket centroid)
        pocket_vectors = self._get_pocket_vectors(pocket_residues, chain)

        # Get vertex angles (for each residue, vertex angle between aforementioned points)
        vertex_angles = self._get_vertex_angles(pocket_vectors)

        # Transform vertex angles into categories
        categories = self._get_categories(vertex_angles)

        # Store categories
        self.features = categories
        # Store categories, vertex angles plus vectors and metadata
        self.features_verbose = pd.concat([pocket_vectors, vertex_angles, categories], axis=1)

    @staticmethod
    def _get_pocket_residues(molecule, chain):
        """
        Get KLIFS pocket residues from PDB structural data: Bio.PDB.Residue.Residue plus metadata,
        i.e. KLIFS residue ID, PDB residue ID, and residue name for all pocket residues.

        Parameters
        ----------
        molecule : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
            Content of mol2 or pdb file as BioPandas object.
        chain : Bio.PDB.Chain.Chain
            Chain from PDB file.

        Returns
        -------
        pandas.DataFrame
            Pocket residues: Bio.PDB.Residue.Residue plus metadata, i.e. KLIFS residue ID,
            PDB residue ID, and residue name (columns) for all pocket residues (rows).
        """

        # Get KLIFS pocket metadata, e.g. PDB residue IDs from mol2 file (DataFrame)
        pocket_residues = pd.DataFrame(
            molecule.df.groupby("klifs_id res_id res_name".split()).groups.keys(),
            columns="klifs_id res_id res_name".split(),
        )
        pocket_residues.set_index("klifs_id", drop=False, inplace=True)

        # Select residues from chain based on PDB residue IDs and add to DataFrame
        pocket_residues_list = []

        for residue_id in pocket_residues.res_id:

            try:  # Standard amino acids
                pocket_residue = chain[residue_id]

            except KeyError:  # Non-standard amino acid
                pocket_residue = [i for i in chain.get_list() if i.get_id()[1] == residue_id][0]

            pocket_residues_list.append(pocket_residue)

        pocket_residues["pocket_residues"] = pocket_residues_list

        return pocket_residues

    def _get_pocket_vectors(self, pocket_residues, chain):
        """
        Get vectors to CA, residue side chain centroid, and pocket centroid.

        Parameters
        ----------
        pocket_residues : pandas.DataFrame
            Pocket residues: Bio.PDB.Residue.Residue plus metadata, i.e. KLIFS residue ID,
            PDB residue ID, and residue name (columns) for all pocket residues (rows).
        chain : Bio.PDB.Chain.Chain
            Chain from PDB file.

        Returns
        -------
        pandas.DataFrame
            Vectors to CA, residue side chain centroid, and pocket centroid for each residue of a
            molecule, alongside with metadata on KLIFS residue ID, PDB residue ID, and residue
            name.
        """

        # Save here values per residue
        data = []

        # Calculate pocket centroid
        if not self.vector_pocket_centroid:
            self.vector_pocket_centroid = self._get_pocket_centroid(pocket_residues)

        # Calculate CA atom and side chain centroid
        for residue in pocket_residues.pocket_residues:

            vector_ca = self._get_ca(residue)
            vector_side_chain_centroid = self._get_side_chain_centroid(residue, chain)

            data.append([vector_ca, vector_side_chain_centroid, self.vector_pocket_centroid])

        data = pd.DataFrame(
            data,
            index=pocket_residues.klifs_id,
            columns="ca side_chain_centroid pocket_centroid".split(),
        )

        metadata = pocket_residues["klifs_id res_id res_name".split()]

        if len(metadata) != len(data):
            raise ValueError(
                f"DataFrames to be concatenated must be of same length: "
                f"Metadata has {len(metadata)} rows, CA/CB/centroid data has {len(data)} rows."
            )

        return pd.concat([metadata, data], axis=1)

    @staticmethod
    def _get_vertex_angles(pocket_vectors):
        """
        Get vertex angles for residues' side chain orientations to the molecule (pocket) centroid.
        Side chain orientation of a residue is defined by the vertex_angle formed by
        (i) the residue's CB atom,
        (ii) the residue's side chain centroid, and
        (iii) the pocket centroid (calculated based on its CA atoms), whereby the CA atom forms the
        vertex.

        Parameters
        ----------
        pocket_vectors : pandas.DataFrame
            Vectors to CA, residue side chain centroid, and pocket centroid for each residue of a
            molecule, alongside with metadata on KLIFS residue ID, PDB residue ID, and residue name
            (columns) for 85 pocket residues.

        Returns
        -------
        pandas.DataFrame
            Vertex angles (column) for up to 85 residues (rows).
        """

        vertex_angles = []

        for index, row in pocket_vectors.iterrows():

            # If all three vectors available, calculate vertex_angle
            # - otherwise set vertex_angle to None

            if row.ca and row.side_chain_centroid and row.pocket_centroid:
                # Calculate vertex vertex_angle: CA atom is vertex
                vertex_angle = np.degrees(
                    calc_angle(row.side_chain_centroid, row.ca, row.pocket_centroid)
                )
                vertex_angles.append(vertex_angle.round(2))
            else:
                vertex_angles.append(None)

        # Cast to DataFrame
        vertex_angles = pd.DataFrame(
            vertex_angles, index=pocket_vectors.klifs_id, columns=["vertex_angle"]
        )

        return vertex_angles

    def _get_categories(self, vertex_angles):
        """
        Get side chain orientation category for pocket residues based on their side chain
        orientation vertex angles.
        The side chain orientation towards the pocket is described with the following three
        categories:
        Inwards (0.0), intermediate (1.0), and outwards (2.0).

        Parameters
        ----------
        vertex_angles : pandas.DataFrame
            Vertex angles (column) for up to 85 residues (rows).

        Returns
        -------
        pandas.DataFrame
            Side chain orientation categories (column) for up to 85 residues (rows).
        """

        if "vertex_angle" not in vertex_angles.columns:
            raise ValueError('Input DataFrame needs column with name "vertex_angle".')

        categories = [
            self._get_category_from_vertex_angle(vertex_angle)
            for vertex_angle in vertex_angles.vertex_angle
        ]

        # Cast from Series to DataFrame and set column name for feature
        categories = pd.DataFrame(categories, index=vertex_angles.index, columns=["sco"])

        return categories

    def _get_category_from_vertex_angle(self, vertex_angle):
        """
        Transform a given vertex angle into a category value, which defines the side chain
        orientation towards the pocket:
        Inwards (category 0.0), intermediate (category 1.0), and outwards (category 2.0).

        Parameters
        ----------
        vertex_angle : float
            Vertex angle between a residue's CA atom (vertex), side chain centroid and pocket
            centroid. Ranges between 0.0 and 180.0.

        Returns
        -------
        float
            Side chain orientation towards pocket:
            Inwards (category 0.0), intermediate (category 1.0), and outwards (category 2.0).
        """

        if 0.0 <= vertex_angle <= 45.0:  # Inwards
            return 0.0
        elif 45.0 < vertex_angle <= 90.0:  # Intermediate
            return 1.0
        elif 90.0 < vertex_angle <= 180.0:  # Outwards
            return 2.0
        elif np.isnan(vertex_angle):
            return np.nan
        else:
            raise ValueError(
                f"Molecule {self.molecule_code}: Unknown vertex angle {vertex_angle}. "
                f"Only values between 0.0 and 180.0 allowed."
            )

    def _get_ca(self, residue):
        """
        Get residue's CA atom.

        Parameters
        ----------
        residue : Bio.PDB.Residue.Residue
            Residue.

        Returns
        -------
        Bio.PDB.vectors.Vector or None
            Residue's CA vector.
        """

        atom_names = [atom.name for atom in residue.get_atoms()]

        # Set CA atom

        if "CA" in atom_names:
            vector_ca = residue["CA"].get_vector()
        else:
            logger.info(f"{self.molecule_code}: SCO: CA atom: Missing in {residue}.")
            vector_ca = None

        return vector_ca

    def _get_side_chain_centroid(self, residue, chain):
        """
        Get residue's side chain centroid.

        Parameters
        ----------
        residue : Bio.PDB.Residue.Residue
            Residue.
        chain : Bio.PDB.Chain.Chain
            Chain from PDB file.

        Returns
        -------
        Bio.PDB.vectors.Vector or None
            Residue's side chain centroid.
        """

        # Select only atoms that are
        # - not part of the backbone
        # - not oxygen atoms (OXT) on the terminal carboxyl group
        # - not H atoms

        selected_atoms = [
            atom
            for atom in residue.get_atoms()
            if (atom.name not in "N CA C O OXT".split()) & (not atom.get_id().startswith("H"))
        ]

        n_atoms = len(selected_atoms)

        # Set side chain centroid
        exception = None

        # Standard residues
        # Normally residue.id[0] == ' '
        # but with PyMol converted pdb files some non-standard residues are entered as standard
        # amino acids
        try:

            n_atoms_cutoff = N_HEAVY_ATOMS_CUTOFF[residue.get_resname()]

            if residue.get_resname() == "GLY":  # GLY residue

                side_chain_centroid = self._get_pcb_from_residue(residue, chain)

                if side_chain_centroid is None:
                    exception = "GLY - None"

            elif residue.get_resname() == "ALA":  # ALA residue

                try:
                    side_chain_centroid = residue["CB"].get_vector()

                except KeyError:
                    side_chain_centroid = self._get_pcb_from_residue(residue, chain)

                    if side_chain_centroid is not None:
                        exception = "ALA - pCB atom"
                    else:
                        exception = "ALA - None"

            elif n_atoms >= n_atoms_cutoff:  # Other standard residues with enough side chain atoms

                side_chain_centroid = Vector(center_of_mass(selected_atoms, geometric=True))

            else:  # Other standard residues with too few side chain atoms

                try:
                    side_chain_centroid = residue["CB"].get_vector()
                    exception = (
                        f"Standard residue - CB atom, only {n_atoms}/{n_atoms_cutoff} residues"
                    )

                except KeyError:
                    side_chain_centroid = self._get_pcb_from_residue(residue, chain)

                    if side_chain_centroid is not None:
                        exception = f"Standard residue - pCB atom, only {n_atoms}/{n_atoms_cutoff} residues"
                    else:
                        exception = (
                            f"Standard residue - None, only {n_atoms}/{n_atoms_cutoff} residues"
                        )

        # Non-standard residues
        except KeyError:

            if n_atoms > 0:
                side_chain_centroid = Vector(center_of_mass(selected_atoms, geometric=True))
                exception = f"Non-standard residue - centroid of {n_atoms} atoms"
            else:
                side_chain_centroid = None
                exception = "Non-standard residue - None"

        if exception:
            logger.info(
                f"{self.molecule_code}: SCO: Side chain centroid for "
                f"residue {residue.get_resname()}, {residue.id} with {n_atoms} atoms is: "
                f"{exception}."
            )

        return side_chain_centroid

    def _get_pocket_centroid(self, pocket_residues):
        """
        Get centroid of pocket CA atoms.

        Parameters
        ----------
        pocket_residues : pandas.DataFrame
            Pocket residues: Bio.PDB.Residue.Residue plus metadata, i.e. KLIFS residue ID,
            PDB residue ID, and residue name (columns) for all pocket residues (rows).

        Returns
        -------
        Bio.PDB.vectors.Vector or None
            Pocket centroid.
        """

        # Initialize list for all CA atoms in pocket
        ca_vectors = []

        # Log missing CA atoms
        ca_atoms_missing = []

        for residue in pocket_residues.pocket_residues:
            try:
                ca_vectors.append(residue["CA"])
            except KeyError:
                ca_atoms_missing.append(residue)

        if len(ca_atoms_missing) > 0:
            logger.info(
                f"{self.molecule_code}: SCO: Pocket centroid: "
                f"{len(ca_atoms_missing)} missing CA atom(s): {ca_atoms_missing}"
            )

        try:
            return Vector(center_of_mass(ca_vectors, geometric=True))
        except ValueError:
            logger.info(
                f"{self.molecule_code}: SCO: Pocket centroid: "
                f"Cannot be calculated. {len(ca_vectors)} CA atoms available."
            )
            return None

    @staticmethod
    def _get_pcb_from_gly(residue):
        """
        Get pseudo-CB atom coordinate for GLY residue.

        Parameters
        ----------
        residue : Bio.PDB.Residue.Residue
            Residue.

        Returns
        -------
        Bio.PDB.vectors.Vector or None
            Pseudo-CB atom vector for GLY centered at CA atom (= pseudo-CB atom coordinate).
        """

        if residue.get_resname() == "GLY":

            # Get pseudo-CB for GLY (vector centered at origin)
            chain = Chain(id="X")  # Set up chain instance
            pcb = HSExposureCB(chain)._get_gly_cb_vector(residue)

            if pcb is None:
                return None

            else:
                # Center pseudo-CB vector at CA atom to get pseudo-CB coordinate
                ca = residue["CA"].get_vector()
                ca_pcb = ca + pcb
                return ca_pcb

        else:
            raise ValueError(f"Residue must be GLY, but is {residue.get_resname()}.")

    def _get_pcb_from_residue(self, residue, chain):
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

    """
    def show_in_nglviewer(self):

        # Get molecule and molecule code
        code = split_klifs_code(self.molecule_code)

        pdb_id = code['pdb_id']
        chain = code['chain']

        viewer = nv.show_pdbid(pdb_id, default=False)
        viewer.clear()
        viewer.add_cartoon(selection=f':{chain}', assembly='AU')
        viewer.center(selection=f':{chain}')

        # Representation parameters
        sphere_radius = {
            'ca': 0.3,
            'side_chain_centroid': 0.3,
            'pocket_centroid': 1
        }

        colors = {
            'ca': [0, 1, 0],
            'side_chain_centroid': [1, 0, 0],
            'pocket_centroid': [0, 0, 1]
        }

        # Show side chain orientation feature per residue
        for index, row in self.features_verbose.iterrows():

            res_id = row.res_id

            viewer.add_representation(repr_type='line', selection=f'{res_id}:{chain}')
            viewer.add_label(selection=f'{res_id}:{chain}.CA')  # TODO: Add angles as label here

            if row.ca:
                ca = list(row.ca.get_array())
                viewer.shape.add_sphere(ca, colors['ca'], sphere_radius['ca'])

            if row.side_chain_centroid:
                side_chain_centroid = list(row.side_chain_centroid.get_array())
                viewer.shape.add_sphere(
                    side_chain_centroid, 
                    colors['side_chain_centroid'], 
                    sphere_radius['side_chain_centroid']
                )

            if row.pocket_centroid:
                pocket_centroid = list(row.pocket_centroid.get_array())
                viewer.shape.add_sphere(
                    pocket_centroid, 
                    colors['pocket_centroid'], 
                    sphere_radius['pocket_centroid']
                )

        viewer.gui_style = 'ngl'

        return viewer
    """