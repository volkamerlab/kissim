"""
kissim.io.data

Defines class for KLIFS data (to be passed to kissim).
"""

import logging

from bravado_core.exception import SwaggerMappingError

from opencadd.databases.klifs import setup_remote
from opencadd.structure.pocket import PocketBase

logger = logging.getLogger(__name__)


class KlifsToKissimData:
    """
    Class for KLIFS data from structure KLIFS ID to prepare data for kissim.

    Attributes
    ----------
    klifs_session : opencadd.databases.klifs.session.Session
        Local or remote KLIFS session.
    structure_klifs_id : int
        KLIFS structure ID.
    text : str
        Structural complex data as string (file content).
    extension : str
        Structural complex data format (file extension).
    residue_ids : list of int
        Pocket residue IDs.
    residue_ixs : list of int
        Pocket residue indices.
    """

    def __init__(self):

        self.klifs_session = None
        self.structure_klifs_id = None
        self.kinase_name = None
        self.text = None
        self.extension = None
        self.residue_ids = None
        self.residue_ixs = None

    @classmethod
    def from_structure_klifs_id(cls, structure_klifs_id, klifs_session=None):
        """
        Get KLIFS data from structure KLIFS ID.

        Parameters
        ----------
        structure_klifs_id : int
            KLIFS structure ID.
        klifs_session : opencadd.databases.klifs.session.Session
            Local or remote KLIFS session.

        Returns
        -------
        kissim.io.KlifsToKissimData
            KLIFS data.
        """

        try:

            data = cls()
            data.structure_klifs_id = structure_klifs_id

            # If no KLIFS session is given, set up remote KLIFS session
            if klifs_session is None:
                klifs_session = setup_remote()
            data.klifs_session = klifs_session

            # Structure KLIFS ID exists
            if not data._structure_klifs_id_exists():
                return None

            # In case of a local KLIFS session, test if complex and pocket structural files exist
            if data.klifs_session._database is not None:
                if not data._local_session_files_exist():
                    return None

            data.text, data.extension = data._get_text_and_extension()
            data.residue_ids, data.residue_ixs = data._get_pocket_residue_ids_and_ixs()
            data.kinase_name = data._get_kinase_name()

            return data

        except ValueError as e:
            logger.error(
                f"The following structure could not be loaded into kissim: {structure_klifs_id}: "
                f"{e}"
            )
            return None

    def _structure_klifs_id_exists(self):
        """
        Check if structure KLIFS ID exists.

        Returns
        -------
        bool
            True if structure KLIFS ID exists, else False.
        """

        structure_klifs_id_exists = True

        if self.klifs_session._client:
            try:
                self.klifs_session.structures.by_structure_klifs_id(self.structure_klifs_id)
            except SwaggerMappingError as e:
                logger.error(
                    f"{self.structure_klifs_id}: Structure KLIFS ID unknown to remote session "
                    f"(KLIFS response: SwaggerMappingError: {e})"
                )
                structure_klifs_id_exists = False
        else:
            try:
                self.klifs_session.structures.by_structure_klifs_id(self.structure_klifs_id)
            except ValueError as e:
                logger.error(
                    f"{self.structure_klifs_id}: Structure KLIFS ID unknown to local session. "
                    f"(ValueError: {e})"
                )
                structure_klifs_id_exists = False

        return structure_klifs_id_exists

    def _local_session_files_exist(self):
        """
        Check if the coordinate files complex.pdb and pocket.pdb exist in local KLIFS session.

        Returns
        -------
        True if files exist, else False.
        """

        # Get path to folder with data for structure KLIFS ID
        path = self.klifs_session.structures.by_structure_klifs_id(self.structure_klifs_id)[
            "structure.filepath"
        ][0]

        # Get path to complex.pdb and pocket.pdb
        complex_filepath = self.klifs_session._path_to_klifs_download / path / "complex.pdb"
        pocket_filepath = self.klifs_session._path_to_klifs_download / path / "pocket.pdb"

        # Do both files exist?
        if complex_filepath.exists() and pocket_filepath.exists():
            return True
        else:
            logger.error(
                f"{self.structure_klifs_id}: Local complex.pdb or pocket.pdb file missing: "
                f"{complex_filepath}"
            )
            return False

    def _get_text_and_extension(self):
        """
        Get structural data for a complex from a KLIFS structure ID as Biopython Structure object.

        Returns
        -------
        text : string
            Complex structural data.
        extension : string
            Complex file extension.
        """

        extension = "pdb"

        if self.klifs_session._database is not None:
            filepath = self.klifs_session.structures.by_structure_klifs_id(
                self.structure_klifs_id
            )["structure.filepath"][0]
            filepath = (
                self.klifs_session._path_to_klifs_download / filepath / f"complex.{extension}"
            )
            structure_klifs_id_or_filepath = filepath
        else:
            structure_klifs_id_or_filepath = self.structure_klifs_id

        text = self.klifs_session.coordinates.to_text(
            structure_klifs_id_or_filepath, "complex", extension
        )
        return text, extension

    def _get_pocket_residue_ids_and_ixs(self):
        """
        Get pocket residues.

        Returns
        -------
        residue_ids : list of int
            Pocket residue PDB IDs.
        residue_ixs : list of int
            Pocket residues indices.
        """

        # TODO check if if-else necessary (API should be the same!)
        if self.klifs_session._client:
            residues = self.klifs_session.pockets.by_structure_klifs_id(self.structure_klifs_id)
        else:
            residues = self.klifs_session.pockets.by_structure_klifs_id(
                self.structure_klifs_id, extension="pdb"
            )
        residue_ids = residues["residue.id"].to_list()
        residue_ixs = residues["residue.klifs_id"].to_list()

        pocket_base = PocketBase()
        residue_ids, residue_ixs = pocket_base._format_residue_ids_and_ixs(
            residue_ids, residue_ixs, "set pocket residues"
        )

        return residue_ids, residue_ixs

    def _get_kinase_name(self):
        """
        TODO docstring + unit test!!
        """

        structures = self.klifs_session.structures.by_structure_klifs_id(self.structure_klifs_id)
        kinase_name = structures.squeeze()["kinase.klifs_name"]
        return kinase_name
