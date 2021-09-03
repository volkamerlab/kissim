"""
kissim.encoding.fingerprint_generator

Defines sequencial and parallel processing of fingerprints from local or remote structures.
"""

import datetime
from itertools import repeat
import json
import logging
from pathlib import Path

from multiprocessing import Pool
import pandas as pd
from opencadd.databases.klifs import setup_remote

from kissim.encoding import Fingerprint, FingerprintNormalized, FingerprintGeneratorBase
from kissim.utils import set_n_cores

logger = logging.getLogger(__name__)


class FingerprintGenerator(FingerprintGeneratorBase):
    @classmethod
    def from_structure_klifs_ids(cls, structure_klifs_ids, klifs_session=None, n_cores=1):
        """
        Calculate fingerprints for one or more KLIFS structures (by structure KLIFS IDs).

        Parameters
        ----------
        structure_klifs_id : int
            Input structure KLIFS ID (output fingerprints may contain less IDs because some
            structures could not be encoded).
        klifs_session : opencadd.databases.klifs.session.Session
            Local or remote KLIFS session.
        n_cores : int or None
            Number of cores to be used for fingerprint generation as defined by the user.

        Returns
        -------
        kissim.encoding.fingerprint_generator
            Fingerprint generator object containing fingerprints.
        """

        logger.info("GENERATE FINGERPRINTS")
        logger.info(f"Number of input structures: {len(structure_klifs_ids)}")

        start_time = datetime.datetime.now()
        logger.info(f"Fingerprint generation started at: {start_time}")

        # Set up KLIFS session if needed
        if klifs_session is None:
            klifs_session = setup_remote()

        # Set number of cores to be used
        n_cores = set_n_cores(n_cores)

        # Initialize FingerprintGenerator object
        fingerprint_generator = cls()
        fingerprint_generator.structure_klifs_ids = structure_klifs_ids
        fingerprint_generator.klifs_session = klifs_session
        fingerprints_list = fingerprint_generator._get_fingerprint_list(n_cores)
        fingerprint_generator.data = {
            i.structure_klifs_id: i
            for i in fingerprints_list
            if i is not None  # Removes emtpy fingerprints
        }

        logger.info(f"Number of output fingerprints: {len(fingerprint_generator.data)}")

        end_time = datetime.datetime.now()
        logger.info(f"Runtime: {end_time - start_time}")

        return fingerprint_generator

    @classmethod
    def from_json(cls, filepath):
        """
        Initialize a FingerprintGenerator object from a json file.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Path to json file.
        """

        filepath = Path(filepath)
        with open(filepath, "r") as f:
            json_string = f.read()
        fingerprints_list = json.loads(json_string)

        data = {}
        for fingerprint_dict in fingerprints_list:
            fingerprint = Fingerprint._from_dict(fingerprint_dict)
            data[fingerprint.structure_klifs_id] = fingerprint

        fingerprint_generator = cls()
        fingerprint_generator.data = data
        fingerprint_generator.structure_klifs_ids = list(fingerprint_generator.data.keys())

        return fingerprint_generator

    def to_json(self, filepath):
        """
        Write FingerprintGenerator class attributes to a json file.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Path to json file.
        """

        fingerprint_list = [
            fingerprint.__dict__ for structure_klifs_id, fingerprint in self.data.items()
        ]
        json_string = json.dumps(fingerprint_list)
        filepath = Path(filepath)
        with open(filepath, "w") as f:
            f.write(json_string)

    def _get_fingerprint_list(self, n_cores):
        """
        Generate fingerprints.

        Parameters
        ----------
        n_cores : int
            Number of cores.

        Returns
        -------
        list of kissim.encoding.fingerprint
            List of fingerprints
        """

        pool = Pool(processes=n_cores)
        fingerprints_list = pool.starmap(
            self._get_fingerprint, zip(self.structure_klifs_ids, repeat(self.klifs_session))
        )
        pool.close()
        pool.join()

        return fingerprints_list

    def _get_fingerprint(self, structure_klifs_id, klifs_session):
        """
        Generate a fingerprint.

        Parameters
        ----------
        structure_klifs_id : int
            Structure KLIFS ID.
        klifs_session : opencadd.databases.klifs.session.Session
            Local or remote KLIFS session.

        Returns
        -------
        kissim.encoding.fingerprint
            Fingerprint.
        """

        logger.info(f"{structure_klifs_id}: Generate fingerprint...")
        fingerprint = Fingerprint.from_structure_klifs_id(structure_klifs_id, klifs_session)
        return fingerprint
