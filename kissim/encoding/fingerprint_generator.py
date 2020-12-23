"""
kissim.encoding.fingerprint

Defines sequencial and parallel processing of fingerprints from local or remote structures.
"""

import datetime
import logging
from itertools import repeat

from multiprocessing import cpu_count, Pool
from opencadd.databases.klifs import setup_remote

logger = logging.getLogger(__name__)


class FingerprintGenerator:
    """
    Generate fingerprints for multiple structures.

    Attributes
    ----------
    TODO
    """

    def __init__(self):

        self.structure_klifs_ids = None
        self.klifs_session = None
        self.data = None

    @classmethod
    def from_structure_klifs_ids(cls, structure_klifs_ids, klifs_session=None, n_cores=None):
        """
        Calculate fingerprints for one or more KLIFS structures (by structure KLIFS IDs).

        Parameters
        ----------
        TODO
        """

        start_time = datetime.datetime.now()

        # Set up KLIFS session if needed
        if klifs_session is None:
            klifs_session = setup_remote()

        # Initialize FingerprintGenerator object
        fingerprint_generator = cls()
        fingerprint_generator.structure_klifs_ids = structure_klifs_ids
        fingerprint_generator.klifs_session = klifs_session

        # Set number of cores to be used
        n_cores = fingerprint_generator._set_n_cores(n_cores)

        # Generate fingerprints
        if n_cores == 1:
            fingerprints_list = fingerprint_generator._process_fingerprints_in_sequence()
        else:
            fingerprints_list = fingerprint_generator._process_fingerprints_in_parallel(n_cores)

        # Add fingerprints to FingerprintGenerator object
        fingerprint_generator.data = {
            i.structure_klifs_id: i
            for i in fingerprints_list
            if i is not None  # Removes emtpy fingerprints
        }

        end_time = datetime.datetime.now()

        logger.info(f"Number of input structures: {len(structure_klifs_ids)}")
        logger.info(f"Number of fingerprints: {len(fingerprints_list)}")
        logger.info(
            f"Number of fingerprints without None: "
            f"{len([i for i in fingerprints_list if i is not None])}"
        )
        logger.info(f"Start of fingerprint generation: {start_time}")
        logger.info(f"End of fingerprint generation: {end_time}")

        return fingerprint_generator

    def _set_n_cores(self, n_cores):
        """TODO"""

        max_n_cores = cpu_count() - 1
        if n_cores is None:
            n_cores = max_n_cores
        else:
            if n_cores > max_n_cores:
                raise ValueError(
                    f"Maximal number of available cores: {max_n_cores}. You chose: {n_cores}."
                )
        logger.info(f"Number of cores used: {n_cores}.")
        return n_cores

    def _process_fingerprints_in_sequence(self):
        """TODO"""

        fingerprints_list = [
            self._get_fingerprint(structure_klifs_id, self.klifs_session)
            for structure_klifs_id in self.structure_klifs_ids
        ]
        return fingerprints_list

    def _process_fingerprints_in_parallel(self, n_cores):
        """TODO"""

        print(list(zip(self.structure_klifs_ids, repeat(self.klifs_session))))

        pool = Pool(processes=n_cores)
        fingerprints_list = pool.starmap(
            self._get_fingerprint, zip(self.structure_klifs_ids, repeat(self.klifs_session))
        )
        pool.close()
        pool.join()
        return fingerprints_list

    def _get_fingerprint(self, structure_klifs_id, klifs_session):
        """TODO"""

        try:
            fingerprint = Fingerprint.from_structure_klifs_id(structure_klifs_id, klifs_session)
            return fingerprint
        except Exception as e:  # TODO too generic!
            logger.info(
                f"Fingerprint for structure with KLIFS ID {structure_klifs_id} could not "
                f"be generated."
            )
            logger.error(e)
            return None
