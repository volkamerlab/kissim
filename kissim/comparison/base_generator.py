"""
kissim.comparison.base_generator

Defines the base class for pairwise distances for a set of fingerprints.
"""

from pathlib import Path

import pandas as pd


class BaseGenerator:
    """
    Generate distances for multiple fingerprint pairs.

    Attributes
    ----------
    data : pandas.DataFrame
        Distance(s) and bit coverage(s) for each structure pair (kinase pair).
    structure_kinase_ids : list of list
        Structure and kinase IDs for structures in dataset.
    """

    def __init__(self):
        self.data = None
        self.structure_kinase_ids = None

    @property
    def structure_ids(self):
        """
        Unique structure IDs associated with all fingerprints (sorted alphabetically).

        Returns
        -------
        list of str or int
            Structure IDs.
        """

        return sorted(
            pd.DataFrame(self.structure_kinase_ids, columns=["structure_id", "kinase_id"])[
                "structure_id"
            ].unique()
        )

    @property
    def kinase_ids(self):
        """
        Unique kinase IDs (e.g. kinase names) associated with all fingerprints (sorted
        alphabetically).

        Returns
        -------
        list of str or int
            Kinase IDs.
        """

        return sorted(
            pd.DataFrame(self.structure_kinase_ids, columns=["structure_id", "kinase_id"])[
                "kinase_id"
            ].unique()
        )

    @property
    def structure_pair_ids(self):
        """
        All structure pair IDs.

        Returns
        -------
        list of list of int
            List of structure pairs.
        """

        return self.data[["structure.1", "structure.2"]].values.tolist()

    @property
    def kinase_pair_ids(self):
        """
        All kinase pair IDs.

        Returns
        -------
        list of list of str
            List of kinases belonging to structure pairs.
        """

        return self.data[["kinase.1", "kinase.2"]].values.tolist()

    @property
    def distances(self):
        """
        All structure pair feature distances.

        Returns
        -------
        np.array
            Fingerprint distances for structure pairs.
        """

        return self.data[self.data.columns[self.data.columns.str.startswith("distance")]].values

    @property
    def bit_coverages(self):
        """
        All structure pair feature bit coverages.

        Returns
        -------
        np.array
            Bit coverages for structure pairs.
        """

        return self.data[
            self.data.columns[self.data.columns.str.startswith("bit_coverage")]
        ].values

    @classmethod
    def from_csv(cls, filepath):
        """TODO"""

        filepath = Path(filepath)

        data = pd.read_csv(filepath)

        base_generator = cls()
        base_generator.data = data
        base_generator.structure_kinase_ids = base_generator._structure_kinase_ids

        return base_generator

    def to_csv(self, filepath):
        """TODO"""

        filepath = Path(filepath)

        self.data.to_csv(filepath, index=False)

    @property
    def _structure_kinase_ids(self):
        """
        Structure and kinase IDs for structures in dataset.

        Returns
        -------
        list of list
            Structure and kinase IDs for structures in dataset.
        """

        return (
            pd.concat(
                [
                    self.data[["structure.1", "kinase.1"]]
                    .rename(columns={"structure.1": "s", "kinase.1": "k"})
                    .drop_duplicates(),
                    self.data[["structure.2", "kinase.2"]]
                    .rename(columns={"structure.2": "s", "kinase.2": "k"})
                    .drop_duplicates(),
                ]
            )
            .drop_duplicates()
            .values.tolist()
        )
