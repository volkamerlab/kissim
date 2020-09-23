"""
kissim.io.complex

Defines classes that convert structural data into kissim complex objects.
"""

from Bio.PDB.HSExposure import HSExposureCB

from opencadd.io import DataFrame, Biopython
from opencadd.databases.klifs import setup_remote
from .core import Base


class Complex(Base):
    """
    Class defining a complex structure object.
    """

    @classmethod
    def from_file(cls, filepath):
        """
        Load complex structure from file.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Path to pdb file.

        Returns
        -------
        kissim.core.Complex
            Kissim complex object.
        """

        complex = ComplexFromFile.from_file(filepath)

        return cls(complex.dataframe, complex._biopython, complex._hse)

    @classmethod
    def from_structure_id(cls, structure_id):
        """
        Load complex structure from KLIFS structure ID.

        Parameters
        ----------
        structure_id : int
            KLIFS structure ID.

        Returns
        -------
        kissim.core.Complex
            Kissim complex object.
        """

        complex = ComplexFromKlifsId.from_structure_id(structure_id)

        return cls(complex.dataframe, complex._biopython, complex._hse)


class ComplexFromFile(Base):
    """
    Class defining a complex structure object, loaded from a file.
    """

    @classmethod
    def from_file(cls, filepath):
        """
        Load structure from file.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Path to pdb file.

        Returns
        -------
        kissim.core.Structure
            Kissim structure object.
        """

        # Structure as DataFrame
        dataframe = DataFrame.from_file(filepath)

        # Structure as Biopython object
        _biopython = Biopython.from_file(filepath)

        # Calculate half-sphere exposure
        _hse = HSExposureCB(_biopython)
        _hse = _hse.property_dict
        _hse = {residue[1][1]: hse for residue, hse in _hse.items()}

        complex = cls(dataframe, _biopython, _hse)

        return complex


class ComplexFromKlifsId(Base):
    """
    Class defining a complex structure object, loaded from a KLIFS structure ID.
    """

    @classmethod
    def from_structure_id(cls, structure_id):
        """
        Load structure from KLIFS structure ID (fetch remotely).

        Parameters
        ----------
        structure_id : int
            KLIFS structure ID.

        Returns
        -------
        kissim.core.Structure
            Kissim structure object.
        """

        # TODO handle temporary files more elegantly

        remote = setup_remote()
        filepath = remote.coordinates.to_file(structure_id, ".", "complex", "pdb")
        complex = ComplexFromFile.from_file(filepath)
        filepath.unlink()

        return complex
