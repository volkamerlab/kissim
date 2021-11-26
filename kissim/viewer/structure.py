"""
kissim.viewer.structure

Visualizes a structure's fingerprint in 3D.
"""

from kissim.viewer.base import _BaseViewer


class StructureViewer(_BaseViewer):
    """
    View a structure's fingerprint in 3D.

    Attributes
    ----------
    Inherited from kissim.viewer.base._BaseViewer
    """

    @classmethod
    def from_structure_klifs_id(cls, structure_klifs_id, klifs_session=None):
        """
        Initialize viewer from structure KLIFS ID: Generate fingerprint and fetch structure in PDB
        format.
        """

        structure_klifs_ids = [structure_klifs_id]
        viewer = cls._from_structure_klifs_ids(structure_klifs_ids, klifs_session)

        return viewer

    def residue_to_color_mapping(self, feature_name, fingerprint, plot_cmap=False):
        """
        Map feature values using color on residues.

        Parameters
        ----------
        feature_name : str
            Name of a continuous feature.
        fingerprint : kissim.encoding.Fingerprint
            Fingerprint.
        plot_cmap : bool
            Plot color map (default: False).

        Returns
        -------
        list of list of [str, str]
            List of color-residue ID pairs. Color given as hex string, residue PDB IDs as string.
        """

        if feature_name in self._fingerprint.physicochemical:
            data = fingerprint.physicochemical[feature_name]
            data.index = fingerprint.residue_ids
            # If residue IDs contain None values, `data.index` will be of type float; we need int
            data.index = data.index.astype("Int32")

            residue_to_color = self._residue_to_color_mapping(
                feature_name,
                data,
                discrete=True,
                divergent=False,
                plot_cmap=plot_cmap,
            )
        elif feature_name in self._fingerprint.distances:
            data = fingerprint.distances[feature_name]
            data.index = fingerprint.residue_ids
            # If residue IDs contain None values, `data.index` will be of type float; we need int
            data.index = data.index.astype("Int32")

            residue_to_color = self._residue_to_color_mapping(
                feature_name,
                data,
                discrete=False,
                divergent=False,
                plot_cmap=plot_cmap,
            )
        else:
            raise ValueError(f"Feature name {feature_name} unknown.")

        return residue_to_color
