import pickle as pickle

import numpy as np
from scipy.spatial.distance import cdist


class OtherMolecules:
    """A class for characterizing the relationships between multiple
    pymolecule."""

    def __init__(self, parent_molecule_object):
        """Initializes the pymolecule.OtherMolecules class.

        Args:
            parent_molecule_object: The pymolecule.Molecule object
                associated with this class.

        """

        self.parent_molecule = parent_molecule_object

    def align_other_molecule_to_this(self, other_mol, tethers):
        """Aligns a molecule to self (this pymolecule.Molecule object) using a
        quaternion RMSD alignment.

        Args:
            other_mol: A pymolecule.Molecule that is to be aligned to this
                one.
            tethers: A tuple of two np.array objects, where each array
                contains the indices of self and other_mol, respectively, such
                that equivalent atoms are listed in the same order. So, for
                example, if (atom 1, self = atom 3, other) and (atom2, self =
                atom6, other) than the tethers would be (np.array([1,2]),
                np.array([3,6])).

        """

        # Adapted from Itzhack Y. Bar-Itzhack. New Method for Extracting the
        # Quaternion from a Rotation Matrix. Journal of Guidance, Control, and
        # Dynamics 2000

        # get the atoms corresponding to the tethers, in tether order
        self_static_atom_coordinates = self.parent_molecule.information.coordinates[
            tethers[0]
        ]
        other_dynamic_atom_coordinates = other_mol.information.coordinates[tethers[1]]

        # translate the tether atoms to the origin
        center_self = np.mean(self_static_atom_coordinates, 0)
        center_other = np.mean(other_dynamic_atom_coordinates, 0)

        self_static_atom_coordinates = self_static_atom_coordinates - center_self
        other_dynamic_atom_coordinates = other_dynamic_atom_coordinates - center_other

        # get optimal rotation
        M = np.dot(
            np.transpose(self_static_atom_coordinates),
            other_dynamic_atom_coordinates,
        )

        # Create symmetric 4x4 matrix K from M
        K = np.array(
            [
                [
                    M[0, 0] + M[1, 1] + M[2, 2],
                    M[1, 2] - M[2, 1],
                    M[2, 0] - M[0, 2],
                    M[0, 1] - M[1, 0],
                ],
                [
                    M[1, 2] - M[2, 1],
                    M[0, 0] - M[1, 1] - M[2, 2],
                    M[1, 0] + M[0, 1],
                    M[2, 0] + M[0, 2],
                ],
                [
                    M[2, 0] - M[0, 2],
                    M[1, 0] + M[0, 1],
                    M[1, 1] - M[0, 0] - M[2, 2],
                    M[1, 2] + M[2, 1],
                ],
                [
                    M[0, 1] - M[1, 0],
                    M[2, 0] + M[0, 2],
                    M[1, 2] + M[2, 1],
                    M[2, 2] - M[0, 0] - M[1, 1],
                ],
            ]
        )

        # Find eigenvector associated with the most positive eigenvalue of K.
        # Multiple quaternions can
        E, V = np.linalg.eig(K)
        index = np.argmax(E)
        eigenvector = V[:, index]
        rot_quat = Quaternion(
            eigenvector[0], eigenvector[1], eigenvector[2], eigenvector[3]
        )

        rot_mat = rot_quat.to_matrix()

        # Apply translation to the other molecule
        new_mol = other_mol.selections.copy()

        new_mol.information.coordinates = new_mol.information.coordinates - center_other
        new_mol.information.coordinates = np.dot(
            new_mol.information.coordinates, rot_mat
        )
        new_mol.information.coordinates = new_mol.information.coordinates + center_self

        return new_mol

    def steric_clash_with_another_molecule(
        self, other_mol, cutoff, pairwise_comparison=True
    ):
        """Detects steric clashes between the pymolecule.Molecule (self) and
        another pymolecule.Molecule.

        Args:
            other_mol: The pymolecule.Molecule object that will be evaluated
                for steric clashes.
            cutoff: A float, the user-defined distance cutoff in Angstroms.
            pairwise_comparison: An optional boolean, whether or not to
                perform a simple pairwise distance comparison (if True) or to
                use a more sophisitcated method (if False). True by default.

        Returns:
            A boolean.  True if steric clashes are present, False if they are
                not.

        """

        if (
            pairwise_comparison
        ):  # so use a simple pairwise comparison to find close atoms
            (
                indices1,
                indices2,
            ) = self.parent_molecule.selections.select_close_atoms_from_different_molecules(
                other_mol, cutoff, True
            )
        else:  # so the more sophisticated heirarchical method
            (
                indices1,
                indices2,
            ) = self.parent_molecule.selections.select_close_atoms_from_different_molecules(
                other_mol, cutoff, False, True
            )  # terminate early is true because you don't want all close ones

        if len(indices1) == 0 and len(indices2) == 0:
            return False
        else:
            return True

    def merge_with_another_molecule(self, other_molecule):
        """Merges two molecular models into a single model.

        Args:
            other_molecule: A molecular model (pymolecule.Molecule object).

        Returns:
            A single pymolecule.Molecule object containing the atoms of this
                model combined with the atoms of other_molecule.

        """

        merged = self.parent_molecule.selections.copy()

        # if masses have been assigned to either molecule, they must be
        # assigned to both
        if (
            "mass" in merged.information.atom_information.dtype.names
            or "mass" in self.parent_molecule.information.atom_information.dtype.names
        ):
            self.parent_molecule.information.assign_masses()
            merged.information.assign_masses()

        merged.filename = ""
        merged.information.remarks.extend(other_molecule.information.remarks)
        merged.information.atom_information = np.lib.recfunctions.stack_arrays(
            (
                merged.information.atom_information,
                other_molecule.information.atom_information,
            ),
            usemask=False,
        )

        merged.information.coordinates = np.vstack(
            (merged.information.coordinates, other_molecule.information.coordinates)
        )

        # if either of the undo points is None, set the merged one to None
        if (
            not merged.information.coordinates_undo_point is None
            and not other_molecule.information.coordinates_undo_point is None
        ):
            merged.information.coordinates_undo_point = np.vstack(
                (
                    merged.information.coordinates_undo_point,
                    other_molecule.information.coordinates_undo_point,
                )
            )
        else:
            merged.information.coordinates_undo_point = None

        # merge the bonds, though note that bonds between the two molecules
        # will not be set
        if (
            not merged.information.bonds is None
            and not other_molecule.information.bonds is None
        ):
            bonds1 = merged.information.bonds.copy()
            bonds2 = other_molecule.information.bonds.copy()

            bonds1_v2 = np.hstack((bonds1, np.zeros((len(bonds1), len(bonds2)))))
            bonds2_v2 = np.hstack((np.zeros((len(bonds2), len(bonds1))), bonds2))

            merged.information.bonds = np.vstack((bonds1_v2, bonds2_v2))
        else:
            merged.information.bonds = None

        # the molecule center will be redefined, so you might as well start
        # the hierarchy all over
        try:
            del merged.information.hierarchy["spheres"]
        except Exception:
            pass

        return merged

    def distance_to_another_molecule(self, other_molecule, pairwise_comparison=True):
        """Computes the minimum distance between any of the atoms of this
        molecular model and any of the atoms of a second specified model.

        Args:
            other_molecule: a pymolecule.Molecule, the other molecular model.
            pairwise_comparison: An optional boolean, whether or not to
                perform a simple pairwise distance comparison (if True) or to
                use a more sophisitcated method (if False). True by default.

        Returns:
            A float, the minimum distance between any two atoms of the two
                specified molecular models (self and other_molecule).

        """

        if pairwise_comparison:
            return np.amin(
                cdist(
                    self.parent_molecule.information.coordinates,
                    other_molecule.information.coordinates,
                )
            )
        else:  # so use the more sofisticated methods for comparison
            # note that this is not the fastest way to do this, but it uses
            # existing functions and is still pretty fast, so I'm going to
            # stick with it.

            # first, get a cutoff distance. Let's just do a quick survey of
            # the two molecules to pick a good one.
            self_tmp = self.parent_molecule.information.coordinates[
                np.arange(
                    0,
                    len(self.parent_molecule.information.coordinates),
                    len(self.parent_molecule.information.coordinates) / 10.0,
                    dtype=int,
                )
            ]
            other_tmp = other_molecule.information.coordinates[
                np.arange(
                    0,
                    len(other_molecule.information.coordinates),
                    len(other_molecule.information.coordinates) / 10.0,
                    dtype=int,
                )
            ]
            cutoff = np.amin(cdist(self_tmp, other_tmp))

            # now get all the indices that come within that cutoff
            (
                self_indices,
                other_indices,
            ) = self.parent_molecule.selections.select_close_atoms_from_different_molecules(
                other_molecule, cutoff, False
            )

            self_coors = self.parent_molecule.information.coordinates[self_indices]
            self_other = other_molecule.information.coordinates[other_indices]

            return np.amin(cdist(self_coors, self_other))

    def rmsd_equivalent_atoms_specified(self, other_mol, tethers):
        """Calculates the RMSD between this pymolecule.Molecle object and
        another, where equivalent atoms are explicitly specified.

        Args:
            other_mol: The other pymolecule.Molecule object.
            tethers: A tuple of two np.array objects, where each array
                contains the indices of self and other_mol, respectively, such
                that equivalent atoms are listed in the same order. So, for
                example, if (atom 1, self = atom 3, other) and (atom2, self =
                atom6, other) than the tethers would be (np.array([1,2]),
                np.array([3,6])).

        Returns:
            A float, the RMSD between self and other_mol.

        """

        if len(self.parent_molecule.information.coordinates) != len(
            other_mol.information.coordinates
        ):
            print("Cannot calculate RMSD: number of atoms are not equal.")
            print(
                "\t"
                + str(len(self.parent_molecule.information.coordinates))
                + " vs. "
                + str(len(other_mol.information.coordinates))
                + " atoms."
            )
            return 99999999.0

        self_coors_in_order = self.parent_molecule.information.coordinates[tethers[0]]
        other_coors_in_order = other_mol.information.coordinates[tethers[1]]

        delta = self_coors_in_order - other_coors_in_order
        norm_squared = np.sum(delta**2, axis=-1)
        rmsd = np.power(np.sum(norm_squared) / len(norm_squared), 0.5)
        return rmsd

    def rmsd_order_dependent(self, other_mol):
        """Calculates the RMSD between two structures, where equivalent atoms
        are listed in the same order.

        Args:
            other_mol: The other pymolecule.Molecule object.

        Returns:
            A float, the RMSD between self and other_mol.

        """

        self_index_in_order = np.arange(
            0, len(self.parent_molecule.information.coordinates), 1, dtype=int
        )
        other_index_in_order = np.arange(
            0, len(other_mol.information.coordinates), 1, dtype=int
        )

        return self.rmsd_equivalent_atoms_specified(
            other_mol, (self_index_in_order, other_index_in_order)
        )
