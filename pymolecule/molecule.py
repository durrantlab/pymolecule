import copy
import pickle as pickle
import sys

import numpy as np
from scipy.spatial.distance import cdist

from .fileio import FileIO
from .geometry import Geometry
from .info import Information
from .manipulation import Manipulation
from .other import OtherMolecules
from .structure import AtomsAndBonds


class Molecule:
    """Loads, saves, and manipulates molecular models. The main pymolecule
    class."""

    def __init__(self):
        """Initializes the variables of the Molecule class."""

        self.io = FileIO(self)
        self.atoms_and_bonds = AtomsAndBonds(self)
        self.selections = Selections(self)
        self.manipulation = Manipulation(self)
        self.information = Information(self)
        self.other_molecule = OtherMolecules(self)
        self.geometry = Geometry(self)

    # surprised this doesn't come with numpy
    def numpy_structured_array_remove_field(self, narray, field_names):
        """Removes a specific field name from a structured numpy array.

        Args:
            narray: A structured numpy array.
            field_names: A list of strings, where each string is one of the
                field names of narray.

        Returns:
            A structured numpy array identical to narray, but with the field
                names in field_names removed.

        """

        # now remove the coordinates from the atom_information object to save
        # memory
        names = list(narray.dtype.names)
        for f in field_names:
            names.remove(f)
        return narray[names].copy()

    def __is_number(self, s):
        """Determines whether or not a string represents a number.

        Args:
            s: A string (e.g., "5.4").

        Returns:
            A boolean, whether or not the string can be represented by a float.

        """

        try:
            float(s)
            return True
        except ValueError:
            return False


class Selections:
    """A class for selecting atoms"""

    ######## selections ########
    def __init__(self, parent_molecule_object):
        """Initializes the pymolecule.Selections class.

        Args:
            parent_molecule_object: The pymolecule.Molecule object associated
                with this class.

        """

        self.parent_molecule = parent_molecule_object

    def select_atoms(self, selection_criteria):
        """Select a set of atoms based on user-specified criteria.

        Args:
            selection_criteria: An dictionary, where the keys correspond to
                keys in the self.parent_molecule.information.atom_information
                structured numpy array, and the values are lists of acceptable
                matches. The selection is a logical "AND" between dictionary
                entries, but "OR" within the value lists themselves. For
                example: {'atom':['CA','O'], 'chain':'A', 'resname':'PRO'}
                would select all atoms with the names CA or O that are located
                in the PRO residues of chain A.

        Returns:
            A np.array containing the indices of the atoms of the
                selection.

        """

        try:
            # start assuming everything is selected
            selection = np.ones(
                len(self.parent_molecule.information.atom_information), dtype=bool
            )

            for key in list(selection_criteria.keys()):

                vals = selection_criteria[key]

                # make sure the vals are in a list
                if not type(vals) is list and not type(vals) is tuple:
                    vals = [vals]  # if it's a single value, put it in a list

                # make sure the vals are in the right format
                if key in self.parent_molecule.information.constants["f8_fields"]:
                    vals = [float(v) for v in vals]
                elif key in self.parent_molecule.information.constants["i8_fields"]:
                    vals = [int(v) for v in vals]
                else:
                    vals = [v.strip() for v in vals]

                # "or" all the vals together start assuming nothing is
                # selected
                subselection = np.zeros(
                    len(self.parent_molecule.information.atom_information), dtype=bool
                )
                for val in vals:
                    subselection = np.logical_or(
                        subselection,
                        (self.parent_molecule.information.atom_information[key] == val),
                    )

                # now "and" that with everything else
                selection = np.logical_and(selection, subselection)

            # now get the indices of the selection
            return np.nonzero(selection)[0]
        except Exception:
            print("ERROR: Could not make the selection. Existing fields:")
            print(
                "\t"
                + ", ".join(
                    self.parent_molecule.information.atom_information.dtype.names
                )
            )
            sys.exit(0)

    def select_atoms_in_bounding_box(self, bounding_box):
        """Selects all the atoms that are within a bounding box.

        Args:
            bounding_box: A 2x3 np.array containing the minimum and
                maximum points of the bounding box. Example:
                np.array([[min_x, min_y, min_z], [max_x, max_y, max_z]]).

        Returns:
            A np.array containing the indices of the atoms that are within
                the bounding box.

        """

        min_pt = bounding_box[0]
        max_pt = bounding_box[1]

        sel1 = np.nonzero(
            (self.parent_molecule.information.coordinates[:, 0] > min_pt[0])
        )[0]
        sel2 = np.nonzero(
            (self.parent_molecule.information.coordinates[:, 0] < max_pt[0])
        )[0]
        sel3 = np.nonzero(
            (self.parent_molecule.information.coordinates[:, 1] > min_pt[1])
        )[0]
        sel4 = np.nonzero(
            (self.parent_molecule.information.coordinates[:, 1] < max_pt[1])
        )[0]
        sel5 = np.nonzero(
            (self.parent_molecule.information.coordinates[:, 2] > min_pt[2])
        )[0]
        sel6 = np.nonzero(
            (self.parent_molecule.information.coordinates[:, 2] < max_pt[2])
        )[0]
        sel = np.intersect1d(sel1, sel2)
        sel = np.intersect1d(sel, sel3)
        sel = np.intersect1d(sel, sel4)
        sel = np.intersect1d(sel, sel5)
        sel = np.intersect1d(sel, sel6)

        return sel

    def select_all_atoms_bound_to_selection(self, selection):
        """Selects all the atoms that are bound to a user-specified selection.

        Args:
            selection: A np.array containing the indices of the
                user-specified selection.

        Returns:
            A np.array containing the indices of the atoms that are bound
                to the user-specified selection. Note that this new selection
                does not necessarily include the indices of the original
                user-specified selection.

        """

        if self.parent_molecule.information.bonds is None:
            print(
                "You need to define the bonds to use select_all_atoms_bound_to_selection()."
            )
            return

        bonds_to_consider = self.parent_molecule.information.bonds[selection]
        return np.unique(np.nonzero(bonds_to_consider)[1])

    def select_branch(self, root_atom_index, directionality_atom_index):
        """Identify an isolated "branch" of a molecular model. Assumes the
        atoms with indices root_atom_index and directionality_atom_index are
        bound to one another and that the branch starts at root_atom_index one
        and "points" in the direction of directionality_atom_index.

        Args:
            root_atom_index: An int, the index of the first atom in the branch
                (the "root").
            directionality_atom_index: An int, the index of the second atom in
                the branch, used to establish directionality

        Returns:
            A numpy array containing the indices of the atoms of the branch.

        """

        # note that this function is mostly retained for legacy reasons. the
        # old version of pymolecule had a branch-identification function.

        if self.parent_molecule.information.bonds is None:
            print(
                "To identify atoms in the same molecule as the atoms of a selection, you need to define the bonds."
            )
            return

        # Make sure atoms are neighboring
        if not directionality_atom_index in self.select_all_atoms_bound_to_selection(
            np.array([root_atom_index])
        ):
            print(
                "The root and directionality atoms, with indices "
                + str(root_atom_index)
                + " and "
                + str(directionality_atom_index)
                + ", respectively, are not neighboring atoms."
            )
            return

        # first, set up the two indices need to manage the growing list of
        # connected atoms. current_index is the index in the list that you're
        # currently considering
        current_index = 1

        # create an "empty" array to store the indices of the connected atoms
        # can't know ahead of time what size, so let's use a python list #
        # -99999 *
        # np.ones(len(self.parent_molecule.information.coordinates),
        # dtype=int) # assume initially that all the atoms belong to this
        # molecule. this list will be shortened, possibly, later if that
        # assumption is incorrect.
        indices_of_this_branch = [root_atom_index, directionality_atom_index]

        while True:
            # get all the neighbors of the current atom
            try:
                current_atom_index = indices_of_this_branch[current_index]
            except Exception:
                break  # this error because you've reached the end of the larger molecule

            neighbors_indices = (
                self.parent_molecule.selections.select_all_atoms_bound_to_selection(
                    np.array([current_atom_index])
                )
            )

            # get the ones in neighbors_indices that are not in
            # indices_of_this_molecule
            new_ones = np.setdiff1d(neighbors_indices, indices_of_this_branch)
            indices_of_this_branch.extend(new_ones)

            # prepare to look at the next atom in the list
            current_index = current_index + 1

        return np.array(indices_of_this_branch)

    def select_atoms_from_same_molecule(self, selection):
        """Selects all the atoms that belong to the same molecule as a
        user-defined selection, assuming that the pymolecule.Molecule object
        actually contains multiple physically distinct molecules that are not
        bound to each other via covalent bonds.

        Args:
            selection: A np.array containing the indices of the
                user-defined selection.

        Returns:
            A np.array containing the indices of the atoms belonging to the
                same molecules as the atoms of the user-defined selection.

        """

        # If your "Molecule" object actually contains several molecules, this
        # one selects all the atoms from any molecule containing any atom in
        # the selection note that bonds must be defined

        if self.parent_molecule.information.bonds is None:
            print(
                "To identify atoms in the same molecule as the atoms of a selection, you need to define the bonds."
            )
            return

        indices = []
        for index in selection:

            # first, set up the two indices need to manage the growing list of
            # connected atoms. current_index is the index in the list that
            # you're currently considering
            current_index = 0

            # create an "empty" array to store the indices of the connected
            # atoms can't know ahead of time what size, so let's use a python
            # list # -99999 *
            # np.ones(len(self.parent_molecule.information.coordinates),
            # dtype=int) # assume initially that all the atoms belong to this
            # molecule. this list will be shortened, possibly, later if that
            # assumption is incorrect.
            indices_of_this_molecule = [index]

            while True:
                # get all the neighbors of the current atom
                try:
                    current_atom_index = indices_of_this_molecule[current_index]
                except Exception:
                    break  # this error because you've reached the end of the larger molecule

                neighbors_indices = (
                    self.parent_molecule.selections.select_all_atoms_bound_to_selection(
                        np.array([current_atom_index])
                    )
                )

                # get the ones in neighbors_indices that are not in
                # indices_of_this_molecule
                new_ones = np.setdiff1d(neighbors_indices, indices_of_this_molecule)
                indices_of_this_molecule.extend(new_ones)

                # prepare to look at the next atom in the list
                current_index = current_index + 1

            # indices_of_this_molecule =
            # indices_of_this_molecule[:current_index-1] # so the list is
            # prunes down.
            indices.append(indices_of_this_molecule)

        # now merge and remove redundancies
        return np.unique(np.hstack(indices))

    def get_selections_of_constituent_molecules(self):
        """Identifies the indices of atoms belonging to separate molecules,
        assuming that the pymolecule.Molecule object actually contains multiple
        physically distinct molecules that are not bound to each other via
        covalent bonds.

        Returns:
            A python list of np.array objects containing the indices of the
                atoms belonging to each molecule of the composite
                pymolecule.Molecule object.

        """

        # If your pymolecule.Molecule object contains multiple molecules
        # (e.g., several chains), this will return a list of selections
        # corresponding to the atoms of each molecule

        atoms_not_yet_considered = self.select_all()
        selections = []

        while len(atoms_not_yet_considered) > 0:
            # add the atoms in the same molecule as the first atom in
            # atoms_not_yet_considered
            this_molecule_atoms = self.select_atoms_from_same_molecule(
                np.array([atoms_not_yet_considered[0]])
            )

            # now remove these from the atoms_not_yet_considered list
            atoms_not_yet_considered = np.setxor1d(
                this_molecule_atoms, atoms_not_yet_considered, True
            )

            # save the atoms of this molecule
            selections.append(this_molecule_atoms)

        return selections

    def select_atoms_near_other_selection(self, selection, cutoff):
        """Selects all atoms that are near the atoms of a user-defined
        selection.

        Args:
            selection: A np.array containing the indices of the
                user-defined selection.
            cutoff: A float, the distance cutoff (in Angstroms).

        Returns:
            A np.array containing the indices of all atoms near the
                    user-defined selection, not including the atoms of the
                    user-defined selection themselves.

        """

        # note that this does not return a selection that includes the input
        # selection. merge selections as required to get a selection that also
        # includes the input.

        invert_selection = self.invert_selection(selection)

        selection_coors = self.parent_molecule.information.coordinates[selection]
        inversion_coors = self.parent_molecule.information.coordinates[invert_selection]

        indices_of_nearby = invert_selection[
            np.unique(np.nonzero(cdist(inversion_coors, selection_coors) < cutoff)[0])
        ]
        return indices_of_nearby

    def select_atoms_in_same_residue(self, selection):
        """Selects all atoms that are in the same residue as any of the atoms
        of a user-defined seleciton. Residues are considered unique if they
        have a unique combination of resname, resseq, and chainid fields.

        Args:
            selection: A np.array containing the indices of the
                user-defined selection.

        Returns:
            A np.array containing the indices of all atoms in the same
                residue as any of the atoms of the user-defined selection.

        """

        # get string ids representing the residues of all atoms
        keys = np.char.add(
            self.parent_molecule.information.atom_information["resname_stripped"], "-"
        )
        keys = np.char.add(
            keys,
            np.array(
                [
                    str(t)
                    for t in self.parent_molecule.information.atom_information["resseq"]
                ]
            ),
        )
        keys = np.char.add(keys, "-")
        keys = np.char.add(
            keys, self.parent_molecule.information.atom_information["chainid_stripped"]
        )

        # get the unique keys of the selection
        unique_keys_of_selection = np.unique(keys[selection])

        # now get all the atoms of these selection keys

        # the below works, but is slow for large systems
        # residues = self.parent_molecule.selections.get_residue_selections()
        # new_selection = np.array([], dtype=int)
        # for key in unique_keys_of_selection:
        #    print key
        #    new_selection = np.append(new_selection, residues[key])

        # let's use this instead, faster for large systems.
        new_selection = np.array([], dtype=int)
        for key in unique_keys_of_selection:
            new_selection = np.append(new_selection, np.nonzero(keys == key)[0])

        return new_selection

    def invert_selection(self, selection):
        """Inverts a user-defined selection (i.e., identifies all atoms that
        are not in the seleciton).

        Args:
            selection: A np.array containing the indices of the
                user-defined selection.

        Returns:
            A np.array containing the indices of all atoms that are not in
                the user-defined seleciton.

        """

        # selection is a list of atom indices
        all_atoms = np.arange(
            0, len(self.parent_molecule.information.atom_information), 1, dtype=int
        )
        remaining_indices = np.delete(all_atoms, selection)
        return remaining_indices

    def select_all(self):
        """Selects all the atoms in a pymolecule.Molecule object.

        Returns:
            A np.array containing the indices of all atoms in the
                pymolecule.Molecule object.

        """

        return self.select_atoms({})

    def select_close_atoms_from_different_molecules(
        self, other_mol, cutoff, pairwise_comparison=True, terminate_early=False
    ):
        """Effectively detects steric clashes between self and another
        pymolecule.Molecule.

        Args:
            other_mol: A pymolecule.Molecule object of the other molecule.
            cutoff: A float, the user-defined distance cutoff in Angstroms.
            pairwise_comparison: An optional boolean, whether or not to
                perform a simple pairwise distance comparison (if True) or to
                use a more sophisitcated method (if False). True by default.
            terminate_early = An optional boolean, whether or not to stop
                looking for steric clashes once one is found. False by default.

        Returns:
            A tuple containing two elements. The first is a np.array
                containing the indices of all nearby atoms from this
                pymolecule.Molecule object (self). The second is a np.array
                containing the indices of all nearby atoms from the other
                molecule.

        """

        if pairwise_comparison:

            dists = cdist(
                self.parent_molecule.information.coordinates,
                other_mol.information.coordinates,
            )
            close_ones = np.nonzero(dists < cutoff)
            close_ones_from_mol_parent_molecule = np.unique(close_ones[0])
            close_ones_from_mol_other_mol = np.unique(close_ones[1])

            return (close_ones_from_mol_parent_molecule, close_ones_from_mol_other_mol)
        else:  # so do the more complex hierarchical comparison
            # first, do some quick and easy checks
            margin = np.array([cutoff, cutoff, cutoff])
            self_min = np.min(self.parent_molecule.information.coordinates, 0) - margin
            other_mol_max = np.max(other_mol.information.coordinates, 0) + margin

            if self_min[0] > other_mol_max[0]:
                return (np.array([]), np.array([]))
            if self_min[1] > other_mol_max[1]:
                return (np.array([]), np.array([]))
            if self_min[2] > other_mol_max[2]:
                return (np.array([]), np.array([]))

            self_max = np.max(self.parent_molecule.information.coordinates, 0) + margin
            other_mol_min = np.min(other_mol.information.coordinates, 0) - margin

            if other_mol_min[0] > self_max[0]:
                return (np.array([]), np.array([]))
            if other_mol_min[1] > self_max[1]:
                return (np.array([]), np.array([]))
            if other_mol_min[2] > self_max[2]:
                return (np.array([]), np.array([]))

            # now assign spheres to the whole molecule, the chains, the
            # residues note that this won't recalculate the data if it's
            # already been calculated
            self.parent_molecule.information.define_molecule_chain_residue_spherical_boundaries()
            other_mol.information.define_molecule_chain_residue_spherical_boundaries()

            # if the whole molecules are too far away, give up
            self_cent = self.parent_molecule.information.hierarchy["spheres"][
                "molecule"
            ]["center"]
            self_rad = self.parent_molecule.information.hierarchy["spheres"][
                "molecule"
            ]["radius"]
            other_cent = other_mol.information.hierarchy["spheres"]["molecule"][
                "center"
            ]
            other_rad = other_mol.information.hierarchy["spheres"]["molecule"]["radius"]
            mol_dist = np.linalg.norm(self_cent - other_cent)

            if mol_dist > self_rad + other_rad + cutoff:
                # the molecules are too far away to clash
                return (np.array([]), np.array([]))

            # check the chains
            chain_distances = cdist(
                self.parent_molecule.information.hierarchy["spheres"]["chains"][
                    "centers"
                ],
                other_mol.information.hierarchy["spheres"]["chains"]["centers"],
            )
            sum1_matrix = np.hstack(
                [
                    np.array(
                        [
                            self.parent_molecule.information.hierarchy["spheres"][
                                "chains"
                            ]["radii"]
                        ]
                    ).T
                    for t in range(
                        len(
                            other_mol.information.hierarchy["spheres"]["chains"][
                                "radii"
                            ]
                        )
                    )
                ]
            )
            sum2_matrix = np.vstack(
                [
                    np.array(
                        [other_mol.information.hierarchy["spheres"]["chains"]["radii"]]
                    )
                    for t in range(
                        len(
                            self.parent_molecule.information.hierarchy["spheres"][
                                "chains"
                            ]["radii"]
                        )
                    )
                ]
            )
            sum_matrix = sum1_matrix + sum2_matrix + cutoff
            indices_of_clashing_chains = np.nonzero(chain_distances < sum_matrix)

            if len(indices_of_clashing_chains[0]) == 0:
                # the chains don't clash, so no atoms can either
                return (np.array([]), np.array([]))

            # check the residues
            residue_distances = cdist(
                self.parent_molecule.information.hierarchy["spheres"]["residues"][
                    "centers"
                ],
                other_mol.information.hierarchy["spheres"]["residues"]["centers"],
            )
            sum1_matrix = np.hstack(
                [
                    np.array(
                        [
                            self.parent_molecule.information.hierarchy["spheres"][
                                "residues"
                            ]["radii"]
                        ]
                    ).T
                    for t in range(
                        len(
                            other_mol.information.hierarchy["spheres"]["residues"][
                                "radii"
                            ]
                        )
                    )
                ]
            )
            sum2_matrix = np.vstack(
                [
                    np.array(
                        [
                            other_mol.information.hierarchy["spheres"]["residues"][
                                "radii"
                            ]
                        ]
                    )
                    for t in range(
                        len(
                            self.parent_molecule.information.hierarchy["spheres"][
                                "residues"
                            ]["radii"]
                        )
                    )
                ]
            )
            sum_matrix = sum1_matrix + sum2_matrix + cutoff

            indices_of_clashing_residues = np.nonzero(residue_distances < sum_matrix)

            if len(indices_of_clashing_residues[0]) == 0:
                # the residues don't clash, so no atoms can either
                return (np.array([]), np.array([]))

            # now time to check the atoms
            self_close_atom_indices = np.array([], dtype=int)
            other_close_atom_indices = np.array([], dtype=int)

            for i in range(len(indices_of_clashing_residues[0])):
                self_res_index = indices_of_clashing_residues[0][i]
                other_res_index = indices_of_clashing_residues[1][i]

                self_res_name = self.parent_molecule.information.hierarchy["spheres"][
                    "residues"
                ]["keys"][self_res_index]
                other_res_name = other_mol.information.hierarchy["spheres"]["residues"][
                    "keys"
                ][other_res_index]

                self_res_indices = self.parent_molecule.information.hierarchy[
                    "residues"
                ]["indices"][self_res_name]
                other_res_indices = other_mol.information.hierarchy["residues"][
                    "indices"
                ][other_res_name]

                self_coors = self.parent_molecule.information.coordinates[
                    self_res_indices
                ]
                other_coors = other_mol.information.coordinates[other_res_indices]

                some_self_indices, some_other_indices = np.nonzero(
                    cdist(self_coors, other_coors) < cutoff
                )
                if (
                    len(some_self_indices) != 0 or len(some_other_indices) != 0
                ):  # so there are some
                    self_close_atom_indices = np.append(
                        self_close_atom_indices, self_res_indices[some_self_indices]
                    )
                    other_close_atom_indices = np.append(
                        other_close_atom_indices, other_res_indices[some_other_indices]
                    )

                    if (
                        terminate_early
                    ):  # so don't keep looking once you've found something
                        return (self_close_atom_indices, other_close_atom_indices)

            # so nothing was found in the end
            return (
                np.unique(self_close_atom_indices),
                np.unique(other_close_atom_indices),
            )

    def create_molecule_from_selection(
        self, selection, serial_reindex=True, resseq_reindex=False
    ):
        """Creates a pymolecule.Molecule from a user-defined atom selection.

        Args:
            selection: A np.array containing the indices of the atoms in
                the user-defined selection.
            serial_reindex: An optional boolean, whether or not to reindex
                the atom serial fields. Default is True.
            resseq_reindex: An optional boolean, whether or not to reindex
                the atom resseq fields. Default is False.

        Returns:
            A pymolecule.Molecule object containing the atoms of the
                user-defined selection.

        """

        new_mol = Molecule()
        new_mol.information.coordinates = self.parent_molecule.information.coordinates[
            selection
        ]

        # try to get the undo coordinates as well, though they may not have
        # been set
        try:
            new_mol.information.coordinates_undo_point = (
                self.parent_molecule.information.coordinates_undo_point[selection]
            )
        except Exception:
            new_mol.information.coordinates_undo_point = None

        new_mol.information.atom_information = (
            self.parent_molecule.information.atom_information[selection]
        )

        if not self.parent_molecule.information.bonds is None:
            new_mol.information.bonds = self.parent_molecule.information.bonds[
                selection
            ]
            new_mol.information.bonds = new_mol.information.bonds[:, selection]
        else:
            new_mol.information.bonds = None

        # note that hierarchy will have to be recalculated

        if serial_reindex:
            new_mol.io.serial_reindex()
        if resseq_reindex:
            new_mol.io.resseq_reindex()
        return new_mol

    def copy(self):
        """Returns an exact copy (pymolecule.Molecule) of this Molecule object.
        Undo points are NOT copied.

        Returns:
            A pymolecule.Molecule, containing to the same atomic information as
                this pymolecule.Molecule object.

        """

        new_molecule = Molecule()
        new_molecule.information.filename = self.parent_molecule.information.filename
        new_molecule.information.remarks = self.parent_molecule.information.remarks[:]
        new_molecule.information.atom_information = (
            self.parent_molecule.information.atom_information.copy()
        )
        new_molecule.information.coordinates = (
            self.parent_molecule.information.coordinates.copy()
        )

        try:
            new_molecule.information.coordinates_undo_point = (
                self.parent_molecule.information.coordinates_undo_point.copy()
            )
        except Exception:
            new_molecule.information.coordinates_undo_point = (
                self.parent_molecule.information.coordinates_undo_point
            )

        if not self.parent_molecule.information.bonds is None:
            new_molecule.information.bonds = (
                self.parent_molecule.information.bonds.copy()
            )
        else:
            new_molecule.information.bonds = None

        new_molecule.information.hierarchy = copy.deepcopy(
            self.parent_molecule.information.hierarchy
        )

        return new_molecule

    def get_chain_selections(self):
        """Identifies the atom selections of each chain.

        Returns:
            A dictionary. The keys of the dictionary correspond to the
                chainids, and the values are np.array objects containing the
                indices of the associated chain atoms.

        """

        # so it hasn't already been calculated
        if not "chains" in list(self.parent_molecule.information.hierarchy.keys()):
            unique_chainids = np.unique(
                self.parent_molecule.information.atom_information["chainid_stripped"]
            )

            self.parent_molecule.information.hierarchy["chains"] = {}
            self.parent_molecule.information.hierarchy["chains"]["indices"] = {}
            for chainid in unique_chainids:
                self.parent_molecule.information.hierarchy["chains"]["indices"][
                    chainid
                ] = self.parent_molecule.selections.select_atoms(
                    {"chainid_stripped": chainid}
                )

        return self.parent_molecule.information.hierarchy["chains"]["indices"]

    def get_residue_selections(self):
        """Identifies the atom selections of each residue.

        Returns:
            A dictionary. The keys of this dictionary correspond to the unique
                resname-resseq-chainid residue identifiers, and the values are
                np.array objects containing the indices of the associated
                residue atoms.

        """

        # so it hasn't already been calculated
        if not "residues" in list(self.parent_molecule.information.hierarchy.keys()):

            keys = np.char.add(
                self.parent_molecule.information.atom_information["resname_stripped"],
                "-",
            )
            keys = np.char.add(
                keys,
                np.array(
                    [
                        str(t)
                        for t in self.parent_molecule.information.atom_information[
                            "resseq"
                        ]
                    ]
                ),
            )
            keys = np.char.add(keys, "-")
            keys = np.char.add(
                keys,
                self.parent_molecule.information.atom_information["chainid_stripped"],
            )

            unique_resnames = np.unique(keys)

            self.parent_molecule.information.hierarchy["residues"] = {}
            self.parent_molecule.information.hierarchy["residues"]["indices"] = {}
            for key in unique_resnames:
                resname, resseq, chainid = key.split("-")
                resseq = int(resseq)

                self.parent_molecule.information.hierarchy["residues"]["indices"][
                    key
                ] = self.parent_molecule.selections.select_atoms(
                    {
                        "chainid_stripped": chainid,
                        "resname_stripped": resname,
                        "resseq": resseq,
                    }
                )

        return self.parent_molecule.information.hierarchy["residues"]["indices"]

    def get_molecule_from_selection(self, selection):
        """Creates a Molecule from a user-defined atom selection.

        Args:
            selection: A np.array containing the indices of the atoms in
                the user-defined selection.

        Returns:
            A Molecule object containing the atoms of the user-defined
                selection.

        """

        new_mol = Molecule()
        new_mol.information.set_coordinates(
            self.parent_molecule.information.get_coordinates()[selection]
        )
        new_mol.information.set_atom_information(
            self.parent_molecule.information.get_atom_information()[selection]
        )

        # note that hierarchy will have to be recalculated

        return new_mol
