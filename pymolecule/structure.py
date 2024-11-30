import pickle as pickle

import numpy as np
import scipy
from loguru import logger
from scipy.spatial.distance import pdist


class AtomsAndBonds:
    """A class for adding and deleting atoms and bonds"""

    def __init__(self, parent_molecule_object):
        """Initializes the pymolecule.AtomsAndBonds class.

        Args:
            parent_molecule_object: The pymolecule.Molecule object
                associated with this class.

        """

        self.parent_molecule = parent_molecule_object

    def create_bonds_by_distance(
        self, remove_old_bond_data=True, delete_excessive_bonds=True
    ):
        """Determines which atoms are bound to each other based on their
        proximity.

        Args:
            remove_old_bond_data: An optional boolean, whether or not to
                discard old bond data before adding in bonds determined by
                distance. True by default.
            delete_excessive_bonds: An optional boolean, whether or not to
                check for and delete excessive bonds. True by default.

        """

        # create/recreate the bond array if needed
        if remove_old_bond_data or self.parent_molecule.information.bonds is None:
            self.parent_molecule.information.bonds = np.zeros(
                (
                    len(self.parent_molecule.information.atom_information),
                    len(self.parent_molecule.information.atom_information),
                )
            )

        # get the longest bond length on record
        max_bond_length = np.max(
            [
                self.parent_molecule.information.constants["bond_length_dict"][key]
                for key in list(
                    self.parent_molecule.information.constants[
                        "bond_length_dict"
                    ].keys()
                )
            ]
        )

        # which ones could possibly be bound (less than the max_bond_length)
        distances = scipy.spatial.distance.squareform(
            pdist(self.parent_molecule.information.coordinates)
        )
        ones_to_consider = np.nonzero(distances < max_bond_length)

        for index in range(len(ones_to_consider[0])):
            index1 = ones_to_consider[0][index]
            index2 = ones_to_consider[1][index]
            element_1 = self.parent_molecule.information.atom_information[
                "element_stripped"
            ][index1].decode("utf-8")
            element_2 = self.parent_molecule.information.atom_information[
                "element_stripped"
            ][index2].decode("utf-8")

            if index1 != index2:  # so an atom is not bound to itself.parent_molecule
                key = f"{element_1}-{element_2}"

                try:
                    bond_dist = self.parent_molecule.information.constants[
                        "bond_length_dict"
                    ][key]
                except Exception:
                    logger.warning(f"Unknown bond between {key}")
                    logger.warning(
                        f"Assuming the maximum bond length of {max_bond_length}"
                    )

                # so they should be bonded
                if (
                    distances[index1][index2] < bond_dist * 1.2
                    and distances[index1][index2] > bond_dist * 0.5
                ):
                    self.parent_molecule.information.bonds[index1][index2] = 1
                    self.parent_molecule.information.bonds[index2][index1] = 1

        if delete_excessive_bonds:
            # now do a sanity check. C cannot have more than 4 bonds, O cannot
            # have more than 2, and N cannot have more than 2 if more, than
            # use ones closest to ideal bond length
            for index in range(len(self.parent_molecule.information.atom_information)):
                # get the info of the inde xatom
                element = self.parent_molecule.information.atom_information[
                    "element_stripped"
                ][index].decode("utf-8")
                bond_partner_indices = (
                    self.parent_molecule.selections.select_all_atoms_bound_to_selection(
                        np.array([index])
                    )
                )
                number_of_bonds = len(bond_partner_indices)

                try:
                    # so this atom has too many bonds
                    if (
                        number_of_bonds
                        > self.parent_molecule.information.constants[
                            "max_number_of_bonds_permitted"
                        ][element]
                    ):
                        # get the distances of this atoms bonds
                        dists = distances[index][bond_partner_indices]

                        # get the ideal distances of those bonds initialize
                        # the vector
                        ideal_dists = np.empty(len(dists))

                        # populate the ideal-bond-length vector
                        for t in range(len(bond_partner_indices)):
                            index_partner = bond_partner_indices[t]
                            element_partner = (
                                self.parent_molecule.information.atom_information[
                                    "element_stripped"
                                ][index_partner]
                            )
                            ideal_dists[t] = self.parent_molecule.information.constants[
                                "bond_length_dict"
                            ][element + "-" + element_partner]
                            # print element, element_partner

                        diff = np.absolute(dists - ideal_dists)  # get the distance

                        # identify the bonds to discard
                        indices_in_order = diff.argsort()
                        indices_to_throw_out = indices_in_order[
                            self.parent_molecule.information.constants[
                                "max_number_of_bonds_permitted"
                            ][element] :
                        ]
                        indices_to_throw_out = bond_partner_indices[
                            indices_to_throw_out
                        ]

                        # discard the extra bonds
                        for throw_out_index in indices_to_throw_out:
                            self.parent_molecule.information.bonds[index][
                                throw_out_index
                            ] = 0
                            self.parent_molecule.information.bonds[throw_out_index][
                                index
                            ] = 0

                except Exception:
                    pass  # element probably wasn't in the dictionary

    def number_of_bond_partners_of_element(self, atom_index, the_element):
        """Counts the number of atoms of a given element bonded to a specified
        atom of interest.

        Args:
            atom_index: An int, the index of the atom of interest.
            the_element: A string describing the element of the neighbors to
                be counted.

        Returns:
            An int, the number of neighboring atoms of the specified element.

        """

        # this function is really here for historical reasons. it's similar to
        # the old number_of_neighors_of_element function. it could be done
        # pretty easily with numpy

        the_element = the_element.strip()
        bond_partners_selection = (
            self.parent_molecule.selections.select_all_atoms_bound_to_selection(
                np.array([atom_index])
            )
        )
        elements = self.parent_molecule.information.atom_information[
            "element_stripped"
        ][bond_partners_selection]
        return len(np.nonzero(elements == the_element)[0])

    def index_of_first_bond_partner_of_element(self, atom_index, the_element):
        """For a given atom of interest, returns the index of the first
        neighbor of a specified element.

        Args:
            atom_index: An int, the index of the atom of interest.
            the_element: A string specifying the desired element of the
                neighbor.

        Returns:
            An int, the index of the first neighbor atom of the specified
                element. If no such neighbor exists, returns -1.

        """

        # this function is really here for historical reasons. it's similar to
        # the old index_of_neighbor_of_element function. it could be done
        # pretty easily with numpy

        the_element = the_element.strip()
        bond_partners_selection = (
            self.parent_molecule.selections.select_all_atoms_bound_to_selection(
                np.array([atom_index])
            )
        )
        elements = self.parent_molecule.information.atom_information[
            "element_stripped"
        ][bond_partners_selection]
        return bond_partners_selection[np.nonzero(elements == the_element)[0]][0]

    def delete_bond(self, index1, index2):
        """Deletes a bond.

        Args:
            index1: An int, the index of the first atom of the bonded pair.
            index2: An int, the index of the second atom of the bonded pair.

        """

        self.parent_molecule.information.bonds[index1][index2] = 0
        self.parent_molecule.information.bonds[index2][index1] = 0

    def add_bond(self, index1, index2, order=1):
        """Adds a bond.

        Args:
            index1: An int, the index of the first atom of the bonded pair.
            index2: An int, the index of the second atom of the bonded pair.
            order: An optional int, the order of the bond. 1 by default.

        """

        self.parent_molecule.information.bonds[index1][index2] = order
        self.parent_molecule.information.bonds[index2][index1] = order

    def delete_atom(self, index):
        """Deletes an atom.

        Args:
            index: An int, the index of the atom to delete.

        """

        # remove the atom information
        self.parent_molecule.information.atom_information = np.delete(
            self.parent_molecule.information.atom_information, index
        )

        # remove the coordinates
        self.parent_molecule.information.coordinates = np.delete(
            self.parent_molecule.information.coordinates, index, axis=0
        )
        try:
            self.parent_molecule.information.coordinates_undo_point = np.delete(
                self.parent_molecule.information.coordinates_undo_point, index, axis=0
            )
        except Exception:
            pass

        # remove the relevant bonds
        self.parent_molecule.information.bonds = np.delete(
            self.parent_molecule.information.bonds, index, 0
        )
        self.parent_molecule.information.bonds = np.delete(
            self.parent_molecule.information.bonds, index, 1
        )

        # the hierarchy will have to be recomputed
        self.hierarchy = {}

    def add_atom(
        self,
        record_name="ATOM",
        serial=1,
        name="X",
        resname="XXX",
        chainid="X",
        resseq=1,
        occupancy=0.0,
        tempfactor=0.0,
        charge="",
        element="X",
        coordinates=np.array([0.0, 0.0, 0.0]),
    ):
        """Adds an atom.

        Args:
            record_name: An optional string, the record name of the atom.
                "ATOM" is the default.
            serial: An optional int, the serial field of the atom. 1 is the
                default.
            name: An optional string, the name of the atom. "X" is the
                default.
            resname: An optional string, the resname of the atom. "XXX" is
                the default.
            chainid: An optional string, chainid of the atom. "X" is the
                default.
            resseq: An optional int, the resseq field of the atom. 1 is the
                default.
            occupancy: An optional float, the occupancy of the atom. 0.0 is
                the default.
            tempfactor: An optional float, the tempfactor of the atom. 0.0
                is the default.
            charge: An optional string, the charge of the atom. "" is the
                default.
            element: An optional string, the element of the atom. "X" is the
                default.
            coordinates: An optional np.array, the (x, y, z) coordinates
                of the atom. np.array([0.0, 0.0, 0.0]) is the default.

        """

        # add the atom information

        if len(record_name) < 6:
            record_name = record_name.ljust(6)
        if len(name) < 5:
            if len(name) < 4:
                name = name.rjust(4) + " "
            else:
                name = name.rjust(5)
        if len(resname) < 4:
            resname = resname.rjust(4)
        if len(chainid) < 2:
            chainid = chainid.rjust(2)
        if len(charge) < 2:
            charge = charge.ljust(2)
        if len(element) < 2:
            element = element.rjust(2)

        name_stripped = name.strip()
        resname_stripped = resname.strip()
        chainid_stripped = chainid.strip()
        element_stripped = element.strip()

        try:
            mass = self.parent_molecule.information.constants["mass_dict"][
                element_stripped
            ]
        except Exception:
            mass = 0.0

        # if there is no atom_information, you need to create it.
        if self.parent_molecule.information.atom_information is None:
            self.parent_molecule.information.atom_information = np.zeros(
                (1,),
                dtype=[
                    ("record_name", "|S6"),
                    ("serial", "<i8"),
                    ("name", "|S5"),
                    ("resname", "|S4"),
                    ("chainid", "|S2"),
                    ("resseq", "<i8"),
                    ("occupancy", "<f8"),
                    ("tempfactor", "<f8"),
                    ("element", "|S2"),
                    ("charge", "|S2"),
                    ("name_stripped", "|S5"),
                    ("resname_stripped", "|S4"),
                    ("chainid_stripped", "|S2"),
                    ("element_stripped", "|S2"),
                ],
            )

        self.parent_molecule.information.atom_information = np.ma.resize(
            self.parent_molecule.information.atom_information,
            self.parent_molecule.information.total_number_of_atoms() + 1,
        )
        self.parent_molecule.information.atom_information["record_name"][
            -1
        ] = record_name
        self.parent_molecule.information.atom_information["name"][-1] = name
        self.parent_molecule.information.atom_information["resname"][-1] = resname
        self.parent_molecule.information.atom_information["chainid"][-1] = chainid
        self.parent_molecule.information.atom_information["charge"][-1] = charge
        self.parent_molecule.information.atom_information["element"][-1] = element
        self.parent_molecule.information.atom_information["name_stripped"][
            -1
        ] = name_stripped
        self.parent_molecule.information.atom_information["resname_stripped"][
            -1
        ] = resname_stripped
        self.parent_molecule.information.atom_information["chainid_stripped"][
            -1
        ] = chainid_stripped
        self.parent_molecule.information.atom_information["element_stripped"][
            -1
        ] = element_stripped
        self.parent_molecule.information.atom_information["serial"][-1] = serial
        self.parent_molecule.information.atom_information["resseq"][-1] = resseq
        self.parent_molecule.information.atom_information["occupancy"][-1] = occupancy
        self.parent_molecule.information.atom_information["tempfactor"][-1] = tempfactor

        if "mass" in self.parent_molecule.information.atom_information.dtype.names:
            self.parent_molecule.information.atom_information["mass"][-1] = mass

        # now add the coordinates
        if self.parent_molecule.information.coordinates is None:
            self.parent_molecule.information.coordinates = np.array([coordinates])
        else:
            self.parent_molecule.information.coordinates = np.vstack(
                (self.parent_molecule.information.coordinates, coordinates)
            )

        # now add places for bonds, though bonds will only be added if done
        # explicitly, not here
        if self.parent_molecule.information.bonds is None:
            self.parent_molecule.information.bonds = np.array([[0]])
        else:
            self.parent_molecule.information.bonds = np.vstack(
                (
                    self.parent_molecule.information.bonds,
                    np.zeros(
                        self.parent_molecule.information.total_number_of_atoms() - 1
                    ),
                )
            )

            self.parent_molecule.information.bonds = np.hstack(
                (
                    self.parent_molecule.information.bonds,
                    np.zeros(
                        (1, self.parent_molecule.information.total_number_of_atoms())
                    ).T,
                )
            )
