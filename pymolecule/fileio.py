import os
import pickle as pickle
import shutil

import numpy as np
from numpy.lib.recfunctions import append_fields


class FileIO:
    """A class for saving and loading molecular data into a pymolecule.Molecule
    object"""

    def __init__(self, parent_molecule_object):
        """Initializes the pymolecule.io class.

        Args:
            parent_molecule_object: The pymolecule.Molecule object
                associated with this class.

        """

        self.parent_molecule = parent_molecule_object

    def load_pym_into(self, filename):
        """Loads the molecular data contained in a pym file into the current
        pymolecule.Molecule object.

        Args:
            filename: A string, the filename of the pym file.

        """

        if filename[-1:] != "/":
            filename = filename + "/"

        # first, get the files that must exist
        self.parent_molecule.information.atom_information = pickle.load(
            open(filename + "atom_information", "rb")
        )
        self.parent_molecule.information.coordinates = np.load(
            filename + "coordinates.npz"
        )["arr_0"]

        # now look for other possible files (optional output)
        if os.path.exists(filename + "remarks"):
            self.parent_molecule.information.remarks = pickle.load(
                open(filename + "remarks", "rb")
            )
        if os.path.exists(filename + "hierarchy"):
            self.parent_molecule.information.hierarchy = pickle.load(
                open(filename + "hierarchy", "rb")
            )
        if os.path.exists(filename + "filename"):
            self.parent_molecule.information.filename = pickle.load(
                open(filename + "filename", "rb")
            )
        if os.path.exists(filename + "bonds.npz"):
            self.parent_molecule.information.bonds = np.load(filename + "bonds.npz")[
                "arr_0"
            ]
        if os.path.exists(filename + "coordinates_undo_point.npz"):
            self.parent_molecule.information.coordinates_undo_point = np.load(
                filename + "coordinates_undo_point.npz"
            )["arr_0"]

        # self.parent_molecule.information = pickle.load( open( filename, "rb" ) )

    def load_pdb_into(
        self,
        filename,
        bonds_by_distance=True,
        serial_reindex=True,
        resseq_reindex=False,
    ):
        """Loads the molecular data contained in a pdb file into the current
        pymolecule.Molecule object.

        Args:
            filename: A string, the filename of the pdb file.
            bonds_by_distance: An optional boolean, whether or not to
                determine atomic bonds based on atom proximity. True by
                default.
            serial_reindex: An optional boolean, whether or not to reindex
                the pdb serial field. True by default.
            resseq_reindex: An optional boolean, whether or not to reindex
                the pdb resseq field. False by default.

        """

        self.parent_molecule.information.filename = filename

        # open/read the file
        afile = open(filename, "r")
        self.load_pdb_into_using_file_object(
            afile, bonds_by_distance, serial_reindex, resseq_reindex
        )
        afile.close()

    def load_pdb_into_using_file_object(
        self,
        file_obj,
        bonds_by_distance=True,
        serial_reindex=True,
        resseq_reindex=False,
    ):
        """Loads molecular data from a python file object (pdb formatted) into
        the current pymolecule.Molecule object. Note that most users will want
        to use the load_pdb_into() function instead, which is identical except
        that it accepts a filename string instead of a python file object.

        Args:
            file_obj: A python file object, containing pdb-formatted data.
            bonds_by_distance: An optional boolean, whether or not to
                determine atomic bonds based on atom proximity. True by
                default.
            serial_reindex: An optional boolean, whether or not to reindex
                the pdb serial field. True by default.
            resseq_reindex: An optional boolean, whether or not to reindex
                the pdb resseq field. False by default.
        """

        source_data = np.genfromtxt(
            file_obj,
            dtype="S6,S5,S5,S5,S1,S4,S4,S8,S8,S8,S6,S6,S10,S2,S3",
            names=[
                "record_name",
                "serial",
                "name",
                "resname",
                "chainid",
                "resseq",
                "empty",
                "x",
                "y",
                "z",
                "occupancy",
                "tempfactor",
                "empty2",
                "element",
                "charge",
            ],
            delimiter=[6, 5, 5, 5, 1, 4, 4, 8, 8, 8, 6, 6, 10, 2, 3],
        )

        # get the remarks, if any. good to hold on to this because some of my
        # programs might retain info via remarks
        remark_indices = np.nonzero(source_data["record_name"] == b"REMARK")[0]
        self.parent_molecule.information.remarks = []
        for index in remark_indices:
            astr = ""
            for name in source_data.dtype.names[1:]:
                astr = astr + source_data[name][index].decode("utf-8")
            self.parent_molecule.information.remarks.append(astr.rstrip())

        if source_data.ndim == 0:
            # in case the pdb file has only one line
            source_data = source_data.reshape(1, -1)

        # get the ones that are ATOM or HETATOM in the record_name
        or_matrix = np.logical_or(
            (source_data["record_name"] == b"ATOM  "),
            (source_data["record_name"] == b"HETATM"),
        )
        indices_of_atom_or_hetatom = np.nonzero(or_matrix)[0]
        self.parent_molecule.information.atom_information = source_data[
            indices_of_atom_or_hetatom
        ]

        # now, some of the data needs to change types first, fields that
        # should be numbers cannot be empty strings
        for field in (
            self.parent_molecule.information.constants["i8_fields"]
            + self.parent_molecule.information.constants["f8_fields"]
        ):
            check_fields = self.parent_molecule.information.atom_information[field]
            check_fields = np.char.strip(check_fields)
            indices_of_empty = np.nonzero(check_fields == "")[0]
            self.parent_molecule.information.atom_information[field][
                indices_of_empty
            ] = "0"

        # now actually change the type
        old_types = self.parent_molecule.information.atom_information.dtype
        descr = old_types.descr
        for field in self.parent_molecule.information.constants["i8_fields"]:
            index = self.parent_molecule.information.atom_information.dtype.names.index(
                field
            )
            descr[index] = (descr[index][0], "i8")
        for field in self.parent_molecule.information.constants["f8_fields"]:
            index = self.parent_molecule.information.atom_information.dtype.names.index(
                field
            )
            descr[index] = (descr[index][0], "f8")
        new_types = np.dtype(descr)
        self.parent_molecule.information.atom_information = (
            self.parent_molecule.information.atom_information.astype(new_types)
        )

        # remove some of the fields that just contain empty data
        self.parent_molecule.information.atom_information = (
            self.parent_molecule.numpy_structured_array_remove_field(
                self.parent_molecule.information.atom_information, ["empty", "empty2"]
            )
        )

        # the coordinates need to be placed in their own special numpy array
        # to facilitate later manipulation
        self.parent_molecule.information.coordinates = np.vstack(
            [
                self.parent_molecule.information.atom_information["x"],
                self.parent_molecule.information.atom_information["y"],
                self.parent_molecule.information.atom_information["z"],
            ]
        ).T
        self.parent_molecule.information.atom_information = self.parent_molecule.numpy_structured_array_remove_field(
            self.parent_molecule.information.atom_information,
            ["x", "y", "z"],
            # now remove the coordinates from the atom_information object to
            # save memory
        )

        # now determine element from atom name for those entries where it's
        # not given note that the
        # molecule.information.assign_elements_from_atom_names function can be
        # used to overwrite this and assign elements based on the atom name
        # only.
        indices_where_element_is_not_defined = np.nonzero(
            np.char.strip(self.parent_molecule.information.atom_information["element"])
            == b""
        )[0]

        self.parent_molecule.information.assign_elements_from_atom_names(
            indices_where_element_is_not_defined
        )

        # string values in self.parent_molecule.information.atom_information
        # should also be provided in stripped format for easier comparison
        fields_to_strip = ["name", "resname", "chainid", "element"]
        for f in fields_to_strip:
            self.parent_molecule.information.atom_information = append_fields(
                self.parent_molecule.information.atom_information,
                f + "_stripped",
                data=np.char.strip(
                    self.parent_molecule.information.atom_information[f]
                ),
            )

        # now, if there's conect data, load it. this part of the code is not
        # that "numpyic"
        conect_indices = np.nonzero(source_data["record_name"] == b"CONECT")[0]
        if len(conect_indices) > 0:
            self.parent_molecule.information.bonds = np.zeros(
                (
                    len(self.parent_molecule.information.atom_information),
                    len(self.parent_molecule.information.atom_information),
                )
            )

            # build serial to index mapping
            serial_to_index = {}
            for index, inf in enumerate(
                self.parent_molecule.information.atom_information["serial"]
            ):
                serial_to_index[inf] = index  # is there a faster way?

            # get the connect data
            for index in conect_indices:
                astr = ""
                for name in source_data.dtype.names[1:]:
                    astr = astr + source_data[name][index].decode("utf-8")
                astr = astr.rstrip()

                indices = []
                for i in range(0, len(astr), 5):
                    indices.append(serial_to_index[int(astr[i : i + 5])])

                for partner_index in indices[1:]:
                    self.parent_molecule.information.bonds[indices[0]][
                        partner_index
                    ] = 1
                    self.parent_molecule.information.bonds[partner_index][
                        indices[0]
                    ] = 1
        # else: # create empty bond array
        #    self.parent_molecule.information.bonds = np.zeros((len(self.parent_molecule.information.atom_information), len(self.parent_molecule.information.atom_information)))

        if bonds_by_distance:
            self.parent_molecule.atoms_and_bonds.create_bonds_by_distance(False)
        if serial_reindex:
            self.serial_reindex()
        if resseq_reindex:
            self.resseq_reindex()

    def save_pym(
        self,
        filename,
        save_bonds=False,
        save_filename=False,
        save_remarks=False,
        save_hierarchy=False,
        save_coordinates_undo_point=False,
    ):
        """Saves the molecular data contained in a pymolecule.Molecule object
        to a pym file.

        Args:
            filename: An string, the filename to use for saving. (Note that
                this is actually a directory, not a file.)
            save_bonds: An optional boolean, whether or not to save
                information about atomic bonds. False by default.
            save_filename: An optional boolean, whether or not to save the
                original (pdb) filename. False by default.
            save_remarks: An optional boolean, whether or not to save remarks
                associated with the molecule. False by default.
            save_hierarchy: An optional boolean, whether or not to save
                information about spheres the bound (encompass) the whole
                molecule, the chains, and the residues. False by default.
            save_coordinates_undo_point: An optional boolean, whether or not
                to save the last coordinate undo point. False by default.

        """

        # Why not just pickle self.parent.information? Because it's a huge
        # file, can't selectively not save bonds, for example, and np.save
        # is faster than cPickle protocol 2 on numpy arrays

        # if the directory already exists, first delete it
        if os.path.exists(filename):
            try:
                shutil.rmtree(filename)
            except Exception:
                pass

            # it could be a file, not a directory
            try:
                os.remove(filename)
            except Exception:
                pass

        # filename is actually a directory, so append separator if needed
        if filename[-1:] != "/":
            filename = filename + "/"

        # make directory
        os.mkdir(filename)

        # save components

        # python objects must be pickled
        if save_hierarchy:
            # note this is a combo of python objects and numpy arrays, so must
            # be pickled.
            pickle.dump(
                self.parent_molecule.information.hierarchy,
                open(filename + "hierarchy", "wb"),
                -1,
            )
        if save_remarks:
            pickle.dump(
                self.parent_molecule.information.remarks,
                open(filename + "remarks", "wb"),
                -1,
            )  # using the latest protocol
        if save_filename:
            pickle.dump(
                self.parent_molecule.information.filename,
                open(filename + "filename", "wb"),
                -1,
            )

        # unfortunately, the speedy np.save doesn't work on masked arrays
        # masked arrays have a dump method, but it just uses cPickle so we're
        # just going to cPickle masked arrays. Could be so much faster if
        # numpy were up to speed... :(not clear that np.ma.dump accepts
        # protocol parameter, so let's just use cPickle directly
        pickle.dump(
            self.parent_molecule.information.atom_information,
            open(filename + "atom_information", "wb"),
            -1,
        )

        # fortunately, coordinates and bonds are regular numpy arrays they can
        # be saved with numpy's speedy np.save function note that I'm
        # compressing them here. benchmarking suggests this takes longer to
        # save, but is much faster to load. so I'm prioritizing load times
        # over save times note also that np.savez can save multiple arrays
        # to a single file, probably speeding up load.

        np.savez(
            filename + "coordinates.npz", self.parent_molecule.information.coordinates
        )
        if save_bonds:
            np.savez(filename + "bonds.npz", self.parent_molecule.information.bonds)
        if save_coordinates_undo_point:
            np.savez(
                filename + "coordinates_undo_point.npz",
                self.parent_molecule.information.coordinates_undo_point,
            )

    def save_pdb(
        self, filename="", serial_reindex=True, resseq_reindex=False, return_text=False
    ):
        """Saves the molecular data contained in a pymolecule.Molecule object
        to a pdb file.

        Args:
            filename: An string, the filename to use for saving.
            serial_reindex: An optional boolean, whether or not to reindex
                the pdb serial field. True by default.
            resseq_reindex: An optional boolean, whether or not to reindex
                the pdb resseq field. False by default.
            return_text: An optional boolean, whether or not to return text
                instead of writing to a file. If True, the filename variable is
                ignored.

        Returns:
            If return_text is True, a PDB-formatted string. Otherwise, returns
                nothing.

        """

        # so the pdb is not empty (if it is empty, don't save)
        if len(self.parent_molecule.information.atom_information) > 0:

            if serial_reindex:
                self.serial_reindex()
            if resseq_reindex:
                self.resseq_reindex()

            if not return_text:
                afile = open(filename, "w")
            else:
                return_string = ""

            # print out remarks
            for line in self.parent_molecule.information.remarks:
                remark = "REMARK" + line + "\n"

                if not return_text:
                    afile.write(remark)
                else:
                    return_string = return_string + remark

            # print out coordinates
            printout = np.char.add(
                self.parent_molecule.information.atom_information["record_name"],
                np.char.rjust(
                    self.parent_molecule.information.atom_information["serial"].astype(
                        "|S5"
                    ),
                    5,
                ),
            )
            printout = np.char.add(
                printout, self.parent_molecule.information.atom_information["name"]
            )
            printout = np.char.add(
                printout, self.parent_molecule.information.atom_information["resname"]
            )
            printout = np.char.add(
                printout, self.parent_molecule.information.atom_information["chainid"]
            )
            printout = np.char.add(
                printout,
                np.char.rjust(
                    self.parent_molecule.information.atom_information["resseq"].astype(
                        "|S4"
                    ),
                    4,
                ),
            )
            printout = np.char.add(printout, "    ")
            printout = np.char.add(
                printout,
                np.char.rjust(
                    np.array(
                        [
                            "%.3f" % t
                            for t in self.parent_molecule.information.coordinates[:, 0]
                        ]
                    ),
                    8,
                ),
            )
            printout = np.char.add(
                printout,
                np.char.rjust(
                    np.array(
                        [
                            "%.3f" % t
                            for t in self.parent_molecule.information.coordinates[:, 1]
                        ]
                    ),
                    8,
                ),
            )
            printout = np.char.add(
                printout,
                np.char.rjust(
                    np.array(
                        [
                            "%.3f" % t
                            for t in self.parent_molecule.information.coordinates[:, 2]
                        ]
                    ),
                    8,
                ),
            )
            printout = np.char.add(
                printout,
                np.char.rjust(
                    np.array(
                        [
                            "%.2f" % t
                            for t in self.parent_molecule.information.atom_information[
                                "occupancy"
                            ]
                        ]
                    ),
                    6,
                ),
            )
            printout = np.char.add(
                printout,
                np.char.rjust(
                    np.array(
                        [
                            "%.2f" % t
                            for t in self.parent_molecule.information.atom_information[
                                "tempfactor"
                            ]
                        ]
                    ),
                    6,
                ),
            )
            printout = np.char.add(printout, "          ")
            printout = np.char.add(
                printout, self.parent_molecule.information.atom_information["element"]
            )
            printout = np.char.add(
                printout, self.parent_molecule.information.atom_information["charge"]
            )

            if not return_text:
                if printout[0][-1:] == "\n":
                    afile.write("".join(printout) + "\n")
                else:
                    afile.write("\n".join(printout) + "\n")
            else:
                if printout[0][-1:] == "\n":
                    return_string = return_string + ("".join(printout) + "\n")
                else:
                    return_string = return_string + ("\n".join(printout) + "\n")

            # print out connect
            if not self.parent_molecule.information.bonds is None:
                for index in range(len(self.parent_molecule.information.bonds)):
                    indices_of_bond_partners = self.parent_molecule.selections.select_all_atoms_bound_to_selection(
                        np.array([index])
                    )
                    if len(indices_of_bond_partners) > 0:

                        if not return_text:
                            afile.write(
                                "CONECT"
                                + str(
                                    self.parent_molecule.information.atom_information[
                                        "serial"
                                    ][index]
                                ).rjust(5)
                                + "".join(
                                    [
                                        str(
                                            self.parent_molecule.information.atom_information[
                                                "serial"
                                            ][
                                                t
                                            ]
                                        ).rjust(5)
                                        for t in indices_of_bond_partners
                                    ]
                                )
                                + "\n"
                            )
                        else:
                            return_string = return_string + (
                                "CONECT"
                                + str(
                                    self.parent_molecule.information.atom_information[
                                        "serial"
                                    ][index]
                                ).rjust(5)
                                + "".join(
                                    [
                                        str(
                                            self.parent_molecule.information.atom_information[
                                                "serial"
                                            ][
                                                t
                                            ]
                                        ).rjust(5)
                                        for t in indices_of_bond_partners
                                    ]
                                )
                                + "\n"
                            )

            if not return_text:
                afile.close()
            else:
                return return_string

        else:
            print(
                'ERROR: Cannot save a Molecule with no atoms (file name "'
                + filename
                + '")'
            )

    def serial_reindex(self):
        """Reindexes the serial field of the atoms in the molecule, starting
        with 1"""

        for i in range(
            len(self.parent_molecule.information.atom_information["serial"])
        ):
            self.parent_molecule.information.atom_information["serial"][i] = i + 1

    def resseq_reindex(self):
        """Reindexes the resseq field of the atoms in the molecule, starting
        with 1"""

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

        keys2 = np.insert(keys, 0, "")[:-1]
        index_of_change = np.nonzero(np.logical_not(keys == keys2))[0]
        index_of_change = np.append(
            index_of_change, len(self.parent_molecule.information.atom_information)
        )

        count = 1
        for t in range(len(index_of_change[:-1])):
            start = index_of_change[t]
            end = index_of_change[t + 1]
            self.parent_molecule.information.atom_information["resseq"][
                np.arange(start, end, 1, dtype="int")
            ] = count
            count = count + 1
