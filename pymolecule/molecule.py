import pickle as pickle

from .fileio import FileIO
from .geometry import Geometry
from .info import Information
from .manipulation import Manipulation
from .other import OtherMolecules
from .selection import Selections
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
