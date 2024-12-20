import pickle as pickle

import numpy as np


class Geometry:
    """A class containing a few gemoetry functions. Note that numpy should be
    used for most geometry functions."""

    def __init__(self, parent_molecule_object):
        """Initializes the pymolecule.Geometry class.

        Args:
        parent_molecule_object: The pymolecule.Molecule object associated
            with this class.

        """

        self.parent_molecule = parent_molecule_object

    def angle_between_three_points(self, pt1, pt2, pt3):
        """Computes the angle (in radians) formed by three points (np.array
        objects).

        Args:
            pt1: A np.array (x, y, z) representing the first of the three
                3D points.
            pt2: A np.array (x, y, z) representing the second of the three
                3D points.
            pt3: A np.array (x, y, z) representing the third of the three
                3D points.

        Returns:
            A float containing the angle between the three points, in radians.

        """

        vector1 = pt1 - pt2
        vector2 = pt3 - pt2

        vector1_mag = np.linalg.norm(vector1)
        vector2_mag = np.linalg.norm(vector2)

        # Make sure vectors aren't <0,0,0>
        if vector1_mag < 1e-10 or vector2_mag < 1e-10:
            print("One of vectors to determine angle is < 0, 0, 0 >...returning 0.")
            return 0

        vector1 = vector1 / vector1_mag
        vector2 = vector2 / vector2_mag
        dot_prod = np.dot(vector1, vector2)

        # Prevent errors that can rarely occur
        if dot_prod > 1.0:
            dot_prod = 1.0
        if dot_prod < -1.0:
            dot_prod = -1.0

        return np.arccos(dot_prod)

    def dihedral_angle(self, pt1, pt2, pt3, pt4):
        """Calculates the dihedral angle formed by four points (np.array
        objects).

        Args:
            pt1: A np.array (x, y, z) representing the first 3D point.
            pt2: A np.array (x, y, z) representing the second 3D point.
            pt3: A np.array (x, y, z) representing the third 3D point.
            pt4: A np.array (x, y, z) representing the fourth 3D point.

        Returns:
            A float containing the dihedral angle between the four points, in
                radians.

        """

        b1 = pt2 - pt1
        b2 = pt3 - pt2
        b3 = pt4 - pt3

        b2Xb3 = np.cross(b2, b3)
        b1Xb2 = np.cross(b1, b2)

        b1XMagb2 = np.linalg.norm(b2) * b1

        return np.arctan2(np.dot(b1XMagb2, b2Xb3), np.dot(b1Xb2, b2Xb3))

    def is_planar(self, pt1, pt2, pt3, pt4, planarity_cutoff=0.2):
        """Checks whether four points (np.array) lie in a common plane.

        Args:
            pt1: A np.array (x, y, z) representing a 3D point.
            pt2: A np.array (x, y, z) representing a 3D point.
            pt3: A np.array (x, y, z) representing a 3D point.
            pt4: A np.array (x, y, z) representing a 3D point.
            planarity_cutoff: An optional float. How much the points can
                deviate (in Angstroms) and still be considered planar. The
                default is 0.2.

        Returns:
            A boolean, whether the 4 points can be considered planar.

        """

        return self.planarity_deviation(pt1, pt2, pt3, pt4) < planarity_cutoff

    def planarity_deviation(self, pt1, pt2, pt3, pt4):
        """Determines how close four points (np.array objects) come to lying
        in a common plane.

        Args:
            pt1: A np.array (x, y, z) representing a 3D point.
            pt2: A np.array (x, y, z) representing a 3D point.
            pt3: A np.array (x, y, z) representing a 3D point.
            pt4: A np.array (x, y, z) representing a 3D point.

        Returns:
            A float, the minimum distance between one point and the plane
                formed by the other three.

        """

        # note that minimal efforts were made to "numpify" this section. It's
        # mostly legacy code.

        x1 = pt1[0]
        y1 = pt1[1]
        z1 = pt1[2]
        x2 = pt2[0]
        y2 = pt2[1]
        z2 = pt2[2]
        x3 = pt3[0]
        y3 = pt3[1]
        z3 = pt3[2]
        x4 = pt4[0]
        y4 = pt4[1]
        z4 = pt4[2]

        A = (y1 * (z2 - z3)) + (y2 * (z3 - z1)) + (y3 * (z1 - z2))
        B = (z1 * (x2 - x3)) + (z2 * (x3 - x1)) + (z3 * (x1 - x2))
        C = (x1 * (y2 - y3)) + (x2 * (y3 - y1)) + (x3 * (y1 - y2))
        D = (
            ((-x1) * ((y2 * z3) - (y3 * z2)))
            + ((-x2) * ((y3 * z1) - (y1 * z3)))
            + ((-x3) * ((y1 * z2) - (y2 * z1)))
        )
        denom = np.sqrt(np.power(A, 2) + np.power(B, 2) + np.power(C, 2))
        if denom == 0:
            return 0  # implies straight line
        distance1 = np.fabs((A * x4) + (B * y4) + (C * z4) + D) / denom

        A1 = (y1 * (z2 - z4)) + (y2 * (z4 - z1)) + (y4 * (z1 - z2))
        B1 = (z1 * (x2 - x4)) + (z2 * (x4 - x1)) + (z4 * (x1 - x2))
        C1 = (x1 * (y2 - y4)) + (x2 * (y4 - y1)) + (x4 * (y1 - y2))
        D1 = (
            ((-x1) * ((y2 * z4) - (y4 * z2)))
            + ((-x2) * ((y4 * z1) - (y1 * z4)))
            + ((-x4) * ((y1 * z2) - (y2 * z1)))
        )
        distance2 = (np.fabs((A1 * x3) + (B1 * y3) + (C1 * z3) + D1)) / (
            np.sqrt(np.power(A1, 2) + np.power(B1, 2) + np.power(C1, 2))
        )

        A2 = (y1 * (z4 - z3)) + (y4 * (z3 - z1)) + (y3 * (z1 - z4))
        B2 = (z1 * (x4 - x3)) + (z4 * (x3 - x1)) + (z3 * (x1 - x4))
        C2 = (x1 * (y4 - y3)) + (x4 * (y3 - y1)) + (x3 * (y1 - y4))
        D2 = (
            ((-x1) * ((y4 * z3) - (y3 * z4)))
            + ((-x4) * ((y3 * z1) - (y1 * z3)))
            + ((-x3) * ((y1 * z4) - (y4 * z1)))
        )
        distance3 = (np.fabs((A2 * x2) + (B2 * y2) + (C2 * z2) + D2)) / (
            np.sqrt(np.power(A2, 2) + np.power(B2, 2) + np.power(C2, 2))
        )

        A3 = (y4 * (z2 - z3)) + (y2 * (z3 - z4)) + (y3 * (z4 - z2))
        B3 = (z4 * (x2 - x3)) + (z2 * (x3 - x4)) + (z3 * (x4 - x2))
        C3 = (x4 * (y2 - y3)) + (x2 * (y3 - y4)) + (x3 * (y4 - y2))
        D3 = (
            ((-x4) * ((y2 * z3) - (y3 * z2)))
            + ((-x2) * ((y3 * z4) - (y4 * z3)))
            + ((-x3) * ((y4 * z2) - (y2 * z4)))
        )
        distance4 = (np.fabs((A3 * x1) + (B3 * y1) + (C3 * z1) + D3)) / (
            np.sqrt(np.power(A3, 2) + np.power(B3, 2) + np.power(C3, 2))
        )

        return np.min(np.array([distance1, distance2, distance3, distance4]))
