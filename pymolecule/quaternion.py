import numpy as np


class Quaternion:
    """A class supporting quaternion arithmetic"""

    def __init__(self, s, x, y, z):
        """Initializes the pymolecule.Quaternion class.

        Args:
            s: ????
            x: ????
            y: ????
            z: ????

        """

        self.v = np.zeros(4)
        self.v[0] = s
        self.v[1] = x
        self.v[2] = y
        self.v[3] = z

    def __str__(self):
        """String containing quaternion information in the form of s x y z

        Returns:
            A string, containing all information about this quaternion

        """

        return (
            ""
            + str(self.v[0])
            + "\t"
            + str(self.v[1])
            + "\t"
            + str(self.v[2])
            + "\t"
            + str(self.v[3])
        )

    def copy_of(self):
        """Returns a copy of self"""

        return Quaternion(self.v[0], self.v[1], self.v[2], self.v[3])

    def load_from_mat(self, m):
        """Converts a rotation matrix that is pure orthogonal (det(matrix)=1)
        into a Quaternion. Adapted from
        http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm

        Args:
            m: A 2D np.array representing a pure orthogonal matrix.

        """
        # Make sure m is a 3x3 array
        if m.shape[0] != 3 or m.shape[1] != 3:
            print("Could not load quaternion from matrix...size is not (3x3)")
            return

        # Check that matrix is orthogonal. m_T = m_inv
        if not np.array_equal(np.transpose(m), np.linalg.inv(m)):
            print("Load Quaternion error. Matrix is not orthogonal")
            return

        # Need to make sure that the matrix is special orthogonal
        if np.fabs(1 - np.linalg.det(m)) > 0.000001:  # Done for rounding errors
            print("Load Quaternion error.  Determinant is not 1")
            return

        # First calculate the sum of the diagonal elements
        t = m.trace()

        if t > 0:
            S = np.sqrt(t + 1.0) * 2
            self.v[0] = 0.25 * S
            self.v[1] = (m[2, 1] - m[1, 2]) / S
            self.v[2] = (m[0, 2] - m[2, 0]) / S
            self.v[3] = (m[1, 0] - m[0, 1]) / S
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            self.v[0] = (m[2, 1] - m[1, 2]) / S
            self.v[1] = 0.25 * S
            self.v[2] = (m[0, 1] + m[1, 0]) / S
            self.v[3] = (m[0, 2] + m[2, 0]) / S
        elif m[1, 1] > m[2, 2]:
            S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            self.v[0] = (m[0, 2] - m[2, 0]) / S
            self.v[1] = (m[0, 1] + m[1, 0]) / S
            self.v[2] = 0.25 * S
            self.v[3] = (m[2, 1] + m[1, 2]) / S
        else:
            S = np.sqrt(1.0) * 2
            self.v[0] = (m[1, 0] - m[0, 1]) / S
            self.v[1] = (m[0, 2] + m[2, 0]) / S
            self.v[2] = (m[2, 1] + m[1, 2]) / S
            self.v[3] = 0.25 * S

    def rep_as_44_matrix(self):
        """Creates a 4x4 matrix representation of the Quaternion.

        Returns:
            A 4x4 numpy array

        """

        n = self.normalize()
        qw = n.v[0]
        qx = n.v[1]
        qy = n.v[2]
        qz = n.v[3]

        return np.array(
            [
                [qw, qx, qy, qz],
                [-qx, qw, -qz, qy],
                [-qy, qz, qw, -qx],
                [-qz, -qy, qx, qw],
            ]
        )

    def to_matrix(self):
        """Converts to a normalized 3x3 matrix.

        Returns:
            A 3x3 np.array, corresponding to the quaternion.

        """

        # First normalize
        n = self.normalize()
        qw = n.v[0]
        qx = n.v[1]
        qy = n.v[2]
        qz = n.v[3]
        return np.array(
            [
                [
                    1.0 - 2.0 * qy * qy - 2.0 * qz * qz,
                    2.0 * qx * qy - 2.0 * qz * qw,
                    2.0 * qx * qz + 2.0 * qy * qw,
                ],
                [
                    2.0 * qx * qy + 2.0 * qz * qw,
                    1.0 - 2.0 * qx * qx - 2.0 * qz * qz,
                    2.0 * qy * qz - 2.0 * qx * qw,
                ],
                [
                    2.0 * qx * qz - 2.0 * qy * qw,
                    2.0 * qy * qz + 2.0 * qx * qw,
                    1.0 - 2.0 * qy * qy - 2.0 * qx * qx,
                ],
            ]
        )

    def add(self, q2):
        """Adds two quaternions.

        Args:
            q2: A quaternion, to be added to self.

        Returns:
            A Quaternion, with the values corresponding to self + q2.

        """

        return Quaternion(
            self.v[0] + q2.v[0],
            self.v[1] + q2.v[1],
            self.v[2] + q2.v[2],
            self.v[3] + q2.v[3],
        )

    def invert(self):
        """Takes the inverse of the quaternion for "division"

        Returns:
        A Quaternion, with the values corresponding to self^-1

        """

        return Quaternion(self.v[0], -1 * self.v[1], -1 * self.v[2], -1 * self.v[3])

    def minus(self, q2):
        """Multiplies two quaternions

        Args:
            q2: A quaternion, to be subtracted from self

        Returns:
            A Quaternion, with the values corresponding to self - q2

        """

        return Quaternion(
            self.v[0] - q2.v[0],
            self.v[1] - q2.v[1],
            self.v[2] - q2.v[2],
            self.v[3] - q2.v[3],
        )

    def multiply(self, q2):
        """Multiplies two quaternions

        Args:
            q2: A quaternion, to be multiplied with self

        Returns:
            A Quaternion, with the values corresponding to self * q2

        """

        return Quaternion(
            self.v[0] * q2.v[0]
            - self.v[1] * q2.v[1]
            - self.v[2] * q2.v[2]
            - self.v[3] * q2.v[3],
            self.v[1] * q2.v[0]
            + self.v[0] * q2.v[1]
            + self.v[2] * q2.v[3]
            - self.v[3] * q2.v[2],
            self.v[0] * q2.v[2]
            - self.v[1] * q2.v[3]
            + self.v[2] * q2.v[0]
            + self.v[3] * q2.v[1],
            self.v[0] * q2.v[3]
            + self.v[1] * q2.v[2]
            - self.v[2] * q2.v[1]
            + self.v[3] * q2.v[0],
        )

    def normalize(self):
        """Normalizes the quaternion.

        Returns:
            A normalized Quaternion.

        """

        # First normalize
        n = np.sqrt(
            np.power(self.v[0], 2)
            + np.power(self.v[1], 2)
            + np.power(self.v[2], 2)
            + np.power(self.v[3], 2)
        )

        return Quaternion(self.v[0] / n, self.v[1] / n, self.v[2] / n, self.v[3] / n)

    def scale(self, scalar):
        """Scales a quaternion.

        Args:
            scalar: the value to scale the quaternion by.

        Returns:
            A Quaternion, with the values corresponding to self * scalar.

        """

        return Quaternion(
            self.v[0] * scalar,
            self.v[1] * scalar,
            self.v[2] * scalar,
            self.v[3] * scalar,
        )
