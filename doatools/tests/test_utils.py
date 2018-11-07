import unittest
import numpy as np
import numpy.testing as npt
import doatools.utils.math as doa_math
import doatools.utils.conversion as doa_conv

class TestMath(unittest.TestCase):

    def test_vec(self):
        x = np.array([[1, 2], [3, 4], [5, 6]])
        actual = doa_math.vec(x)
        expected = np.array([[1], [3], [5], [2], [4], [6]])
        npt.assert_array_equal(actual, expected)

    def test_khatri_rao(self):
        A = np.array([[1.5, 0.5], [3.0, 2.0]])
        B = np.array([[-1.0, 1.0], [-2.5, 2.5], [-5.0, 5.0]])
        actual = doa_math.khatri_rao(A, B)
        expected = np.array([
            [-1.5, 0.5], [-3.75, 1.25], [-7.5, 2.5],
            [-3.0, 2.0], [-7.5, 5.0], [-15.0, 10.0]
        ])
        npt.assert_allclose(actual, expected)

    def test_projm(self):
        # Real
        A = np.array([[0.3, 0.4], [0.7, -0.1], [0.5, 0.9]])
        PA_expected = np.array([
            [1.792687e-01,  8.448704e-02,  3.741569e-01],
            [8.448704e-02,  9.913028e-01, -3.851615e-02],
            [3.741569e-01, -3.851615e-02,  8.294285e-01]
        ])
        PA_actual = doa_math.projm(A)
        npt.assert_allclose(PA_actual, PA_expected, rtol=1e-6)
        # Complex
        B = A + 0.1j
        PB_expected = np.array([
            [1.874574e-1             ,  8.725290e-2+2.044990e-2j,  3.776414e-1+4.089980e-2j],
            [8.725290e-2-2.044990e-2j,  9.901159e-1             , -4.158146e-2+5.112474e-3j],
            [3.776414e-1-4.089980e-2j, -4.158146e-2-5.112474e-3j,  8.224267e-1             ]
        ])
        PB_actual = doa_math.projm(B)
        npt.assert_allclose(PB_actual, PB_expected, rtol=1e-6)

    def test_cartesian(self):
        ax1 = np.array([1, 2])
        ax2 = np.array([3, 4, 5])
        ax3 = np.array([6, 7])
        actual = doa_math.cartesian(ax1, ax2, ax3)
        expected = np.array([
            [1, 3, 6], [1, 3, 7], [1, 4, 6], [1, 4, 7], [1, 5, 6], [1, 5, 7],
            [2, 3, 6], [2, 3, 7], [2, 4, 6], [2, 4, 7], [2, 5, 6], [2, 5, 7]
        ])
        npt.assert_array_equal(actual, expected)

    def assert_unique_rows(self, x, y, atol, rtol):
        """Asserts that the rows are unique.

        Args:
            x: Original matrix.
            y: Matrix of unique rows.
            atol: Absolute tolerance.
            rtol: Relative tolerance.
        """
        tol = atol + rtol * (np.abs(x)).max()
        n = y.shape[0]
        for i in range(n):
            for k in range(i + 1, n):
                self.assertTrue(np.any(np.abs(y[i, :] - y[k, :]) > tol), 'Difference too small when comparing {0} and {1}'.format(y[i, :], y[k, :]))

    def test_unique_rows(self):
        # 1D
        data1 = np.array([1.0, 1.01, 2.2, 2.0, 0.99, -10.0, 1.95]).reshape((-1, 1))
        atol1 = 0.0
        rtol1 = 1e-2
        unique1, indices1 = doa_math.unique_rows(data1, atol1, rtol1, True)
        self.assert_unique_rows(data1, unique1, atol1, rtol1)
        npt.assert_array_equal(unique1, data1[indices1, :])
        # 2D
        data2 = np.array([
            [0.1, 0.1, 0.1],
            [0.1, 0.11, 0.1],
            [0.1, 0.09, 0.5],
            [0.1, 0.1, 0.5],
            [0.1, 0.1, 0.5],
            [1.0, 1.0, 1.0]
        ])
        atol2 = 0.0
        rtol2 = 1e-2
        unique2, indices2 = doa_math.unique_rows(data2, atol2, rtol2, True)
        self.assert_unique_rows(data2, unique2, atol2, rtol2)
        npt.assert_array_equal(unique2, data2[indices2, :])
        # Randomly generated 2D
        np.random.seed(1123)
        centers = np.array([
            [0, -1, 1],
            [0, 1, -1],
            [1, 1, 1]
        ])
        data3 = np.vstack([
            centers[2, :] + np.random.uniform(-0.1, 0.1, (100, 3)),
            centers[1, :] + np.random.uniform(-0.1, 0.1, (100, 3)),
            centers[0, :] + np.random.uniform(-0.1, 0.1, (100, 3))
        ])
        atol3 = 0.0
        rtol3 = 0.2
        unique3, indices3 = doa_math.unique_rows(data3, atol3, rtol3, True, True)
        self.assert_unique_rows(data3, unique3, atol3, rtol3)
        npt.assert_array_equal(unique3, data3[indices3, :])
        npt.assert_allclose(unique3, centers, atol=0.1)

class TestConversion(unittest.TestCase):

    def test_cartesian_to_spherical(self):
        # 1D
        coords1 = np.array([[1.], [-2.], [3.]])
        s1_expected = np.array([
            [1., 0., 0.],
            [2., np.pi, 0.],
            [3., 0., 0.]
        ])
        npt.assert_allclose(doa_conv.cart2spherical(coords1), s1_expected)
        # 2D
        coords2 = np.array([
            [ 1.0,  1.0],
            [ 0.0, -1.0],
            [-2.0, -2.0]
        ])
        s2_expected = np.array([
            [np.sqrt(2), np.pi/4, 0.0],
            [1.0, -np.pi/2, 0.0],
            [np.sqrt(8), -np.pi*3/4, 0.0]
        ])
        npt.assert_allclose(doa_conv.cart2spherical(coords2), s2_expected)
        # 3D
        coords3 = np.array([
            [0.0, 0.0, 1.0],
            [-1.0, 1.0, 1.0],
            [3/4, np.sqrt(3)/4, 1/2]
        ])
        s3_expected = np.array([
            [1.0, 0.0, np.pi/2],
            [np.sqrt(3), np.pi*3/4, np.arctan2(1.0, np.sqrt(2))],
            [1.0, np.pi/6, np.pi/6]
        ])
        npt.assert_allclose(doa_conv.cart2spherical(coords3), s3_expected)

if __name__ == '__main__':
    unittest.main()
