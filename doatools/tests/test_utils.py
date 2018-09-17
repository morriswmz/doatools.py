import unittest
import numpy as np
import numpy.testing as npt
import doatools.utils.math as doa_math

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
        npt.assert_array_almost_equal(actual, expected)

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

if __name__ == '__main__':
    unittest.main()
