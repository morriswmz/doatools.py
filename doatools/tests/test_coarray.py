import unittest
import numpy as np
import numpy.testing as npt
from doatools.model.arrays import UniformLinearArray, CoPrimeArray
from doatools.model.coarray import WeightFunction1D

class TestWeightFunction(unittest.TestCase):

    def test_ula(self):
        ula = UniformLinearArray(4, 0.5)
        wf = WeightFunction1D(ula)
        npt.assert_array_equal(
            wf.differences(),
            np.array([-3, -2, -1, 0, 1, 2, 3])
        )
        npt.assert_array_equal(
            wf.weights(),
            np.array([1, 2, 3, 4, 3, 2, 1])
        )
        self.assertEqual(wf.get_central_ula_size(), 7)
        indices_expected = {
            -3: [12],
            -2: [8, 13],
            -1: [4, 9, 14],
            0: [0, 5, 10, 15],
            1: [1, 6, 11],
            2: [2, 7],
            3: [3]
        }
        for diff, indices in indices_expected.items():
            self.assertListEqual(wf.indices_of(diff), indices)
    
    def test_coprime_array(self):
        # [0, 2, 4, 3, 6, 9]
        cpa = CoPrimeArray(2, 3, 0.5)
        wf = WeightFunction1D(cpa)
        npt.assert_array_equal(
            wf.differences(),
            np.array([-9, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 9])
        )
        npt.assert_array_equal(
            wf.weights(),
            np.array([1, 1, 2, 1, 2, 3, 3, 2, 6, 2, 3, 3, 2, 1, 2, 1, 1])
        )
        indices_expected = {
            -9: [30],
            -7: [31],
            -6: [24, 33],
            -5: [32],
            -4: [12, 25],
            -3: [18, 27, 34],
            -2: [6, 13, 26],
            -1: [15, 19],
            0: [0, 7, 14, 21, 28, 35],
            1: [9, 20],
            2: [1, 8, 16],
            3: [3, 22, 29],
            4: [2, 10],
            5: [17],
            6: [4, 23],
            7: [11],
            9: [5]
        }
        for diff, indices in indices_expected.items():
            self.assertListEqual(wf.indices_of(diff), indices)
        F_expected = np.zeros((15, 36))
        for i, diff in enumerate(range(-7, 8)):
            F_expected[i, indices_expected[diff]] = 1.0 / len(indices_expected[diff])
        npt.assert_allclose(wf.get_coarray_selection_matrix(), F_expected)
        npt.assert_allclose(wf.get_coarray_selection_matrix(True), F_expected[7:, :])

if __name__ == '__main__':
    unittest.main()
