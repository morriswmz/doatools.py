import unittest
import numpy as np
import numpy.testing as npt
from doatools.model.arrays import UniformLinearArray
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
        self.assertEqual(wf.central_ula_size(), 7)
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

if __name__ == '__main__':
    unittest.main()
