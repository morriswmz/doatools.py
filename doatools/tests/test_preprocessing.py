import unittest
import numpy as np
import numpy.testing as npt
from doatools.estimation.preprocessing import spatial_smooth

class TestPreprocessing(unittest.TestCase):

    def test_spatial_smoothing_real(self):
        R = np.array([
            [3., 2., 1., 1., 1.],
            [2., 4., 2., 2., 1.],
            [1., 2., 5., 3., 2.],
            [1., 2., 3., 4., 3.],
            [1., 1., 2., 3., 5.]
        ])
        # l = 1 special case
        npt.assert_array_almost_equal(spatial_smooth(R, 1), R)
        npt.assert_array_almost_equal(spatial_smooth(R, 1, True), 0.5 * (R + np.flip(R)))
        # l > 1
        l = 3
        Rf_expected = np.array([
            [4.000000, 2.333333, 1.666667],
            [2.333333, 4.333333, 2.666667],
            [1.666667, 2.666667, 4.666667]
        ])
        Rfb_expected = np.array([
            [4.333333, 2.500000, 1.666667],
            [2.500000, 4.333333, 2.500000],
            [1.666667, 2.500000, 4.333333]
        ])
        npt.assert_array_almost_equal(spatial_smooth(R, l), Rf_expected)
        npt.assert_array_almost_equal(spatial_smooth(R, l, True), Rfb_expected)

    def test_spatial_smoothing_complex(self):
        R = np.array([
            [3.    , 2.+2.j, 1.+1.j, 1.+1.j, 1.+1.j],
            [2.-2.j, 4.    , 2.+2.j, 2.+2.j, 1.+1.j],
            [1.-1.j, 2.-2.j, 5.    , 3.+3.j, 2.+2.j],
            [1.-1.j, 2.-2.j, 3.-3.j, 4.    , 3.+3.j],
            [1.-1.j, 1.-1.j, 2.-2.j, 3.-3.j, 5.    ]
        ])
        # l = 1 special case
        npt.assert_array_almost_equal(spatial_smooth(R, 1), R)
        npt.assert_array_almost_equal(spatial_smooth(R, 1, True), 0.5 * (R + np.flip(R).conj()))
        # l > 1
        l = 3
        Rf_expected = np.array([
            [4.000000+0.000000j, 2.333333+2.333333j, 1.666667+1.666667j],
            [2.333333-2.333333j, 4.333333+0.000000j, 2.666667+2.666667j],
            [1.666667-1.666667j, 2.666667-2.666667j, 4.666667+0.000000j]
        ])
        Rfb_expected = np.array([
            [4.333333+0.000000j, 2.500000+2.500000j, 1.666667+1.666667j],
            [2.500000-2.500000j, 4.333333+0.000000j, 2.500000+2.500000j],
            [1.666667-1.666667j, 2.500000-2.500000j, 4.333333+0.000000j]
        ])
        npt.assert_array_almost_equal(spatial_smooth(R, l), Rf_expected)
        npt.assert_array_almost_equal(spatial_smooth(R, l, True), Rfb_expected)

if __name__ == '__main__':
    unittest.main()
