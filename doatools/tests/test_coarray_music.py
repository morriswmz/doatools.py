import unittest
import numpy as np
import numpy.testing as npt
from doatools.model.arrays import CoPrimeArray
from doatools.model.sources import FarField1DSourcePlacement
from doatools.estimation.music import RootMUSIC1D
from doatools.estimation.coarray import CoarrayACMBuilder1D


class TestCoarrayMUSIC(unittest.TestCase):

    def setUp(self):
        self.wavelength = 1

    def test_coprime(self):
        cpa = CoPrimeArray(3, 5, self.wavelength/2)
        sources = FarField1DSourcePlacement(np.linspace(-np.pi/3, np.pi/3, 11))
        A = cpa.steering_matrix(sources, self.wavelength)
        # Compute the ideal covariance matrix at 0dB SNR.
        R = A @ A.T.conj() + np.eye(cpa.size)
        # Transform to the augmented covariance matrix.
        ss_transform = CoarrayACMBuilder1D(cpa)
        rmusic = RootMUSIC1D(self.wavelength)
        for method in ['ss', 'da']:
            Ra = ss_transform(R, method)
            # Estimate the DOAs using root-MUSIC.
            resolved, estimates = rmusic.estimate(Ra, sources.size, cpa.d0)
            self.assertTrue(resolved)
            npt.assert_allclose(estimates.locations, sources.locations, rtol=1e-6, atol=1e-8)

if __name__ == '__main__':
    unittest.main()
    