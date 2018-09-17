import unittest
import numpy as np
import numpy.testing as npt
from doatools.model.arrays import CoPrimeArray
from doatools.model.sources import FarField1DSourcePlacement
from doatools.estimation.music import RootMUSIC1D
from doatools.estimation.covariance import ACMTransformer1D


class TestCoarrayMUSIC(unittest.TestCase):

    def setUp(self):
        self.wavelength = 1

    def test_coprime(self):
        design = CoPrimeArray(3, 5, self.wavelength/2)
        sources = FarField1DSourcePlacement(np.linspace(-np.pi/3, np.pi/3, 11))
        A = design.steering_matrix(sources, self.wavelength)
        # Compute the ideal covariance matrix at 0dB SNR.
        R = A @ A.T.conj() + np.eye(design.size)
        # Transform to the augmented covariance matrix.
        ss_transform = ACMTransformer1D(design)
        for method in ['ss', 'da']:
            Ra, vula = ss_transform(R, method)
            # Estimate the DOAs using root-MUSIC.
            rmusic = RootMUSIC1D(vula, self.wavelength)
            resolved, estimates = rmusic.estimate(Ra, sources.n_sources)
            self.assertTrue(resolved)
            npt.assert_array_almost_equal(estimates.locations, sources.locations)

if __name__ == '__main__':
    unittest.main()
    