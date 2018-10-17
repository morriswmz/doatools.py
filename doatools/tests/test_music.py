import unittest
from doatools.model.arrays import UniformLinearArray
from doatools.model.sources import FarField1DSourcePlacement
from doatools.model.signals import ComplexStochasticSignal
from doatools.estimation.grid import FarField1DSearchGrid
from doatools.estimation.music import MUSIC, RootMUSIC1D
import numpy as np
import numpy.testing as npt

class TestMUSIC(unittest.TestCase):

    def setUp(self):
        self.wavelength = 1.

    def test_music_1d(self):
        np.random.seed(42)
        ula = UniformLinearArray(10, self.wavelength / 2)
        n_sources = 5
        sources = FarField1DSourcePlacement(np.linspace(-np.pi/3, np.pi/3, n_sources))
        A = ula.steering_matrix(sources, self.wavelength)
        # Compute the ideal covariance matrix at SNR=0dB
        R = A @ A.T.conj() + np.eye(ula.size)
        # Both MUSIC and root-MUSIC should give high precision results.
        # MUSIC
        music = MUSIC(ula, self.wavelength, FarField1DSearchGrid())
        resolved, estimates = music.estimate(R, n_sources)
        self.assertTrue(resolved)
        npt.assert_allclose(estimates.locations, sources.locations, rtol=1e-6, atol=1e-8)
        # MUSIC with refinements
        resolved, estimates_refined = music.estimate(R, n_sources, refine_estimates=True)
        self.assertTrue(resolved)
        mse_original = np.sum((estimates.locations - sources.locations)**2)
        mse_refined = np.sum((estimates_refined.locations - sources.locations)**2)
        self.assertLess(mse_refined, mse_original)
        # root-MUSIC
        rmusic = RootMUSIC1D(self.wavelength)
        resolved, estimates = rmusic.estimate(R, n_sources, ula.d0)
        self.assertTrue(resolved)
        npt.assert_allclose(estimates.locations, sources.locations, rtol=1e-6, atol=1e-8)

if __name__ == '__main__':
    unittest.main()