import unittest
from doatools.model.arrays import UniformLinearArray
from doatools.model.sources import FarField1DSourcePlacement, NearField2DSourcePlacement
from doatools.model.signals import ComplexStochasticSignal
from doatools.estimation.grid import FarField1DSearchGrid, NearField2DSearchGrid
from doatools.estimation.beamforming import BartlettBeamformer, MVDRBeamformer
import numpy as np
import numpy.testing as npt

class TestBeamforming(unittest.TestCase):

    def setUp(self):
        self.wavelength = 1.

    def test_beamforming_1d(self):
        ula = UniformLinearArray(16, self.wavelength / 2)
        n_sources = 6
        sources = FarField1DSourcePlacement(np.linspace(-np.pi/3, np.pi/3, n_sources))
        # Compute the ideal covariance matrix at SNR = 0 dB
        A = ula.steering_matrix(sources, self.wavelength)
        R = A @ A.T.conj() + np.eye(ula.size)
        grid = FarField1DSearchGrid(start=-np.pi/2, stop=np.pi/2, size=5761)
        # MVDR
        mvdr = MVDRBeamformer(ula, self.wavelength, grid)
        resolved, estimates = mvdr.estimate(R, n_sources)
        self.assertTrue(resolved)
        npt.assert_allclose(sources.locations, estimates.locations, rtol=1e-2)
        # Bartlett
        bartlett = BartlettBeamformer(ula, self.wavelength, grid)
        resolved, estimates = bartlett.estimate(R, n_sources)
        self.assertTrue(resolved)
        npt.assert_allclose(sources.locations, estimates.locations, rtol=1e-2)

if __name__ == '__main__':
    unittest.main()
