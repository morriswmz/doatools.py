import unittest
import numpy as np
import numpy.testing as npt
from doatools.model.arrays import UniformLinearArray
from doatools.model.sources import FarField1DSourcePlacement
from doatools.estimation.esprit import Esprit1D
import itertools

class TestEsprit(unittest.TestCase):

    def setUp(self):
        self.wavelength = 1.0

    def test_esprit_1d(self):
        ula = UniformLinearArray(24, self.wavelength / 2.0)
        estimator = Esprit1D(self.wavelength)
        # 4 sources with random shifts
        np.random.seed(42)
        for i in range(10):
            offset = np.random.uniform(-np.pi/64, np.pi/64)
            sources = FarField1DSourcePlacement(
                np.linspace(-np.pi/16, np.pi/16, 3) + offset)
            # Use the ideal covariance matrix at SNR = 0 dB
            A = ula.steering_matrix(sources, self.wavelength)
            R = A @ A.conj().T + np.eye(ula.size)
            # Test different settings
            displacements = range(1, 4)
            formulations = ['ls', 'tls']
            row_weights = ['none', 'default', 'custom']
            for d, f, rw in itertools.product(displacements, formulations, row_weights):
                if rw == 'custom':
                    # All weights set to 2.0. Should be equivalent to the no
                    # weighting case.
                    rw = np.full((ula.size - d), 2.0)
                _, estimates = estimator.estimate(R, sources.size, ula.d0, d, f, rw)
                err_msg = 'displacement={0}, formulation={1}, row_weights={2}'.format(d, f, rw)
                npt.assert_allclose(estimates.locations, sources.locations,
                                    rtol=1e-2, atol=1e-8, err_msg=err_msg)

if __name__ == '__main__':
    unittest.main()
