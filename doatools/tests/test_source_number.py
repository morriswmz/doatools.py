import unittest
import numpy as np
from doatools.estimation.source_number import aic, mdl, sorte
from doatools.model.arrays import UniformLinearArray
from doatools.model.sources import FarField1DSourcePlacement
from doatools.model.signals import ComplexStochasticSignal
from doatools.model.snapshots import get_narrowband_snapshots

class TestSourceNumberDetection(unittest.TestCase):

    def test_ideal_case(self):
        n = 20 # Number of sensors
        # Because this test is for ideal case, we set n_snapshots to a very
        # large number.
        n_snapshots = 10000
        # Vary number of sources
        for k in range(1, 11):
            # Ideal eigenvalues in ascending order.
            l = np.ones((n,))
            l[:-k] = 0.1 # 10 dB SNR
            self.assertEqual(aic(l, n_snapshots), k, 'AIC k = {0}'.format(k))
            self.assertEqual(mdl(l, n_snapshots), k, 'MDL k = {0}'.format(k))
            self.assertEqual(sorte(l), k, 'SORTE k = {0}'.format(k))

    def test_stochastic_case(self):
        np.random.seed(723)
        n = 20
        wavelength = 1.0
        n_snapshots = 200
        ula = UniformLinearArray(n, wavelength / 2)
        # Vary number of sources
        for k in range(1, 11):
            sources = FarField1DSourcePlacement(
                np.linspace(-np.pi/3, np.pi/3, k)
            )
            source_signal = ComplexStochasticSignal(sources.size, 1.0)
            noise_signal = ComplexStochasticSignal(ula.size, 0.1) # 10 dB
            y, R = get_narrowband_snapshots(
                ula, sources, wavelength, source_signal, noise_signal,
                n_snapshots, return_covariance=True
            )
            self.assertEqual(aic(R, n_snapshots), k, 'AIC k = {0}'.format(k))
            self.assertEqual(mdl(R, n_snapshots), k, 'MDL k = {0}'.format(k))
            self.assertEqual(sorte(R), k, 'SORTE k = {0}'.format(k))

if __name__ == '__main__':
    unittest.main()
