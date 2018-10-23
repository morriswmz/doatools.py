import unittest
import numpy as np
import numpy.testing as npt
from doatools.model.arrays import UniformLinearArray, NestedArray
from doatools.model.sources import FarField1DSourcePlacement
from doatools.model.signals import ComplexStochasticSignal
from doatools.model.snapshots import get_narrowband_snapshots
from doatools.performance.mse import ecov_music_1d, ecov_coarray_music_1d
from doatools.estimation.music import RootMUSIC1D
from doatools.estimation.coarray import CoarrayACMBuilder1D

class TestMSE(unittest.TestCase):

    def setUp(self):
        self.wavelength = 1

    def test_ecov_music_1d(self):
        array = UniformLinearArray(12, self.wavelength / 2.0)
        sources = FarField1DSourcePlacement(np.linspace(-np.pi / 3, np.pi / 4, 4))
        n_snapshots = 100
        power_source = np.array([1.0, 2.0, 3.0, 2.0])
        power_noise = 1.0
        C = ecov_music_1d(array, sources, self.wavelength, power_source, power_noise, n_snapshots)
        C_expected = np.array([
            [ 1.58491227e-05, -1.29740954e-08, -2.07219245e-09, -2.31572888e-08],
            [-1.29740954e-08,  2.31249079e-06, -1.83104065e-09, -1.28782667e-09],
            [-2.07219245e-09, -1.83104065e-09,  1.30339574e-06, -9.13662010e-10],
            [-2.31572888e-08, -1.28782667e-09, -9.13662010e-10,  3.85004236e-06]
        ])
        npt.assert_allclose(C, C_expected, rtol=1e-6)

    def test_ecov_coarray_music_1d(self):
        array = NestedArray(3, 3, self.wavelength / 2.0)
        sources = FarField1DSourcePlacement(np.linspace(-np.pi / 3, np.pi / 4, 4))
        n_snapshots = 100
        power_source = np.array([1.0, 2.0, 3.0, 2.0])
        power_noise = 1.0
        C = ecov_coarray_music_1d(array, sources, self.wavelength, power_source, power_noise, n_snapshots)
        C_expected = np.array([
            [ 4.411638e-04, -3.959079e-05, -3.204349e-05, -2.384947e-05],
            [-3.959079e-05,  2.354386e-05, -3.359402e-06,  7.594943e-06],
            [-3.204349e-05, -3.359402e-06,  1.524470e-05, -1.336803e-05],
            [-2.384947e-05,  7.594943e-06, -1.336803e-05,  5.516666e-05]
        ])
        npt.assert_allclose(C, C_expected, rtol=1e-6)
    
    def test_ecov_coarray_music_1d_convergence(self):
        array = NestedArray(3, 3, self.wavelength / 2.0)
        sources = FarField1DSourcePlacement(np.linspace(-np.pi / 3, np.pi / 4, 4))
        n_snapshots = 1000
        power_source = 10.0
        power_noise = 1.0
        source_signal = ComplexStochasticSignal(sources.size, power_source)
        noise_signal = ComplexStochasticSignal(array.size, power_noise)
        transform = CoarrayACMBuilder1D(array)
        estimator = RootMUSIC1D(self.wavelength)
        # Collect empirical results.
        np.random.seed(42)
        n_repeats = 1000    
        C_emp = np.zeros((sources.size, sources.size))
        for rr in range(n_repeats):
            _, R = get_narrowband_snapshots(array, sources, self.wavelength,
                source_signal, noise_signal, n_snapshots, True)
            Rss = transform(R)
            _, estimates = estimator.estimate(Rss, sources.size, array.d0)
            err = estimates.locations - sources
            C_emp += np.outer(err, err)
        C_emp /= n_repeats
        # Compare with analytical results.
        C = ecov_coarray_music_1d(array, sources, self.wavelength, power_source, power_noise, n_snapshots)
        npt.assert_allclose(np.diag(C_emp), np.diag(C), rtol=1e-1)

if __name__ == '__main__':
    unittest.main()
