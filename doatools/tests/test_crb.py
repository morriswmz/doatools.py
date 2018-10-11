import unittest
import numpy.testing as npt
import numpy as np

from doatools.model import UniformLinearArray, FarField1DSourcePlacement
from doatools.performance import crb_det_farfield_1d, crb_sto_farfield_1d, crb_stouc_farfield_1d

class TestCRB(unittest.TestCase):

    def setUp(self):
        self.wavelength = 1.0

    def test_det_farfield_1d(self):
        ula = UniformLinearArray(10, self.wavelength / 2)
        sources = FarField1DSourcePlacement(np.linspace(-np.pi/3, np.pi/4, 3))
        P = np.array([
            [10.0    , 1.0+0.3j, 0.5-0.1j],
            [1.0-0.3j,     11.0, 0.9-0.2j],
            [0.5+0.1j, 0.9+0.2j,      9.0]
        ])
        sigma = 1.0
        n_snapshots = 100
        CRB_actual = crb_det_farfield_1d(ula, sources, self.wavelength, P, sigma, n_snapshots)
        CRB_expected = np.array([
            [ 2.636339e-06, -2.947961e-08, -2.894480e-08],
            [-2.947961e-08,  5.868361e-07, -1.254243e-08],
            [-2.894480e-08, -1.254243e-08,  1.495266e-06]
        ])
        npt.assert_allclose(CRB_actual, CRB_expected, rtol=1e-6)

    def test_sto_farfield_1d(self):
        ula = UniformLinearArray(10, self.wavelength / 2)
        sources = FarField1DSourcePlacement(np.linspace(-np.pi/3, np.pi/4, 3))
        P = np.array([
            [10.0    , 1.0+0.3j, 0.5-0.1j],
            [1.0-0.3j,     11.0, 0.9-0.2j],
            [0.5+0.1j, 0.9+0.2j,      9.0]
        ])
        sigma = 1.0
        n_snapshots = 100
        CRB_actual = crb_sto_farfield_1d(ula, sources, self.wavelength, P, sigma, n_snapshots)
        CRB_expected = np.array([
            [ 2.663109e-06, -3.037374e-08, -2.994223e-08],
            [-3.037374e-08,  5.922466e-07, -1.287051e-08],
            [-2.994223e-08, -1.287051e-08,  1.512018e-06]
        ])
        npt.assert_allclose(CRB_actual, CRB_expected, rtol=1e-6)

    def test_stouc_farfield_1d(self):
        ula = UniformLinearArray(10, self.wavelength / 2)
        sources = FarField1DSourcePlacement(np.linspace(-np.pi/3, np.pi/4, 3))
        p = np.array([2.0, 3.0, 1.0])
        sigma = 1.0
        n_snapshots = 100
        CRB_actual = crb_stouc_farfield_1d(ula, sources, self.wavelength, p, sigma, n_snapshots)
        CRB_expected = np.array([
            [ 1.3757938e-05, -3.7302575e-09, 4.3845873e-08],
            [-3.7302575e-09,  2.2173076e-06, 5.0642214e-09],
            [ 4.3845873e-08,  5.0642214e-09, 1.4740719e-05]
        ])
        npt.assert_allclose(CRB_actual, CRB_expected, rtol=1e-6)

    def test_convergence_farfield_1d(self):
        # The three CRBs should converge when SNR is sufficiently high.
        ula = UniformLinearArray(16, self.wavelength / 2)
        sources = FarField1DSourcePlacement(np.linspace(-np.pi/3, np.pi/4, 5))
        p = np.diag(np.full((sources.size,), 1000.0))
        sigma = 1.0 # 30 dB SNR
        n_snapshots = 10
        CRB_stouc = crb_stouc_farfield_1d(ula, sources, self.wavelength, p, sigma, n_snapshots)
        CRB_sto = crb_sto_farfield_1d(ula, sources, self.wavelength, p, sigma, n_snapshots)
        CRB_det = crb_det_farfield_1d(ula, sources, self.wavelength, p, sigma, n_snapshots)
        npt.assert_allclose(np.diag(CRB_sto), np.diag(CRB_stouc), rtol=1e-2)
        npt.assert_allclose(np.diag(CRB_det), np.diag(CRB_stouc), rtol=1e-2)

if __name__ == '__main__':
    unittest.main()
