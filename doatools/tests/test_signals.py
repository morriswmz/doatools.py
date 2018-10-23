import unittest
import numpy as np
import numpy.testing as npt
from doatools.model.signals import ComplexStochasticSignal, RandomPhaseSignal

class TestSignals(unittest.TestCase):

    def test_complex_stochastic(self):
        np.random.seed(42)
        p = np.array([1., 2., 3., 2.])
        signal = ComplexStochasticSignal(p.size, p)
        n_snapshots = 10000
        S = signal.emit(n_snapshots)
        S = (S @ S.T.conj()) / n_snapshots
        self.assertLessEqual(np.linalg.norm(S - np.diag(p), 'fro'), 2e-1)

    def test_random_phase(self):
        np.random.seed(42)
        amps = np.array([1., 2., 3., 1.])
        signal = RandomPhaseSignal(amps.size, amps)
        n_snapshots = 10000
        S = signal.emit(n_snapshots)
        S = (S @ S.T.conj()) / n_snapshots
        self.assertLessEqual(np.linalg.norm(S - np.diag(amps**2), 'fro'), 1e-1)

if __name__ == '__main__':
    unittest.main()
