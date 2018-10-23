from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import sqrtm
from ..utils.math import randcn

class SignalGenerator(ABC):
    """Abstrace base class for all signal generators.
    
    Extend this class to create your own signal generators.
    """

    @property
    @abstractmethod
    def dim(self):
        """Retrieves the dimension of the signal generator."""
        pass

    @abstractmethod
    def emit(self, n):
        """Emits the signal matrix.

        Generates a k x n matrix where k is the dimension of the signal and
        each column represents a sample.
        """
        pass

class ComplexStochasticSignal(SignalGenerator):
    """Creates a signal generator that generates zero-mean complex
    circularly-symmetric Gaussian signals.

    Args:
        dim (int): Dimension of the complex Gaussian distribution. Must match
            the size of ``C`` if ``C`` is not a scalar.
        C: Covariance matrix of the complex Gaussian distribution.
            Can be specified by

            1. A full covariance matrix.
            2. An real vector denoting the diagonals of the covariance
               matrix if the covariance matrix is diagonal.
            3. A scalar if the covariance matrix is diagonal and all
               diagonal elements share the same value. In this case,
               parameter n must be specified.
            
            Default value is `1.0`.
    """

    def __init__(self, dim, C=1.0):
        self._dim = dim
        if np.isscalar(C):
            # Scalar
            self._C2 = np.sqrt(C)
            self._generator = lambda n: self._C2 * randcn((self._dim, n))
        elif C.ndim == 1:
            # Vector
            if C.size != dim:
                raise ValueError('The size of C must be {0}.'.format(dim))
            self._C2 = np.sqrt(C).reshape((-1, 1))
            self._generator = lambda n: self._C2 * randcn((self._dim, n))
        elif C.ndim == 2:
            # Matrix
            if C.shape[0] != dim or C.shape[1] != dim:
                raise ValueError('The shape of C must be ({0}, {0}).'.format(dim))
            self._C2 = sqrtm(C)
            self._generator = lambda n: self._C2 @ randcn((self._dim, n))
        else:
            raise ValueError(
                'The covariance must be specified by a scalar, a vector of'
                'size {0}, or a matrix of {0}x{0}.'.format(dim)
            )
        self._C = C
    
    @property
    def dim(self):
        return self._dim

    def emit(self, n):
        return self._generator(n)

class RandomPhaseSignal(SignalGenerator):
    r"""Creates a random phase signal generator.

    The phases are uniformly and independently sampled from :math:`[-\pi, \pi]`.

    Args:
        dim (int): Dimension of the signal (usually equal to the number of
            sources).
        amplitudes: Amplitudes of the signal. Can be specified by
            
            1. A scalar if all sources have the same amplitude.
            2. A vector if the sources have different amplitudes.
    """

    def __init__(self, dim, amplitudes=1.0):
        self._dim = dim
        if np.isscalar(amplitudes):
            self._amplitudes = np.full(amplitudes, (dim, 1))
        else:
            if amplitudes.size != dim:
                raise ValueError("The size of 'amplitudes' does not match the value of 'dim'.")
            self._amplitudes = amplitudes.reshape((-1, 1))

    @property
    def dim(self):
        return self._dim

    def emit(self, n):
        phases = np.random.uniform(-np.pi, np.pi, (self._dim, n))
        c = np.sin(phases) * 1j
        c += np.cos(phases)
        return self._amplitudes * c
