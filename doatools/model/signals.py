from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import sqrtm

class SignalGenerator(ABC):

    @property
    @abstractmethod
    def dim(self):
        '''
        Retrieves the dimension of the signal generator.
        '''
        pass

    @abstractmethod
    def emit(self, n):
        '''
        Generates a k x n matrix where k is the dimension of the signal and
        each column represents a sample.
        '''
        pass

class ComplexStochasticSignal(SignalGenerator):

    def __init__(self, C, n=None):
        '''
        Creates a signal generator that generates zero-mean complex
        circularly-symmetric Gaussian signals.

        Args:
            C: Covariance matrix of the complex Gaussian distribution.
                Can be specified by
                1. A full covariance matrix.
                2. An real vector denoting the diagonals of the covariance
                   matrix if the covariance matrix is diagonal.
                3. A scalar if the covariance matrix is diagonal and all
                   diagonal elements share the same value. In this case,
                   parameter n must be specified.
            n: Dimension of the complex Gaussian distribution. Only need to be
                specified when `C` is a scalar.
        '''
        self._C = C
        if np.isscalar(C):
            self._dim = n
        else:
            self._dim = C.shape[0]
    
    @property
    def dim(self):
        return self._dim
    
    @property
    def covariance(self):
        '''Retrieves the covariance of the complex Gaussian distribution.'''
        return self._C
    
    @covariance.setter
    def covariance(self, C):
        '''Sets the covariance of the complex Gaussian distribution.'''
        if not np.isscalar(C):
            if C.ndim > 2 or any(map(lambda x: x != self._dim, C.shape)):
                raise ValueError('Expecting a scalar, an 1D vector of length {0}, or a matrix of size {0}x{0}'.format(self._dim))
        self._C = C

    def emit(self, n):
        k = self.dim
        S = 1j * np.random.randn(k, n)
        S += np.random.randn(k, n)
        # Apply the covariance transform.
        if np.isscalar(self._C):
            S *= np.sqrt(self._C / 2.)
        elif self._C.ndim == 1:
            S = np.sqrt(self._C.reshape((-1, 1)) / 2.) * S
        else:
            S = sqrtm(self._C / 2.) @ S
        return S

