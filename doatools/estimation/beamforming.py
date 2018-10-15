import numpy as np
from .core import SpectrumBasedEstimatorBase

def f_bartlett(A, R):
    r'''
    Compute the spectrum output of the Bartlett beamformer.

    P_{\mathrm{Bartlett}}(\theta) = a(\theta)^H R a(\theta)

    Args:
        A: M x K steering matrix of candidate direction-of-arrivals, where
            M is the number of sensors and K is the number of candidate
            direction-of-arrivals.
        R: M x M covariance matrix.
    '''
    return np.sum(A.conj() * (R @ A), axis=0).real

def f_mvdr(A, R):
    r'''
    Compute the spectrum output of the Bartlett beamformer.

    P_{\mathrm{Bartlett}}(\theta) = 1/(a(\theta)^H R^{-1} a(\theta))

    Args:
        A: M x K steering matrix of candidate direction-of-arrivals, where
            M is the number of sensors and K is the number of candidate
            direction-of-arrivals.
        R: M x M covariance matrix.
    '''
    return 1.0 / np.sum(A.conj() * np.linalg.lstsq(R, A, None)[0], axis=0).real

def _validate_covariance_input(design, R):
    '''
    A helper function that ensures the size of R matches the given design.
    '''
    if R.shape[0] != design.size or R.shape[0] != design.size:
        raise ValueError('The dimension of the sample covariance matrix does not match the array size.')

class BartlettBeamformer(SpectrumBasedEstimatorBase):

    def __init__(self, design, wavelength, search_grid, **kwargs):
        '''
        Creates a Barlett-beamformer based estimator. The spectrum is computed
        on a predefined-grid, and the source locations are estimated by
        identifying the peaks.

        Args:
            design (ArrayDesign): Array design.
            wavelength (float): Wavelength of the carrier wave.
            search_grid (SearchGrid): The search grid used to locate the
                sources.

        References:
        [1] H. L. Van Trees, Optimum array processing. New York: Wiley, 2002.
        '''
        super().__init__(design, wavelength, search_grid, **kwargs)
        
    def estimate(self, R, k, return_spectrum=False):
        '''
        Estimates the source locations from the given covariance matrix.

        Args:
            R (ndarray): Covariance matrix input. The size of R must match that
                of the array design used when creating this estimator.
            k (int): Expected number of sources.
            return_spectrum (bool): Set to True to also output the spectrum for
                visualization. Default value if False.
        
        Returns:
            resolved (bool): A boolean indicating if the desired number of sources
                are resolved. If resolved is False, both `estimates` and
                `spectrum` will be None.
            estimates (SourcePlacement): A SourcePlacement instance of the same
                type as the one used in the search grid, represeting the
                estimated DOAs. Will be `None` if resolved is False.
            spectrum (ndarray): A numpy array of the same shape of the
                specified search grid, consisting of values evaluated at the
                grid points. Will be `None` if resolved is False. Only present
                if `return_spectrum` is True.
        '''
        _validate_covariance_input(self._design, R)
        return self._estimate(lambda A: f_bartlett(A, R), k, return_spectrum)

class MVDRBeamformer(SpectrumBasedEstimatorBase):

    def __init__(self, design, wavelength, search_grid, **kwargs):
        '''
        Creates a MVDR-beamformer based estimator. The spectrum is computed
        on a predefined-grid, and the source locations are estimated by
        identifying the peaks.

        Args:
            design (ArrayDesign): Array design.
            wavelength (float): Wavelength of the carrier wave.
            search_grid (SearchGrid): The search grid used to locate the
                sources.

        References:
        [1] H. L. Van Trees, Optimum array processing. New York: Wiley, 2002.
        '''
        super().__init__(design, wavelength, search_grid, **kwargs)
        
    def estimate(self, R, k, return_spectrum=False):
        '''
        Estimates the source locations from the given covariance matrix.

        Args:
            R (ndarray): Covariance matrix input. The size of R must match that
                of the array design used when creating this estimator.
            k (int): Expected number of sources.
            return_spectrum (bool): Set to True to also output the spectrum for
                visualization. Default value if False.
        
        Returns:
            resolved (bool): A boolean indicating if the desired number of sources
                are resolved. If resolved is False, both `estimates` and
                `spectrum` will be None.
            estimates (SourcePlacement): A SourcePlacement instance of the same
                type as the one used in the search grid, represeting the
                estimated DOAs. Will be `None` if resolved is False.
            spectrum (ndarray): A numpy array of the same shape of the
                specified search grid, consisting of values evaluated at the
                grid points. Will be `None` if resolved is False. Only present
                if `return_spectrum` is True.
        '''
        _validate_covariance_input(self._design, R)
        return self._estimate(lambda A: f_mvdr(A, R), k, return_spectrum)
