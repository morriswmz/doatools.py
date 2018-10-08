import numpy as np
from math import ceil
from scipy.signal import find_peaks
import warnings
from ..model.arrays import UniformLinearArray
from ..model.sources import FarField1DSourcePlacement
from .core import SpectrumBasedEstimatorBase, get_noise_subspace

def f_music(A, En):
    r'''
    Function to evaluate the MUSIC spectrum. This is a vectorized
    implementation of the spectrum function:

    P_{\mathrm{MUSIC}}(\theta) = \frac{1}{a^H(\theta) E_n E_n^H a(\theta)}

    Args:
        A: M x K steering matrix of candidate direction-of-arrivals, where
            M is the number of sensors and K is the number of candidate
            direction-of-arrivals.
        En: M x D matrix of noise eigenvectors, where D is the dimension of the
            noise subspace.
    '''
    v = En.T.conj() @ A
    return np.reciprocal(np.sum(v * v.conj(), axis=0).real)

def _validate_estimation_input(design, R, k):
    '''
    A helper function that ensures the size of R matches the given design and
    the number of expected sources k is less than the size of R.
    '''
    if k >= design.size:
        raise ValueError('Too many sources.')
    if R.shape[0] != design.size or R.shape[0] != design.size:
        raise ValueError('The dimension of the sample covariance matrix does not match the array size.')

class MUSIC(SpectrumBasedEstimatorBase):

    def __init__(self, design, wavelength, search_grid, **kwargs):
        '''
        Creates a spectrum-based MUSIC estimator. The MUSIC spectrum is computed
        on a predefined-grid, and the source locations are estimated by
        identifying the peaks.

        Args:
            design (ArrayDesign): Array design.
            wavelength (float): Wavelength of the carrier wave.
            search_grid (SearchGrid): The search grid used to locate the
                sources.
        
        Example:


        References:
        [1] R. Schmidt, "Multiple emitter location and signal parameter
            estimation," IEEE Transactions on Antennas and Propagation,
            vol. 34, no. 3, pp. 276-280, Mar. 1986.
        '''
        super().__init__(design, wavelength, search_grid, **kwargs)
        
    def estimate(self, R, k, output_spectrum=False):
        '''
        Estimates the source locations from the given covariance matrix.

        Args:
            R (ndarray): Covariance matrix input. The size of R must match that
                of the array design used when creating this estimator.
            k (int): Expected number of sources.
            output_spectrum (bool): Set to True to also output the spectrum for
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
                if `output_spectrum` is True.
        '''
        _validate_estimation_input(self._design, R, k)
        En = get_noise_subspace(R, k)
        return self._estimate(lambda A: f_music(A, En), k, output_spectrum)

class RootMUSIC1D:

    def __init__(self, design, wavelength):
        '''
        Create a root-MUSIC estimator for the given uniform linear array.

        Args:
            design (ArrayDesign): A uniform linear array design.
            wavelength (float): Wavelength of the carrier wave.
        '''
        if not isinstance(design, UniformLinearArray):
            raise ValueError('Root-MUSIC currently only supports uniform linear arrays.')
        if design.has_perturbation('location_errors'):
            warnings.warn('Root-MUSIC does not consider location errors. Result may be inaccurate.')
        self._design = design
        self._wavelength = wavelength

    def estimate(self, R, k):
        '''
        Estimates the DOAs of 1D far-field sources from the give covariance
        matrix.

        Args:
            R (ndarray): Covariance matrix input. The size of R must match that of the
                array design used when creating this estimator.
            k (int): Expected number of sources.
        '''
        _validate_estimation_input(self._design, R, k)
        m = self._design.size
        En = get_noise_subspace(R, k)
        # Compute the coefficients for the polynomial.
        C = En @ En.T.conj()
        coeff = np.zeros((m - 1,), dtype=np.complex_)
        for i in range(1, m):
            coeff[i - 1] += np.sum(np.diag(C, i))
        coeff = np.hstack((coeff[::-1], np.sum(np.diag(C)), coeff.conj()))
        # Find the roots of the polynomial.
        z = np.roots(coeff)
        # Find k points inside the unit circle that are also closest to the unit
        # circle.
        nz = len(z)
        mask = np.ones((nz,), dtype=np.bool_)
        for i in range(nz):
            absz = abs(z[i])
            if absz > 1.0:
                # Outside the unit circle.
                mask[i] = False
            elif absz == 1.0:
                # On the unit circle. Need to find the closest point and remove
                # it.
                idx = -1
                dist = np.inf
                for j in range(nz):
                    if j != i and mask[j]:
                        cur_dist = abs(z[i] - z[j])
                        if cur_dist < dist:
                            dist = cur_dist
                            idx = j
                if idx < 0:
                    raise RuntimeError('Unpaired point found on the unit circle, which is impossible.')
                mask[idx] = False
        z = z[mask]
        sorted_indices = np.argsort(1.0 - np.abs(z))
        if len(z) < k:
            return False, None
        else:
            c = 2 * np.pi * self._design.d0 / self._wavelength
            phases = np.angle(z[sorted_indices[:k]]) / c
            locations = np.arcsin(phases)
            locations.sort()
            return True, FarField1DSourcePlacement(locations)
