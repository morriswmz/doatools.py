from collections import namedtuple
from abc import ABC, abstractmethod
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter

def find_peaks_simple(x):
    if x.ndim == 1:
        # Delegate to scipy's peak finder.
        return find_peaks(x)[0],
    else:
        # Use maximum filter for peak finding.
        y = maximum_filter(x, 3)
        return np.where(x == y)

def get_noise_subspace(R, k):
    '''
    Gets the noise eigenvectors.

    Args:
        R: Covariance matrix.
        k: Number of sources.
    '''
    _, E = np.linalg.eigh(R)
    # Note: eigenvalues are sorted in ascending order.
    return E[:,:-k]

class SpectrumBasedEstimatorBase:

    def __init__(self, design, wavelength, search_grid,
                 peak_finder=find_peaks_simple, enable_caching=True):
        '''Base class for a spectrum-based estimator.

        Args:
            design: Array design.
            wavelength: Wavelength of the carrier wave.
            search_grid: The search grid used to locate the sources.
            peak_finder: A callable object that accepts an ndarray and returns
                a tuple containing the indices representing the peak locations,
                where the length of this tuple should be the number of
                dimensions of the input ndarray.
            enable_caching: If set to True, the steering matrix for the given
                search grid will be cached. Otherwise the steering matrix will
                be computed everything `estimate()` is called. Because the array
                and the search grid are supposed to remain unchanged, caching
                the steering matrix will save a lot of computations for dense
                grids in Monte Carlo simulations. Default value is True.
        '''
        self._design = design
        self._wavelength = wavelength
        self._search_grid = search_grid
        self._peak_finder = peak_finder
        self._enable_caching = enable_caching
        self._A = None

    def _get_steering_matrix(self):
        if self._A is not None:
            return self._A
        A = self._design.steering_matrix(
            self._search_grid.source_placement,
            self._wavelength, perturbations='known'
        )
        if self._enable_caching:
            self._A = A
        return A

    def _estimate(self, f_sp, k, return_spectrum):
        '''
        A generic implementation of the estimation process: compute the spectrum
        -> identify the peaks -> locate the largest peaks as estimates.

        Subclasses can implement `f_sp` and call this method to obtain the
        estimates.

        Args:
            f_sp: A callable object that accepts the steering matrix as the
                parameter and return a 1D numpy array representing the computed
                spectrum.
            k (int): Expected number of sources. 
            return_spectrum: Set to True to also output the spectrum for
                visualization.
        
        Returns:
            resolved (bool): A boolean indicating if the desired number of
                sources are found. This flag does not guarantee that the
                estimated source locations are correct. The estimated source
                locations may be completely wrong!
                If resolved is False, both `estimates` and `spectrum` will be
                None.
            estimates (SourcePlacement): A SourcePlacement instance of the same
                type as the one used in the search grid, represeting the
                estimated DOAs. Will be `None` if resolved is False.
            spectrum (ndarray): A numpy array of the same shape of the
                specified search grid, consisting of values evaluated at the
                grid points. Will be `None` if resolved is False. Only present
                if `return_spectrum` is True.
        '''
        sp = f_sp(self._get_steering_matrix())
        # Restores the shape of the spectrum.
        sp = sp.reshape(self._search_grid.shape)
        # Find peak locations.
        peak_indices = self._peak_finder(sp)
        # The peak finder returns a tuple whose length is at least one. Hence
        # we can get the number of peaks by checking the length of the first
        # element in the tuple.
        n_peaks = len(peak_indices[0])
        if n_peaks < k:
            # Not enough peaks.
            if return_spectrum:
                return False, None, None
            else:
                return False, None
        else:
            # Obtain the peak values for sorting. Remember that `peak_indices`
            # is a tuple of 1D numpy arrays, and `sp` has been reshaped.
            peak_values = [sp[t] for t in zip(*peak_indices)]
            # Identify the k largest peaks.
            top_indices = np.argsort(peak_values)[-k:]
            # Filter out the peak indices of the k largest peaks.
            peak_indices = [axis[top_indices] for axis in peak_indices]
            # Obtain the estimates.
            # Note that we need to convert n-d indices to flattened indices.
            # We sorted the flattened indices here to respect the ordering of
            # source locations in the search grid.
            flattened_indices = np.ravel_multi_index(peak_indices, self._search_grid.shape)
            flattened_indices.sort()
            estimates = self._search_grid.source_placement[flattened_indices]
            if return_spectrum:
                return True, estimates, sp
            else:
                return True, estimates
        
    def _refine_estimates(self, f_sp, estimates):
        pass



