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
    noise_dim = R.shape[0] - k
    return E[:,:noise_dim]

class SpectrumBasedEstimatorBase:

    def __init__(self, design, wavelength, search_grid, peak_finder=find_peaks_simple):
        '''
        Base class for a spectrum-based estimator.

        Args:
            design: Array design.
            wavelength: Wavelength of the carrier wave.
            search_grid: The search grid used to locate the sources.
            peak_finder: A callable object that accepts an ndarray and returns
                a tuple containing the indices representing the peak locations,
                where the length of this tuple should be the number of
                dimensions of the input ndarray.
        '''
        self._design = design
        self._wavelength = wavelength
        self._search_grid = search_grid
        self._peak_finder = peak_finder

    def _estimate(self, f_sp, k, output_spectrum):
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
            output_spectrum: Set to True to also output the spectrum for
                visualization.
        
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
        A_grid = self._design.steering_matrix(
            self._search_grid.source_placement,
            self._wavelength, perturbations='known'
        )
        sp = f_sp(A_grid)
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
            if output_spectrum:
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
            if output_spectrum:
                return True, estimates, sp
            else:
                return True, estimates
        
    def _refine_estimates(self, f_sp, estimates):
        pass



