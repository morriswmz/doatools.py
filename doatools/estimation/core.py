from collections import namedtuple
from abc import ABC, abstractmethod
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter

# Helper functions for validating inputs.
def ensure_covariance_size(R, array):
    """Ensures the size of R matches the given array design."""
    m = array.size
    if R.ndim != 2:
        raise ValueError('Expecting a matrix.')
    if R.shape[0] != m or R.shape[0] != m:
        raise ValueError(
            'The shape of the covariance matrix does not match the array size.'
            'Expected shape is {0}. Got {1}'
            .format((m, m), R.shape)
        )

def ensure_n_resolvable_sources(k, max_k):
    """Checks if the number of expected sources exceeds the maximum resolvable sources."""
    if k > max_k:
        raise ValueError(
            'Too many sources. Maximum number of resolvable sources is {0}'
            .format(max_k)
        )

def find_peaks_simple(x):
    if x.ndim == 1:
        # Delegate to scipy's peak finder.
        return find_peaks(x)[0],
    else:
        # Use maximum filter for peak finding.
        y = maximum_filter(x, 3)
        return np.where(x == y)

def get_noise_subspace(R, k):
    """
    Gets the noise eigenvectors.

    Args:
        R: Covariance matrix.
        k: Number of sources.
    """
    _, E = np.linalg.eigh(R)
    # Note: eigenvalues are sorted in ascending order.
    return E[:,:-k]

class SpectrumBasedEstimatorBase(ABC):

    def __init__(self, array, wavelength, search_grid,
                 peak_finder=find_peaks_simple, enable_caching=True):
        """Base class for a spectrum-based estimator.

        Args:
            array: Array design.
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
        """
        self._array = array
        self._wavelength = wavelength
        self._search_grid = search_grid
        self._peak_finder = peak_finder
        self._enable_caching = enable_caching
        self._atom_matrix = None
    
    def _compute_atom_matrix(self, grid):
        """Computes the atom matrix for spectrum computation.
        
        An atom matrix, A, is an M x K matrix, where M is the number of sensors
        and K is equal to the size of the search grid. For instance, in MUSIC,
        the atom matrix is just the steering matrix. The spectrum output for
        the k-th grid point is given by |a_k E_n|^2, where a_k is the k-th
        column of A.

        Because A is actually the steering matrix in many spectrum based
        estimators (e.g., MVDR, MUSIC), the default implementation will create
        steering matrice. 

        Args:
            grid: The search grid used to generate the atom matrix.
        """
        # Default implementation: steering matrix.
        return self._array.steering_matrix(
            grid.source_placement, self._wavelength,
            perturbations='known'
        )

    def _get_atom_matrix(self, alt_grid=None):
        """Retrieves the atom matrix for spectrum computation.

        See `_compute_atom_matrix` for more details on the atom matrix.

        Args:
            alt_grid: If specified, will retrieve the atom matrix for this
                grid instead of the default search_grid. Used in the grid
                refinement process. Default value is None and the atom matrix
                for the default search grid is returned.
        """
        if alt_grid is not None:
            return self._compute_atom_matrix(alt_grid)
        # Check cached version of the default search grid if possible.
        if self._atom_matrix is not None:
            return self._atom_matrix
        A = self._compute_atom_matrix(self._search_grid)
        if self._enable_caching:
            self._atom_matrix = A
        return A

    def _estimate(self, f_sp, k, return_spectrum=False, refine_estimates=False,
                  refinement_density=10, refinement_iters=3):
        """
        A generic implementation of the estimation process: compute the spectrum
        -> identify the peaks -> locate the largest peaks as estimates.

        Subclasses can implement `f_sp` and call this method to obtain the
        estimates.

        Args:
            f_sp: A callable object that accepts the atom matrix as the
                parameter and return a 1D numpy array representing the computed
                spectrum.
            k (int): Expected number of sources. 
            return_spectrum: Set to True to also output the spectrum for
                visualization.
            refine_estimates: Set to True to enable grid refinement to obtain
                potentially more accurate estimates.
            refinement_density: Density of the refinement grids. Higher density
                values lead to denser refinement grids and increased
                computational complexity. Default value is 10.
            refinement_iters: Number of refinement iterations. More iterations
                generally lead to better results, at the cost of increased
                computational complexity. Default value is 3.
        
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
                grid points. Only present if `return_spectrum` is True.
        """
        sp = f_sp(self._get_atom_matrix())
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
                return False, None, sp
            else:
                return False, None
        else:
            # Obtain the peak values for sorting. Remember that `peak_indices`
            # is a tuple of 1D numpy arrays, and `sp` has been reshaped.
            peak_values = sp[peak_indices]
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
            if refine_estimates:
                # Convert sorted flattened indices back to a tuple of coordinate
                # arrays.
                peak_indices = np.unravel_index(flattened_indices, self._search_grid.shape)
                self._refine_estimates(f_sp, estimates, peak_indices,
                                       refinement_density, refinement_iters)
            if return_spectrum:
                return True, estimates, sp
            else:
                return True, estimates
        
    def _refine_estimates(self, f_sp, est0, peak_indices, density=10, n_iters=3):
        """Refines the estimates.
        
        Given the i-th estimate, a refined grid will be created around it. The
        spectrum function will be evaluated on this refined grid and a new peak
        will be located to update the i-th estimate. This process is repeated
        several times.

        Args:
            f_sp: A callable object that accepts the steering matrix as the
                parameter and return a 1D numpy array representing the computed
                spectrum.
            est0: Initial estimates.
            peak_indices: A tuple of indices arrays representing the coordinates
                of the initial estimates on the original search grid.
            density: Refinement density.
            n_iters: Number of refinement iterations.
        """
        # We modify the estimated locations **in-place** here.
        locations = est0.locations
        # Create initial refined grids.
        subgrids = self._search_grid.create_refined_grids_at(*peak_indices, density=density)
        for r in range(n_iters):
            for i in range(len(subgrids)):
                g = subgrids[i]
                # Refine the i-th estimate.
                A = self._get_atom_matrix(g)
                sp = f_sp(A)
                i_max = sp.argmax() # argmax for the flattened spectrum.
                # Update the initial estimates in-place.
                locations[i] = g.source_placement[i_max]
                if r == n_iters - 1:
                    continue
                # Continue to create finer grids.
                peak_coord = np.unravel_index(i_max, g.shape)
                subgrids[i] = g.create_refined_grid_at(peak_coord, density=density)
