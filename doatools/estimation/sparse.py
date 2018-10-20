import warnings
from abc import ABC, abstractmethod
import numpy as np
from .core import SpectrumBasedEstimatorBase, ensure_covariance_size
from ..optim.l1lsq import L1RegularizedLeastSquaresProblem
from ..utils.math import khatri_rao, vec

class SparseCovarianceMatching(SpectrumBasedEstimatorBase):

    def __init__(self, array, wavelength, search_grid, noise_known=False,
                 formulation='penalizedl1', **kwargs):
        r"""Creates a source location estimator based on matching the sparse
        representation of the covariance matrix.

        The sources are assumed to be uncorrelated. After discretizing the
        parameter space of source locations into a fine grid, the vectorized
        covariance matrix can be represented by
            r = [A^* \odot A, vec(I)][ p ]
                                     [ s ]
        where A is the steering matrix of the discretized source locations, p is
        a sparse vector whose non-zero locations correspond to potential source
        locations, and s is the noise variance.

        Let Phi = [A^* \odot A, vec(I)], and x = [p^T s]^T. We have r = Phi x,
        where x is sparse. We can this expression the sparse representation of
        the covariance matrix. Given the estimates of r, r_est, we can formulate
        a sparse recovery problem to recovery the sparse vector x, and then
        recover the source locations.

        Args:
            array (ArrayDesign): Array design.
            wavelength (float): Wavelength of the carrier wave.
            search_grid (SearchGrid): The search grid used to locate the
                sources.
            formulation (str): 'penalizedl1', 'constrainedl1', or
                'constrainedl2'

        References:
        [1] D. Malioutov, M. Cetin, and A. S. Willsky, "A sparse signal
            reconstruction perspective for source localization with sensor
            arrays," IEEE Transactions on Signal Processing, vol. 53, no. 8,
            pp. 3010-3022, Aug. 2005.
        [2] Y. D. Zhang, M. G. Amin, and B. Himed, "Sparsity-based DOA
            estimation using co-prime arrays," in 2013 IEEE International
            Conference on Acoustics, Speech and Signal Processing (ICASSP),
            2013, pp. 3967-3971.
        [3] Z. Tan and A. Nehorai, "Sparse direction of arrival estimation using
            co-prime arrays with off-grid targets," IEEE Signal Processing
            Letters, vol. 21, no. 1, pp. 26-29, Jan. 2014.
        """
        super().__init__(array, wavelength, search_grid, **kwargs)
        self._formulation = formulation
        self._noise_known = noise_known
        # vec(R) -> m*m elements, real + image -> 2*m*m
        m = 2 * self._array.size**2
        k = self._search_grid.size
        # If noise is not known, we need a additional column for vec(I).
        if not self._noise_known:
            k += 1
        # Initialize the problem.
        self._problem = L1RegularizedLeastSquaresProblem(m, k, formulation, True)

    def _get_atom_matrix(self, alt_grid=None):
        if alt_grid is not None:
            sources = alt_grid.source_placement
            need_compute = True
        else:
            sources = self._search_grid.source_placement
            need_compute = self._atom_matrix is None
        if need_compute:
            A = self._array.steering_matrix(sources, self._wavelength,
                                             perturbations='known')
            Phi = khatri_rao(A.conj(), A)
            if not self._noise_known:
                Phi = np.hstack((Phi, vec(np.eye(self._array.size))))
            Phi = np.vstack((Phi.real, Phi.imag))
            # Cache when possible.
            if alt_grid is None and self._enable_caching:
                self._atom_matrix = Phi
            return Phi
        else:
            return self._atom_matrix

    def _call_solver(self, Phi, R, l, solver_options):
        r = vec(R)
        r = np.vstack((r.real, r.imag))
        sol = self._problem.solve(Phi, r, l, **solver_options).flatten()
        if not self._noise_known:
            # The last element is the noise variance estimate.
            # TODO: output noise estimate?
            sol = sol[:-1]
        return sol
    
    def estimate(self, R, k, l, sigma=None, solver_options={}, **kwargs):
        r"""Estimates the source locations from the given covariance matrix.

        When the sources are uncorrelated, 

        Args:
            R (ndarray): Covariance matrix input. The size of R must match that
                of the array design used when creating this estimator.
            k (int): Expected number of sources.
            l (float): The meaning of this parameter depends on the formulation:
                * 'penalizedl1': regularization parameter of the l1 penalty
                  term.
                * 'constrainedl1': upper bound of the l1 norm of the signal
                  vector.
                * 'constrainedl2': upper bound of the l2 norm of the residual.
            solver_options: A dictionary of additional keyword arguments to be
                passed to the optimization problem solver. For instance, you can
                specify the solver or set the verbosity.
            return_spectrum (bool): Set to True to also output the spectrum for
                visualization. Default value if False.
        
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
        if 'refine_estimates' in kwargs:
            raise ValueError('Grid refinement is not supported.')
        ensure_covariance_size(R, self._array)
        if self._noise_known:
            if sigma is None:
                raise ValueError('sigma must be specified when noise variance is assumed known.')
            # Do not modify R in-place!
            R = R - np.eye(self._array.size) * sigma
        return self._estimate(lambda Phi: self._call_solver(Phi, R, l, solver_options), k, **kwargs)

