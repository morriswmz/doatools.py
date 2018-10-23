import warnings
from abc import ABC, abstractmethod
import numpy as np
from .core import SpectrumBasedEstimatorBase, ensure_covariance_size
from ..optim.l1lsq import L1RegularizedLeastSquaresProblem, \
                          L21RegularizedLeastSquaresProblem
from ..utils.math import khatri_rao, vec

class SparseCovarianceMatching(SpectrumBasedEstimatorBase):
    r"""Creates a source location estimator based on matching the sparse
    representation of the covariance matrix.

    The sources are assumed to be uncorrelated. Take 1D far-field sources
    as an example. After discretizing the range of source locations into a fine
    grid of :math:`G` points,
    :math:`\mathbf{\theta} = [\theta_1, \ldots, \theta_G]^T`, the vectorized
    covariance matrix can be expressed as

    .. math::

        \mathbf{r} = \begin{bmatrix}
            \mathbf{A}^*(\mathbf{\theta}) \odot \mathbf{A}(\mathbf{\theta}) &
            \mathrm{vec}(\mathbf{I})
        \end{bmatrix}
        \begin{bmatrix} \mathbf{p} \\ \sigma^2 \end{bmatrix}
    
    where :math:`\mathbf{A}` is the steering matrix of the discretized source
    locations, :math:`\mathbf{p} \in \mathbb{R}_+^G` is a sparse vector of the
    source powers, and :math:`\sigma^2` is the noise variance. If all sources
    are located on the grid, :math:`p_k` should be non-zero if there is a
    source located at :math:`\theta_k` and zero otherwise.

    Let :math:`\mathbf{\Phi} = \lbrack \mathbf{A}^* \odot \mathbf{A}, \mathrm{vec}(\mathbf{I}) \rbrack`,
    and :math:`\mathbf{x} = \lbrack \mathbf{p}^T, \sigma^2 \rbrack^T`. We have
    :math:`\mathbf{r} = \mathbf{\Phi} \mathbf{x}`, where :math:`\mathbf{x}` is
    sparse. We call this expression the sparse representation of the covariance
    matrix. Given the estimate of :math:`\mathbf{r}`, :math:`\hat{\mathbf{r}}`,
    we can formulate a sparse recovery problem to recovery the sparse vector
    :math:`\mathbf{x}`, and then recover the source locations.

    Args:
        array (~doatools.model.arrays.ArrayDesign): Array design.
        wavelength (float): Wavelength of the carrier wave.
        search_grid (~doatools.estimation.grid.SearchGrid): The search grid
            used to locate the sources.
        formulation (str): ``'penalizedl1'``, ``'constrainedl1'``, or
            ``'constrainedl2'``.
        **kwargs: Other keyword arguments supported by
            :class:`~doatools.estimation.core.SpectrumBasedEstimatorBase`.

    References:
        [1] J. Yin and T. Chen, "Direction-of-Arrival Estimation Using a Sparse
        Representation of Array Covariance Vectors," IEEE Transactions on
        Signal Processing, vol. 59, no. 9, pp. 4489–4493, Sep. 2011.

        [2] Y. D. Zhang, M. G. Amin, and B. Himed, "Sparsity-based DOA
        estimation using co-prime arrays," in 2013 IEEE International
        Conference on Acoustics, Speech and Signal Processing (ICASSP),
        2013, pp. 3967-3971.
        
        [3] Z. Tan and A. Nehorai, "Sparse direction of arrival estimation using
        co-prime arrays with off-grid targets," IEEE Signal Processing
        Letters, vol. 21, no. 1, pp. 26-29, Jan. 2014.
    """

    def __init__(self, array, wavelength, search_grid, noise_known=False,
                 formulation='penalizedl1', **kwargs):
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

    def _compute_atom_matrix(self, grid):
        A = self._array.steering_matrix(
            grid.source_placement, self._wavelength,
            perturbations='known'
        )
        Phi = khatri_rao(A.conj(), A)
        if not self._noise_known:
            Phi = np.hstack((Phi, vec(np.eye(self._array.size))))
        Phi = np.vstack((Phi.real, Phi.imag))
        return Phi

    def _get_sparse_spectrum(self, Phi, R, l, solver_options):
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

        Args:
            R (~numpy.ndarray): Covariance matrix input. The size of R must
                match that of the array design used when creating this
                estimator.
            
            k (int): Expected number of sources.
            
            l (float): The regularization parameter. The meaning of this
                parameter depends on the formulation:
                
                * ``'penalizedl1'``: regularization parameter of the l1 penalty
                  term. Larger values of ``l`` usually leads to more sparse
                  solutions, at the cost of increased biases.
                * ``'constrainedl1'``: upper bound of the l1 norm of the signal
                  vector. Smaller values of ``l`` usually leads to more sparse
                  solutions, at the cost of increased biases.
                * ``'constrainedl2'``: upper bound of the l2 norm of the
                  residual. Smaller values of ``l`` usually leads to better
                  reconstruction results. However the optimization problem
                  will become infeasible if ``l`` is too small.
                
                See :class:`~doatools.optim.l1lsq.L1RegularizedLeastSquaresProblem`
                for more details.
            
            solver_options (dict): A dictionary of additional keyword arguments
                to be passed to the optimizer. For instance, you can specify the
                solver or set the verbosity.
            
            return_spectrum (bool): Set to ``True`` to also output the spectrum
                for visualization. Default value if ``False``.
                
        Returns:
            A tuple with the following elements.

            * resolved (:class:`bool`): ``True`` is the solver successfully
              obtained the sparse solution and the peak finder successfully
              identified the desired number of peaks. This flag does **not**
              guarantee that the estimated source locations are correct. The
              estimated source locations may be completely wrong!
              If resolved is False, both ``estimates`` and ``spectrum`` will be
              ``None``.
            * estimates (:class:`~doatools.model.sources.SourcePlacement`):
              A :class:`~doatools.model.sources.SourcePlacement` instance of the
              same type as the one used in the search grid, represeting the
              estimated source locations. Will be ``None`` if resolved is
              ``False``.
            * spectrum (:class:`~numpy.ndarray`): An numpy array of the same
              shape of the specified search grid, consisting of values evaluated
              at the grid points. Only present if ``return_spectrum`` is
              ``True``.
        """
        if 'refine_estimates' in kwargs:
            raise ValueError('Grid refinement is not supported.')
        ensure_covariance_size(R, self._array)
        if self._noise_known:
            if sigma is None:
                raise ValueError('sigma must be specified when noise variance is assumed known.')
            # Do not modify R in-place!
            R = R - np.eye(self._array.size) * sigma
        f_sp = lambda Phi: self._get_sparse_spectrum(Phi, R, l, solver_options)
        return self._estimate(f_sp, k, **kwargs)

class GroupSparseEstimator(SpectrumBasedEstimatorBase):
    r"""Creates a group-sparsity based estimator.

    The group-sparsity based estimator considers the following mulitple
    measurement vector (MMV) model:

    .. math::
    
        \mathbf{Y} = \mathbf{A} \mathbf{X} + \mathbf{N},

    where each column of :math:`\mathbf{Y}` represents a single snapshot
    vector, each column of :math:`\mathbf{X}` represents a single source signal
    vector, each column of :math:`\mathbf{N}` represents a single noise vector.

    Similar to :class:`SparseCovarianceMatching`, we discretize the range of
    source locations into a fine grid, and :math:`\mathbf{A}` is the steering
    matrix of the discretized source locations. If we can find out the non-zero
    rows of :math:`\mathbf{X}`, we can then recover the source locations.

    The group-sparsity based estimator solves the following mixed-norm
    optimization problem:

    .. math::

        \min_{\mathbf{X}} \frac{1}{2}\| \mathbf{A}\mathbf{X} - \mathbf{Y} \|_2
        + \lambda \| \mathbf{X} \|_{2,1},

    where :math:`\lambda` is the regularization parameter, and

    .. math::
    
        \| \mathbf{X} \|_{2,1}
        =\sum_{i} \left(\sum_{j} |X_{ij}|^2\right)^{\frac{1}{2}}.
    
    After recovering :math:`\mathbf{X}`, the :math:`l_2` norms of the rows of
    :math:`\mathbf{X}` forms a pseudo spectrum. We can then find the source
    locations by identifying the largest peaks.

    Args:
        array (~doatools.model.arrays.ArrayDesign): Array design.
        wavelength (float): Wavelength of the carrier wave.
        search_grid (~doatools.estimation.grid.SearchGrid): The search grid
            used to locate the sources.
        n_snapshots (int): Number of snapshots used.
        **kwargs: Other keyword arguments supported by
            :class:`~doatools.estimation.core.SpectrumBasedEstimatorBase`.
    
    References:
        [1] D. Malioutov, M. Cetin, and A. S. Willsky, "A sparse signal
        reconstruction perspective for source localization with sensor
        arrays," IEEE Transactions on Signal Processing, vol. 53, no. 8,
        pp. 3010-3022, Aug. 2005.
    """

    def __init__(self, array, wavelength, search_grid, n_snapshots, **kwargs):
        super().__init__(array, wavelength, search_grid, **kwargs)
        self._n_snapshots = n_snapshots
        self._problem = L21RegularizedLeastSquaresProblem(
            array.size, search_grid.size, n_snapshots, True
        )
    
    def _get_sparse_spectrum(self, A, Y, l, solver_options):
        X = self._problem.solve(A, Y, l, **solver_options)
        return np.linalg.norm(X, ord=2, axis=1)

    def estimate(self, Y, k, l, solver_options={}, **kwargs):
        """Estimates the source locations from the given measurements.

        Args:
            Y (~numpy.ndarray): The matrix of measurements, each column of which
                represents a single snapshot.
            k (int): Expected number of sources.
            l (float): The regularization parameter. Larger values of ``l``
                usually leads to more sparse solutions, at the cost of increased
                biases.
            solver_options (dict): A dictionary of additional keyword arguments
                to be passed to the optimizer. For instance, you can specify the
                solver or set the verbosity.
            return_spectrum (bool): Set to ``True`` to also output the spectrum
                for visualization. Default value if ``False``.
                
        Returns:
            A tuple with the following elements.

            * resolved (:class:`bool`): ``True`` is the solver successfully
              obtained the sparse solution and the peak finder successfully
              identified the desired number of peaks. This flag does **not**
              guarantee that the estimated source locations are correct. The
              estimated source locations may be completely wrong!
              If resolved is False, both ``estimates`` and ``spectrum`` will be
              ``None``.
            * estimates (:class:`~doatools.model.sources.SourcePlacement`):
              A :class:`~doatools.model.sources.SourcePlacement` instance of the
              same type as the one used in the search grid, represeting the
              estimated source locations. Will be ``None`` if resolved is
              ``False``.
            * spectrum (:class:`~numpy.ndarray`): An numpy array of the same
              shape of the specified search grid, consisting of values evaluated
              at the grid points. Only present if ``return_spectrum`` is
              ``True``.
        """
        if 'refine_estimates' in kwargs:
            raise ValueError('Grid refinement is not supported.')
        if Y.shape[0] != self._array.size:
            raise ValueError('The number of rows of Y must be equal to the array size.')
        if Y.shape[1] != self._n_snapshots:
            raise ValueError('The number of columns of Y must be equal to the number of snapshots.')
        f_sp = lambda A: self._get_sparse_spectrum(A, Y, l, solver_options)
        return self._estimate(f_sp, k, **kwargs)
