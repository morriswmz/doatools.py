import warnings
import numpy as np
import cvxpy as cvx
from .core import SpectrumBasedEstimatorBase
from ..utils.math import khatri_rao, vec

def _create_penalized_l1(A, b, x, l):
    obj_func = 0.5 * cvx.sum_squares(cvx.matmul(A, x) - b) + l * cvx.norm1(x)
    constraints = [x >= 0]
    problem = cvx.Problem(cvx.Minimize(obj_func), constraints)
    return obj_func, constraints, problem

def _create_constrained_l1(A, b, x, l):
    obj_func = cvx.sum_squares(cvx.matmul(A, x) - b)
    constraints = [cvx.norm1(x) <= l, x >= 0]
    problem = cvx.Problem(cvx.Minimize(obj_func), constraints)
    return obj_func, constraints, problem

def _create_constrained_l2(A, b, x, l):
    obj_func = cvx.norm1(x)
    constraints = [cvx.norm(cvx.matmul(A, x) - b) <= l, x >= 0]
    problem = cvx.Problem(cvx.Minimize(obj_func), constraints)
    return obj_func, constraints, problem

_PROBLEM_CREATORS = {
    'penalizedl1': _create_penalized_l1,
    'constrainedl1': _create_constrained_l1,
    'constrainedl2': _create_constrained_l2
}

class _SparseRecoveryProblem:

    def __init__(self, m, k, formulation='penalized'):
        r'''
        Creates a reusable sparse recovery problem.
        Let A be an m x k real matrix, b be an m x 1 vector, x be an k x 1
        vector, l be a nonnegative scalar. We formulate a sparse recovery
        problem using the one of the following formulations:
        
        1. Penalized l1 ('penalizedl1'):

        min_{x} 0.5 \| Ax - b \|_2^2 + l * \| x \|_1,
        s.t. x >= 0.

        2. Constrained l1 ('constrainedl1'):

        min_{x} \| Ax - b \|_2^2,
        s.t. \| x \|_1 <= l,
             x >= 0.

        3. Constrained l2 ('constrainedl2'):

        min_{x} \| x \|_1,
        s.t. \| Ax - b \|_2 <= l,
             x >= 0.

        Args:
            m (int): Dimension of the observation vector b.
            k (int): Dimension of the sparse vector x.
            formulation (str): 'penalizedl1', 'constrainedl1' or
                'constrainedl2'.
        '''
        if formulation not in _PROBLEM_CREATORS.keys():
            raise ValueError('Formulation must be one of the following: {0}.'.format(', '.join(_PROBLEM_CREATORS.keys())))
        # Initialize parameters and variable.
        self._formulation = formulation
        self._A = cvx.Parameter((m, k), name='A')
        self._b = cvx.Parameter((m, 1), name='b')
        self._l = cvx.Parameter(nonneg=True, name='lambda')
        self._x = cvx.Variable((k, 1), name='x')
        self._obj_func, self._constraints, self._problem = \
            _PROBLEM_CREATORS[formulation](self._A, self._b, self._x, self._l)

    def solve(self, A, b, l, **kwargs):
        '''
        Solves the sparse recovery problem with the specified parameters.

        Args:
            A: Dictionary matrix.
            b: Observation vector.
            l: Regularization/constraint parameter.
            **kwargs: Other keyword arguments to be passed to the solver.
        '''
        self._A.value = A
        self._b.value = b
        self._l.value = l
        self._problem.solve(**kwargs)
        if self._problem.status != 'optimal':
            warnings.warn('Optimal solution cannot be obtained.')
            return np.zeros((self._x.size,))
        return self._x.value

class SparseBPDN(SpectrumBasedEstimatorBase):

    def __init__(self, design, wavelength, search_grid, formulation='penalizedl1', **kwargs):
        r'''
        Creates a source location estimator based on sparse basis pursuit
        denoising.

        Args:
            design (ArrayDesign): Array design.
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
        '''
        super().__init__(design, wavelength, search_grid, **kwargs)
        m = self._design.size
        k = self._search_grid.size
        # vec(R) -> m*m elements, real + image -> 2*m*m, noise + 1
        self._problem = _SparseRecoveryProblem(2*m*m, k + 1, formulation)

    def _solve_bpdn(self, A, R, reg_param, **kwargs):
        A = khatri_rao(A.conj(), A)
        A = np.hstack((A, vec(np.eye(self._design.size))))
        A = np.vstack((A.real, A.imag))
        b = vec(R)
        b = np.vstack((b.real, b.imag))
        return self._problem.solve(A, b, reg_param, **kwargs).flatten()[:-1]
    
    def estimate(self, R, k, l, output_spectrum=False, **kwargs):
        '''
        Estimates the source locations from the given covariance matrix.

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
            output_spectrum (bool): Set to True to also output the spectrum for
                visualization. Default value if False.
            **kwargs: Additional keyword arguments to be passed to the
                optimization problem solver. For instance, you can specify the
                solver or set the verbosity.
        
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
        return self._estimate(lambda A: self._solve_bpdn(A, R, l, **kwargs), k, output_spectrum)

