import numpy as np
import warnings
try:
    import cvxpy as cvx
    cvx_available = True
except ImportError:
    warnings.warn('Cannot import cvxpr. Some sparse recovery based estimators will not be usable.')
    cvx_available = False

class L1RegularizedLeastSquaresProblem:

    def __init__(self, m, k, formulation='penalizedl1', nonnegative=False):
        r"""Creates a reusable l1-regularized least squares problem.

        Let A be an m x k real dictionary matrix, b be an m x 1 observation
        vector, x be an k x 1 sparse vector, l be a nonnegative scalar. Let lb
        equal to 0 if x must be nonnegative and -inf if x can be any real
        number.
        
        The default formulation, named 'penalizedl1', is given by

        min_{x} 0.5 \| Ax - b \|_2^2 + l * \| x \|_1,
        s.t. x >= lb

        This formulation can be efficiently solved with QP or FISTA.

        One common variant, namely the 'constraintedl1' formulation, is given by

        min_{x} \| Ax - b \|_2^2,
        s.t. \| x \|_1 <= l,
             x >= lb.

        This formulation can be efficiently solved with QP.

        A less common variant, namely the 'constrainedl2' formulation, is given
        by (Note that the l2 error is upper bounded by l. If l is too small this
        problem may be infeasible.)

        min_{x} \| x \|_1,
        s.t. \| Ax - b \|_2 <= l,
             x >= lb.
            
        This formulation can be converted to a SOCP problem.

        Args:
            m (int): Dimension of the observation vector b.
            k (int): Dimension of the sparse vector x (or the number of columns
                of the dictionary matrix, A).
            formulation (str): 'penalizedl1', 'constrainedl1' or
                'constrainedl2'. Default value is 'penalizedl1'.
            nonnegative (bool): Specifies whether x must be nonnegative.
        """
        if not cvx_available:
            raise RuntimeError('Cannot initialize when cvxpy is not available.')
        # Initialize parameters and variables
        A = cvx.Parameter((m, k))
        b = cvx.Parameter((m, 1))
        l = cvx.Parameter(nonneg=True)
        x = cvx.Variable((k, 1))
        # Create the problem
        if formulation == 'penalizedl1':
            obj_func = 0.5 * cvx.sum_squares(cvx.matmul(A, x) - b) + l * cvx.norm1(x)
            constraints = []
        elif formulation == 'constrainedl1':
            obj_func = cvx.sum_squares(cvx.matmul(A, x) - b)
            constraints = [cvx.norm1(x) <= l]
        elif formulation == 'constrainedl2':
            obj_func = cvx.norm1(x)
            constraints = [cvx.norm(cvx.matmul(A, x) - b) <= l]
        else:
            raise ValueError("Unknown formulation '{0}'.".format(formulation))
        if nonnegative:
            constraints.append(x >= 0)
        problem = cvx.Problem(cvx.Minimize(obj_func), constraints)
        self._formulation = formulation
        self._A = A
        self._b = b
        self._l = l
        self._x = x
        self._obj_func = obj_func
        self._constraints = constraints
        self._problem = problem

    def solve(self, A, b, l, **kwargs):
        """Solves the problem with the specified parameters.

        Args:
            A: Dictionary matrix.
            b: Observation vector.
            l: Regularization/constraint parameter.
            **kwargs: Other keyword arguments to be passed to the solver.
        """
        self._A.value = A
        self._b.value = b
        self._l.value = l
        self._problem.solve(**kwargs)
        if self._problem.status != 'optimal':
            warnings.warn('Optimal solution cannot be obtained.')
            return np.zeros((self._x.size,))
        return self._x.value

class L21RegularizedLeastSquaresProblem:
    r"""l-21 norm regularized least squares problem.
    
    The l-21 norm of a matrix variable X is given by
        \|X\|_{2,1} = \sum_{i=1}^M \|X_i\|_2,
    where X_i is the i-th row of X.

    The l-21 norm regularized least squares problem is given by
        min_X \frac{1}{2} \|AX - B\|_F^2 + l \|X\|_{2,1},
    where A is M x K, X is K x L, B is M x L, and l is the regularization
    parameter. Usually A is the dictionary matrix, X is the sparse signal to be
    reconstructed, and B is the observation matrix where each column of B
    represents a single observation.

    Args:
        m (int): Number of rows of the dictionary matrix A.
        k (int): Number of rows of X (or the number of columns of the dictionary
            matrix, A).
        n (int): Number of observations (or the number of columns of the
            observation matrix, B)
    """

    def __init__(self, m, k, n, complex=False):
        if not cvx_available:
            raise RuntimeError('Cannot initialize when cvxpy is not available.')
        # Initialize parameters and variables
        A = cvx.Parameter((m, k), complex=complex)
        B = cvx.Parameter((m, n), complex=complex)
        l = cvx.Parameter(nonneg=True)
        X = cvx.Variable((k, n), complex=complex)
        # Create the problem
        # CVXPY issue:
        #   cvx.norm does not work if axis is not 0.
        # Workaround:
        #   use cvx.norm(X.T, 2, axis=0) instead of cvx.norm(X, 2, axis=1)
        obj_func = 0.5 * cvx.norm(cvx.matmul(A, X) - B, 'fro')**2 + \
                   l * cvx.sum(cvx.norm(X.T, 2, axis=0))
        self._problem = cvx.Problem(cvx.Minimize(obj_func))
        self._A = A
        self._B = B
        self._l = l
        self._X = X

    def solve(self, A, B, l, **kwargs):
        """Solves the problem with the specified parameters.

        Args:
            A: Dictionary matrix.
            B: Observation matrix.
            l: Regularization parameter.
            **kwargs: Other keyword arguments to be passed to the solver.
        """
        self._A.value = A
        self._B.value = B
        self._l.value = l
        self._problem.solve(**kwargs)
        if self._problem.status != 'optimal':
            warnings.warn('Optimal solution cannot be obtained.')
            return np.zeros(self._X.shape)
        return self._X.value
