import numpy as np
import warnings
try:
    import cvxpy as cvx
    cvx_available = True
except ImportError:
    warnings.warn('Cannot import cvxpr. Some sparse recovery based estimators will not be usable.')
    cvx_available = False

class L1RegularizedLeastSquaresProblem:
    r"""Creates a reusable :math:`l_1`-regularized least squares problem.

    Let :math:`\mathbf{A}` be an :math:`M \times L` real dictionary matrix,
    :math:`\mathbf{b}` be an :math:`M \times 1` observation vector,
    :math:`\mathbf{x}` be a :math:`K \times 1` sparse vector, :math:`l` be
    a nonnegative scalar. Let :math:`c` equal to 0 if :math:`\mathbf{x}`
    must be nonnegative and :math:`-\infty` if :math:`\mathbf{x}` can be any
    real number.
    
    The default formulation, named ``'penalizedl1'``, is given by

    .. math::

        \begin{aligned}
        \min_{\mathbf{x}}&
        \frac{1}{2} \| \mathbf{A}\mathbf{x} - \mathbf{b} \|_2^2 +
            l \| \mathbf{x} \|_1,\\
        \text{s.t. }& x \geq c
        \end{aligned}

    This formulation can be efficiently solved with QP or FISTA.

    One common variant, namely the ``'constraintedl1'`` formulation, is
    given by

    .. math::

        \begin{aligned}
        \min_{\mathbf{x}}& \| \mathbf{A}\mathbf{x} - \mathbf{b} \|_2^2,\\
        \text{s.t. }& \| \mathbf{x} \|_1 \leq l, \mathbf{x} \geq c.
        \end{aligned}

    This formulation can be efficiently solved with QP.

    A less common variant, namely the ``'constrainedl2'`` formulation, is
    given by

    .. math::

        \begin{aligned}
        \min_{\mathbf{x}}& \| \mathbf{x} \|_1,\\
        \text{s.t. }& \| \mathbf{A}\mathbf{x} - \mathbf{b} \|_2 \leq l,
            \mathbf{x} \geq c.
        \end{aligned}
    
    Note that the :math:`l_2` error is upper bounded by l. If l is too small
    this problem may be infeasible. This formulation can be converted to a
    SOCP problem.

    Args:
        m (int): Dimension of the observation vector :math:`\mathbf{b}`.
        k (int): Dimension of the sparse vector :math:`\mathbf{x}` (or the
            number of columns of the dictionary matrix, :math:`\mathbf{A}`).
        formulation (str): ``'penalizedl1'``, ``'constrainedl1'`` or
            ``'constrainedl2'``. Default value is ``'penalizedl1'``.
        nonnegative (bool): Specifies whether :math:`\mathbf{x}` must be
            nonnegative. Default value is ``False``.
    """

    def __init__(self, m, k, formulation='penalizedl1', nonnegative=False):
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
            A (~numpy.ndarray): Dictionary matrix.
            b (~numpy.ndarray): Observation vector.
            l (float): Regularization/constraint parameter.
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
    r"""Creates an :math:`l_{2,1}`-norm regularized least squares problem.
    
    The :math:`l_{2,1}`-norm of a matrix variable
    :math:`\mathbf{X} \in \mathbb{C}^{K \times L}` is given by
    
    .. math::
        
        \| \mathbf{X} \|_{2,1}
        = \sum_{i=1}^K \left(\sum_{j=1}^L |X_{ij}|^2\right)^{\frac{1}{2}}.
    
    The :math:`l_{2,1}`-norm regularized least squares problem is given by

    .. math::

        \min_{\mathbf{X}}
        \frac{1}{2} \| \mathbf{A}\mathbf{X} - \mathbf{B} \|_F^2 +
        l \| \mathbf{X} \|_{2,1},
    
    where :math:`\mathbf{A}` is :math:`M \times K`, :math:`\mathbf{X}` is
    :math:`K \times L`, :math:`\mathbf{B}` is :math:`M \times L`, and :math:`l`
    is the regularization parameter. Usually :math:`\mathbf{A}` is the
    dictionary matrix, :math:`\mathbf{X}` is the sparse signal to be
    reconstructed, and :math:`\mathbf{B}` is the observation matrix where each
    column of :math:`\mathbf{B}` represents a single observation.

    Args:
        m (int): Number of rows of the dictionary matrix :math:`\mathbf{A}`.
        k (int): Number of rows of :math:`\mathbf{X}` (or the number of columns
            of the dictionary matrix, :math:`\mathbf{A}`).
        n (int): Number of observations (the number of columns of the
            observation matrix, :math:`\mathbf{B}`)
        complex (bool): Specifies whether all matrices are complex. Default
            value is ``False``.
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
            A (~numpy.ndarray): Dictionary matrix.
            B (~numpy.ndarray): Observation matrix.
            l (~numpy.ndarray): Regularization parameter.
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
