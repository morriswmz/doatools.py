import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter

def vec(x):
    """Vectorizes a matrix by stacking the columns.
    
    NumPy arrays use row major ordering, while MATLAB uses column major
    ordering. Therefore in NumPy `reshape((-1, 1))` stacks the rows instead of
    columns.
    
    This function is just a shorthand for `reshape((-1, 1), order='F')`.

    Args:
        x: An ndarray to be vectorized.
    """
    return x.reshape((-1, 1), order='F')

def abs_squared(x):
    """Computes Re(x)^2 + Im(x)^2.
    
    Args:
        x: An complex ndarray.
    """
    return x.real**2 + x.imag**2

def khatri_rao(a, b):
    """Evaluates the Khatri-Rao (i.e., column-wise Kronecker product) between
    the two given matrices."""
    n1, k1 = a.shape
    n2, k2 = b.shape
    if k1 != k2:
        raise ValueError('Two input matrices must have the same number of columns.')
    c = np.zeros((n1 * n2, k1), dtype=np.result_type(a.dtype, b.dtype))
    for i in range(k1):
        c[:,i] = np.outer(a[:,i], b[:,i]).flatten()
    return c

def projm(A, use_pinv=False):
    """Computes the projection matrix of the input matrix.
    
    Given a full column rank matrix A, the projection matrix of A is given by
        A (A^H A)^{-1} A^H

    Args:
        A: An ndarray.
        use_pinv: If set to true, will use `pinv` instead `solve` to compute
            the projection matrix. Set this to True if `A` is close to singular.
    """
    if use_pinv:
        return A @ np.linalg.pinv(A)
    if np.iscomplexobj(A):
        A_H = A.conj().T
        return A @ np.linalg.solve(A_H @ A, A_H)
    else:
        return A @ np.linalg.solve(A.T @ A, A.T)

def cartesian(*xi):
    """Evaluates the Cartesian product among the input vectors.

    For instance, if the inputs are [1, 2] and [3, 4, 5], the result will be
    
    [[1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5]]

    Args:
        *xi: 1D arrays.
    
    Returns:
        prod: An ndarray array containing the Cartesian product.
    """
    yi = np.meshgrid(*xi, indexing='ij')
    return np.vstack([y.flatten() for y in yi]).T

def randcn(shape):
    """Samples from complex circularly-symmetric normal distribution.

    Args:
        shape (tuple): Shape of the output.
    
    Returns:
        ~numpy.ndarray: A complex :class:`~numpy.ndarray` containing the
        samples.
    """
    x = 1j * np.random.randn(*shape)
    x += np.random.randn(*shape)
    x *= np.sqrt(0.5)
    return x

def unique_rows(x, atol=0.0, rtol=1e-8, return_index=False, sort=False):
    """Obtains the unique rows within the specified tolerance.
    
    This function is designed to obtain unique rows from a matrix while
    considering floating-point errors. Hence, the tolerance is usually set to
    small values. This function matches rows in a greedy manner.

    Two rows x[i,:] and x[k,:] are considered to be equal if
        abs(x[i,l] - x[k,l]) <= atol + rtol * nanmax(abs(x))
    for all l.

    When the tolerance values are relatively large, the behavior of this
    function is not well-defined. For instance, if x is a column vector of the
    following elements: [0.1, 0.2, 0.3, 0.4], and absolute tolerance is set to
    0.2, there exists multiple solutions. In such cases, the problem of unique
    rows is related to clique cover problem, which is not trivial to solve.

    If the input matrix contains infinities, relative tolerance will not be
    effective because `rtol * nanmax(abs(x))` becomes infinity.

    Args:
        x: An ndarray representing the input matrix.
        atol: Absolute tolerance. Default value is 0.0.
        rtol: Relative tolerance. Default value is 1e-6.
        return_index: Set to True to return the row indices in the original
            matrix that maps to the output rows.
        sort: Set to True to sort the output in lexicographical order. Default
            value if False.

    Returns:
        y: An ndarray consists of the unique rows.
        indices: An ndarray of indices representing the rows in the input matrix
            that are used to construct the output.
    """
    if x.ndim != 2:
        raise ValueError('Matrix input expected.')
    n, m = x.shape
    # Handle the empty case
    if n == 0:
        return (x.copy(), np.zeros((0, m))) if return_index else x.copy()
    # Scale the relative tolerance according to the maximum absolute value in x.
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    tol = atol + rtol * max(abs(xmin), abs(xmax))
    # Find unique rows greedily
    # This is a O(N^2) process. May be optimized via k-d tree.
    processed = [False] * n
    unique_indices = []
    ii = 0
    while ii >= 0:
        next_ii = -1
        cur_row = x[ii, :]
        unique_indices.append(ii)
        for kk in range(ii + 1, n):
            if processed[kk]:
                continue
            if np.all(np.abs(cur_row - x[kk, :]) <= tol):
                # Found a close one.
                processed[kk] = True
            else:
                # Not close enough
                if next_ii < 0:
                    next_ii = kk
        ii = next_ii
    y = x[unique_indices, :]
    if sort:
        # Lexicographical sorting
        indices = np.lexsort(np.flipud(y.T))
        y = y[indices, :]
        if return_index:
            unique_indices = [unique_indices[i] for i in indices]
    if return_index:
        return y, np.array(unique_indices, dtype=np.int32)
    else:
        return y
