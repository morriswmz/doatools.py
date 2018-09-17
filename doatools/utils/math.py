import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter

def vec(x):
    '''
    Vectorizes a matrix by stacking the columns. Numpy arrays use row major
    ordering, while MATLAB uses column major ordering. Therefore in numpy
    `reshape((-1, 1))` stacks the rows instead of columns. This function is just
    a shorthand for `reshape((-1, 1), order='F')`.

    Args:
        x: A ndarray to be vectorized.
    '''
    return x.reshape((-1, 1), order='F')

def khatri_rao(a, b):
    '''
    Evaluates the Khatri-Rao (i.e., column-wise Kronecker product) between the
    two given matrices.
    '''
    n1, k1 = a.shape
    n2, k2 = b.shape
    if k1 != k2:
        raise ValueError('Two input matrices must have the same number of columns.')
    c = np.zeros((n1 * n2, k1), dtype=np.result_type(a.dtype, b.dtype))
    for i in range(k1):
        c[:,i] = np.outer(a[:,i], b[:,i]).flatten()
    return c

def cartesian(*xi):
    '''
    Evaluates the Cartesian product among the input vectors. For instance, if
    the inputs are [1, 2] and [3, 4, 5], the result will be
    
    [[1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5]]

    Args:
        *xi: 1D arrays.
    
    Returns:
        prod: A numpy array containing the Cartesian product.
    '''
    yi = np.meshgrid(*xi, indexing='ij')
    return np.vstack([y.flatten() for y in yi]).T
