import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter

def vec(x):
    '''
    Vectorize a matrix by stacking the columns. Numpy arrays use row major
    ordering, while MATLAB uses column major ordering. Therefore in numpy
    `reshape((-1, 1))` stacks the rows instead of columns. This function is just
    a shorthand for `reshape((-1, 1), order='F')`.

    Args:
        x: A ndarray to be vectorized.
    '''
    return x.reshape((-1, 1), order='F')

def khatri_rao(a, b):
    '''
    Evaluate the Khatri-Rao (i.e., column-wise Kronecker product) between the
    two given matrices.
    '''
    n, k = a.shape
    if b.shape[1] != k:
        raise ValueError('Two input matrices must have the same number of columns.')
    c = np.zeros(n * n, k, dtype=np.result_type(a.dtype, b.dtype))
    for i in range(k):
        c[:,i] = np.outer(a[:,i], b[:,i]).flatten()
    return c
