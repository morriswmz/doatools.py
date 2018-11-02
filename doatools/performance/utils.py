import numpy as np

def unify_p_to_matrix(p, k):
    """ Unifies the source power input into a matrix.
    
    Args:
        p: A scalar, list, or a numpy array.
        k: Number of sources.
    """
    if np.isscalar(p):
        P = np.zeros((k, k))
        np.fill_diagonal(P, p)
        return P
    if isinstance(p, list) or p.ndim == 1:
        if len(p) != k:
            raise ValueError('The length of the input vector does not match the number of sources.')
        return np.diag(p)
    elif p.ndim == 2:
        if p.shape[0] != k or p.shape[1] != k:
            raise ValueError('The shape of the input matrix does not match the number of sources.')
        return p
    else:
        raise ValueError('Expecting a scalar, a 1D vector or a 2D matrix.')

def unify_p_to_vector(p, k):
    """Unifies the source power input into a vector.
    
    Args:
        p: A scalar, list, or a numpy array.
        k: Number of sources.
    """
    if np.isscalar(p):
        return np.full((k,), p)
    if isinstance(p, list) or p.ndim == 1:
        if len(p) != k:
            raise ValueError('The length of the input vector does not match the number of sources.')
        return np.array(p)
    elif p.ndim == 2:
        if p.shape[0] != k or p.shape[1] != k:
            raise ValueError('The shape of the input matrix does not match the number of sources.')
        return np.diag(p)
    else:
        raise ValueError('Expecting a scalar, a 1D vector or a 2D matrix.')

def reduce_output_matrix(x, mode):
    """Reduces the output CRB/covariance matrix according to ``mode``.

    Args:
        x (~numpy.ndarray): The output matrix.
        mode (str): Can be ``'full'``, ``'diag'``, or ``'mean_diag'``.
    """
    if mode == 'full':
        return x
    elif mode == 'diag':
        return np.diag(x)
    elif mode == 'mean_diag':
        return np.mean(np.diag(x))
    else:
        raise ValueError("Unknown mode '{0}'.".format(mode))
