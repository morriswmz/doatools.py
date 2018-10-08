import numpy as np

def unify_p_to_matrix(p, k):
    '''
    Unifies the source power input into a matrix.
    
    Args:
        p: A scalar, list, or a numpy array.
        k: Number of sources.
    '''
    if np.isscalar(p):
        P = np.zeros((k, k))
        np.fill_diagonal(P, p)
        return P
    if isinstance(p, list) or p.ndim == 1:
        if len(p) != k:
            raise ValueError('The length of the input vector does not match the number of sources.')
        return np.diag(p)
    elif p.ndim == 2:
        return p
    else:
        raise ValueError('Expecting a scalar, an 1D vector or a 2D matrix.')

def unify_p_to_vector(p, k):
    '''
    Unifies the source power input into a vector.
    
    Args:
        p: A scalar, list, or a numpy array.
        k: Number of sources.
    '''
    if np.isscalar(p):
        return np.full((k,), p)
    if isinstance(p, list) or p.ndim == 1:
        if len(p) != k:
            raise ValueError('The length of the input vector does not match the number of sources.')
        return np.array(p)
    elif p.ndim == 2:
        return np.diag(p)
    else:
        raise ValueError('Expecting a scalar, an 1D vector or a 2D matrix.')
