import numpy as np

def spatial_smooth(R, l, fb=False):
    """
    Applys spatial smoothing to the given covariance matrix.
    
    Reference:
    [1] S. U. Pillai and B. H. Kwon, "Forward/backward spatial smoothing
        techniques for coherent signal identification," IEEE Transactions on 
        Acoustics, Speech, and Signal Processing, vol. 37, no. 1, pp. 8-15,
        Jan. 1989.
    
    Args:
        R: Input covariance matrix. Both real and complex covariance matrices
            are supported.
        l: Number of subarrays. Must be an integer within [1, M], where M is
            the size of the covariance matrix.
        fb: Set to True to enable forward-backward spatial smoothing. Default
            value is False and only forward mode spatial smoothing is computed.
    
    Returns:
        Rs: (M - l + 1) x (M - l + 1) spatially-smoothed covariance matrix,
            where M is the size of the input covariance matrix.
    """
    m = R.shape[0]
    if l < 1 or l > m:
        raise ValueError('The number of subarrays must be within [1, {0}].'.format(m))
    
    # Forward pass
    Rf = R[:m-l+1, :m-l+1].copy()
    for i in range(1, l):
        Rf += R[i:i+m-l+1, i:i+m-l+1]
    Rf /= l
    if not fb:
        return Rf
    
    # Adds backward pass
    if np.iscomplexobj(Rf):
        return 0.5 * (Rf + np.flip(Rf).conj())
    else:
        return 0.5 * (Rf + np.flip(Rf))

def l1_svd(y, k):
    """Performs l1-SVD to help reduce the dimensionality.
    
    The original multiple measurement model is given by
        y = Ax + n. (1)
    When the number of snapshots is large, the resulting problem may be
    computationally expensive to solve. Note that rank(Ax) = k <= m. We can
    decompose y into two parts: the part corresponding to the signal subspace,
    and the part corresponding to the noise subspace. We can keep just the part
    corresponding to the signal subspace to reduce the dimensionality without
    losing too much information.

    To do so, we compute the SVD of y = USV^H. Let T_k = [I_k 0]. Multiplying
    both sides of (1) by V T_k, we obtain
        y_s = A x_s + n_s, (2)
    where y_s = y V T_k, x_s = x V T_k, n_s = n V T_k. We can then use (2)
    instead of (1) to perform DOA estimation.

    Args:
        y: An m x l measurement matrix, where m denotes the number of sensors
            and l denotes the number of snapshots.
        k: Dimension of the signal subspace, cannot be greater than min(m,l).

    Returns:
        An m x k matrix.
    """
    U, s, Vh = np.linalg.svd(y, full_matrices=False)
    return U[:,:k] * s[:k]
