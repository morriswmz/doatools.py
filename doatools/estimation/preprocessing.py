import numpy as np

def spatial_smooth(R, l, fb=False):
    '''
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
    '''
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
