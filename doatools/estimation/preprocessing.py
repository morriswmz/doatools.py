import numpy as np

def spatial_smooth(R, l, fb=False):
    """Applies spatial smoothing to the given covariance matrix.

    Spatial smoothing can decorrelate coherent sources.
    
    Args:
        R: Input covariance matrix. Both real and complex covariance matrices
            are supported.
        l (int): Number of subarrays. Must be an integer that is greater than 0
            and less than or equal to the size of ``R``.
        fb (bool): Set to ``True`` to enable forward-backward spatial smoothing.
            Default value is False and only forward mode spatial smoothing is
            computed.
    
    Returns:
        An (m - l + 1) x (m - l + 1) spatially-smoothed covariance matrix, where
        m is the size of the input covariance matrix.

    References:
        [1] S. U. Pillai and B. H. Kwon, "Forward/backward spatial smoothing
        techniques for coherent signal identification," IEEE Transactions on 
        Acoustics, Speech, and Signal Processing, vol. 37, no. 1, pp. 8-15,
        Jan. 1989.
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
    r"""Performs :math:`l_1`-SVD to help reduce the dimensionality.
    
    Consider the following measurement model

    .. math::

        \mathbf{Y} = \mathbf{A}\mathbf{X} + \mathbf{N}, (1)
    
    where :math:`\mathbf{Y} \in \mathbb{C}^{M \times N}` is the matrix of
    snapshots, :math:`\mathbf{A} \in \mathbb{C}^{M \times K}` is the steering
    matrix, :math:`\mathbf{X} \in \mathbb{C}^{K \times N}` consists of source
    signals, and :math:`\mathbf{N} \in \mathbb{C}^{M \times N}` consists of
    additive noise.

    When the number of snapshots, :math:`N`, is large, the resulting problem may
    be computationally expensive to solve. Note that
    :math:`\mathrm{rank}(\mathbf{A}\mathbf{X}) = K <= M`. We can
    decompose :math:`\mathbf{Y}` into two parts: the part corresponding to the
    signal subspace, and the part corresponding to the noise subspace. We can
    keep just the part corresponding to the signal subspace to reduce the
    dimensionality without losing too much information.

    To do so, we compute the SVD of :math:`\mathbf{Y}` as
    :math:`\mathbf{U}\mathbf{S}\mathbf{V}^H`. Let
    :math:`\mathbf{T}_K = [\mathbf{I}_K \mathbf{0}]`. Multiplying both sides of
    (1) by :math:`\mathbf{V} \mathbf{T}_K`, we obtain

    .. math::

        \mathbf{Y}_\mathrm{sv}
        = \mathbf{A}\mathbf{X}_\mathrm{sv} + \mathbf{N}_\mathrm{sv}, (2)
    
    where :math:`\mathbf{Y}_\mathrm{sv} = \mathbf{Y}\mathbf{V}\mathbf{T}_K`,
    :math:`\mathbf{X}_\mathrm{sv} = \mathbf{X}\mathbf{V}\mathbf{T}_K`,
    and :math:`\mathbf{N}_\mathrm{sv} = \mathbf{N}\mathbf{V}\mathbf{T}_K`.
    We can then use (2) instead of (1) as our new measurement model to perform
    DOA estimation.

    Args:
        y: The snapshot matrix, each column of which represents a single
            snapshot.
        k: Dimension of the signal subspace, cannot be greater than either the
            number of snapshots or the number of sensors.

    Returns:
        The reduced measurement matrix.

    References:
        [1] D. Malioutov, M. Cetin, and A. S. Willsky, "A sparse signal
        reconstruction perspective for source localization with sensor arrays,"
        IEEE Transactions on Signal Processing, vol. 53, no. 8, pp. 3010-3022,
        Aug. 2005.
    """
    U, s, Vh = np.linalg.svd(y, full_matrices=False)
    return U[:,:k] * s[:k]
