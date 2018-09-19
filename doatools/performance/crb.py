import numpy as np

def _unify_source_power_input(p, k):
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
    if isinstance(p, list):
        p = np.array(p)
    if p.ndim == 1:
        if p.size != k:
            raise ValueError('The length of the input vector does not match the number of sources.')
        return np.diag(p)
    elif p.ndim == 2:
        return p
    else:
        raise ValueError('Expecting a scalar, an 1D vector or a 2D matrix.')


def crb_sto_farfield_1d(design, wavelength, sources, p, sigma, n_snapshots=1):
    r'''
    Computes the stochastic CRB for 1D farfield sources. Under the stochastic
    signal model, the source signal is assumed to be complex Gaussian, and the
    noise signal is assumed complex white Gaussian.

    Args:
        design: Array design.
        wavelength: Wavelength of the carrier wave.
        sources: A FarField1DSourcePlacement instance representing the source
            locations.
        p: The power of the source signals. Can be
            1. A scalar if all sources are uncorrelated and share the same
               power.
            2. A 1D numpy array if all sources are uncorrelated but have
               different powers.
            3. A 2D numpy array representing the source covariance matrix.
        sigma: Variance of the additive noise.
        n_snapshots: Number of snapshots.

    Returns:
        crb: k x k CRB matrix where k is the number of sources.
    
    References:
    [1] P. Stoica and A. Nehorai, "Performance study of conditional and
        unconditional direction-of-arrival estimation," IEEE Transactions on
        Acoustics, Speech and Signal Processing, vol. 38, no. 10,
        pp. 1783-1795, Oct. 1990.
    '''
    k = sources.size
    P = _unify_source_power_input(p, k)
    A, D = design.steering_matrix(sources, wavelength, True)
    A_H = A.T.conj()
    I = np.eye(design.size)
    # Compute the covairance matrix: R = A P A^H + \sigma I
    R = A @ P @ A_H + sigma * I
    # Compute the projection matrix: A (A^H A)^{-1} A^H
    P_A = A @ np.linalg.solve(A_H @ A, A_H)
    # Compute the H matrix.
    H = D.T.conj() @ (I - P_A) @ D
    # Compute the CRB
    CRB = (H * ((P @ (A_H @ np.linalg.solve(R, A)) @ P).T)).real
    CRB = np.linalg.inv(CRB) * (sigma / n_snapshots / 2)
    return 0.5 * (CRB + CRB.T)

def crb_det_farfield_1d(design, wavelength, sources, p, sigma, n_snapshots=1):
    pass

def crb_stouc_farfield_1d(design, wavelength, sources, p, sigma, n_snapshots=1,
                          output_fim=False):
    pass
