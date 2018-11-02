import numpy as np
from ..model.sources import FarField1DSourcePlacement
from .utils import unify_p_to_matrix, unify_p_to_vector, reduce_output_matrix
from ..utils.math import projm

def crb_sto_farfield_1d(array, sources, wavelength, p, sigma, n_snapshots=1,
                        return_mode='full'):
    r"""Computes the stochastic CRB for 1D farfield sources.
    
    Under the stochastic signal model, the source signal is assumed to be
    complex Gaussian, and the noise signal is assumed complex white Gaussian.
    The unknown parameters include:

    * Source locations.
    * Real and imaginary parts of the source covariance matrix.
    * Noise variance.
    
    This function only computes the CRB for source locations.
    Because the unknown parameters do not include array perturbation parameters,
    all array perturbation parameters are assumed known during the computation.

    Args:
        array (~doatools.model.arrays.ArrayDesign): Array design.
        wavelength (float): Wavelength of the carrier wave.
        sources (~doatools.model.sources.FarField1DSourcePlacement):
            Source locations.
        p (float or ~numpy.ndarray): The power of the source signals. Can be

            1. A scalar if all sources are uncorrelated and share the same
               power.
            2. A 1D numpy array if all sources are uncorrelated but have
               different powers.
            3. A 2D numpy array representing the source covariance matrix.
        
        sigma (float): Variance of the additive noise.
        n_snapshots (int): Number of snapshots. Default value is 1.
        return_mode (str): Can be one of the following:
            
            1. ``'full'``: returns the full CRB matrix.
            2. ``'diag'``: returns only the diagonals of the CRB matrix.
            3. ``'mean_diag'``: returns the mean of the diagonals of the CRB
               matrix.

            Default value is ``'full'``.
    
    Returns:
        Depending on ``'return_mode'``, can be the full CRB matrix, the
        diagonals of the CRB matrix, or the mean of the diagonals of the CRB
        matrix.
    
    References:
        [1] P. Stoica and A. Nehorai, "Performance study of conditional and
        unconditional direction-of-arrival estimation," IEEE Transactions on
        Acoustics, Speech and Signal Processing, vol. 38, no. 10,
        pp. 1783-1795, Oct. 1990.
    """
    if not isinstance(sources, FarField1DSourcePlacement):
        raise ValueError('Sources must be far-field and 1D.')
    k = sources.size
    P = unify_p_to_matrix(p, k)
    A, D = array.steering_matrix(sources, wavelength, True, 'all')
    A_H = A.T.conj()
    I = np.eye(array.size)
    # Compute the covairance matrix: R = A P A^H + \sigma I
    R = A @ P @ A_H + sigma * I
    # Compute the projection matrix: A (A^H A)^{-1} A^H
    P_A = A @ np.linalg.solve(A_H @ A, A_H)
    # Compute the H matrix.
    H = D.T.conj() @ (I - P_A) @ D
    # Compute the CRB
    CRB = (H * ((P @ (A_H @ np.linalg.solve(R, A)) @ P).T)).real
    CRB = np.linalg.inv(CRB) * (sigma / n_snapshots / 2)
    return reduce_output_matrix(0.5 * (CRB + CRB.T), return_mode)

def crb_det_farfield_1d(array, sources, wavelength, P, sigma, n_snapshots=1,
                        return_mode='full'):
    r"""Computes the deterministic CRB for 1D farfield sources.
    
    Under the deterministic signal model, the source signal is assumed to be
    deterministic unknown, and the noise signal is assumed complex white
    Gaussian. The unknown parameters include:

    * Source locations.
    * Real and imaginary parts of the source signals.
    * Noise variance.

    This function only computes the CRB for source locations.
    Because the unknown parameters do not include array perturbation parameters,
    all array perturbation parameters are assumed known during the computation.

    Args:
        array (~doatools.model.arrays.ArrayDesign): Array design.
        wavelength (float): Wavelength of the carrier wave.
        sources (~doatools.model.sources.FarField1DSourcePlacement):
            Source locations.
        P: The sample covariance matrix of the source signals. Suppose there are
            :math:`T` snapshots,
            :math:`\mathbf{x}(1), \mathbf{x}(2), \ldots, \mathbf{x}(T)`.
            The sample covariance matrix of the source signals is obtained as
            follows:

            .. math::
                
                \frac{1}{T} \sum_{t=1}^T \mathbf{x}(t) \mathbf{x}^H(t).
            
        sigma (float): Variance of the additive noise.
        n_snapshots (int): Number of snapshots. Default value is 1.
        return_mode (str): Can be one of the following:
            
            1. ``'full'``: returns the full CRB matrix.
            2. ``'diag'``: returns only the diagonals of the CRB matrix.
            3. ``'mean_diag'``: returns the mean of the diagonals of the CRB
               matrix.

            Default value is ``'full'``.
    
    Returns:
        Depending on ``'return_mode'``, can be the full CRB matrix, the
        diagonals of the CRB matrix, or the mean of the diagonals of the CRB
        matrix.
    
    References:
        [1] P. Stoica and A. Nehorai, "Performance study of conditional and
        unconditional direction-of-arrival estimation," IEEE Transactions on
        Acoustics, Speech and Signal Processing, vol. 38, no. 10,
        pp. 1783-1795, Oct. 1990.
    """
    if not isinstance(sources, FarField1DSourcePlacement):
        raise ValueError('Sources must be far-field and 1D.')
    k = sources.size
    m = array.size
    if P.ndim != 2 or P.shape[0] != k or P.shape[1] != k:
        raise ValueError('The sample covariance matrix of the source signals must be a K x K matrix, where K is the number of sources.')
    A, D = array.steering_matrix(sources, wavelength, True, 'all')
    # Compute the projection matrix: A (A^H A)^{-1} A^H
    P_A = projm(A)
    # Compute the H matrix.
    H = D.conj().T @ (np.eye(m) - P_A) @ D
    # Compute the CRB.
    CRB = np.linalg.inv((H * P.T).real) * (sigma / n_snapshots / 2)
    return reduce_output_matrix(0.5 * (CRB + CRB.T), return_mode)

def crb_stouc_farfield_1d(array, sources, wavelength, p, sigma, n_snapshots=1,
                          return_mode='full'):
    r"""Computes the stochastic CRB for 1D farfield sources with the assumption
    that the sources are uncorrelated.
    
    Under the stochastic signal model, the source signal is assumed to be
    uncorrelated complex Gaussian, and the noise signal is assumed complex
    white Gaussian. The unknown parameters include:

    * Source locations.
    * Diagonals of the source covariance matrix.
    * Noise variance.
    
    This function only computes the CRB for source locations.
    Because the unknown parameters do not include array perturbation parameters,
    all array perturbation parameters are assumed known during the computation.

    Args:
        array (~doatools.model.arrays.ArrayDesign): Array design.
        wavelength (float): Wavelength of the carrier wave.
        sources (~doatools.model.sources.FarField1DSourcePlacement):
            Source locations.
        p (float or ~numpy.ndarray): The power of the source signals. Can be

            1. A scalar if all sources are uncorrelated and share the same
               power.
            2. A 1D numpy array if all sources are uncorrelated but have
               different powers.
            3. A 2D numpy array representing the source covariance matrix.
               Only the diagonal elements will be used.
        
        sigma (float): Variance of the additive noise.
        n_snapshots (int): Number of snapshots. Default value is 1.
        return_mode (str): Can be one of the following:
            
            1. ``'full'``: returns the full CRB matrix.
            2. ``'diag'``: returns only the diagonals of the CRB matrix.
            3. ``'mean_diag'``: returns the mean of the diagonals of the CRB
               matrix.

            Default value is ``'full'``.
    
    Returns:
        Depending on ``'return_mode'``, can be the full CRB matrix, the
        diagonals of the CRB matrix, or the mean of the diagonals of the CRB
        matrix.
    
    References:
        [1] H. L. Van Trees, Optimum array processing. New York: Wiley, 2002.

        [2] M. Wang and A. Nehorai, "Coarrays, MUSIC, and the Cramér-Rao Bound,"
        IEEE Transactions on Signal Processing, vol. 65, no. 4, pp. 933-946,
        Feb. 2017.

        [3] C-L. Liu and P. P. Vaidyanathan, "Cramér-Rao bounds for coprime and
        other sparse arrays, which find more sources than sensors," Digital
        Signal Processing, vol. 61, pp. 43-61, 2017.
    """
    if not isinstance(sources, FarField1DSourcePlacement):
        raise ValueError('Sources must be far-field and 1D.')
    k = sources.size
    p = unify_p_to_vector(p, k)
    m = array.size
    # We need to compute each submatrix of the FIM.
    A, DA = array.steering_matrix(sources, wavelength, True, 'all')
    A_H = A.conj().T
    DA_H = DA.conj().T
    R = (A * p) @ A_H + sigma * np.eye(m)
    R_inv = np.linalg.inv(R)
    R_inv = 0.5 * (R_inv + R_inv.conj().T)
    DRD = DA_H @ R_inv @ DA
    DRA = DA_H @ R_inv @ A
    ARD = A_H @ R_inv @ DA
    ARA = A_H @ R_inv @ A
    PP = np.outer(p, p)
    FIM_tt = 2.0 * ((DRD.T * ARA + DRA.conj() * ARD) * PP).real
    FIM_pp = (ARA.conj().T * ARA).real
    R_inv2 = R_inv @ R_inv
    FIM_ss = np.trace(R_inv2).real
    # diag(A @ B) = np.sum(A^T * B, axis=0, keepdims=True)
    FIM_tp = 2.0 * (DRA.conj() * (p[:, np.newaxis] * ARA)).real
    FIM_ts = 2.0 * (p * np.sum(DA.conj() * (R_inv2 @ A), axis=0)).real[:, np.newaxis]
    FIM_ps = np.sum(A.conj() * (R_inv2 @ A), axis=0).real[:, np.newaxis]
    FIM = np.block([
        [FIM_tt,          FIM_tp,          FIM_ts],
        [FIM_tp.conj().T, FIM_pp,          FIM_ps],
        [FIM_ts.conj().T, FIM_ps.conj().T, FIM_ss]
    ])
    CRB = np.linalg.inv(FIM)[:k, :k] / n_snapshots
    return reduce_output_matrix(0.5 * (CRB + CRB.T), return_mode)
