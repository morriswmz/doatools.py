import numpy as np
from ..model.arrays import UniformLinearArray
from ..model.sources import FarField1DSourcePlacement
from ..model.coarray import WeightFunction1D
from .utils import unify_p_to_matrix, unify_p_to_vector, reduce_output_matrix

def ecov_music_1d(array, sources, wavelength, P, sigma, n_snapshots=1,
                  perturbations='all', return_mode='full'):
    """Computes the asymptotic covariance matrix of the estimation errors of
    the classical MUSIC algorithm.

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
        perturbations (str): Specifies which perturbations are considered when
            constructing the steering matrix. Possible values include ``'all'``,
            ``'known'``, and ``'none'``. Default value is ``'all'``.
            See :meth:`~doatools.model.arrays.ArrayDesign.steering_matrix`
            for more details.
        return_mode (str): Can be one of the following:
            
            1. ``'full'``: returns the full covariance matrix.
            2. ``'diag'``: returns only the diagonals of the covariance matrix.
            3. ``'mean_diag'``: returns the mean of the diagonals of the
               covariance matrix.
        
            Default value is ``'full'``.
    
    Returns:
        Depending on ``'return_mode'``, can be the full covariance matrix, the
        diagonals of the covariance matrix, or the mean of the diagonals of the
        covariance matrix.

    References:
        [1] P. Stoica and A. Nehorai, "MUSIC, maximum likelihood, and Cramér-Rao
        bound: further results and comparisons," IEEE Transactions on Acoustics,
        Speech and Signal Processing, vol. 38, no. 12, pp. 2140-2150, Dec. 1990.
        
        [2] P. Stoica and A. Nehorai, "MUSIC, maximum likelihood, and Cramér-Rao
        bound," IEEE Transactions on Acoustics, Speech and Signal Processing,
        vol. 37, no. 5, pp. 720-741, May 1989.
    """
    if not isinstance(sources, FarField1DSourcePlacement):
        raise ValueError('Sources must be far-field and 1D.')
    k = sources.size
    if k >= array.size:
        raise ValueError('Too many sources.')
    P = unify_p_to_matrix(P, k)
    A, D = array.steering_matrix(sources, wavelength, True, perturbations)
    A_H = A.conj().T
    AHA = A_H @ A
    # Compute the H matrix.
    # H = D^H (I - A (A^H A)^{-1} A^H) D
    H = D.conj().T @ (np.eye(array.size) - A @ np.linalg.solve(AHA, A_H)) @ D
    # Compute the B matrix.
    # B = P^{-1} + \sigma P^{-1} (A^H A)^{-1} P^{-1} 
    P_inv = np.linalg.inv(P)
    B = P_inv @ (np.eye(k) + sigma * np.linalg.solve(AHA, P_inv))
    # Compute the asymptotic covariance.
    # C = \sigma/(2N) (H \odot I)^{-1} Re(H \odot B^T) (H \odot I)^{-1}
    h = np.reciprocal(np.diag(H).real)
    C = (sigma / 2.0 / n_snapshots) * ((H * B.T).real * np.outer(h, h))
    return reduce_output_matrix(C, return_mode)

def ecov_coarray_music_1d(array, sources, wavelength, p, sigma, n_snapshots=1,
                          return_mode='full'):
    """Computes the asymptotic covariance matrix of the estimation errors of the
    coarray-based MUSIC algorithm, SS-MUSIC or DA-MUSIC.

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
            
            1. ``'full'``: returns the full covariance matrix.
            2. ``'diag'``: returns only the diagonals of the covariance matrix.
            3. ``'mean_diag'``: returns the mean of the diagonals of the
               covariance matrix.
        
            Default value is ``'full'``.
    
    Returns:
        Depending on ``'return_mode'``, can be the full covariance matrix, the
        diagonals of the covariance matrix, or the mean of the diagonals of the
        covariance matrix.
        
    References:
        [1] M. Wang and A. Nehorai, "Coarrays, MUSIC, and the Cramér-Rao Bound,"
        IEEE Transactions on Signal Processing, vol. 65, no. 4, pp. 933-946,
        Feb. 2017.
    """
    if not isinstance(sources, FarField1DSourcePlacement):
        raise ValueError('Sources must be far-field and 1D.')
    if array.is_perturbed:
        raise ValueError('Array cannot have any perturbation.')
    k = sources.size
    p = unify_p_to_vector(p, k)
    m = array.size
    # TODO: potential optimizations with sparse matrices
    # Generate the coarray selection matrix.
    wf = WeightFunction1D(array)
    F = wf.get_coarray_selection_matrix()
    # Check the number of sources.
    m_v = (F.shape[0] + 1) // 2
    if k >= m_v:
        raise ValueError('Too many sources.')
    # Compute the covariance matrix of the physical array.
    A = array.steering_matrix(sources, wavelength)
    R_ideal = (A * p) @ A.conj().T + sigma * np.eye(m)
    # Compute virtual array related variable.
    arr_virtual = UniformLinearArray(m_v, wavelength / 2.0)
    Av, DAv = arr_virtual.steering_matrix(sources, wavelength, True)
    Rss_ideal = (Av * p) @ Av.conj().T + sigma * np.eye(m_v)
    # Construct the subarray selection matrices.
    G = np.zeros((m_v * m_v, 2 * m_v - 1))
    for ii in range(0, m_v):
        for jj in range(0, m_v):
            G[ii * m_v + jj, m_v - ii + jj - 1] = 1.0
    # Get the noise subspace of the ideal augmented covariance matrix.
    _, E = np.linalg.eigh(Rss_ideal)
    En = E[:,:-k]
    En_H = En.conj().T
    proj_En = En @ En_H
    # Evalute xi_k.
    GF_T = (G @ F).T
    Xi_g = np.zeros((m * m, k), dtype=np.complex_)
    pinv_Av = np.linalg.pinv(Av)
    for kk in range(k):
        d_ak = DAv[:,kk]
        alpha_k = -pinv_Av[kk, :].T / p[kk]
        beta_k = proj_En @ d_ak
        gamma_k = (beta_k.conj().T @ beta_k).real
        Xi_g[:, kk] = GF_T @ np.kron(beta_k, alpha_k) / gamma_k
    # Evaluate C.
    C = (Xi_g.conj().T @ np.kron(R_ideal, R_ideal.T) @ Xi_g).real / n_snapshots
    return reduce_output_matrix(C, return_mode)
