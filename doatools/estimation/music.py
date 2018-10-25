import numpy as np
from math import ceil
from scipy.signal import find_peaks
import warnings
from ..model.sources import FarField1DSourcePlacement
from .core import SpectrumBasedEstimatorBase, get_noise_subspace, \
                  ensure_covariance_size, ensure_n_resolvable_sources

def f_music(A, En):
    r"""Computes the classical MUSIC spectrum
    
    This is a vectorized implementation of the spectrum function:

    .. math::
        P_{\mathrm{MUSIC}}(\theta)
        = \frac{1}{\mathbf{a}^H(\theta) \mathbf{E}_\mathrm{n}
                   \mathbf{E}_\mathrm{n}^H \mathbf{a}(\theta)}

    Args:
        A: m x k steering matrix of candidate direction-of-arrivals, where
            m is the number of sensors and k is the number of candidate
            direction-of-arrivals.
        En: m x d matrix of noise eigenvectors, where d is the dimension of the
            noise subspace.
    """
    v = En.T.conj() @ A
    return np.reciprocal(np.sum(v * v.conj(), axis=0).real)

class MUSIC(SpectrumBasedEstimatorBase):
    """Creates a spectrum-based MUSIC estimator.
    
    The MUSIC spectrum is computed on a predefined-grid using
    :meth:`~doatools.estimation.music.f_music`, and the source locations are
    estimated by identifying the peaks.

    Args:
        array (~doatools.model.arrays.ArrayDesign): Array design.
        wavelength (float): Wavelength of the carrier wave.
        search_grid (~doatools.estimation.grid.SearchGrid): The search grid
            used to locate the sources.
        **kwargs: Other keyword arguments supported by
            :class:`~doatools.estimation.core.SpectrumBasedEstimatorBase`.
    
    References:
        [1] R. Schmidt, "Multiple emitter location and signal parameter
        estimation," IEEE Transactions on Antennas and Propagation,
        vol. 34, no. 3, pp. 276-280, Mar. 1986.
    """

    def __init__(self, array, wavelength, search_grid, **kwargs):
        super().__init__(array, wavelength, search_grid, **kwargs)
        
    def estimate(self, R, k, **kwargs):
        """Estimates the source locations from the given covariance matrix.

        Args:
            R (~numpy.ndarray): Covariance matrix input. The size of R must
                match that of the array design used when creating this
                estimator.
            k (int): Expected number of sources.
            return_spectrum (bool): Set to ``True`` to also output the spectrum
                for visualization. Default value if ``False``.
            refine_estimates (bool): Set to True to enable grid refinement to
                obtain potentially more accurate estimates.
            refinement_density (int): Density of the refinement grids. Higher
                density values lead to denser refinement grids and increased
                computational complexity. Default value is 10.
            refinement_iters (int): Number of refinement iterations. More
                iterations generally lead to better results, at the cost of
                increased computational complexity. Default value is 3.
        
        Returns:
            A tuple with the following elements.

            * resolved (:class:`bool`): A boolean indicating if the desired
              number of sources are found. This flag does **not** guarantee that
              the estimated source locations are correct. The estimated source
              locations may be completely wrong!
              If resolved is False, both ``estimates`` and ``spectrum`` will be
              ``None``.
            * estimates (:class:`~doatools.model.sources.SourcePlacement`):
              A :class:`~doatools.model.sources.SourcePlacement` instance of the
              same type as the one used in the search grid, represeting the
              estimated source locations. Will be ``None`` if resolved is
              ``False``.
            * spectrum (:class:`~numpy.ndarray`): An numpy array of the same
              shape of the specified search grid, consisting of values evaluated
              at the grid points. Only present if ``return_spectrum`` is
              ``True``.
        """
        ensure_covariance_size(R, self._array)
        ensure_n_resolvable_sources(k, self._array.size - 1)
        En = get_noise_subspace(R, k)
        return self._estimate(lambda A: f_music(A, En), k, **kwargs)

class RootMUSIC1D:
    """Creates a root-MUSIC estimator for uniform linear arrays.

    Args:
        wavelength (float): Wavelength of the carrier wave.

    References:
        [1] A. Barabell, "Improving the resolution performance of
        eigenstructure-based direction-finding algorithms," ICASSP '83. IEEE
        International Conference on Acoustics, Speech, and Signal Processing,
        Boston, Massachusetts, USA, 1983, pp. 336-339.

        [2] B. D. Rao and K. V. S. Hari, "Performance analysis of Root-Music,"
        IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 37,
        no. 12, pp. 1939-1949, Dec. 1989.
    """

    def __init__(self, wavelength):
        self._wavelength = wavelength

    def estimate(self, R, k, d0=None, unit='rad'):
        """Estimates the direction-of-arrivals of 1D far-field sources.

        Args:
            R (~numpy.ndarray): Covariance matrix input. This covariance matrix
                must be obtained using a uniform linear array.
            k (int): Expected number of sources.
            d0 (float): Inter-element spacing of the uniform linear array used
                to obtain ``R``. If not specified, it will be set to one half
                of the ``wavelength`` used when creating this estimator.
                Default value is ``None``.
            unit (str): Unit of the estimates. Default value is ``'rad'``.
                See :class:`~doatools.model.sources.FarField1DSourcePlacement`
                for more details on valid units.
        
        Returns:
            A tuple with the following elements.

            * resolved (:class:`bool`): ``True`` only if the rooting algorithm
              successfully finds ``k`` roots inside the unit circle. This flag
              does **not** guarantee that the estimated source locations are
              correct. The estimated source locations may be completely wrong!
              If resolved is False, ``estimates`` will be ``None``.
            * estimates (:class:`~doatools.model.sources.FarField1DSourcePlacement`):
              A :class:`~doatools.model.sources.FarField1DSourcePlacement`
              recording the estimated source locations. Will be ``None`` if
              resolved is ``False``.
        """
        if R.ndim != 2 or R.shape[0] != R.shape[1]:
            raise ValueError('R should be a square matrix.')
        m = R.shape[0]
        ensure_n_resolvable_sources(k, m - 1)
        if d0 is None:
            d0 = self._wavelength / 2.0
        En = get_noise_subspace(R, k)
        # Compute the coefficients for the polynomial.
        C = En @ En.T.conj()
        coeff = np.zeros((m - 1,), dtype=np.complex_)
        for i in range(1, m):
            coeff[i - 1] += np.sum(np.diag(C, i))
        coeff = np.hstack((coeff[::-1], np.sum(np.diag(C)), coeff.conj()))
        # Find the roots of the polynomial.
        z = np.roots(coeff)
        # Find k points inside the unit circle that are also closest to the unit
        # circle.
        nz = len(z)
        mask = np.ones((nz,), dtype=np.bool_)
        for i in range(nz):
            absz = abs(z[i])
            if absz > 1.0:
                # Outside the unit circle.
                mask[i] = False
            elif absz == 1.0:
                # On the unit circle. Need to find the closest point and remove
                # it.
                idx = -1
                dist = np.inf
                for j in range(nz):
                    if j != i and mask[j]:
                        cur_dist = abs(z[i] - z[j])
                        if cur_dist < dist:
                            dist = cur_dist
                            idx = j
                if idx < 0:
                    raise RuntimeError('Unpaired point found on the unit circle, which is impossible.')
                mask[idx] = False
        z = z[mask]
        sorted_indices = np.argsort(1.0 - np.abs(z))
        if len(z) < k:
            return False, None
        else:
            z = z[sorted_indices[:k]]
            return True, FarField1DSourcePlacement.from_z(z, self._wavelength, d0, unit)
