import numpy as np
from .core import SpectrumBasedEstimatorBase, get_noise_subspace, \
                  ensure_covariance_size, ensure_n_resolvable_sources
from ..utils.math import abs_squared

class MinNorm(SpectrumBasedEstimatorBase):
    """Creates a spectrum-based Min-Norm estimator.
    
    The Min-Norm spectrum is computed on a predefined-grid, and the source
    locations are estimated by identifying the peaks.

    Args:
        array (~doatools.model.arrays.ArrayDesign): Array design.
        wavelength (float): Wavelength of the carrier wave.
        search_grid (~doatools.estimation.grid.SearchGrid): The search grid
            used to locate the sources.
        **kwargs: Other keyword arguments supported by
            :class:`~doatools.estimation.core.SpectrumBasedEstimatorBase`.
    
    Notes:
        The Min-Norm algorithm is sometimes referred to in the literature as a
        weighted MUSIC algorithm because it uses weighted eigenspaces.
    
    References:
        [1] R. Kumaresan and D. W. Tufts, "Estimating the angles of arrival of
        multiple plane waves," IEEE Trans. Aerospace Electron. Syst.,
        vol. 19, pp. 134-139, Jan. 1983.
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
        # We compute the d vector from the noise subspace.
        # d = En c^* / |c|^2
        En = get_noise_subspace(R, k)
        c = En[0, :]
        w = c.conj() / (np.linalg.norm(c, 2)**2)
        d = (En @ w).conj()
        # Spectrum = 1/|d^H a(\theta)|^2
        f_sp = lambda A: np.reciprocal(abs_squared(d @ A))
        return self._estimate(f_sp, k, **kwargs)
