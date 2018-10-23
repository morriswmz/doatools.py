import numpy as np
from .core import SpectrumBasedEstimatorBase, ensure_covariance_size

def f_bartlett(A, R):
    r"""Computes the spectrum output of the Bartlett beamformer.

    .. math::
        P_{\mathrm{Bartlett}}(\theta)
        = \mathbf{a}(\theta)^H \mathbf{R} \mathbf{a}(\theta)

    Args:
        A: m x k steering matrix of candidate direction-of-arrivals, where
            m is the number of sensors and k is the number of candidate
            direction-of-arrivals.
        R: m x m covariance matrix.
    """
    return np.sum(A.conj() * (R @ A), axis=0).real

def f_mvdr(A, R):
    r"""Compute the spectrum output of the Bartlett beamformer.

    .. math::
        P_{\mathrm{MVDR}}(\theta)
        = \frac{1}{\mathbf{a}(\theta)^H \mathbf{R}^{-1} \mathbf{a}(\theta)}

    Args:
        A: m x k steering matrix of candidate direction-of-arrivals, where
            m is the number of sensors and k is the number of candidate
            direction-of-arrivals.
        R: m x m covariance matrix.
    """
    return 1.0 / np.sum(A.conj() * np.linalg.lstsq(R, A, None)[0], axis=0).real

class BartlettBeamformer(SpectrumBasedEstimatorBase):
    """Creates a Barlett-beamformer based estimator.

    This estimator is also named beamscan based estimator.
    
    The spectrum is computed on a predefined-grid using
    :meth:`~doatools.estimation.beamforming.f_bartlett`, and the source
    locations are estimated by identifying the peaks.

    Args:
        array (~doatools.model.arrays.ArrayDesign): Array design.
        wavelength (float): Wavelength of the carrier wave.
        search_grid (~doatools.estimation.grid.SearchGrid): The search grid
            used to locate the sources.
        **kwargs: Other keyword arguments supported by
            :class:`~doatools.estimation.core.SpectrumBasedEstimatorBase`.

    References:
        [1] H. L. Van Trees, Optimum array processing. New York: Wiley, 2002.
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
        return self._estimate(lambda A: f_bartlett(A, R), k, **kwargs)

class MVDRBeamformer(SpectrumBasedEstimatorBase):
    """Creates a MVDR-beamformer based estimator.
    
    The spectrum is computed on a predefined-grid using
    :meth:`~doatools.estimation.beamforming.f_mvdr`, and the source locations
    are estimated by identifying the peaks.

    Args:
        array (~doatools.model.arrays.ArrayDesign): Array design.
        wavelength (float): Wavelength of the carrier wave.
        search_grid (~doatools.estimation.grid.SearchGrid): The search grid
            used to locate the sources.
        **kwargs: Other keyword arguments supported by
            :class:`~doatools.estimation.core.SpectrumBasedEstimatorBase`.

    References:
        [1] H. L. Van Trees, Optimum array processing. New York: Wiley, 2002.
    """

    def __init__(self, array, wavelength, search_grid, **kwargs):
        super().__init__(array, wavelength, search_grid, **kwargs)
        
    def estimate(self, R, k, **kwargs):
        """
        Estimates the source locations from the given covariance matrix.

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
        return self._estimate(lambda A: f_mvdr(A, R), k, **kwargs)
