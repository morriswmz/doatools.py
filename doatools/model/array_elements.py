from abc import ABC, abstractmethod
import numpy as np

class ArrayElement(ABC):
    """Base class for array elements."""

    @property
    @abstractmethod
    def output_size(self):
        """Retrieves the output size of this array element.
        
        For scalar sensors, the output size is one. For vector sensors, the
        output size is greater than one.
        """
        raise NotImplementedError()

    @property
    def is_scalar(self):
        """Retrieves whether this array element has a scalar output."""
        return self.output_size == 1

    @property
    @abstractmethod
    def is_isotropic(self):
        """Retrieves whether this array element is isotropic."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def is_polarized(self):
        """Retrieves whether this array element measures polarized waves."""
        raise NotImplementedError()

    def calc_spatial_response(self, r, az, el, polarization=None):
        """Calculates the spatial response of for given sources configurations.

        Args:
            r (float or ~numpy.ndarray): A single range value or an array of
                range values. Must have the same shape as ``az`` and ``el``.
            az (float or ~numpy.ndarray): A single azimuth angle or an array of
                azimuth angles. Must have the same shape as ``az`` and ``el``.
            el (float or ~numpy.ndarray): A single elevation angle or an array
                of elevation angles. Must have the same shape as ``az`` and
                ``el``.
            polarization (~numpy.ndarray or None): Polarization information.
                Suppose ``r``, ``az``, ``el`` share the same shape
                ``(d1, d2, ..., dn)``. Then ``polarization`` should have a shape
                of ``(d1, d2, ..., dn, l)``, where ``l`` is the number of
                polarization parameters for each source. Default value is
                ``None``.

        Returns:
            ~numpy.ndarray: A spatial response tensor. For a scalar element,
            the shape should be the same as that of ``r``, ``az``, or ``el``.
            For a vector element (``output_size > 1``), the shape is given by
            ``(l, d1, d2, ..., dn)``, where ``l`` is equal to ``output_size``
            and ``(d1, d2, ..., dn)`` is the shape of ``r``, ``az``, or ``el``.
        """
        # Validate inputs.
        input_shape = np.shape(r)
        if np.shape(az) != input_shape or np.shape(el) != input_shape:
            raise ValueError('r, az, and el must share the same shape.')
        if polarization is not None:
            if not self.is_polarized:
                raise ValueError(
                    '{0} does not support polarized sources.'
                    .format(self.__class__.__name__)
                )
            expected_p_shape = input_shape + (polarization.shape[-1],)
            if expected_p_shape != polarization.shape:
                raise ValueError(
                    'The shape of the polarization data does not match that of '
                    'r, az, or el. Expecting {0}. Got {1}.'
                    .format(expected_p_shape, polarization.shape)
                )
        # Call the actual implementation.
        return self._calc_spatial_response(r, az, el, polarization)
    
    @abstractmethod
    def _calc_spatial_response(self, r, az, el, polarization):
        """Actual implementation of spatial response calculations.
        
        The inputs are guaranteed to have valid shapes.
        """
        raise NotImplementedError()

class IsotropicScalarSensor(ArrayElement):
    """Creates an isotropic scalar array element."""

    @property
    def output_size(self):
        return 1

    @property
    def is_isotropic(self):
        return True

    @property
    def is_polarized(self):
        return False
    
    def _calc_spatial_response(self, r, az, el, polarization):
        if np.isscalar(r):
            return 1.
        else:
            return np.ones_like(r)

#: An isotropic scalar sensor.
ISOTROPIC_SCALAR_SENSOR = IsotropicScalarSensor()

class CustomNonisotropicSensor(ArrayElement):
    """Creates a customize non-isotropic sensor.

    Args:
        f_sr (~collection.abc.Callable): Custom spatial response function.
            It accepts four inputs: ``r``, ``az``, ``el``, and ``polarization``,
            and outputs the spatial response. See
            :meth:`~doatools.model.array_elements.ArrayElement.calc_spatial_response`
            for more details.
        output_size (int): Output size of the sensor. Must be consistent with
            the output of ``f_sr``. Default value is ``1``.
        polarized (bool): Specifies whether the sensor measures polarized waves.
            Must be consistent with the implemention in ``f_sr``. Default value
            is ``False``.
    """

    def __init__(self, f_sr, output_size=1, polarized=False):
        self._f_sr = f_sr
        self._output_size = output_size
        self._is_polarized = polarized

    @property
    def output_size(self):
        return self._output_size

    @property
    def is_isotropic(self):
        return False

    @property
    def is_polarized(self):
        return self._is_polarized

    def _calc_spatial_response(self, r, az, el, polarization):
        return self._f_sr(r, az, el, polarization)
