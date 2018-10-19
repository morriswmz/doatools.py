from enum import Enum
from abc import ABC, abstractmethod
import copy
import warnings
import numpy as np
from scipy.spatial.distance import cdist
from ..utils.conversion import convert_angles

def _validate_sensor_location_ndim(sensor_locations):
    if sensor_locations.shape[1] < 1 or sensor_locations.shape[1] > 3:
        raise ValueError('Sensor locations can only consists of 1D, 2D or 3D coordinates.')

class SourcePlacement(ABC):
    '''Represents the placement of several sources.'''

    def __init__(self, locations, units):
        self._locations = locations
        self._units = units

    def __len__(self):
        '''Returns the number of sources.'''
        return self._locations.shape[0]

    def __getitem__(self, key):
        '''Accesses a specific source location or obtains a subset of source
        placement via slicing.

        Implementation notice: this is a generic implementation. When `key` is
        a scalar, key is treated as an index and normal indexing operation
        follows. When `key` is not a scalar, we need to return a new instance
        with source locations specified by `key`. First, a shallow copy is made
        with `copy.copy()`. Then the shallow copy's `_locations` field is set to
        the source locations specified by `key`. Finally, the shallow copy is
        returned.

        Args:
            key : An integer, slice, or 1D numpy array of indices/boolean masks.
        '''
        if np.isscalar(key):
            return self._locations[key]
        if isinstance(key, slice):
            # Slicing results a view. We force a copy here.
            locations = self._locations[key].copy()
        elif isinstance(key, list):
            locations = self._locations[key]
        elif isinstance(key, np.ndarray):
            if key.ndim != 1:
                raise ValueError('1D array expected.')
            locations = self._locations[key]
        else:
            raise KeyError('Unsupported index.')
        new_copy = copy.copy(self)
        new_copy._locations = locations
        return new_copy

    @property
    def size(self):
        '''Retrieves the number of sources.'''
        return len(self)

    @property
    def locations(self):
        '''Retrives the source locations.
        
        While this property provides read/write access to the underlying ndarray
        storing the source locations. Modifying the underlying ndarray is
        discourage because modified values are not checked for validity.
        '''
        return self._locations

    @property
    def units(self):
        '''Retrives a tuple consisting of units used for each dimension.'''
        return self._units

    @property
    @abstractmethod
    def valid_ranges(self):
        '''Retrieves the valid ranges for each dimension.

        Returns:
            A tuple of 2 element tuples: ((min_1, max_1), ...).
        '''
        raise NotImplementedError()
    
    @abstractmethod
    def as_unit(self, new_unit):
        '''Creates a copy with the source locations converted to the new unit.'''
        raise NotImplementedError()

    @abstractmethod
    def phase_delay_matrix(self, sensor_locations, wavelength, derivatives=False):
        '''Computes the M x K phase delay matrix.

        The phase delay matrix D is an M x K matrix, where D[m,k] is the
        relative phase delay between the m-th sensor and the k-th source
        (usually using the first sensor as the reference).

        Notes: the phase delay matrix is used in constructing the steering
        matrix. This method is decoupled from the steering matrix method
        because the phase delays are calculated differently for different
        types of sources (e.g. far-field vs. near-field).

        Args:
            sensor_locations: An M x d (d = 1, 2, 3) matrix representing the
                sensor locations using the Cartesian coordinate system.
            wavelength: Wavelength of the carrier wave.
            derivatives: If set to true, also outputs the derivative matrix (or
                matrices) with respect to the source locations. Default value
                is False.
        '''
        pass

class FarField1DSourcePlacement(SourcePlacement):

    VALID_RANGES = {
        'rad': (-np.pi/2, np.pi/2),
        'deg': (-90.0, 90.0),
        'sin': (-1.0, 1.0)
    }

    def __init__(self, locations, unit='rad'):
        r'''Creates a far-field 1D source placement.

        Args:
            locations: A list or 1D numpy array representing the source
                locations.
            unit: Can be 'rad', 'deg' or 'sin'. 'sin' is a special unit where
                sine of the arrival angle is used instead of the arrival angle
                itself. 
        '''
        if isinstance(locations, list):
            locations = np.array(locations)
        if locations.ndim > 1:
            raise ValueError('1D numpy array expected.')
        if unit not in FarField1DSourcePlacement.VALID_RANGES:
            raise ValueError(
                'Unit can only be one of the following: {0}.'
                .format(', '.join(FarField1DSourcePlacement.VALID_RANGES.keys()))
            )
        lb, ub = FarField1DSourcePlacement.VALID_RANGES[unit]
        if np.any(locations < lb) or np.any(locations > ub):
            raise ValueError(
                "When unit is '{0}', source locations must be within [{0}, {1}]."
                .format_map(unit, lb, ub)
            )
        super().__init__(locations, (unit,))

    @staticmethod
    def from_z(z, wavelength, d0, unit='rad'):
        '''Creates a far-field 1D source placement from complex roots.
        
        Used in rooting based DOA estimators such as root-MUSIC and ESPRIT.

        Args:
            z: A ndarray of complex roots.
            wavelength (float): Wavelength of the carrier wave.
            d0 (float): Inter-element spacing of the uniform linear array.
        
        Returns:
            A FarField1DSourcePlacement instance.
        '''
        c = 2 * np.pi * d0 / wavelength
        sin_vals = np.angle(z) / c
        if unit == 'sin':
            sin_vals.sort()
            return FarField1DSourcePlacement(sin_vals, 'sin')
        locations = np.arcsin(sin_vals)
        locations.sort()        
        if unit == 'rad':
            return FarField1DSourcePlacement(locations)
        else:
            return FarField1DSourcePlacement(np.rad2deg(locations), 'deg')

    @property
    def valid_ranges(self):
        return FarField1DSourcePlacement.VALID_RANGES[self._units[0]],

    def as_unit(self, new_unit):
        return FarField1DSourcePlacement(
            convert_angles(self._locations, self._units[0], new_unit),
            new_unit
        )

    def phase_delay_matrix(self, sensor_locations, wavelength, derivatives=False):
        '''Computes the M x K phase delay matrix for one-dimensional far-field
        sources.
        
        The phase delay matrix D is an M x K matrix, where D[m,k] is the
        relative phase delay between the m-th sensor and the k-th far-field
        source (usually using the first sensor as the reference).

        Args:
            sensor_locations: An M x d (d = 1, 2, 3) matrix representing the
                sensor locations using the Cartesian coordinate system. When the
                sensor locations are 2D or 3D, the DOAs are assumed to be within
                the xy-plane.
            wavelength: Wavelength of the carrier wave.
            derivatives: If set to true, also outputs the derivative matrix (or
                matrices) with respect to the source locations. Default value
                is False.

        Returns:
            A: The steering matrix.
            D: The derivative matrix. Only returns when `derivatives` is True.
        '''
        _validate_sensor_location_ndim(sensor_locations)
        
        if self._units[0] == 'sin':
            return self._phase_delay_matrix_sin(sensor_locations, wavelength, derivatives)
        else:
            return self._phase_delay_matrix_rad(sensor_locations, wavelength, derivatives)
        
    def _phase_delay_matrix_rad(self, sensor_locations, wavelength, derivatives=False):
        # Unit can only be 'rad' or 'deg'.
        # Unify to radians.
        if self._units[0] == 'deg':
            locations = np.deg2rad(self._locations)
        else:
            locations = self._locations
        
        locations = locations[np.newaxis]
        s = 2 * np.pi / wavelength
        if sensor_locations.shape[1] == 1:
            # D[i,k] = sensor_location[i] * sin(doa[k])
            D = s * np.outer(sensor_locations, np.sin(locations))
            if derivatives:
                DD = s * np.outer(sensor_locations, np.cos(locations))
        else:
            # The sources are assumed to be within the xy-plane. The offset
            # along the z-axis of the sensors does not affect the delays.
            # D[i,k] = sensor_location_x[i] * sin(doa[k])
            #          + sensor_location_y[i] * cos(doa[k])
            D = s * (np.outer(sensor_locations[:, 0], np.sin(locations)) +
                     np.outer(sensor_locations[:, 1], np.cos(locations)))
            if derivatives:
                DD = s * (np.outer(sensor_locations[:, 0], np.cos(locations)) -
                          np.outer(sensor_locations[:, 1], np.sin(locations)))
        if self._units[0] == 'deg' and derivatives:
            DD *= np.pi / 180.0 # Do not forget the scaling when unit is 'deg'.
        return (D, DD) if derivatives else D

    def _phase_delay_matrix_sin(self, sensor_locations, wavelength, derivatives=False):
        sin_vals = self._locations
        s = 2 * np.pi / wavelength
        if sensor_locations.shape[1] == 1:
            # D[i,k] = sensor_location[i] * sin_val[k]
            D = s * (sensor_locations * sin_vals)
            if derivatives:
                # Note that if x = \sin\theta then
                # \frac{\partial cx}{\partial x} = c
                # This is different from the derivative w.r.t. \theta:
                # \frac{\partial cx}{\partial \theta} = c\cos\theta
                DD = np.tile(s * sensor_locations, (1, self._locations.size))
        else:
            # The sources are assumed to be within the xy-plane. The offset
            # along the z-axis of the sensors does not affect the delays.
            cos_vals = np.sqrt(1.0 - sin_vals * sin_vals)
            D = s * (np.outer(sensor_locations[:, 0], sin_vals) +
                     np.outer(sensor_locations[:, 1], cos_vals))
            if derivatives:
                # If x = \sin\theta, \theta \in (-\pi/2, \pi/2)
                # a \sin\theta + b \cos\theta = ax + b\sqrt{1-x^2}
                # d/dx(ax + b\sqrt{1-x^2}) = a - bx/\sqrt{1-x^2}
                # sensor_locations[:, 0, np.newaxis] will be a column
                # vector and broadcasting will be utilized.
                DD = s * (sensor_locations[:, 0, np.newaxis] -
                          np.outer(sensor_locations[:, 1], sin_vals / cos_vals))
        return (D, DD) if derivatives else D


class FarField2DSourcePlacement(SourcePlacement):

    VALID_RANGES = {
        'rad': ((-np.pi, np.pi), (-np.pi/2, np.pi/2)),
        'deg': ((-180.0, 180.0), (-90.0, 90.0))
    }

    def __init__(self, locations, unit='rad'):
        '''Creates a far-field 2D source placement.

        Args:
            locations: An K x 2 numpy array representing the source locations,
                where K is the number of sources, and the k-th row consists of
                the azimuth and elevation angle of the k-th source. Should never
                be modified after creation.
            unit: Can be 'rad' or 'deg'.
        '''
        if isinstance(locations, list):
            locations = np.array(locations)
        if locations.ndim != 2 or locations.shape[1] != 2:
            raise ValueError('Expecting an K x 2 numpy array.')
        if unit not in FarField2DSourcePlacement.VALID_RANGES:
            raise ValueError(
                'Unit can only be one of the following: {0}.'
                .format(', '.join(FarField2DSourcePlacement.VALID_RANGES.keys()))
            )
        (min_az, max_az), (min_el, max_el) = FarField2DSourcePlacement.VALID_RANGES[unit]
        if np.any(locations[:, 0] < min_az) or np.any(locations[:, 0] > max_az):
            raise ValueError(
                "When unit is '{0}', azimuth angles must be within [{1}, {2}]."
                .format(unit, min_az, max_az)
            )
        if np.any(locations[:, 1] < min_el) or np.any(locations[:, 1] > max_el):
            raise ValueError(
                "When unit is '{0}', elevation angles must be within [{1}, {2}]."
                .format(unit, min_el, max_el)
            )
        super().__init__(locations, (unit, unit))

    @property
    def valid_ranges(self):
        return FarField2DSourcePlacement.VALID_RANGES[self._units[0]]

    def as_unit(self, new_unit):
        return FarField2DSourcePlacement(
            convert_angles(self._locations, self._units[0], new_unit),
            new_unit
        )

    def phase_delay_matrix(self, sensor_locations, wavelength, derivatives=False):
        '''Computes the M x K phase delay matrix for two-dimensional far-field
        sources.
        
        The phase delay matrix D is an M x K matrix, where D[m,k] is the
        relative phase delay between the m-th sensor and the k-th far-field
        source (usually using the first sensor as the reference).

        Args:
            sensor_locations: An M x d (d = 1, 2, 3) matrix representing the
                sensor locations using the Cartesian coordinate system. Linear
                arrays (1D arrays) are assumed to be placed along the x-axis.
            wavelength: Wavelength of the carrier wave.
            derivatives: If set to true, also outputs the derivative matrix (or
                matrices) with respect to the source locations. Default value
                is False.
        '''
        _validate_sensor_location_ndim(sensor_locations)
        
        if derivatives:
            raise ValueError('Derivative matrix computation is not supported for far-field 2D DOAs.')

        # Unify to radians.
        if self._units[0] == 'deg':
            locations = np.deg2rad(self._locations)
        else:
            locations = self._locations
        
        s = 2 * np.pi / wavelength
        cos_el = np.cos(locations[:, 1])
        if sensor_locations.shape[1] == 1:
            # Linear arrays are assumed to be placed along the x-axis
            # Need to convert azimuth-elevation pairs to broadside angles.
            D = s * np.outer(sensor_locations, cos_el * np.cos(locations[:, 0]))
        else:
            cc = cos_el * np.cos(locations[:, 0])
            cs = cos_el * np.sin(locations[:, 0])
            if sensor_locations.shape[1] == 2:
                D = s * (np.outer(sensor_locations[:, 0], cc) +
                         np.outer(sensor_locations[:, 1], cs))
            else:
                D = s * (np.outer(sensor_locations[:, 0], cc) +
                         np.outer(sensor_locations[:, 1], cs) +
                         np.outer(sensor_locations[:, 2], np.sin(locations[:, 1])))
        
        return D

class NearField2DSourcePlacement(SourcePlacement):

    def __init__(self, locations):
        '''Creates a near-field 2D source placement.

        Args:
            locations: An K x 2 numpy array representing the source locations,
                where K is the number of sources and the k-th row consists of
                the x and y coordinates of the k-th source. Should never be
                modified after creation.
        '''
        if isinstance(locations, list):
            locations = np.array(locations)
        if locations.ndim != 2 or locations.shape[1] != 2:
            raise ValueError('Expecting an K x 2 numpy array.')
        super().__init__(locations, ('m', 'm'))

    @property
    def valid_ranges(self):
        return (-np.inf, np.inf), (-np.inf, np.inf)

    def as_unit(self, new_unit):
        if new_unit != 'm':
            raise ValueError("new_unit must be 'm'.")
        return NearField2DSourcePlacement(self._locations.copy())

    def phase_delay_matrix(self, sensor_locations, wavelength, derivatives=False):
        '''Computes the M x K phase delay matrix for two-dimensional near-field
        sources.
        
        The phase delay matrix D is an M x K matrix, where D[m,k] is the
        relative phase delay between the m-th sensor and the k-th near-field
        source (usually using the first sensor as the reference).

        Args:
            sensor_locations: An M x d (d = 1, 2, 3) matrix representing the
                sensor locations using the Cartesian coordinate system. Linear
                arrays (1D arrays) are assumed to be placed along the x-axis,
                and 2D arrays are assumed to be placed in the xy-plane.
            wavelength: Wavelength of the carrier wave.
            derivatives: If set to true, also outputs the derivative matrix (or
                matrices) with respect to the source locations. Default value
                is False.
        '''
        _validate_sensor_location_ndim(sensor_locations)
        
        if derivatives:
            raise ValueError('Derivative matrix computation is not supported for near-field 2D DOAs.')

        # Align the number of dimensions
        source_locations = self._locations
        if sensor_locations.shape[1] < 2:
            # 1D arrays
            sensor_locations = np.pad(sensor_locations, ((0, 0), (0, 1)), 'constant')
        elif sensor_locations.shape[1] > 2:
            # 3D arrays
            source_locations = np.pad(source_locations, ((0, 0), (0, 1)), 'constant')

        s = 2 * np.pi / wavelength
        # Compute the pair-wise Euclidean distance.
        M = cdist(sensor_locations, source_locations, 'euclidean')
        M -= M[0, :].copy() # Use the first sensor as the reference sensor.
        return s * M
