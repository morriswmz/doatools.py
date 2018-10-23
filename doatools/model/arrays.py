from math import gcd
from ..utils.math import cartesian
import numpy as np
import warnings
import copy

PERTURBATION_TYPES = [
    'location_errors',
    'gain_errors',
    'phase_errors',
    'mutual_coupling'
]

def validate_location_errors(m, params):
    if params.ndim != 2:
        raise ValueError('Expecting a 2D array.')
    if params.shape[1] > 3:
        raise ValueError('Locations errors cannot be more than 3-dimensional.')
    if params.shape[0] != m:
        raise ValueError('The shape of the location errors matrix does not much the array size.')

def validate_gain_and_phase_errors(m, params):
    if params.ndim != 1:
        raise ValueError('Expecting an 1D array.')
    if params.size != m:
        raise ValueError('The size of the gain/phase errors vector does not much the array size.')

def validate_mutual_coupling(m, params):
    if params.ndim != 2:
        raise ValueError('Expecting a 2D array.')
    if params.shape[0] != m or params.shape[1] != m:
        raise ValueError('The mutual coupling matrix must be {0}x{0}.'.format(m))

PERTURBATION_VALIDATORS = {
    'location_errors': validate_location_errors,
    'gain_errors': validate_gain_and_phase_errors,
    'phase_errors': validate_gain_and_phase_errors,
    'mutual_coupling': validate_mutual_coupling
}

class ArrayDesign:
    """Base class for all array designs.

    Arrays can be 1D, 2D, or 3D. Consider the standard cartesian coordinate
    system. We define 1D, 2D, and 3D arrays as follows:

    * 1D arrays are linear arrays along the x-axis.
    * 2D arrays are planar arrays lying within the xy-plane.
    * 3D arrays are not restricted, whose element can exist anywhere in the 3D
      space.

    We store the element locations with an m x d array, where m denotes the
    number of elements (size) of the array.

    * For 1D arrays, d equals one, and the i-th row of the m x 1 array stores
      the x coordinate of the i-th element.
    * For 2D arrays, d equals two, and the i-th row of the m x 2 array stores
      the x and y coordinates of the i-th element.
    * For 3D arrays, d equals three, and the i-th row of the m x 3 array stores
      the x, y, and z coordinates of the i-th element.

    While this class is generally intended for internal use. You can also use
    this class to create custom arrays. Just to make sure that you do not modify
    ``locations`` after creating the array.

    Args:
        locations: A list or ndarray specifying the element locations. For
            1D arrays, ``locations`` can be either a 1D list/ndarray, or
            an m x 1 list/ndarray, where m is the number of elements. For 2D
            or 3D arrays, ``locations`` must be a 2D list/ndarray of shape
            m x d, where d is 2 or 3. If the input is an ndarray, it will
            not be copied and should not be changed after creating the
            array design.
        name (str): Name of the array design.
        perturbations (dict): A dictionary containing the perturbation 
            parameters. The keys should be among the following:

            * ``'location_errors'``
            * ``'gain_errors'`` (relative, -0.2 means 0.8 * original gain)
            * ``'phase_errors'`` (in radians)
            * ``'mutual_coupling'``

            The values are two-element tuples where the first element is an
            ndarray representing the parameters and the second element is
            a bool specifying whether these parameters are known in prior.

    .. note::
        Array designs are supposed to be **immutable**. Because array
        design objects are passed around when computing steering matrices,
        weight functions, etc., having a mutable internal state leads to more
        complexities and potential unexpected results. Although the internal
        states are generally accessible in Python, please refrain from modifying
        them.
    """

    def __init__(self, locations, name, perturbations={}):
        if not isinstance(locations, np.ndarray):
            locations = np.array(locations)
        if locations.ndim > 2:
            raise ValueError('Expecting an 1D vector or a 2D matrix.')
        if locations.ndim == 1:
            locations = locations.reshape((-1, 1))
        elif locations.shape[1] > 3:
            raise ValueError('Array can only be 1D, 2D or 3D.')
        self._locations = locations
        self._name = name
        # Validate perturbations
        self._perturbations = self._validate_and_copy_perturbations(perturbations)
    
    @property
    def name(self):
        """Retrieves the name of this array."""
        return self._name
    
    @property
    def element_count(self):
        """(Deprecated) Retrieves the number of elements in the array."""
        warnings.warn('Use size instead of element_count in the future.', DeprecationWarning)
        return self.size

    @property
    def size(self):
        """Retrieves the number of elements in the array."""
        return self._locations.shape[0]
    
    @property
    def element_locations(self):
        """Retrieves the nominal element locations.

        Returns:
            An M x d matrix, where M is the number of elements and d is the
            number of dimensions of the nominal array.
        """
        return self._locations.copy()

    @property
    def actual_element_locations(self):
        """Retrieves the actual element locations, considering location errors.

        Returns:
            An M x d matrix, where M is the number of elements and d is the
            maximum of the following two:
            
            1. number of dimensions of the nominal array;
            2. number of dimensions of the sensor location errors.
        """
        if 'location_errors' in self._perturbations:
            return self._compute_actual_locations(self._perturbations['location_errors'][0])
        else:
            return self.element_locations
    
    def _compute_actual_locations(self, location_errors):
        actual_ndim = self.ndim
        loc_err_dim = location_errors.shape[1]
        if loc_err_dim <= actual_ndim:
            # It is possible that the location errors only exist along the
            # first one or two axis.
            actual_locations = self._locations.copy()
            actual_locations[:, :loc_err_dim] += location_errors
        else:
            # Actual dimension is higher. This happens if a linear array,
            # which is 1D, has location errors along both x- and y-axis.
            actual_locations = location_errors.copy()
            actual_locations[:, :actual_ndim] += self._locations
        return actual_locations

    @property
    def is_perturbed(self):
        """Returns if the array contains perturbations."""
        return len(self._perturbations) > 0

    @property
    def ndim(self):
        """Retrieves the number of dimensions of the nominal array.

        The number of dimensions is defined as the number of columns of the
        ndarray storing the nominal array element locations. It does not
        reflect the number of dimensions of the minimal subspace in which the
        nominal array lies. For instance, if the element locations are given by
        ``[[0, 0], [1, 1], [2, 2]]``, ``ndim`` equals to 2 instead of 1, despite
        the fact that this array is a linear array.

        Perturbations do not affect this value.
        """
        return self._locations.shape[1]
    
    @property
    def actual_ndim(self):
        """Retrieves the number of dimensions of the array, considering location errors."""
        if 'location_errors' in self._perturbations:
            return max(self._perturbations['location_errors'][0].shape[1], self.ndim)
        else:
            return self.ndim
    
    def has_perturbation(self, ptype):
        """Checks if the array has the given type of perturbation."""
        return ptype in self._perturbations
    
    def is_perturbation_known(self, ptype):
        """Checks if the specified perturbation is known in prior."""
        return self._perturbations[ptype][1]
    
    def get_perturbation_params(self, ptype):
        """Retrieves the parameters for the specified perturbation type."""
        return self._perturbations[ptype][0]
    
    @property
    def perturbations(self):
        """Retrieves a copy of the dictionary of all perturbations."""
        # Here we have a deep copy.
        return copy.deepcopy(self._perturbations)
    
    def _validate_and_copy_perturbations(self, perturbations):
        p_copy = {}
        for k, v in perturbations.items():
            if k not in PERTURBATION_TYPES:
                raise ValueError('Unsupported perturbation type "{0}".'.format(k))
            if not isinstance(v, tuple) and len(v) != 2:
                raise ValueError('Perturbation details should be specified by a two-element tuple.')
            # Validate and copy
            PERTURBATION_VALIDATORS[k](self.size, v[0])
            params_copy = v[0].copy()
            params_copy.setflags(write=False)
            p_copy[k] = (params_copy, v[1])
        return p_copy

    def get_perturbed_copy(self, perturbations, new_name=None):
        """Returns a copy of this array design but with the specified
        perturbations.
        
        The specified perturbations will replace the existing ones.

        Args:
            perturbations (dict): A dictionary containing the perturbation
                parameters. The keys should be among the following:

                * 'location_errors'
                * 'gain_errors' (relative, -0.2 means 0.8 * original gain)
                * 'phase_errors' (in radians)
                * 'mutual_coupling'

                The values are two-element tuples where the first element is an
                ndarray representing the parameters and the second element is
                a bool specifying whether these parameters are known in prior.
            new_name (str): An optional new name for the resulting array design.
                If not provided, the name of the original array design will be
                used.
        """
        array = self.get_perturbation_free_copy(new_name)
        # Merge perturbation parameters.
        perturbations = self._validate_and_copy_perturbations(perturbations)
        array._perturbations = {**self._perturbations, **perturbations}
        return array

    def get_perturbation_free_copy(self, new_name=None):
        """Returns a perturbation-free copy of this array design.

        Args:
            new_name (str): An optional new name for the resulting array design.
                If not provided, the name of the original array design will be
                used.
        """
        if new_name is None:
            new_name = self._name
        array = copy.copy(self)
        array._perturbations = {}
        array._name = new_name
        return array

    def steering_matrix(self, sources, wavelength, compute_derivatives=False,
                        perturbations='all'):
        r"""Creates the steering matrix for the given DOAs.

        Args:
            sources: An instance of :class:`~doatools.model.sources.SourcePlacement`.

                1. 1D arrays are placed along the x-axis. 2D arrays are placed
                   within the xy-plane.
                2. If you pass in 1D DOAs for an 2D or 3D array, these DOAs will
                   be assumed to be within the xy-plane. The azimuth angles are
                   calculated as :math:`\pi/2` minus the original 1D DOA values
                   (broadside -> azimuth). The elevation angles are set to zeros
                   (within the xy-plane).
            
            wavelength: Wavelength of the carrier wave.
            compute_derivatives: If set to True, also outputs the derivative
                matrix DA with respect to the DOAs, where the k-th column of
                DA is the derivative of the k-th column of A with respect to the
                k-th DOA. DA is used when computing the CRBs. Only available to
                1D DOAs.
            perturbations: Specifies which perturbations are considered when
                constructing the steering matrix:

                * ``'all'`` - All perturbations are considered. This is the
                  default value.
                * ``'known'`` - Only known perturbations (we have prior
                  knowledge of the perturbation parameters) are considered. This
                  option is used by DOA estimators when the exact knowledge of
                  these perturbations are known in prior.
                * ``'none'`` - None of the perturbations are considered.
        
        Notes:
            The steering matrix calculation is bound to array designs. This is
            a generic implementation, which can be overridden for special types
            of arrays.
        """
        # Filter perturbations.
        if perturbations == 'all':
            perturb_dict = self._perturbations
        elif perturbations == 'known':
            perturb_dict = {k: v for k, v in self._perturbations.items() if v[1]}
        elif perturbations == 'none':
            perturb_dict = {}
        else:
            raise ValueError('Perturbation can only be "all", "known", or "none".')
        
        if 'location_errors' in perturb_dict:
            actual_locations = self._compute_actual_locations(perturb_dict['location_errors'][0])
        else:
            actual_locations = self._locations

        # Compute the steering matrix
        T = sources.phase_delay_matrix(actual_locations, wavelength, compute_derivatives)
        if compute_derivatives:
            A = np.exp(1j * T[0])
            DA = [A * (1j * X) for X in T[1:]]
        else:
            A = np.exp(1j * T)
        
        # Apply other perturbations
        if 'gain_errors' in perturb_dict:
            gain_coeff = 1. + perturb_dict['gain_errors'][0]
            A = gain_coeff * A
            if compute_derivatives:
                DA = [gain_coeff * X for X in DA]
        if 'phase_errors' in perturb_dict:
            phase_coeff = np.exp(1j * perturb_dict['phase_errors'][0])
            A = phase_coeff * A
            if compute_derivatives:
                DA = [phase_coeff * X for X in DA]
        if 'mutual_coupling' in perturb_dict:
            A = perturb_dict['mutual_coupling'][0] @ A
            if compute_derivatives:
                DA = [perturb_dict['mutual_coupling'][0] @ X for X in DA]

        if compute_derivatives:
            return (A,) + tuple(DA)
        else:
            return A

class GridBasedArrayDesign(ArrayDesign):
    """Base class for all grid-based array designs.
    
    For grid based arrays, each elements is placed on a predefined grid.

    Args:
        indices (ndarray): m x d matrix denoting the grid indices
            of each element. e.g., if indices is ``[1, 2, 3,]``, then the
            actual locations will be ``[d0, 2*d0, 3*d0]``. The input ndarray is
            not copied and should never be changed after creating this array
            design.
        d0 (float): Grid size (or base inter-element spacing). For 2D and 3D
            arrays, d0 can either be a scalar (if the base inter-element
            spacing remains the same along all axes), or a list-like object
            such that d0[i] specifies the base inter-element spacing along
            the i-th axis.
        name (str): Name of the array design.
        **kwargs: Other keyword arguments supported by :class:`ArrayDesign`.
    """
    
    def __init__(self, indices, d0, name, **kwargs):
        if not np.isscalar(d0):
            d0 = np.array(d0)
            if indices.shape[1] < d0.size:
                raise ValueError(
                    'd0 must be a scalar, or a 1D ndarray of size 1 or {0}.'
                    .format(indices.shape[1])
                )
        super().__init__(indices * d0, name, **kwargs)
        self._element_indices = indices
        self._d0 = d0

    @property
    def d0(self):
        """Retrieves the base inter-element spacing."""
        return self._d0

    @property
    def element_indices(self):
        """Retrieves the element indices."""
        return self._element_indices.copy()

class UniformLinearArray(GridBasedArrayDesign):
    """Creates an n-element uniform linear array (ULA).
        
    The ULA is placed along the x-axis, whose the first sensor is placed at
    the origin.

    Args:
        n (int): Number of elements.
        d0 (float): Fundamental inter-element spacing (usually smallest).
        name (str): Name of the array design.
        **kwargs: Other keyword arguments supported by :class:`ArrayDesign`.
    """

    def __init__(self, n, d0, name=None, **kwargs):
        if name is None:
            name = 'ULA ' + str(n)
        super().__init__(np.arange(n).reshape((-1, 1)), d0, name, **kwargs)

class NestedArray(GridBasedArrayDesign):
    """Creates an 1D nested array.

    Args:
        n1 (int): Parameter N1.
        n2 (int): Parameter N2.
        d0 (float): Fundamental inter-element spacing (usually smallest).
        name (str): Name of the array design.
        **kwargs: Other keyword arguments supported by :class:`ArrayDesign`.

    References:
        [1] P. Pal and P. P. Vaidyanathan, "Nested arrays: A novel approach to
        array processing with enhanced degrees of freedom," IEEE Transactions on
        Signal Processing, vol. 58, no. 8, pp. 4167-4181, Aug. 2010.
    """

    def __init__(self, n1, n2, d0, name=None, **kwargs):
        if name is None:
            name = 'Nested ({0},{1})'.format(n1, n2)
        indices = np.concatenate((
            np.arange(0, n1),
            np.arange(1, n2 + 1) * (n1 + 1) - 1
        ))
        self._n1 = n1
        self._n2 = n2
        super().__init__(indices.reshape((-1, 1)), d0, name, **kwargs)

    @property
    def n1(self):
        """Retrieves the parameter, N1, used when creating this nested array."""
        return self._n1
    
    @property
    def n2(self):
        """Retrieves the parameter, N2, used when creating this nested array."""
        return self._n2

class CoPrimeArray(GridBasedArrayDesign):
    """Creates an 1D co-prime array.

    Args:
        m (int): The smaller number in the co-prime pair.
        n (int): The larger number in the co-prime pair.
        d0 (float): Fundamental inter-element spacing (usually smallest).
        mode (str): Either ``'m'`` or ``'2m'``.
        name (str): Name of the array design.
        **kwargs: Other keyword arguments supported by :class:`ArrayDesign`.
    
    References:
        [1] P. Pal and P. P. Vaidyanathan, "Coprime sampling and the music
        algorithm," in 2011 Digital Signal Processing and Signal Processing
        Education Meeting (DSP/SPE), 2011, pp. 289-294.
    """

    def __init__(self, m, n, d0, mode='2m', name=None, **kwargs):
        if name is None:
            name = 'Co-prime ({0},{1})'.format(m, n)
        if m > n:
            warnings.warn('m > n. Swapped.')
            m, n = n, m
        if gcd(m, n) != 1:
            raise ValueError('{0} and {1} are not co-prime.'.format(m, n))
        self._coprime_pair = (m, n)
        mode = mode.lower()
        if mode == '2m':
            indices = np.concatenate((
                np.arange(0, n) * m,
                np.arange(1, 2*m) * n
            ))
        elif mode == 'm':
            indices = np.concatenate((
                np.arange(0, n) * m,
                np.arange(1, m) * n
            ))
        else:
            raise ValueError('Unknown mode "{0}"'.format(mode))
        self._mode = mode
        super().__init__(indices.reshape((-1, 1)), d0, name, **kwargs)

    @property
    def coprime_pair(self):
        """Retrieves the co-prime pair used when creating this co-prime array."""
        return self._coprime_pair

    @property
    def mode(self):
        """Retrieves the mode used when creating this co-prime array."""
        return self._mode

_MRLA_PRESETS = [
    [0],
    [0, 1],
    [0, 1, 3],
    [0, 1, 4, 6],
    [0, 1, 4, 7, 9],
    [0, 1, 6, 9, 11, 13],
    [0, 1, 8, 11, 13, 15, 17],
    [0, 1, 4, 10, 16, 18, 21, 23],
    [0, 1, 4, 10, 16, 22, 24, 27, 29],
    [0, 1, 4, 10, 16, 22, 28, 30, 33, 35],
    [0, 1, 6, 14, 22, 30, 32, 34, 37, 39, 41],
    [0, 1, 6, 14, 22, 30, 38, 40, 42, 45, 47, 49],
    [0, 1, 6, 14, 22, 30, 38, 46, 48, 50, 53, 55, 57],
    [0, 1, 6, 14, 22, 30, 38, 46, 54, 56, 58, 61, 63, 65],
    [0, 1, 6, 14, 22, 30, 38, 46, 54, 62, 64, 66, 69, 71, 73],
    [0, 1, 8, 18, 28, 38, 48, 58, 68, 70, 72, 74, 77, 79, 81, 83],
    [0, 1, 8, 18, 28, 38, 48, 58, 68, 78, 80, 82, 84, 87, 89, 91, 93],
    [0, 1, 8, 18, 28, 38, 48, 58, 68, 78, 88, 90, 92, 94, 97, 99, 101, 103],
    [0, 1, 8, 18, 28, 38, 48, 58, 68, 78, 88, 98, 100, 102, 104, 107, 109, 111, 113],
    [0, 1, 10, 22, 34, 46, 58, 70, 82, 94, 106, 108, 110, 112, 114, 117, 119, 121, 123, 125]
]

class MinimumRedundancyLinearArray(GridBasedArrayDesign):
    """Creates an n-element minimum redundancy linear array (MRLA).

    Args:
        n (int): Number of elements. Up to 20.
        d0 (float): Fundamental inter-element spacing (usually smallest).
        name (str): Name of the array design.
        **kwargs: Other keyword arguments supported by :class:`ArrayDesign`.
    
    References:
        [1] M. Ishiguro, "Minimum redundancy linear arrays for a large number of
        antennas," Radio Sci., vol. 15, no. 6, pp. 1163-1170, Nov. 1980.
        
        [2] A. Moffet, "Minimum-redundancy linear arrays," IEEE Transactions on
        Antennas and Propagation, vol. 16, no. 2, pp. 172-175, Mar. 1968.
    """

    def __init__(self, n, d0, name=None, **kwargs):
        if n < 1 or n >= len(_MRLA_PRESETS):
            raise ValueError('The MRLA presets only support up to 20 elements.')
        if name is None:
            name = 'MRLA {0}'.format(n)
        super().__init__(np.array(_MRLA_PRESETS[n - 1])[:, np.newaxis], d0, name, **kwargs)
        
class UniformCircularArray(ArrayDesign):
    """Creates a uniform circular array (UCA).
    
    The UCA is centered at the origin, in the xy-plane.

    Args:
        n (int): Number of elements.
        r (float): Radius of the circle.
        name (str): Name of the array design.
        **kwargs: Other keyword arguments supported by :class:`ArrayDesign`.
    """

    def __init__(self, n, r, name=None, **kwargs):
        if name is None:
            name = 'UCA ' + str(n)
        self._r = r
        theta = np.linspace(0., np.pi * (2.0 - 2.0 / n), n)
        locations = np.vstack((r * np.cos(theta), r * np.sin(theta))).T
        super().__init__(locations, name)

    @property
    def radius(self):
        """Retrieves the radius of the uniform circular array."""
        return self._r

class UniformRectangularArray(GridBasedArrayDesign):
    """Creates an m x n uniform rectangular array (URA).
    
    The URA is placed on the xy-plane, and the (0,0)-th sensor is placed
    at the origin.

    Args:
        m (int): Number of elements along the x-axis.
        n (int): Number of elements along the y-axis.
        d0 (float): Fundamental inter-element spacing. Can be either a
            scalar or a two-element list-like object.
        name (str): Name of the array design.
        **kwargs: Other keyword arguments supported by :class:`ArrayDesign`.
    """

    def __init__(self, m, n, d0, name=None, **kwargs):
        if name is None:
            name = 'URA {0}x{1}'.format(m, n)
        self._shape = (m, n)
        indices = cartesian(np.arange(m), np.arange(n))
        super().__init__(indices, d0, name, **kwargs)
    
    @property
    def shape(self):
        """Retrieves the shape of this uniform rectangular array."""
        return self._shape
