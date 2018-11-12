from math import gcd
from ..utils.math import cartesian
import numpy as np
import warnings
import copy
from .array_elements import ISOTROPIC_SCALAR_SENSOR
from .perturbations import LocationErrors, GainErrors, PhaseErrors, \
                           MutualCoupling

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
        perturbations (list or dict): If a list is given, it should be a list of
            :class:`~doatools.model.perturbations.ArrayPerturbation`.

            If a dictionary is given, it should be a dictionary containing the
            perturbation definitions. Correspinding
            :class:`~doatools.model.perturbations.ArrayPerturbation` will be
            created automatically. The keys should be among the following:

            * ``'location_errors'``
            * ``'gain_errors'`` (relative, -0.2 means 0.8 * original gain)
            * ``'phase_errors'`` (in radians)
            * ``'mutual_coupling'``

            The values in the dictionary are two-element tuples. The first
            element is an :class:`~numpy.ndarray` representing the parameters,
            and the second element is a boolean specifying whether these
            parameters are known in prior.
        element (~doatools.model.array_elements.ArrayElement): Array element
            (sensor) used in this array. Default value is an instance of
            :class:`~doatools.model.array_elements.IsotropicScalarSensor`.

    Notes:
        Array designs are generally not changed after creation. Because array
        design objects are passed around when computing steering matrices,
        weight functions, etc., having a mutable internal state leads to more
        complexities and potential unexpected results. Although the internal
        states are generally accessible in Python, please refrain from modifying
        them unless you are aware of the side effects.
    """

    def __init__(self, locations, name, perturbations=[],
                 element=ISOTROPIC_SCALAR_SENSOR):
        if not isinstance(locations, np.ndarray):
            locations = np.array(locations)
        if locations.ndim > 2:
            raise ValueError('Expecting a 1D vector or a 2D matrix.')
        if locations.ndim == 1:
            locations = locations.reshape((-1, 1))
        elif locations.shape[1] > 3:
            raise ValueError('Array can only be 1D, 2D or 3D.')
        self._locations = locations
        self._name = name
        self._element = element
        # Validate and add perturbations
        self._perturbations = {}
        self._add_perturbation_from_list(self._parse_input_perturbations(perturbations))
    
    @property
    def name(self):
        """Retrieves the name of this array."""
        return self._name

    @property
    def size(self):
        """Retrieves the number of elements in the array."""
        return self._locations.shape[0]

    @property
    def output_size(self):
        """Retrieves the output size of the array.
        
        In generate, the output size should be equal to ``size``. However, for
        vector sensor arrays, the output size is greater than the array size.
        """
        return self.size
    
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
        locations = self._locations
        for p in self._perturbations.values():
            locations = p.perturb_sensor_locations(locations)
        return locations

    @property
    def element(self):
        """Retrieves the array element."""
        return self._element
    
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
        return self.actual_element_locations.shape[1]
    
    def has_perturbation(self, ptype):
        """Checks if the array has the given type of perturbation."""
        return ptype in self._perturbations
    
    def is_perturbation_known(self, ptype):
        """Checks if the specified perturbation is known in prior."""
        return self._perturbations[ptype].is_known
    
    def get_perturbation_params(self, ptype):
        """Retrieves the parameters for the specified perturbation type."""
        return self._perturbations[ptype].params
    
    @property
    def perturbations(self):
        """Retrieves a list of all perturbations."""
        # Here we have a deep copy.
        return list(self._perturbations.values())

    def _add_perturbation_from_list(self, perturbations, raise_on_override=True):
        """Adds perturbations from a list of perturbations.
        
        Args:
            perturbations (list): A list of
                :class:`~doatools.model.perturbations.ArrayPerturbation.`.
            raise_on_override: Specifies whether an error should be raised when
                a new perturbation of the same type overrides the existing one.
        """
        for p in perturbations:
            applicable, msg = p.is_applicable_to(self)
            if not applicable:
                raise RuntimeError(msg)
            p_class = p.__class__
            if p_class in self._perturbations and raise_on_override:
                raise RuntimeError(
                    'Cannot have more than one perturbations of the same type. '
                    'Attempting to add another perturbation of the type {0}.'
                    .format(p_class.__name__)
                )
            self._perturbations[p_class] = p
    
    def _parse_input_perturbations(self, perturbations):
        if isinstance(perturbations, dict):
            factories = {
                'location_errors': (lambda p, k: LocationErrors(p, k)),
                'gain_errors': (lambda p, k: GainErrors(p, k)),
                'phase_errors': (lambda p, k: PhaseErrors(p, k)),
                'mutual_coupling': (lambda p, k: MutualCoupling(p, k))
            }
            perturbations = [factories[k](v[0], v[1]) for k, v in perturbations.items()]
        return perturbations
        
    def get_perturbed_copy(self, perturbations, new_name=None):
        """Returns a copy of this array design but with the specified
        perturbations.
        
        The specified perturbations will replace the existing ones.

        Notes:
            The default implementation performs a shallow copy of all existing
            fields using :meth:``~copy.copy``. Override this method if special
            operations are required.

        Args:
            perturbations (list or dict): If a list is given, it should be a
                list of
                :class:`~doatools.model.perturbations.ArrayPerturbation`.

                If a dictionary is given, it should be a dictionary containing
                the perturbation definitions. Correspinding
                :class:`~doatools.model.perturbations.ArrayPerturbation` will be
                created automatically. The keys should be among the following:

                * ``'location_errors'``
                * ``'gain_errors'`` (relative, -0.2 means 0.8 * original gain)
                * ``'phase_errors'`` (in radians)
                * ``'mutual_coupling'``

                The values in the dictionary are two-element tuples. The first
                element is an :class:`~numpy.ndarray` representing the
                parameters, and the second element is a boolean specifying
                whether these parameters are known in prior.
            new_name (str): An optional new name for the resulting array design.
                If not provided, the name of the original array design will be
                used.
        """
        array = self.get_perturbation_free_copy(new_name)
        # Merge perturbation parameters.
        new_perturbations = self._parse_input_perturbations(perturbations)
        array._perturbations = self._perturbations.copy()
        array._add_perturbation_from_list(new_perturbations, False)
        return array

    def get_perturbation_free_copy(self, new_name=None):
        """Returns a perturbation-free copy of this array design.

        Notes:
            The default implementation performs a shallow copy of all existing
            fields using :meth:``~copy.copy``. Override this method if special
            operations are required.

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
                        perturbations='all', flatten=True):
        r"""Creates the steering matrix for the given DOAs.

        Given :math:`K` sources, denoted by :math:`\mathbf{\theta}`, and
        :math:`M` sensors whose the actual sensor locations (after considering
        locations errors if exist) are denoted by :math:`\mathbf{d}`,
        the steering matrix is calculated as

        .. math::

            \mathbf{A} = \mathbf{C} (\mathbf{F}(\mathbf{\theta}, \mathbf{d})
                \odot \mathbf{A}_0(\mathbf{\theta}, \mathbf{d})).

        Here :math:`\odot` denotes the Hadamard product.

        * :math:`\mathbf{A}_0` is an :math:`M \times K` matrix calculated from
          the phase delays between the sourcess, :math:`\mathbf{\theta}`, and
          the sensor locations, :math:`\mathbf{d}`:

          .. math::

                \mathbf{A}_0 = \begin{bmatrix}
                    \mathbf{a}_0(\mathbf{\theta}_1, \mathbf{d}) &
                    \mathbf{a}_0(\mathbf{\theta}_2, \mathbf{d}) &
                    \cdots &
                    \mathbf{a}_0(\mathbf{\theta}_K, \mathbf{d})
                \end{bmatrix}.
            
        * :math:`\mathbf{F}` is the spatial response matrix. For isotropic
          scalar sensors, :math:`\mathbf{F}` is an :math:`M \times K` matrix of
          ones. For vectors sensor arrays, each sensor's output is a vector of
          size :math:`L`. Consequently, :math:`\mathbf{F}` is an
          :math:`L \times M \times K` tensor and the broadcasting rule applies
          when computing the element-wise multiplication between
          :math:`\mathbf{F}` and :math:`\mathbf{A}_0`.
        * :math:`\mathbf{C}(\cdot)` is a matrix function that applies other
          perturbations such as gain errors, phase errors, and mutual coupling.

        When ``compute_derivatives`` is ``True``, this method also computes the
        derivative matrices associated with the source location parameters.
        The :math:`k`-th column of the steering matrix, :math:`\mathbf{A}`,
        should depend only on the :math:`k`-th source. Consequently,
        :math:`\mathbf{A}` can be expressed as

        .. math::

            \mathbf{A} = \begin{bmatrix}
                \mathbf{a}(\mathbf{\theta}_1) &
                \mathbf{a}(\mathbf{\theta}_2) &
                \cdots &
                \mathbf{a}(\mathbf{\theta}_K)
            \end{bmatrix}.
        
        Then the :math:`i`-th derivative matrix is computed as

        .. math::

            \dot{\mathbf{A}}_i = \begin{bmatrix}
                \frac{\partial \mathbf{a}(\mathbf{\theta}_1)}{\partial \theta_{1i}} &
                \frac{\partial \mathbf{a}(\mathbf{\theta}_2)}{\partial \theta_{2i}} &
                \cdots &
                \frac{\partial \mathbf{a}(\mathbf{\theta}_K)}{\partial \theta_{Ki}}
            \end{bmatrix},
        
        where :math:`\theta_{ki}` is the :math:`i`-th parameter of the
        :math:`k`-th source location.

        The current implementation cannot compute the derivative matrices
        when the array element is non-isotropic or non-scalar.
        
        Args:
            sources (~doatools.model.sources.SourcePlacement):
                An instance of :class:`~doatools.model.sources.SourcePlacement`.

                1. 1D arrays are placed along the x-axis. 2D arrays are placed
                   within the xy-plane.
                2. If you pass in 1D DOAs for an 2D or 3D array, these DOAs will
                   be assumed to be within the xy-plane. The azimuth angles are
                   calculated as :math:`\pi/2` minus the original 1D DOA values
                   (broadside -> azimuth). The elevation angles are set to zeros
                   (within the xy-plane).
            
            wavelength (float): Wavelength of the carrier wave.
            compute_derivatives (bool): If set to True, also outputs the
                derivative matrices with respect to the DOAs. The k-th column of
                the i-th derivative matrix contains the derivatives of the k-th
                column of A with respect to the i-th parameter associated with
                the k-th DOA. The derivative matrices are used when computing
                the CRBs. Not always available.
            perturbations (str): Specifies which perturbations are considered
                when constructing the steering matrix:

                * ``'all'`` - All perturbations are considered. This is the
                  default value.
                * ``'known'`` - Only known perturbations (we have prior
                  knowledge of the perturbation parameters) are considered. This
                  option is used by DOA estimators when the exact knowledge of
                  these perturbations are known in prior.
                * ``'none'`` - None of the perturbations are considered.
            flatten (bool): Specifies whether the output should be flattend to
                matrices. This option does not have any effect if the array
                element has a scalar output. For an array element of non-scalar
                outputs (e.g., a vector sensor), the resulting steering matrix
                is actually a :math:`L \times M \times K` tensor, where]
                :math:`L` is the output size of each array element, :math:`M` is
                the array size, and :math:`K` is the number of sources. Setting
                ``flatten`` to ``True`` will flatten the tensor into a
                :math:`LM \times K` matrix. Default value is ``True``.
        
        Notes:
            The steering matrix calculation is bound to array designs. This is
            a generic implementation, which can be overridden for special types
            of arrays.
        """
        # Filter perturbations.
        if perturbations == 'all':
            perturb_list = self._perturbations.values()
        elif perturbations == 'known':
            perturb_list = [v for v in self._perturbations.values() if v.is_known]
        elif perturbations == 'none':
            perturb_list = []
        else:
            raise ValueError('Perturbation can only be "all", "known", or "none".')
        # Check array element.
        if not self._element.is_isotropic or not self._element.is_scalar:
            require_spatial_response = True
            if compute_derivatives:
                raise RuntimeError(
                    'Derivative computation is not supported when the array '
                    'elements are non-isotropic or non-scalar.'
                )
        else:
            require_spatial_response = False
        # Compute actual element locations.
        actual_locations = self._locations
        for p in perturb_list:
            actual_locations = p.perturb_sensor_locations(actual_locations)
        # Compute the steering matrix
        T = sources.phase_delay_matrix(actual_locations, wavelength, compute_derivatives)
        if compute_derivatives:
            A = np.exp(1j * T[0])
            DA = [A * (1j * X) for X in T[1:]]
        else:
            A = np.exp(1j * T)
            DA = []
        # Apply spatial response
        if require_spatial_response:
            if sources.is_far_field:
                # For far-field sources, the source locations do not matter.
                r, az, el = sources.calc_spherical_coords(np.zeros((1, 1)))
            else:
                r, az, el = sources.calc_spherical_coords(actual_locations)
            S = self._element.calc_spatial_response(r, az, el)
            A = S * A
        # Apply other perturbations
        for p in perturb_list:
            A, DA = p.perturb_steering_matrix(A, DA)
        # Prepare for returns.
        if A.ndim > 2 and flatten:
            A = A.reshape((-1, sources.size))
        if compute_derivatives:
            return (A,) + tuple(DA)
        else:
            return A

class GridBasedArrayDesign(ArrayDesign):
    r"""Base class for all grid-based array designs.
    
    For grid based arrays, each elements is placed on a predefined grid. A
    :math:`d`-dimensional grid in a :math:`k`-dimensional space
    (:math:`d \leq k`) is generated by :math:`d` :math:`k`-dimensional basis
    vectors: :math:`\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_d`. The
    location of the :math:`(i_1, i_2, \ldots, i_d)`-th element is given by

    .. math::

        i_1 \mathbf{v}_1 + i_2 \mathbf{v}_2 + \cdots + i_d \mathbf{v}_d.

    Args:
        indices (ndarray): m x d matrix denoting the grid indices
            of each element. The input ndarray is not copied and should never be
            changed after creating this array design.
        d0 (float): Grid size (or base inter-element spacing). For 2D and 3D
            arrays, d0 can either be a scalar (if the base inter-element
            spacing remains the same along all axes), or a list-like object
            such that d0[i] specifies the base inter-element spacing along
            the i-th axis. When using ``d0`` to specify the grid size, the
            grid is assumed to be aligned with the x-, y-, and z-axis.
        bases (~numpy.ndarray): Grid bases. Each row represents a basis vector.
            Given a ``d``-dimensional grid, the element with the grid index
            ``(i1, i2,...,id)`` is located at
            ``i1 * bases[0,:] + ... + id * bases[d-1, :]``. When ``bases`` is
            specified, ``d0`` will be ignored. Use ``bases`` instead of ``d0``
            if the underlying grid is not aligned with the x-, y-, and z-axis.
        name (str): Name of the array design.
        **kwargs: Other keyword arguments supported by :class:`ArrayDesign`.
    """
    
    def __init__(self, indices, d0=None, name=None, bases=None, **kwargs):
        if bases is None:
            # Use d0 to generate bases.
            if np.isscalar(d0):
                d0 = np.full((indices.shape[1],), d0, dtype=np.float_)
            else:
                d0 = np.array(d0, dtype=np.float_)
            if d0.ndim > 1 or indices.shape[1] < d0.size:
                raise ValueError(
                    'd0 must be a scalar or a list-like object of length {0}.'
                    .format(indices.shape[1])
                )
            bases = np.eye(d0.size) * d0
        else:
            if indices.shape[1] != bases.shape[0]:
                raise ValueError(
                    'The number of rows of the bases matrix does not match '
                    'the number of columns of the indices ({0} != {1}).'
                    .format(bases.shape[0], indices.shape[1])
                )
            # Calculate d0 from bases
            d0 = np.linalg.norm(bases, ord=2, axis=1)
        super().__init__(indices @ bases, name, **kwargs)
        self._element_indices = indices
        self._bases = bases
        self._d0 = d0

    @property
    def d0(self):
        """Retrieves the base inter-element spacing(s) along each grid axis.
        
        You are not supposed to modified the returned array.

        Returns:
            ~numpy.ndarray: A 1D vector containing the inter-element spacings
            along each grid axis.
        """
        return self._d0

    @property
    def bases(self):
        """Retrieves the basis vectors for the grid.
        
        You are not supposed to modify the returned array.
        
        Returns:
            ~numpy.ndarray: A matrix where each row respresents a basis vector.
        """
        return self._bases

    @property
    def element_indices(self):
        """Retrieves the element indices.
        
        You are not supposed to modify the returned array.
        """
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
    """Creates a 1D nested array.

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
    """Creates a 1D co-prime array.

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
