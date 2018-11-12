import warnings
import numpy as np

class ArrayPerturbation:
    """Creates an array perturbation.

    In practice, sensor arrays are not perfectly calibrated and various
    array imperfections exist. These imperfections/perturbations are generally
    modelled with parameters independent of the sources.

    Args:
        params: Parameters associated with this perturbation. Usually an
            :class:`~numpy.ndarray`.
        known (bool): Specifies whether this perturbation is known in prior.
            Default value is ``False``.
    """

    def __init__(self, params, known=False):
        self._params = params
        self._is_known = known

    @property
    def params(self):
        """Retrieves the parameters used to model this perturbation."""
        return self._params

    @property
    def is_known(self):
        """Retrieves whether this perturbation is known."""
        return self._is_known

    def is_applicable_to(self, array):
        """Checks if this perturbation is applicable to the given array.

        Args:
            array (~doatools.model.arrays.ArrayDesign): Array design.
        
        Returns:
            tuple: A two-element tuple of the format ``(bool, str)``. The first
            element is a boolean indicating whether this perturbation is
            applicable. The second element is a string describing why this
            perturbation is not applicable.
        """
        return True

    def perturb_sensor_locations(self, locations):
        """Perturbs the given sensor locations.

        Args:
            locations (~numpy.ndarray): A matrix containing the locations of
                array elements/sensors, where the :math:`i`-th row consists of
                the Cartesian coordinates of the :math:`i`-th array
                element/sensor.
        
        Notes:
            The number of dimensions of the sensor locations and number of
            dimensions of the perturbations do not need to match. It is
            perfectly fine to apply 3D location errors to 1D arrays.
        """
        return locations

    def perturb_steering_matrix(self, A, DA):
        r"""Perturbs the given steering matrix (and derivative matrices).

        Given the steering matrix

        .. math::

            \mathbf{A} = \begin{bmatrix}
                \mathbf{a}(\mathbf{\theta}_1) &
                \mathbf{a}(\mathbf{\theta}_2) &
                \cdots &
                \mathbf{a}(\mathbf{\theta}_K)
            \end{bmatrix},

        The d-th derivative matrix is computed as

        .. math::

            \dot{\mathbf{A}}_d = \begin{bmatrix}
                \frac{\partial \mathbf{a}(\mathbf{\theta}_1)}{\partial \theta_{1d}} &
                \frac{\partial \mathbf{a}(\mathbf{\theta}_2)}{\partial \theta_{2d}} &
                \cdots &
                \frac{\partial \mathbf{a}(\mathbf{\theta}_K)}{\partial \theta_{Kd}}
            \end{bmatrix},

        where :math:`\theta_{kd}` is the :math:`d`-th parameter of the
        :math:`k`-th source location.

        Denote the perturbation as a function :math:`C(\cdots)`. The perturbed
        steering matrix can then be expressed as

        .. math::

            \tilde{\mathbf{A}} = \begin{bmatrix}
                C(\mathbf{a}(\mathbf{\theta}_1)) &
                C(\mathbf{a}(\mathbf{\theta}_2)) &
                \cdots &
                C(\mathbf{a}(\mathbf{\theta}_K))
            \end{bmatrix}.

        The :math:`d`-th derivative matrix should be computed as

        .. math::

            \dot{\tilde{\mathbf{A}}}_d = \begin{bmatrix}
                \frac{\partial C(\mathbf{a}(\mathbf{\theta}_1))}{\partial \theta_{1d}} &
                \frac{\partial C(\mathbf{a}(\mathbf{\theta}_2))}{\partial \theta_{2d}} &
                \cdots &
                \frac{\partial C(\mathbf{a}(\mathbf{\theta}_K))}{\partial \theta_{Kd}}
            \end{bmatrix}.
        
        Args:
            A (~numpy.ndarray): Steering matrix input.
            DA (list): A list of derivative matrices. If this list is empty,
                there is no need to consider the derivative matrices.
        
        Returns:
            tuple: A two element tuple. The first element is the perturbed
            steering matrix. The second element is a list of perturbed
            derivative matrices. If ``DA`` is an empty list, the second element
            should also be an empty list.
        """
        return A, DA

class LocationErrors(ArrayPerturbation):
    """Creates an array perturbation that models sensor location errors.
    
    Args:
        location_errors: Location error matrix.
        known (bool): Specifies whether this perturbation is known in prior.
            Default value is ``False``.
    """

    def __init__(self, location_errors, known=False):
        if not isinstance(location_errors, np.ndarray):
            location_errors = np.array(location_errors)
        if location_errors.ndim != 2:
            raise ValueError('Location errors should be stored in a matrix.')
        if location_errors.shape[1] < 1 or location_errors.shape[1] > 3:
            raise ValueError('Location errors can only be 1D, 2D, or 3D.')
        super().__init__(location_errors, known)

    def is_applicable_to(self, array):
        m = self._params.shape[0]
        if array.size != m:
            return False, 'Expecting an array of size {0}'.format(m)
        else:
            return True, ''

    def perturb_sensor_locations(self, locations):
        array_dim = locations.shape[1]
        loc_err_dim = self._params.shape[1]
        if loc_err_dim <= array_dim:
            # It is possible that the location errors only exist along the
            # first one or two axis.
            perturbed_locations = locations.copy()
            perturbed_locations[:, :loc_err_dim] += self._params
        else:
            # Actual dimension is higher. For instance, this can happen if a
            # linear array, which is 1D, has location errors along both
            # x- and y-axis.
            perturbed_locations = self._params.copy()
            perturbed_locations[:, :array_dim] += locations
        return perturbed_locations

class GainErrors(ArrayPerturbation):
    """Creates an array perturbation that models gain errors.

    Args:
        gain_errors: A real vector storing the gain errors. The values are
            relative. For instance, a gain error of ``-0.1`` means the actual
            gain is ``0.9``.
        known (bool): Specifies whether this perturbation is known in prior.
            Default value is ``False``.
    """
    def __init__(self, gain_errors, known=False):
        if not isinstance(gain_errors, np.ndarray):
            gain_erros = np.array(gain_errors)
        if gain_errors.ndim != 1:
            raise ValueError('Expecting a vector.')
        super().__init__(gain_errors, known)

    def is_applicable_to(self, array):
        if not array.element.is_scalar:
            return False, 'The array element must have a scalar output.'
        m = self._params.shape[0]
        if array.size != m:
            return False, 'Expecting an array of size {0}'.format(m)
        else:
            return True, ''

    def perturb_steering_matrix(self, A, DA):
        r"""Perturbs the steering matrix with gain errors.
        
        Given the gain error vector, :math:`\mathbf{g}`, the perturbed steering
        matrix is computed as

        .. math::

            \tilde{\mathbf{A}} = (\mathbf{g} + \mathbf{I}) \mathbf{A}.

        The derivative matrices are computed as

        .. math::

            \dot{\tilde{\mathbf{A}}}_i = (\mathbf{g} + \mathbf{I})
                \dot{\mathbf{A}}_i.

        Args:
            A (~numpy.ndarray): Steering matrix input.
            DA (list): A list of derivative matrices. If this list is empty,
                there is no need to consider the derivative matrices.
        
        Returns:
            tuple: A two element tuple. The first element is the perturbed
            steering matrix. The second element is a list of perturbed
            derivative matrices. If ``DA`` is an empty list, the second element
            should also be an empty list.
        """
        g = 1.0 + self._params[:, np.newaxis]
        return g * A, [g * X for X in DA]

class PhaseErrors(ArrayPerturbation):
    """Creates an array perturbation that models phase errors.

    Args:
        phase_errors: A real vector storing the phase errors. The values are
            in radians.
        known (bool): Specifies whether this perturbation is known in prior.
            Default value is ``False``.
    """
    def __init__(self, phase_errors, known=False):
        if not isinstance(phase_errors, np.ndarray):
            gain_erros = np.array(phase_errors)
        if phase_errors.ndim != 1:
            raise ValueError('Expecting a vector.')
        super().__init__(phase_errors, known)

    def is_applicable_to(self, array):
        if not array.element.is_scalar:
            return False, 'The array element must have a scalar output.'
        m = self._params.shape[0]
        if array.size != m:
            return False, 'Expecting an array of size {0}'.format(m)
        else:
            return True, ''

    def perturb_steering_matrix(self, A, DA):
        r"""Perturbs the steering matrix with phase errors.

        Given the phase error vector, :math:`\mathbf{\phi}`, the perturbed
        steering matrix is computed as

        .. math::

            \tilde{\mathbf{A}} = \mathrm{diag}(\exp(j\mathrm{\phi})) \mathbf{A},

        where the :math:`\exp(\cdot)` denote element-wise exponentiation.

        The derivative matrices are computed as

        .. math::

            \dot{\tilde{\mathbf{A}}}_i = \mathrm{diag}(\exp(j\mathrm{\phi}))
                \dot{\mathbf{A}}_i.

        Args:
            A (~numpy.ndarray): Steering matrix input.
            DA (list): A list of derivative matrices. If this list is empty,
                there is no need to consider the derivative matrices.
        
        Returns:
            tuple: A two element tuple. The first element is the perturbed
            steering matrix. The second element is a list of perturbed
            derivative matrices. If ``DA`` is an empty list, the second element
            should also be an empty list.
        """
        phi = np.exp(1j * self._params[:, np.newaxis])
        return phi * A, [phi * X for X in DA]

class MutualCoupling(ArrayPerturbation):
    """Creates an array perturbation that models mutual coupling.

    Args:
        C: Mutual coupling matrix.
        known (bool): Specifies whether this perturbation is known in prior.
            Default value is ``False``.
    """
    def __init__(self, C, known=False):
        if not isinstance(C, np.ndarray):
            C = np.array(C)
        if C.ndim != 2 and C.shape[0] != C.shape[1]:
            raise ValueError('Expecting a square matrix.')
        super().__init__(C, known)

    def is_applicable_to(self, array):
        if not array.element.is_scalar:
            return False, 'The array element must have a scalar output.'
        m = self._params.shape[0]
        if array.size != m:
            return False, 'Expecting an array of size {0}'.format(m)
        else:
            return True, ''

    def perturb_steering_matrix(self, A, DA):
        r"""Perturbs the steering matrix with mutual coupling.

        Given the mutual coupling matrix, :math:`\mathbf{C}`, the perturbed
        steering matrix is computed as

        .. math::

            \tilde{\mathbf{A}} = \mathbf{C} \mathbf{A}.

        The derivative matrices are computed as

        .. math::

            \dot{\tilde{\mathbf{A}}}_i = \mathbf{C} \dot{\mathbf{A}}_i.
        
        Args:
            A (~numpy.ndarray): Steering matrix input.
            DA (list): A list of derivative matrices. If this list is empty,
                there is no need to consider the derivative matrices.
        
        Returns:
            tuple: A two element tuple. The first element is the perturbed
            steering matrix. The second element is a list of perturbed
            derivative matrices. If ``DA`` is an empty list, the second element
            should also be an empty list.
        """
        return self._params @ A, [self._params @ X for X in DA]

