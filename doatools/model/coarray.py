import numpy as np
from .arrays import GridBasedArrayDesign
from ..utils.math import unique_rows

def compute_location_differences(locations):
    r"""Computes all locations differences, including duplicates.
    
    Suppose ``locations`` is :math:`m \times d`, then the result will be an
    :math:`m^2 \times d` matrix such that ``locations[i] - locations[j]`` is
    stored in the ``(i + j * m)``-th row of the resulting matrix.

    For instance, if ``locations`` is ``[[0, 1], [1, 3]]``, then the output will
    be ``[[0, 0], [1, 2], [-1, -2], [0, 0]]``.

    Args:
        locations (~numpy.ndarray): An m x d array of sensor locations.
    """
    m, d = locations.shape
    # Use broadcasting to compute pairwise differences.
    D = locations.reshape((1, m, d)) - locations.reshape((m, 1, d))
    return D.reshape((-1, d))

def compute_unique_location_differences(locations, atol=0.0, rtol=1e-8):
    """Computes all unique locations differences.

    Unlike :meth:`compute_location_differences`, duplicates within the
    specified tolerance are removed.

    Args:
        locations: An m x d array of sensor locations.
    """
    return unique_rows(compute_location_differences(locations), atol, rtol)

class WeightFunction1D:
    """Creates a 1D weight function.

    Args:
        array (~doatools.model.arrays.ArrayDesign): Array design.

    References:
        [1] P. Pal and P. P. Vaidyanathan, "Nested arrays: A novel approach to
        array processing with enhanced degrees of freedom," IEEE Transactions on
        Signal Processing, vol. 58, no. 8, pp. 4167-4181, Aug. 2010.
    """

    def __init__(self, array):
        if array.ndim != 1 or not isinstance(array, GridBasedArrayDesign):
            raise ValueError('Expecting a 1D grid-based array.')
        self._m = array.size
        self._mv = None
        self._build_map(array)

    def __call__(self, diff):
        """Evaluates the weight function at the given difference."""
        return self.weight_of(diff)

    def __len__(self):
        """Retrieves the number of unique differences."""
        return len(self._index_map)

    def differences(self):
        """Retrieves a 1D array of unique differences in ascending order.
        
        The ordering of elements returned by :meth:`differences` and the
        ordering of elements returned by :meth:`weights` are the same.
        """
        return self._differences.copy()

    def weights(self):
        """Retrieves a 1D array of weights.

        The ordering of elements returned by :meth:`differences` and the
        ordering of elements returned by :meth:`weights` are the same.
        """
        return np.array([len(self._index_map[x]) for x in self._differences])

    def weight_of(self, diff):
        """Evaluates the weight function at the given difference."""
        if diff in self._index_map:
            return len(self._index_map[diff])
        else:
            return 0

    def indices_of(self, diff):
        """Retrieves the list of indices of elements in the vectorized
        difference matrix that correspond to the given difference. If the given
        difference does not exist, an empty list will be returned.

        Args:
            diff (int): Difference.
        """
        if diff in self._index_map:
            return self._index_map[diff][:]
        else:
            return []

    def get_central_ula_size(self, exclude_negative_part=False):
        r"""Gets the size of the central ULA in the difference coarray.
        
        Args:
            exclude_negative_part (bool): Set to ``True`` to exclude the
                virtual array elements corresponding to negative differences.
                The central ULA part is symmetric with respect to the origin and
                can be represented with

                .. math::

                    \lbrack
                    -M_\mathrm{v}+1, \ldots, -1, 0, 1, \ldots, M_\mathrm{v}
                    \rbrack d_0

                After excluding the negative part, the remaining array elements
                are given by

                .. math::

                    \lbrack
                    0, 1, \ldots, M_\mathrm{v}
                    \rbrack d_0
                
                Default value is ``False``.
        """
        if self._mv is None:
            mv = 0
            while mv in self._index_map:
                mv += 1
            self._mv = mv
        return self._mv if exclude_negative_part else self._mv * 2 - 1 
    
    def get_coarray_selection_matrix(self, exclude_negative_part=False):
        r"""Gets the coarray selection matrix.

        Let the central ULA size be :math:`2M_{\mathrm{v}} - 1` and the original
        array size be :math:`M`. :math:`\mathbf{F}` is defined as an
        :math:`(2M_\mathrm{v} - 1) \times M^2` matrix that transforms the
        vectorized sample covariance matrix, :math:`\mathrm{vec}(\mathbf{R})`,
        to the virtual observation vector of the central ULA,
        :math:`\mathbf{z}`, via redundancy averaging:

        .. math::

            \mathbf{z} = \mathbf{F} \mathrm{vec}(\mathbf{R}).

        Args:
            exclude_negative_part: If set to ``True``, only the nonnegative part
                of the central ULA (i.e.,
                :math:`\lbrack 0, 1, \ldots, M_\mathrm{v} - 1\rbrack`) will be
                considered, and the resulting :math:`\mathbf{F}` will be
                :math:`M_\mathrm{v} \times M^2`. Default value is ``False``.
        
        Returns:
            The coarray selection matrix.
        
        References:
            [1] M. Wang and A. Nehorai, "Coarrays, MUSIC, and the Cramér-Rao
            Bound," IEEE Transactions on Signal Processing, vol. 65, no. 4,
            pp. 933-946, Feb. 2017.
        """
        m_v = self.get_central_ula_size(exclude_negative_part=True)
        if exclude_negative_part:
            m_ula = m_v
            diff_range = range(0, m_v)
        else:
            m_ula = 2 * m_v - 1
            diff_range = range(-m_v + 1, m_v)
        F = np.zeros((m_ula, self._m**2))
        for i, diff in enumerate(diff_range):
            F[i, self.indices_of(diff)] = 1.0 / self.weight_of(diff)
        return F
    
    def _build_map(self, array):
        # Maps difference -> indices in the vectorized difference matrix 
        index_map = {}
        diffs = compute_location_differences(array.element_indices).flatten()
        for i, diff in enumerate(diffs):
            if diff in index_map:
                index_map[diff].append(i)
            else:
                index_map[diff] = [i]
        # Collect all unique differences and sort them
        differences = np.fromiter(index_map.keys(),
            dtype=np.int_, count=len(index_map))
        differences.sort()
        self._index_map = index_map
        self._differences = differences
