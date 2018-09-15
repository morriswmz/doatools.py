import numpy as np
from .arrays import GridBasedArrayDesign

def compute_location_differences(locations):
    '''
    Computes all locations differences. Suppose `locations` is m x d, then
    the result will be a m^2 x d matrix such that `locations[i] - locations[j]`
    is stored in the (i + j * m)-th row of the resulting matrix.

    For instance, if `locations` is [[0, 1], [1, 3]], then the output will be

    [[0, 0], [1, 2], [-1, -2], [0, 0]]

    Args:
        locations: An m x d array of sensor locations.
    '''
    m, d = locations.shape
    # Use broadcasting to compute pairwise differences.
    D = locations.reshape((1, m, d)) - locations.reshape((m, 1, d))
    return D.reshape((-1, d))

class WeightFunction1D:

    def __init__(self, design):
        '''
        Creates an 1D weight function.

        Args:
            design: Array design.

        References:
        [1] P. Pal and P. P. Vaidyanathan, "Nested arrays: A novel approach to
            array processing with enhanced degrees of freedom," IEEE
            Transactions on Signal Processing, vol. 58, no. 8, pp. 4167-4181,
            Aug. 2010.
        '''
        if design.ndim != 1 or not isinstance(design, GridBasedArrayDesign):
            raise ValueError('Expecting an 1D grid-based array.')
        self._build_map(design)

    def __call__(self, diff):
        '''
        Evaluates the weight function at the given difference.
        '''
        if diff in self._index_map:
            return len(self._index_map[diff])
        else:
            return 0

    def __len__(self):
        '''
        Retrieves the number of unique differences.
        '''
        return len(self._index_map)

    def differences(self):
        '''
        Retrieves a 1D array of unqiue differences, sorted in ascending order.
        The ordering of elements returned by `differences()` and the
        ordering of elements returned by `weights()` are the same.
        '''
        return self._differences.copy()

    def weights(self):
        '''
        Retrieves a 1D array of weights.
        The ordering of elements returned by `differences()` and the
        ordering of elements returned by `weights()` are the same.
        '''
        return np.array([len(self._index_map[x]) for x in self._differences])

    def indices_of(self, diff):
        '''
        Retrieves the list of indices of elements in the vectorized difference
        matrix that correspond to the given difference. If the given difference
        does not exist, an empty list will be returned.

        Args:
            diff (int) - Difference.
        '''
        if diff in self._index_map:
            return self._index_map[diff][:]
        else:
            return []

    def central_ula_size(self):
        '''
        Gets the size of the central ULA in the difference coarray.
        '''
        mv = 0
        while mv in self._index_map:
            mv += 1
        return mv * 2 - 1
    
    def _build_map(self, design):
        # Maps difference -> indices in the vectorized difference matrix 
        index_map = {}
        diffs = compute_location_differences(design.element_indices).flatten()
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
