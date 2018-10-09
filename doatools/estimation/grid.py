
from abc import ABC, abstractmethod
import numpy as np
from ..model.sources import FarField1DSourcePlacement, FarField2DSourcePlacement

def merge_intervals(intervals):
    '''
    Merges closed intervals.

    Args:
        intervals: A list of two-element tuples representing the intervals.
    
    Returns:
        merged: A list of two-element tuples representing the intervals after
            merging.
    '''
    if len(intervals) == 0:
        return []
    merged = []
    intervals = sorted(intervals)
    cur_start, cur_stop = intervals[0]
    for i in range(1, len(intervals)):
        if intervals[i][0] <= cur_start:
            cur_stop = max(cur_stop, intervals[i][1])
        else:
            merged.append((cur_start, cur_stop))
            cur_start, cur_stop = intervals[i]
    merged.append(cur_start, cur_stop)
    return merged

class SearchGrid(ABC):

    @property
    @abstractmethod
    def ndim(self):
        '''
        Retrieves the number of dimensions of this search grid.
        '''
        pass

    @property
    @abstractmethod
    def size(self):
        '''
        Retrieves the number of elements on this search grid.
        '''
        pass

    @property
    @abstractmethod
    def shape(self):
        '''
        Retrives the shape of this search grid.

        Returns:
            shape: A tuple representing the shape.
        '''
        pass

    @property
    @abstractmethod
    def source_placement(self):
        '''
        Retrieves the source placement based on this grid. For a
        multi-dimensional search grid with shape (d1, d2,..., dn), the
        returned SourcePlacement instance will contain d1 x d2 x ... x dn
        elements, which are ordered in such a way that the first dimension
        changes the slowest, the second dimension changes the second slowest,
        and so on. For instance, the elements in the following 2x3 grid

        (1, 3) (1, 4) (1, 5)
        (2, 3) (2, 4) (2, 5)

        will be ordered as

        (1, 3) (1, 4) (1, 5) (2, 3) (2, 4) (2, 5)

        Do not modify the returned SourcePlacement instance.
        '''
        pass

    @property
    @abstractmethod
    def unit(self):
        '''
        Retrieves the unit used.
        '''
        pass

    @property
    @abstractmethod
    def axes(self):
        '''
        Retrieves a tuple of 1D numpy vectors representing the axes.
        Do NOT modify.
        '''
        pass

    @abstractmethod
    def create_refined_grid_at(self, *indices, **kwargs):
        '''
        Creates a new search grid by subdividing a subset of the the current
        search grid into finer ones.

        Args:
            *indices: Sequences consist of the indices of the grid points.
                Refinement will be performed around these points.
            density: Controls number of new intervals between two adjacent
                points in the original grid.
            span: Controls how many adjacent intervals will be considered
                around the points specified by `indices` when performing the
                refinement.
        '''
        pass

class FarField1DSearchGrid(SearchGrid):

    def __init__(self, start=-np.pi/2, stop=np.pi/2, size=181, unit='rad'):
        '''
        Creates a search grid for 1D far-field source localization.

        When both `start` and `stop` are scalars, the resulting search grid
        consists only one uniform grid. When both `start` and `stop` are lists
        the resulting search grid is a combination of multiple uniform grids
        specified by `start[k]`, `stop[k]`, and `size[k]`. 

        Args:
            start: A scalar of the starting angle or a list of starting angles.
            stop: A scalar of the stopping angle or a list of stopping angles.
            size: Specifies the grid size. If both `start` and `stop` are
                lists, `size` must also be a list such that 'size[k]' specifies
                the number of grid points between `start[k]` and `stop[k]`.
            unit: Can be 'rad' (default), 'deg' or 'sin'.
        
        Returns:
            grid: A search grid for 1D far-field source localization.
        '''
        if np.isscalar(start):
            locations = np.linspace(start, stop, size)
        else:
            n_points = sum(size)
            locations = np.zeros((n_points, 1))
            offset = 0
            for k in range(len(start)):
                locations[offset:offset+size[k], 0] = np.linspace(start[k], stop[k], size[k])
        self._sources = FarField1DSourcePlacement(locations, unit)

    @property
    def ndim(self):
        return 1

    @property
    def size(self):
        return self._sources.size

    @property
    def shape(self):
        return self._sources.size,

    @property
    def source_placement(self):
        return self._sources

    @property
    def unit(self):
        return self._sources.unit

    @property
    def axes(self):
        return self._sources.locations,

    def create_refined_grid_at(self, *indices, **kwargs):
        # TODO: rethink the refining process
        '''
        Creates a new search grid by subdividing a subset of the the current
        search grid into finer ones.

        Suppose the original grid is [1, 2, 3, 4, 5], the `indices` is [0, 3],
        `density` is 4, and `span` is 1. the new grid will be

            4 new intervals        4 new intervals     4 new intervals
         |---------+---------|  |---------+---------|---------+---------|
        [1, 1.25, 1.5, 1.75, 2, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5]
         ^                                          ^
        index 0 ---span 1--->|  |<---span 1---   index 3   ---span 1--->|

        Args:
            indices: A sequence consists of the indices of the grid points.
                Refinement will be performed around these points.
            density: Controls number of new intervals between two adjacent
                points in the original grid.
            span: Controls how many adjacent intervals will be considered
                around the points specified by `indices` when performing the
                refinement.
        '''
        density = kwargs['density'] if 'density' in kwargs else 10
        span = kwargs['span'] if 'span' in kwargs else 1
        # Compute the intervals to be refined.
        intervals = []
        max_index = self._sources.size - 1
        for ind in indices[0]:
            l = max(0, ind - span)
            r = min(max_index, ind + span)
            intervals.append((l, r))
        intervals = merge_intervals(intervals)
        # Generate the refined grid.
        starts = [self._sources[i[0]] for i in intervals]
        stops = [self._sources[i[1]] for i in intervals]
        sizes = [(i[1] - i[0]) * density + 1 for i in intervals]
        return FarField1DSearchGrid(starts, stops, sizes)
