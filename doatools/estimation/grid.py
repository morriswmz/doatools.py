
from abc import ABC, abstractmethod
import numpy as np
from ..model.sources import FarField1DSourcePlacement, FarField2DSourcePlacement, NearField2DSourcePlacement
from ..utils.math import cartesian

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
    '''Base class for all search grids. Provides standard implementation.'''

    def __init__(self, axes, axis_names, units, spfactory):
        '''Creates a search grid.

        Args:
            axes: A tuple of 1D ndarrays representing the axes of this search
                grid. The source locations on this search grid will be generated
                from these axes.
            axis_names: A tuple of strings denoting the names of the axes.
            units: A tuple of strings representing the unit used for each axis.
            spfactory: A callable object that accepts `axes` and `units` as
                two parameters and returns a SourcePlacement instance for this
                search grid.
        '''
        self._axes = axes
        self._shape = tuple(len(ax) for ax in axes)
        self._axis_names = axis_names
        self._units = units
        self._spfactory = spfactory
        self._sources = None

    @property
    def ndim(self):
        '''Retrieves the number of dimensions of this search grid.'''
        return len(self._axes)

    @property
    def size(self):
        '''Retrieves the number of elements on this search grid.'''
        return np.prod(self.shape)

    @property
    def shape(self):
        '''Retrives the shape of this search grid.

        Returns:
            shape: A tuple representing the shape.
        '''
        return self._shape

    @property
    def source_placement(self):
        '''Retrieves the source placement based on this grid.
        
        For a multi-dimensional search grid with shape (d1, d2,..., dn), the
        returned SourcePlacement instance will contain d1 x d2 x ... x dn
        elements, which are ordered in such a way that the first dimension
        changes the slowest, the second dimension changes the second slowest,
        and so on. For instance, the elements in the following 2x3 grid

        (1, 1) (1, 2) (1, 3)
        (2, 1) (2, 2) (2, 3)

        will be ordered as

        (1, 1) (1, 2) (1, 3) (2, 1) (2, 2) (2, 3)

        Do not modify the returned SourcePlacement instance.
        '''
        if self._sources is None:
            self._sources = self._spfactory(self._axes, self._units)
        return self._sources

    @property
    def units(self):
        '''Retrieves a tuple of strings representing the unit used for each axis.'''
        return self._units

    @property
    def axes(self):
        '''Retrieves a tuple of 1D numpy vectors representing the axes such
        that: source_locations = cartesian(axes[0], axes[1], ...).

        Do NOT modify.
        '''
        return self._axes
    
    @property
    def axis_names(self):
        '''Retrieves a tuple of strings representing the axis names.'''
        return self._axis_names

    @abstractmethod
    def create_refined_grid_at(self, *loc, **kwargs):
        '''
        Creates a new search grid by subdividing a subset of the the current
        search grid into finer ones.

        Args:
            *loc: A sequence representing the coordinate of the grid point
                around which the refinement will be performed.
            density: Controls number of new intervals between two adjacent
                points in the original grid.
            span: Controls how many adjacent intervals will be considered
                around the point specified by `loc` when performing the
                refinement.
        '''
        raise NotImplementedError()

class FarField1DSearchGrid(SearchGrid):

    def __init__(self, start=None, stop=None, size=180, unit='rad', axes=None):
        '''Creates a search grid for 1D far-field source localization.

        When both `start` and `stop` are scalars, the resulting search grid
        consists only one uniform grid. When both `start` and `stop` are lists
        the resulting search grid is a combination of multiple uniform grids
        specified by `start[k]`, `stop[k]`, and `size[k]`. 

        Args:
            start: A scalar of the starting angle or a list of starting angles.
                If not specified, the following default values will be used
                depending on `unit`:
                    'rad': -pi/2, 'deg': -90, 'sin': -1
            stop: A scalar of the stopping angle or a list of stopping angles.
                This angle is not included in the grid. If not specified, the
                following default values will be used depending on `unit`:
                    'rad': pi/2, 'deg': 90, 'sin': 1
            size: Specifies the grid size. If both `start` and `stop` are
                lists, `size` must also be a list such that 'size[k]' specifies
                the number of grid points between `start[k]` and `stop[k]`.
                Default value is 180.
            unit: Can be 'rad' (default), 'deg' or 'sin'.
            axes: A tuple of 1D ndarrays representing the axes of the search
                grid. If specified, `start`, `stop`, and `size` will be ignored
                and the search grid will be generated based only on `axes` and
                `units`. Default value is None.
        
        Returns:
            grid: A search grid for 1D far-field source localization.
        '''
        spfactory = lambda axes, units: FarField1DSourcePlacement(axes[0], units[0])
        if axes is not None:
            super().__init__(axes, ('DOA',), (unit,), spfactory)
        else:
            default_ranges = {
                'rad': (-np.pi / 2, np.pi / 2),
                'deg': (-90.0, 90.0),
                'sin': (-1.0, 1.0)
            }
            if start is None:
                start = default_ranges[unit][0]
            if stop is None:
                stop = default_ranges[unit][1]
            if np.isscalar(start):
                locations = np.linspace(start, stop, size, endpoint=False)
            else:
                n_points = sum(size)
                locations = np.zeros((n_points, 1))
                offset = 0
                for k in range(len(start)):
                    locations[offset:offset+size[k], 0] = np.linspace(start[k], stop[k], size[k], endpoint=False)
            super().__init__((locations,), ('DOA',), (unit,), spfactory)

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

class FarField2DSearchGrid(SearchGrid):

    def __init__(self, start=None, stop=None, size=(360, 90), unit='rad',
                 axes=None):
        '''Creates a search grid for 2D far-field source localization.

        The first dimension corresponds to the azimuth angle, and the second
        dimension corresponds to the elevation angle.

        Args:
            start: A two-element list-like object containing the starting
                azimuth and elevation angles. If not specified, the following
                default values will be used depending on `unit`:
                    'rad': (0, 0), 'deg': (0, 0)
            stop: A two-element list-like object containing the stopping
                azimuth and elevation angles. These two angles are not included
                in the search grid. If not specified, the following default
                values will be used depending on `unit`:
                    'rad': (2*pi, pi/2),
            size: A scalar or a two-element list-like object specifying the
                size of the search grid. If `size` is a scalar, a `size`x`size`
                grid will be created. If `size` is a two-element list-like
                object, a `size[0]`x`size[1]` grid will be created. Default
                value is `(360, 90)`. 
            unit: Can be 'rad' (default), 'deg'.
            axes: A tuple of 1D ndarrays representing the axes of the search
                grid. If specified, `start`, `stop`, and `size` will be ignored
                and the search grid will be generated based only on `axes` and
                `units`. Default value is None.
        
        Returns:
            grid: A search grid for 2D far-field source localization.
        '''
        spfactory = lambda axes, units: FarField2DSourcePlacement(cartesian(*axes), units[0])
        axis_names = ('Azimuth', 'Elevation')
        if axes is not None:
            super().__init__(axes, axis_names, (unit, unit), spfactory)
        else:
            default_ranges = {
                'rad': ((0.0, 0.0), (np.pi*2, np.pi/2)),
                'deg': ((0.0, 0.0), (360.0, 90.0))
            }
            if start is None:
                start = default_ranges[unit][0]
            if stop is None:
                stop = default_ranges[unit][1]
            if np.isscalar(size):
                size = (size, size)
            az = np.linspace(start[0], stop[0], size[0], False)
            el = np.linspace(start[1], stop[1], size[1], False)
            super().__init__((az, el), axis_names, (unit, unit), spfactory)

    def create_refined_grid_at(self, *indices, **kwargs):
        raise NotImplementedError()

class NearField2DSearchGrid(SearchGrid):

    def __init__(self, start, stop, size, axes=None):
        '''Creates a search grid for 2D near-field source localization.

        The first dimension corresponds to the x coordinate, and the second
        dimension corresponds to the y coordinate.

        Args:
            start: A two-element list-like object containing the starting
                x and y coordinates.
            stop: A two-element list-like object containing the stopping
                x and y coordinates.. These two coordinates are not included
                in the search grid.
            size: A scalar or a two-element list-like object specifying the
                size of the search grid. If `size` is a scalar, a `size`x`size`
                grid will be created. If `size` is a two-element list-like
                object, a `size[0]`x`size[1]` grid will be created.
            axes: A tuple of 1D ndarrays representing the axes of the search
                grid. If specified, `start`, `stop`, and `size` will be ignored
                and the search grid will be generated based only on `axes` and
                `units`. Default value is None.
        
        Returns:
            grid: A search grid for 2D near-field source localization.
        '''
        spfactory = lambda axes, units: NearField2DSourcePlacement(cartesian(*axes))
        axis_names = ('x', 'y')
        if axes is not None:
            super().__init__(axes, axis_names, ('m', 'm'), spfactory)
        else:
            if np.isscalar(size):
                size = (size, size)
            x = np.linspace(start[0], stop[0], size[0], False)
            y = np.linspace(start[1], stop[1], size[1], False)
            super().__init__((x, y), axis_names, ('m', 'm'), spfactory)

    def create_refined_grid_at(self, *indices, **kwargs):
        raise NotImplementedError()
