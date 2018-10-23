
from abc import ABC, abstractmethod
import numpy as np
from ..model.sources import FarField1DSourcePlacement, FarField2DSourcePlacement, NearField2DSourcePlacement
from ..utils.math import cartesian

class SearchGrid(ABC):
    """Base class for all search grids. Provides standard implementation.
    
    Args:
        axes: A tuple of 1D ndarrays representing the axes of this search
            grid. The source locations on this search grid will be generated
            from these axes.
        axis_names: A tuple of strings denoting the names of the axes.
        units (str): A tuple of strings representing the unit used for each
            axis.
    """

    def __init__(self, axes, axis_names, units):
        if not isinstance(axes, tuple):
            raise ValueError('axes should be a tuple.')
        if not isinstance(axis_names, tuple):
            raise ValueError('axis_names should be a tuple.')
        if not isinstance(units, tuple):
            raise ValueError('units should be a tuple.')
        self._axes = axes
        self._shape = tuple(len(ax) for ax in axes)
        self._axis_names = axis_names
        self._units = units
        self._sources = None

    @property
    def ndim(self):
        """Retrieves the number of dimensions of this search grid."""
        return len(self._axes)

    @property
    def size(self):
        """Retrieves the number of elements on this search grid."""
        return np.prod(self.shape)

    @property
    def shape(self):
        """Retrieves the shape of this search grid.

        Returns:
            A tuple of integers representing the shape.
        """
        return self._shape

    @property
    def source_placement(self):
        r"""Retrieves the source placement based on this grid.
        
        For a multi-dimensional search grid with shape
        :math:`(d_1, d_2, \ldots, d_n)`, the returned
        :class:`~doatools.model.sources.SourcePlacement` instance will contain
        :math:`d_1 \times d_2 \times \cdots \times d_n` elements, which are
        ordered in such a way that the first dimension changes the slowest, the
        second dimension changes the second slowest, and so on. For instance,
        the elements in the following 2x3 grid

        ::

            (1, 1) (1, 2) (1, 3)
            (2, 1) (2, 2) (2, 3)

        will be ordered as

        ::

            (1, 1) (1, 2) (1, 3) (2, 1) (2, 2) (2, 3)

        Do not **modify**.
        """
        if self._sources is None:
            self._sources = self._create_source_placement()
        return self._sources

    @property
    def units(self):
        """Retrieves a tuple of strings representing the unit used for each axis."""
        return self._units

    @property
    def axes(self):
        """Retrieves a tuple of 1D numpy vectors representing the axes.
        
        The sources locations can be recovered with the Cartesian product over
        ``(axes[0], axes[1], ...)``.

        Do **not** modify.
        """
        return self._axes
    
    @property
    def axis_names(self):
        """Retrieves a tuple of strings representing the axis names."""
        return self._axis_names
    
    @abstractmethod
    def _create_source_placement(self):
        """Creates the source placement instance for this grid.
        
        Notes:
            Implement this method in a subclass to create the source placement
            instance of the desired type.
        """
        raise NotImplementedError()

    def create_refined_axes_at(self, coord, density, span):
        """Creates a new set of axes by subdividing the grids around the input
        coordinate into finer grids. These new axes can then be used to create
        refined grids.

        For instance, suppose that the original grid is a 2D grid with the axes:

        ==========  ===================
        Axis name   Axis data
        ==========  =================== 
        Azimuth     [0, 10, 20, 30, 40]
        Elevation   [0, 20, 40]
        ==========  ===================
        
        Suppose that ``coord`` is (3, 1), ``density`` is 4, and ``span`` is 1.
        Then the following set of axes will be created:

        Refined axes around the coordinate (3, 1) (or azimuth = 30,
        elevation = 20):

        =========  ==============================================
        Axis name  Axis data
        =========  ============================================== 
        Azimuth    [20, 22.5, 25.0, 27.5, 30, 32.5, 35, 37.5, 40]
        Elevation  [0, 5, 10, 15, 20, 25, 30, 35, 40]
        =========  ==============================================

        Args:
            coord: A tuple of integers representing a single coordinate within
                this grid.
            density (int): Controls number of new intervals between two adjacent
                points in the original grid.
            span (int): Controls how many adjacent intervals in the original
                grid will be considered around the point specified by ``coord``
                when performing the refinement.
        
        Returns:
            A tuple of ndarrays representing the refined axes.
        """
        if density < 1:
            raise ValueError('Density must be greater than or equal to 1.')
        if span < 1:
            raise ValueError('Span must be greater than or equal to 1.')
        if len(coord) != self.ndim:
            raise ValueError(
                'Incorrect number of coordinate elements. Expecting {0}. Got {1}.'
                .format(self.ndim, len(coord))
            )
        axes = []
        for j in range(self.ndim):
            # Lower bound and upper bound indices.
            i_lb = max(0, coord[j] - span)
            i_ub = min(self._shape[j] - 1, coord[j] + span)
            # Convert to actual values.
            lb = self._axes[j][i_lb]
            ub = self._axes[j][i_ub]
            axes.append(np.linspace(lb, ub, (i_ub - i_lb) * density + 1))
        return tuple(axes)

    def create_refined_grids_at(self, *coords, **kwargs):
        """Creates multiple new search grids around the given coordinates.

        Args:
            *coords: A sequence of list-like objects representing the
                coordinates of the grid points around which the refinement will
                be performed. The length of ``coords`` should be equal to the
                number of dimensions of this grid. The list-like objects in
                ``coords`` should share the same length. ``coords[j][i]``
                denotes the j-th element of the i-th coordinate.
            density (int): Controls number of new intervals between two adjacent
                points in the original grid.
            span (int): Controls how many adjacent intervals in the original
                grid will be considered around the point specified by ``coords``
                when performing the refinement.
        
        Returns:
            A list of refined grids.
        """
        return [self.create_refined_grid_at(coord, **kwargs) for coord in zip(*coords)]
    
    @abstractmethod
    def create_refined_grid_at(self, coord, density, span):
        """Creates a finer search grid around the given coordinate.

        Args:
            coord: A tuple of integers representing a single coordinate within
                this grid.
            density (int): Controls number of new intervals between two adjacent
                points in the original grid.
            span (int): Controls how many adjacent intervals in the original
                grid will be considered around the point specified by ``coord``
                when performing the refinement.
        
        Returns:
            A refined grid.
        """
        raise NotImplementedError()

class FarField1DSearchGrid(SearchGrid):
    r"""Creates a search grid for 1D far-field source localization.

    When both ``start`` and ``stop`` are scalars, the resulting search grid
    consists only one uniform grid. When both ``start`` and ``stop`` are lists
    the resulting search grid is a combination of multiple uniform grids
    specified by ``start[k]``, ``stop[k]``, and ``size[k]``. 

    Args:
        start (float): A scalar of the starting angle or a list of starting
            angles. If not specified, the following default values will be used
            depending on ``unit``:

            * ``'rad'``: :math:`-\pi/2`
            * ``'deg'``: -90,
            * ``'sin'``: -1

        stop (float): A scalar of the stopping angle or a list of stopping
            angles. This angle is not included in the grid. If not specified,
            the following default values will be used depending on ``unit``:
            
            * ``'rad'``: :math:`\pi/2`
            * ``'deg'``: 90
            * ``'sin'``: 1

        size (int): Specifies the grid size. If both ``start`` and ``stop`` are
            lists, `size` must also be a list such that 'size[k]' specifies the
            number of grid points between ``start[k]`` and ``stop[k]``. Default
            value is 180.
        
        unit (str): Can be ``'rad'`` (default), ``'deg'`` or ``'sin'``.
        
        axes: A tuple of 1D ndarrays representing the axes of the search grid.
            If specified, ``start``, ``stop``, and ``size`` will be ignored
            and the search grid will be generated based only on ``axes`` and
            ``units``. Default value is ``None``.
    
    Returns:
        A search grid for 1D far-field source localization.
    """

    def __init__(self, start=None, stop=None, size=180, unit='rad', axes=None):
        if axes is not None:
            super().__init__(axes, ('DOA',), (unit,))
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
            super().__init__((locations,), ('DOA',), (unit,))
    
    def _create_source_placement(self):
        return FarField1DSourcePlacement(self._axes[0], self._units[0])

    def create_refined_grid_at(self, coord, density=10, span=1):
        """Creates a finer search grid for 1D far-field sources.
        
        Args:
            coord: A tuple of integers representing a single coordinate within
                this grid.
            density (int): Controls number of new intervals between two adjacent
                points in the original grid. Default value is 10.
            span (int): Controls how many adjacent intervals in the original
                grid will be considered around the point specified by ``coord``
                when performing the refinement. Default value is 1.
        
        Returns:
            A refined 1D far-field search grid.
        """
        axes = self.create_refined_axes_at(coord, density, span)
        return FarField1DSearchGrid(unit=self._units[0], axes=axes)

class FarField2DSearchGrid(SearchGrid):
    r"""Creates a search grid for 2D far-field source localization.

    The first dimension corresponds to the azimuth angle, and the second
    dimension corresponds to the elevation angle.

    Args:
        start: A two-element list-like object containing the starting azimuth
            and elevation angles. If not specified, the following default values
            will be used depending on ``unit``:

            * ``'rad'``: (:math:`-\pi`, 0)
            * ``'deg'``: (-180, 0)

        stop: A two-element list-like object containing the stopping azimuth and
            elevation angles. These two angles are not included in the search
            grid. If not specified, the following default values will be used
            depending on ``unit``:

            * ``'rad'``: (:math:`\pi`, :math:`\pi/2`)
            * ``'deg'``: (180, 90)

        size: A scalar or a two-element list-like object specifying the size of
            the search grid. If ``size`` is a scalar, a ``size`` by ``size``
            grid will be created. If ``size`` is a two-element list-like object,
            a ``size[0]`` by ``size[1]`` grid will be created. Default value is
            ``(360, 90)``.

        unit (str): Can be ``'rad'`` (default) or ``'deg'``.

        axes: A tuple of 1D ndarrays representing the axes of the search grid.
            If specified, ``start``, ``stop``, and ``size`` will be ignored and
            the search grid will be generated based only on ``axes`` and
            ``units``. Default value is ``None``.
    
    Returns:
        A search grid for 2D far-field source localization.
    """

    def __init__(self, start=None, stop=None, size=(360, 90), unit='rad',
                 axes=None):
        axis_names = ('Azimuth', 'Elevation')
        if axes is not None:
            super().__init__(axes, axis_names, (unit, unit))
        else:
            default_ranges = {
                'rad': ((-np.pi, 0.0), (np.pi, np.pi/2)),
                'deg': ((-180.0, 0.0), (180.0, 90.0))
            }
            if start is None:
                start = default_ranges[unit][0]
            if stop is None:
                stop = default_ranges[unit][1]
            if np.isscalar(size):
                size = (size, size)
            az = np.linspace(start[0], stop[0], size[0], False)
            el = np.linspace(start[1], stop[1], size[1], False)
            super().__init__((az, el), axis_names, (unit, unit))

    def _create_source_placement(self):
        return FarField2DSourcePlacement(cartesian(*self._axes), self._units[0])

    def create_refined_grid_at(self, coord, density=10, span=1):
        """Creates a finer search grid for 2D far-field sources.
        
        Args:
            coord: A tuple of integers representing a single coordinate within
                this grid.
            density (int): Controls number of new intervals between two adjacent
                points in the original grid. Default value is 10.
            span (int): Controls how many adjacent intervals in the original
                grid will be considered around the point specified by ``coord``
                when performing the refinement. Default value is 1.
        
        Returns:
            A refined 2D far-field search grid.
        """
        axes = self.create_refined_axes_at(coord, density, span)
        return FarField2DSearchGrid(unit=self._units[0], axes=axes)

class NearField2DSearchGrid(SearchGrid):
    """Creates a search grid for 2D near-field source localization.

    The first dimension corresponds to the x coordinate, and the second
    dimension corresponds to the y coordinate.

    Args:
        start: A two-element list-like object containing the starting x and y
            coordinates.

        stop: A two-element list-like object containing the stopping x and y
            coordinates. These two coordinates are not included in the search
            grid.

        size: A scalar or a two-element list-like object specifying the size of
            the search grid. If ``size`` is a scalar, a ``size`` by ``size``
            grid will be created. If ``size`` is a two-element list-like object,
            a ``size[0]`` by ``size[1]`` grid will be created. Default value is
            ``(360, 90)``.

        axes: A tuple of 1D ndarrays representing the axes of the search grid.
            If specified, ``start``, ``stop``, and ``size`` will be ignored and
            the search grid will be generated based only on ``axes`` and
            ``units``. Default value is ``None``.
    
    Returns:
        A search grid for 2D near-field source localization.
    """

    def __init__(self, start=None, stop=None, size=None, axes=None):
        axis_names = ('x', 'y')
        if axes is not None:
            super().__init__(axes, axis_names, ('m', 'm'))
        else:
            if np.isscalar(size):
                size = (size, size)
            x = np.linspace(start[0], stop[0], size[0], False)
            y = np.linspace(start[1], stop[1], size[1], False)
            super().__init__((x, y), axis_names, ('m', 'm'))

    def _create_source_placement(self):
        return NearField2DSourcePlacement(cartesian(*self._axes))

    def create_refined_grids_at(self, coord, density=10, span=1):
        """Creates a finer search grid for 2D near-field sources.
        
        Args:
            coord: A tuple of integers representing a single coordinate within
                this grid.
            density (int): Controls number of new intervals between two adjacent
                points in the original grid. Default value is 10.
            span (int): Controls how many adjacent intervals in the original
                grid will be considered around the point specified by ``coord``
                when performing the refinement. Default value is 1.
        
        Returns:
            A refined 2D near-field search grid.
        """
        axes = self.create_refined_axes_at(coord, density, span)
        return NearField2DSearchGrid(axes=axes)
