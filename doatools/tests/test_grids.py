import unittest
import numpy as np
import numpy.testing as npt
from doatools.estimation.grid import FarField1DSearchGrid, FarField2DSearchGrid
from doatools.utils.math import cartesian

class Test1DGrids(unittest.TestCase):

    def check_1d_grid_with_axes(self, grid, axis_expected, axis_names_expected,
                                units_expected):
        """Helper method to check whether the grid matches the expectations."""
        self.assertEqual(grid.ndim, 1)
        self.assertEqual(grid.size, axis_expected.size)
        self.assertEqual(grid.shape, (axis_expected.size,))
        self.assertEqual(grid.units, units_expected)
        self.assertEqual(grid.axis_names, axis_names_expected)
        npt.assert_allclose(grid.axes[0], axis_expected)
        npt.assert_allclose(grid.source_placement.locations, axis_expected)

    def test_far_field_1d(self):
        default_sets = {
            'rad': (-np.pi/2, np.pi/2, 180),
            'deg': (-90, 90, 180),
            'sin': (-1, 1, 180)
        }
        for k, v in default_sets.items():
            grid = FarField1DSearchGrid(unit=k)
            ax_expected = np.linspace(v[0], v[1], v[2], endpoint=False)
            self.check_1d_grid_with_axes(grid, ax_expected, ('DOA',), (k,))

    def test_far_field_1d_custom(self):
        custom_sets = {
            'rad': (-np.pi/3, np.pi/5, 60),
            'deg': (-30, 50, 80),
            'sin': (-0.5, 0.5, 100)
        }
        for k, v in custom_sets.items():
            grid = FarField1DSearchGrid(start=v[0], stop=v[1], size=v[2], unit=k)
            ax_expected = np.linspace(v[0], v[1], v[2], endpoint=False)
            self.check_1d_grid_with_axes(grid, ax_expected, ('DOA',), (k,))

    def test_far_field_1d_from_axes(self):
        ax = np.linspace(-30, 40, 60)
        grid = FarField1DSearchGrid(axes=(ax,), unit='deg')
        self.check_1d_grid_with_axes(grid, ax, ('DOA',), ('deg',))

    def test_far_field_1d_refinement(self):
        # [0, 10, 20, 30, 40, 50]
        grid = FarField1DSearchGrid(start=0, stop=60, size=6, unit='deg')
        # Single
        refined_single = grid.create_refined_grid_at((1,), density=5, span=2)
        axis_expected = np.linspace(0, 30, 16)
        self.check_1d_grid_with_axes(refined_single, axis_expected, grid.axis_names, grid.units)
        # Multiple
        refined_multi = grid.create_refined_grids_at([0, 3, 5], density=4, span=1)
        axes_expected = [
            np.linspace(0, 10, 5),
            np.linspace(20, 40, 9),
            np.linspace(40, 50, 5)
        ]
        for i, g in enumerate(refined_multi):
            self.check_1d_grid_with_axes(
                g, axes_expected[i], grid.axis_names, grid.units
            )

class Test2DGrids(unittest.TestCase):
    
    def check_2d_grid_with_axes(self, grid, axes_expected, axis_names_expected,
                                units_expected):
        ax1_expected, ax2_expected = axes_expected
        self.assertEqual(grid.ndim, 2)
        self.assertEqual(grid.size, ax1_expected.size * ax2_expected.size)
        self.assertEqual(grid.shape, (ax1_expected.size, ax2_expected.size))
        self.assertEqual(grid.units, units_expected)
        self.assertEqual(grid.axis_names, axis_names_expected)
        npt.assert_allclose(grid.axes[0], ax1_expected)
        npt.assert_allclose(grid.axes[1], ax2_expected)
        npt.assert_allclose(
            grid.source_placement.locations,
            cartesian(ax1_expected, ax2_expected)
        )

    def test_far_field_2d(self):
        default_sets = {
            'rad': ((-np.pi, 0), (np.pi, np.pi/2), (360, 90)),
            'deg': ((-180, 0), (180, 90), (360, 90))
        }
        for k, v in default_sets.items():
            grid = FarField2DSearchGrid(unit=k)
            az_expected = np.linspace(v[0][0], v[1][0], v[2][0], endpoint=False)
            el_expected = np.linspace(v[0][1], v[1][1], v[2][1], endpoint=False)
            self.check_2d_grid_with_axes(
                grid, (az_expected, el_expected),
                ('Azimuth', 'Elevation'), (k, k)
            )

    def test_far_field_2d_refinement(self):
        # az: [0, 10, 20, 30, 40, 50]
        # el: [0, 20, 40]
        grid = FarField2DSearchGrid(start=(0, 0), stop=(60, 60), size=(6, 3), unit='deg')
        # Single
        refined_single = grid.create_refined_grid_at((0, 1), density=4, span=2)
        axes_expected = (np.linspace(0, 20, 9), np.linspace(0, 40, 9))
        self.check_2d_grid_with_axes(refined_single, axes_expected, grid.axis_names, grid.units)
        # Multiple
        refined_multi = grid.create_refined_grids_at([0, 3, 5], [0, 1, 1], density=4, span=1)
        axes_expected = [
            # (0, 0): (0, 0)
            (np.linspace(0, 10, 5), np.linspace(0, 20, 5)),
            # (3, 1): (30, 20)
            (np.linspace(20, 40, 9), np.linspace(0, 40, 9)),
            # (5, 1): (50, 20)
            (np.linspace(40, 50, 5), np.linspace(0, 40, 9))
        ]
        for i, g in enumerate(refined_multi):
            self.check_2d_grid_with_axes(
                g, axes_expected[i], grid.axis_names, grid.units
            )

if __name__ == '__main__':
    unittest.main()
