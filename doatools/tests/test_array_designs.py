import unittest
import numpy as np
import numpy.testing as npt
from doatools.model.arrays import UniformLinearArray, CoPrimeArray, NestedArray, UniformCircularArray
from doatools.model.sources import FarField1DSourcePlacement

class Test1DArrays(unittest.TestCase):

    def setUp(self):
        self.wavelength = 1

    def test_ula(self):
        custom_name = 'TestULA'
        design = UniformLinearArray(6, 2., custom_name)
        self.assertEqual(design.d0, 2.)
        self.assertEqual(design.size, 6)
        self.assertEqual(design.ndim, 1)
        self.assertEqual(design.name, custom_name)
        npt.assert_array_equal(
            design.element_indices,
            np.array([0, 1, 2, 3, 4, 5]).reshape((-1, 1))
        )
        npt.assert_array_equal(
            design.element_locations,
            np.array([0., 2., 4., 6., 8., 10.]).reshape((-1, 1))
        )

    def test_nested(self):
        design = NestedArray(4, 3, 1.)
        self.assertEqual(design.d0, 1.)
        self.assertEqual(design.size, 7)
        self.assertEqual(design.ndim, 1)
        npt.assert_array_equal(
            design.element_indices,
            np.array([0, 1, 2, 3, 4, 9, 14]).reshape((-1, 1))
        )
        npt.assert_array_equal(
            design.element_locations,
            np.array([0., 1., 2., 3., 4., 9., 14.]).reshape((-1, 1))
        )

    def test_coprime(self):
        # M
        design1 = CoPrimeArray(3, 5, 0.5, 'm')
        self.assertEqual(design1.d0, 0.5)
        self.assertEqual(design1.size, 7)
        self.assertEqual(design1.ndim, 1)
        npt.assert_array_equal(
            design1.element_indices,
            np.array([0, 3, 6, 9, 12, 5, 10]).reshape((-1, 1))
        )
        npt.assert_array_almost_equal(
            design1.element_locations,
            np.array([0., 1.5, 3., 4.5, 6., 2.5, 5.]).reshape((-1, 1))
        )
        # 2M
        design2 = CoPrimeArray(3, 5, 0.5, '2m')
        self.assertEqual(design2.d0, 0.5)
        self.assertEqual(design2.size, 10)
        self.assertEqual(design2.ndim, 1)
        npt.assert_array_equal(
            design2.element_indices,
            np.array([0, 3, 6, 9, 12, 5, 10, 15, 20, 25]).reshape((-1, 1))
        )
        npt.assert_array_almost_equal(
            design2.element_locations,
            np.array([0., 1.5, 3., 4.5, 6., 2.5, 5., 7.5, 10., 12.5]).reshape((-1, 1))
        )
    
    def test_uca(self):
        custom_name = 'TestUCA'
        n = 4
        r = 2.0
        uca = UniformCircularArray(n, r, custom_name)
        self.assertEqual(uca.size, n)
        self.assertEqual(uca.ndim, 2)
        self.assertEqual(uca.name, custom_name)
        self.assertEqual(uca.radius, r)
        locations_expected = np.array([
            [2., 0.], [0., 2.], [-2., 0.], [0., -2.]
        ])
        npt.assert_array_almost_equal(uca.element_locations, locations_expected)

    def test_steering_matrix_without_perturbations(self):
        design = CoPrimeArray(2, 3, self.wavelength / 2)
        sources = FarField1DSourcePlacement(np.linspace(-np.pi/3, np.pi/3, 3))
        A, DA = design.steering_matrix(sources, self.wavelength, True)
        A_expected = np.array([
            [ 1.000000+0.000000j, 1.000000+0.000000j,  1.000000+0.000000j],
            [ 0.666131+0.745835j, 1.000000+0.000000j,  0.666131-0.745835j],
            [-0.112539+0.993647j, 1.000000+0.000000j, -0.112539-0.993647j],
            [-0.303263-0.952907j, 1.000000+0.000000j, -0.303263+0.952907j],
            [-0.816063+0.577964j, 1.000000+0.000000j, -0.816063-0.577964j],
            [ 0.798227+0.602356j, 1.000000+0.000000j,  0.798227-0.602356j]
        ])
        DA_expected = np.array([
            [ 0.000000+ 0.000000j, 0.000000+ 0.000000j,  0.000000+ 0.000000j],
            [-2.343109+ 2.092712j, 0.000000+ 6.283185j,  2.343109+ 2.092712j],
            [-6.243270- 0.707105j, 0.000000+12.566371j,  6.243270- 0.707105j],
            [ 4.490467- 1.429095j, 0.000000+ 9.424778j, -4.490467- 1.429095j],
            [-5.447178- 7.691209j, 0.000000+18.849556j,  5.447178- 7.691209j],
            [-8.515612+11.284673j, 0.000000+28.274334j,  8.515612+11.284673j]
        ])
        npt.assert_array_almost_equal(A, A_expected)
        npt.assert_array_almost_equal(DA, DA_expected)

    def test_steering_matrix_with_perturbations(self):
        pass



if __name__ == '__main__':
    unittest.main()
