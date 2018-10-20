import unittest
import numpy as np
import numpy.testing as npt
from scipy.linalg import toeplitz
from doatools.model.arrays import UniformLinearArray, CoPrimeArray, \
                                  NestedArray, MinimumRedundancyLinearArray, \
                                  UniformCircularArray, UniformRectangularArray
from doatools.model.sources import FarField1DSourcePlacement

class Test1DArrays(unittest.TestCase):

    def setUp(self):
        self.wavelength = 1

    def test_ula(self):
        custom_name = 'TestULA'
        ula = UniformLinearArray(6, 2., custom_name)
        self.assertEqual(ula.d0, 2.)
        self.assertEqual(ula.size, 6)
        self.assertEqual(ula.ndim, 1)
        self.assertEqual(ula.name, custom_name)
        npt.assert_array_equal(
            ula.element_indices,
            np.array([0, 1, 2, 3, 4, 5]).reshape((-1, 1))
        )
        npt.assert_array_equal(
            ula.element_locations,
            np.array([0., 2., 4., 6., 8., 10.]).reshape((-1, 1))
        )

    def test_nested(self):
        nea = NestedArray(4, 3, 1.)
        self.assertEqual(nea.n1, 4)
        self.assertEqual(nea.n2, 3)
        self.assertEqual(nea.d0, 1.)
        self.assertEqual(nea.size, 7)
        self.assertEqual(nea.ndim, 1)
        npt.assert_array_equal(
            nea.element_indices,
            np.array([0, 1, 2, 3, 4, 9, 14]).reshape((-1, 1))
        )
        npt.assert_array_equal(
            nea.element_locations,
            np.array([0., 1., 2., 3., 4., 9., 14.]).reshape((-1, 1))
        )

    def test_coprime(self):
        # M
        cpa1 = CoPrimeArray(3, 5, 0.5, 'm')
        self.assertEqual(cpa1.coprime_pair, (3, 5))
        self.assertEqual(cpa1.mode, 'm')
        self.assertEqual(cpa1.d0, 0.5)
        self.assertEqual(cpa1.size, 7)
        self.assertEqual(cpa1.ndim, 1)
        npt.assert_array_equal(
            cpa1.element_indices,
            np.array([0, 3, 6, 9, 12, 5, 10]).reshape((-1, 1))
        )
        npt.assert_allclose(
            cpa1.element_locations,
            np.array([0., 1.5, 3., 4.5, 6., 2.5, 5.]).reshape((-1, 1))
        )
        # 2M
        cpa2 = CoPrimeArray(3, 5, 0.5, '2m')
        self.assertEqual(cpa2.coprime_pair, (3, 5))
        self.assertEqual(cpa2.mode, '2m')
        self.assertEqual(cpa2.d0, 0.5)
        self.assertEqual(cpa2.size, 10)
        self.assertEqual(cpa2.ndim, 1)
        npt.assert_array_equal(
            cpa2.element_indices,
            np.array([0, 3, 6, 9, 12, 5, 10, 15, 20, 25]).reshape((-1, 1))
        )
        npt.assert_allclose(
            cpa2.element_locations,
            np.array([0., 1.5, 3., 4.5, 6., 2.5, 5., 7.5, 10., 12.5]).reshape((-1, 1))
        )
    
    def test_mra(self):
        custom_name = 'TestMRA'
        d0 = self.wavelength / 2
        mra = MinimumRedundancyLinearArray(5, d0, custom_name)
        self.assertEqual(mra.d0, 0.5)
        self.assertEqual(mra.size, 5)
        self.assertEqual(mra.ndim, 1)
        npt.assert_array_equal(
            mra.element_indices,
            np.array([0, 1, 4, 7, 9]).reshape((-1, 1))
        )
        npt.assert_allclose(
            mra.element_locations,
            np.array([0.0, 0.5, 2.0, 3.5, 4.5]).reshape((-1, 1))
        )

    def test_steering_matrix_without_perturbations(self):
        cpa = CoPrimeArray(2, 3, self.wavelength / 2)
        sources = FarField1DSourcePlacement(np.linspace(-np.pi/3, np.pi/3, 3))
        A, DA = cpa.steering_matrix(sources, self.wavelength, True)
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
        npt.assert_allclose(A, A_expected, rtol=1e-6)
        npt.assert_allclose(DA, DA_expected, rtol=1e-6)

    def test_steering_matrix_with_perturbations(self):
        pass

class Test2DArrays(unittest.TestCase):

    def setUp(self):
        self.wavelength = 1

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
        npt.assert_allclose(uca.element_locations, locations_expected, atol=1e-8)

    def test_ura(self):
        custom_name = 'TestURA'
        m, n = 3, 4
        indices_expected = np.array([
            [0, 0], [0, 1], [0, 2], [0, 3],
            [1, 0], [1, 1], [1, 2], [1, 3],
            [2, 0], [2, 1], [2, 2], [2, 3]
        ])
        # Square cells
        d0 = self.wavelength / 2
        ura1 = UniformRectangularArray(m, n, d0, custom_name)
        self.assertEqual(ura1.size, m * n)
        self.assertEqual(ura1.ndim, 2)
        self.assertEqual(ura1.name, custom_name)
        self.assertEqual(ura1.shape, (m, n))
        npt.assert_array_equal(ura1.element_indices, indices_expected)
        npt.assert_allclose(ura1.element_locations, indices_expected * d0)
        # Rectangular cells
        d0 = (self.wavelength / 2, self.wavelength / 3)
        ura2 = UniformRectangularArray(m, n, d0, custom_name)
        self.assertEqual(ura2.size, m * n)
        self.assertEqual(ura2.ndim, 2)
        self.assertEqual(ura2.name, custom_name)
        self.assertEqual(ura2.shape, (m, n))
        npt.assert_array_equal(ura2.element_indices, indices_expected)
        npt.assert_allclose(
            ura2.element_locations,
            indices_expected * np.array(d0)
        )

class TestArrayPerturbations(unittest.TestCase):

    def setUp(self):
        self.wavelength = 1
    
    def test_array_perturbations(self):
        d0 = self.wavelength / 2
        ula = UniformLinearArray(5, d0)
        # No perturbations yet.
        self.assertFalse(ula.is_perturbed)
        for ptype in ['gain_errors', 'phase_errors', 'location_errors', 'mutual_coupling']:
            self.assertFalse(ula.has_perturbation(ptype))
        # Now we add perturbations.         
        gain_errors = np.random.uniform(-0.5, 0.5, (ula.size,))
        phase_errors = np.random.uniform(-np.pi, np.pi, (ula.size,))
        mutual_coupling = toeplitz([1.0, 0.4+0.2j, 0.0, 0.0, 0.0])
        perturbed_name = 'PerturbedULA'
        # Test for 1D, 2D, 3D location errors.
        for ndim in [1, 2, 3]:
            location_errors = np.random.uniform(-0.1 * d0, 0.1 * d0, (ula.size, ndim))
            perturbations = {
                'gain_errors': (gain_errors, True),
                'phase_errors': (phase_errors, True),
                'location_errors': (location_errors, False),
                'mutual_coupling': (mutual_coupling, True)
            }
            ula_perturbed = ula.get_perturbed_copy(perturbations, perturbed_name)
            self.assertEqual(ula_perturbed.name, perturbed_name)
            self.assertTrue(ula_perturbed.is_perturbed)
            for k, v in perturbations.items():
                self.assertEqual(ula_perturbed.has_perturbation(k), True)
                npt.assert_allclose(ula_perturbed.get_perturbation_params(k), v[0])
                self.assertEqual(ula_perturbed.is_perturbation_known(k), v[1])
            # Verify location error calulations.
            self.assertEqual(ula_perturbed.actual_ndim, ndim)
            npt.assert_allclose(
                ula_perturbed.actual_element_locations,
                np.pad(ula.element_locations, ((0, 0), (0, ndim - 1)), 'constant') + location_errors
            )
            # The `perturbation` property should return a dictionary of
            # perturbations.
            perturbations_actual = ula_perturbed.perturbations
            self.assertEqual(len(perturbations_actual), len(perturbations))
            for k, v in perturbations.items():
                npt.assert_allclose(perturbations_actual[k][0], v[0])
                self.assertEqual(perturbations_actual[k][1], v[1])
            # Perturbation-free copies should not have perturbations.
            self.assertFalse(ula_perturbed.get_perturbation_free_copy().is_perturbed)

    def test_perturbation_updates(self):
        d0 = self.wavelength / 2
        ula = UniformLinearArray(5, d0)
        gain_errors = np.random.uniform(-0.5, 0.5, (ula.size,))
        ula_perturbed = ula.get_perturbed_copy({
            'gain_errors': (gain_errors, True)
        })
        for known in [False, True, True, False]:
            phase_errors = np.random.uniform(-np.pi, np.pi, (ula.size, ))
            ula_perturbed = ula_perturbed.get_perturbed_copy({
                'phase_errors': (phase_errors, known)
            })
            # The gain errors should remain there.
            self.assertTrue(ula_perturbed.has_perturbation('gain_errors'))
            self.assertTrue(ula_perturbed.is_perturbation_known('gain_errors'))
            npt.assert_allclose(
                ula_perturbed.get_perturbation_params('gain_errors'),
                gain_errors
            )
            # The phase errors should be updated.
            self.assertTrue(ula_perturbed.has_perturbation('phase_errors'))
            self.assertEqual(ula_perturbed.is_perturbation_known('phase_errors'), known)
            npt.assert_allclose(
                ula_perturbed.get_perturbation_params('phase_errors'),
                phase_errors
            )


if __name__ == '__main__':
    unittest.main()
