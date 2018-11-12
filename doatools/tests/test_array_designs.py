import unittest
import numpy as np
import numpy.testing as npt
from scipy.linalg import toeplitz
from doatools.model.array_elements import CustomNonisotropicSensor
from doatools.model.perturbations import LocationErrors, GainErrors, \
                                         PhaseErrors, MutualCoupling
from doatools.model.arrays import GridBasedArrayDesign
from doatools.model.arrays import UniformLinearArray, CoPrimeArray, \
                                  NestedArray, MinimumRedundancyLinearArray, \
                                  UniformCircularArray, UniformRectangularArray
from doatools.model.sources import FarField1DSourcePlacement

class Test1DArrayDesigns(unittest.TestCase):

    def setUp(self):
        self.wavelength = 1

    def test_ula(self):
        d0 = 2.
        custom_name = 'TestULA'
        ula = UniformLinearArray(6, d0, custom_name)
        self.assertEqual(ula.size, 6)
        self.assertEqual(ula.ndim, 1)
        self.assertEqual(ula.name, custom_name)
        npt.assert_allclose(ula.d0, np.array([d0]))
        npt.assert_allclose(ula.bases, np.array([[d0]]))
        npt.assert_array_equal(
            ula.element_indices,
            np.array([0, 1, 2, 3, 4, 5]).reshape((-1, 1))
        )
        npt.assert_array_equal(
            ula.element_locations,
            np.array([0., 2., 4., 6., 8., 10.]).reshape((-1, 1))
        )

    def test_nested(self):
        d0 = 1.
        nea = NestedArray(4, 3, d0)
        self.assertEqual(nea.n1, 4)
        self.assertEqual(nea.n2, 3)
        self.assertEqual(nea.size, 7)
        self.assertEqual(nea.ndim, 1)
        npt.assert_allclose(nea.d0, np.array([d0]))
        npt.assert_allclose(nea.bases, np.array([[d0]]))
        npt.assert_array_equal(
            nea.element_indices,
            np.array([0, 1, 2, 3, 4, 9, 14]).reshape((-1, 1))
        )
        npt.assert_array_equal(
            nea.element_locations,
            np.array([0., 1., 2., 3., 4., 9., 14.]).reshape((-1, 1))
        )

    def test_coprime(self):
        d0 = self.wavelength / 2
        # M
        cpa1 = CoPrimeArray(3, 5, d0, 'm')
        self.assertEqual(cpa1.coprime_pair, (3, 5))
        self.assertEqual(cpa1.mode, 'm')
        self.assertEqual(cpa1.size, 7)
        self.assertEqual(cpa1.ndim, 1)
        npt.assert_array_equal(cpa1.d0, np.array([d0]))
        npt.assert_array_equal(cpa1.bases, np.array([[d0]]))
        npt.assert_array_equal(
            cpa1.element_indices,
            np.array([0, 3, 6, 9, 12, 5, 10]).reshape((-1, 1))
        )
        npt.assert_allclose(
            cpa1.element_locations,
            np.array([0., 1.5, 3., 4.5, 6., 2.5, 5.]).reshape((-1, 1))
        )
        # 2M
        cpa2 = CoPrimeArray(3, 5, d0, '2m')
        self.assertEqual(cpa2.coprime_pair, (3, 5))
        self.assertEqual(cpa2.mode, '2m')
        self.assertEqual(cpa2.size, 10)
        self.assertEqual(cpa2.ndim, 1)
        npt.assert_array_equal(cpa2.d0, np.array([d0]))
        npt.assert_array_equal(cpa2.bases, np.array([[d0]]))
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
        self.assertEqual(mra.size, 5)
        self.assertEqual(mra.ndim, 1)
        npt.assert_array_equal(mra.d0, np.array([d0]))
        npt.assert_array_equal(mra.bases, np.array([[d0]]))
        npt.assert_array_equal(
            mra.element_indices,
            np.array([0, 1, 4, 7, 9]).reshape((-1, 1))
        )
        npt.assert_allclose(
            mra.element_locations,
            np.array([0.0, 0.5, 2.0, 3.5, 4.5]).reshape((-1, 1))
        )

class Test2DArrayDesigns(unittest.TestCase):

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
        npt.assert_allclose(ura1.d0, np.array([d0, d0]))
        npt.assert_allclose(ura1.bases, np.eye(2) * d0)
        npt.assert_array_equal(ura1.element_indices, indices_expected)
        npt.assert_allclose(ura1.element_locations, indices_expected * d0)
        # Rectangular cells
        d0 = (self.wavelength / 2, self.wavelength / 3)
        ura2 = UniformRectangularArray(m, n, d0, custom_name)
        self.assertEqual(ura2.size, m * n)
        self.assertEqual(ura2.ndim, 2)
        self.assertEqual(ura2.name, custom_name)
        self.assertEqual(ura2.shape, (m, n))
        npt.assert_allclose(ura2.d0, np.array(d0))
        npt.assert_allclose(ura2.bases, np.diag(d0))
        npt.assert_array_equal(ura2.element_indices, indices_expected)
        npt.assert_allclose(
            ura2.element_locations,
            indices_expected * np.array(d0)
        )

class TestGeneralGridBasedArrays(unittest.TestCase):

    def test_3d(self):
        bases = np.array([
            [0., 0.5, 0.],
            [1.,  0., 0.],
            [0.,  0., 2.]
        ])
        indices = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 1, 1]
        ])
        locations_expected = indices @ bases
        array = GridBasedArrayDesign(indices, bases=bases)
        self.assertEqual(array.size, indices.shape[0])
        self.assertEqual(array.ndim, bases.shape[1])
        npt.assert_allclose(array.d0, np.linalg.norm(bases, ord=2, axis=1))
        npt.assert_allclose(array.element_indices, indices)
        npt.assert_allclose(array.bases, bases)
        npt.assert_allclose(array.element_locations, locations_expected)

class TestSteeringMatrix(unittest.TestCase):

    def setUp(self):
        self.wavelength = 1.0

    def test_without_perturbations(self):
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

    def test_with_perturbations(self):
        pass

    def test_custom_nonisotropic_1d(self):
        # Sine response for azimuth angles (cosine for broadside angles)
        f_sr = lambda r, az, el, pol: np.sin(az)
        element = CustomNonisotropicSensor(f_sr)
        ula = UniformLinearArray(5, self.wavelength / 2, element=element)
        sources = FarField1DSourcePlacement(np.linspace(-np.pi/3, np.pi/4, 3))
        A_expected = np.array([
            [ 5.000000e-1+0.000000e+0j,  9.914449e-1+0.000000e+0j,  7.071068e-1+0.000000e+0j],
            [-4.563621e-1-2.042881e-1j,  9.092510e-1-3.952538e-1j, -4.282945e-1+5.626401e-1j],
            [ 3.330655e-1+3.729174e-1j,  6.762975e-1-7.249721e-1j, -1.882710e-1-6.815820e-1j],
            [-1.516317e-1-4.764534e-1j,  3.312097e-1-9.344854e-1j,  6.563659e-1+2.630282e-1j],
            [-5.626959e-2+4.968236e-1j, -6.879475e-2-9.890552e-1j, -6.068505e-1+3.629497e-1j]
        ])
        A = ula.steering_matrix(sources, self.wavelength)
        npt.assert_allclose(A, A_expected, rtol=1e-6)
    
    def test_custom_vector_sensor_1d(self):
        # Each sensor has three outputs with different gains.
        gains = [1.0, 0.5, 0.1]
        output_size = len(gains)
        def f_sr(r, az, el, pol):
            # Sine response.
            res = np.sin(az)
            return np.stack([res * g for g in gains])
        element = CustomNonisotropicSensor(f_sr, output_size=output_size)
        ula = UniformLinearArray(4, self.wavelength / 2, element=element)
        sources = FarField1DSourcePlacement(np.linspace(-np.pi/3, np.pi/4, 3))
        A_expected = np.array([
            [ 5.000000e-1+0.000000e+0j, 9.914449e-1+0.000000e+0j,  7.071068e-1+0.000000e+0j],
            [-4.563621e-1-2.042881e-1j, 9.092510e-1-3.952538e-1j, -4.282945e-1+5.626401e-1j],
            [ 3.330655e-1+3.729174e-1j, 6.762975e-1-7.249721e-1j, -1.882710e-1-6.815820e-1j],
            [-1.516317e-1-4.764534e-1j, 3.312097e-1-9.344854e-1j,  6.563659e-1+2.630282e-1j],
            [ 2.500000e-1+0.000000e+0j, 4.957224e-1+0.000000e+0j,  3.535534e-1+0.000000e+0j],
            [-2.281810e-1-1.021441e-1j, 4.546255e-1-1.976269e-1j, -2.141472e-1+2.813200e-1j],
            [ 1.665327e-1+1.864587e-1j, 3.381488e-1-3.624861e-1j, -9.413548e-2-3.407910e-1j],
            [-7.581586e-2-2.382267e-1j, 1.656049e-1-4.672427e-1j,  3.281829e-1+1.315141e-1j],
            [ 5.000000e-2+0.000000e+0j, 9.914449e-2+0.000000e+0j,  7.071068e-2+0.000000e+0j],
            [-4.563621e-2-2.042881e-2j, 9.092510e-2-3.952538e-2j, -4.282945e-2+5.626401e-2j],
            [ 3.330655e-2+3.729174e-2j, 6.762975e-2-7.249721e-2j, -1.882710e-2-6.815820e-2j],
            [-1.516317e-2-4.764534e-2j, 3.312097e-2-9.344854e-2j,  6.563659e-2+2.630282e-2j]
        ])
        A = ula.steering_matrix(sources, self.wavelength)
        npt.assert_allclose(A, A_expected, rtol=1e-6)

class TestArrayPerturbations(unittest.TestCase):

    def setUp(self):
        self.wavelength = 1
    
    def test_array_perturbations(self):
        d0 = self.wavelength / 2
        ula = UniformLinearArray(5, d0)
        ptype2str = {
            LocationErrors: 'location_errors',
            GainErrors: 'gain_errors',
            PhaseErrors: 'phase_errors',
            MutualCoupling: 'mutual_coupling'
        }
        str2ptype = {v: k for k, v in ptype2str.items()}
        # No perturbations yet.
        self.assertFalse(ula.is_perturbed)
        for ptype in ptype2str.keys():
            self.assertFalse(ula.has_perturbation(ptype))
        # Now we add perturbations.         
        gain_errors = np.random.uniform(-0.5, 0.5, (ula.size,))
        phase_errors = np.random.uniform(-np.pi, np.pi, (ula.size,))
        mutual_coupling = toeplitz([1.0, 0.4+0.2j, 0.0, 0.0, 0.0])
        perturbed_name = 'PerturbedULA'
        # Test for 1D, 2D, 3D location errors.
        for ndim in [1, 2, 3]:
            location_errors = np.random.uniform(-0.1 * d0, 0.1 * d0, (ula.size, ndim))
            perturb_defs = {
                'gain_errors': (gain_errors, True),
                'phase_errors': (phase_errors, True),
                'location_errors': (location_errors, False),
                'mutual_coupling': (mutual_coupling, True)
            }
            ula_perturbed = ula.get_perturbed_copy(perturb_defs, perturbed_name)
            self.assertEqual(ula_perturbed.name, perturbed_name)
            self.assertTrue(ula_perturbed.is_perturbed)
            for k, v in perturb_defs.items():
                cur_ptype = str2ptype[k]
                self.assertEqual(ula_perturbed.has_perturbation(cur_ptype), True)
                npt.assert_allclose(ula_perturbed.get_perturbation_params(cur_ptype), v[0])
                self.assertEqual(ula_perturbed.is_perturbation_known(cur_ptype), v[1])
            # Verify location error calculations.
            self.assertEqual(ula_perturbed.actual_ndim, ndim)
            npt.assert_allclose(
                ula_perturbed.actual_element_locations,
                np.pad(ula.element_locations, ((0, 0), (0, ndim - 1)), 'constant') + location_errors
            )
            # The `perturbation` property should return a list of perturbations.
            perturbs_actual = ula_perturbed.perturbations
            self.assertEqual(len(perturbs_actual), len(perturb_defs))
            for perturb in perturbs_actual:
                params_expected, known_expected = perturb_defs[ptype2str[perturb.__class__]]
                npt.assert_allclose(perturb.params, params_expected)
                self.assertEqual(perturb.is_known, known_expected)
            # Perturbation-free copies should not have perturbations.
            self.assertFalse(ula_perturbed.get_perturbation_free_copy().is_perturbed)

    def test_perturbation_updates(self):
        d0 = self.wavelength / 2
        ula = UniformLinearArray(5, d0)
        gain_errors = np.random.uniform(-0.5, 0.5, (ula.size,))
        ula_perturbed = ula.get_perturbed_copy([
            GainErrors(gain_errors, True)
        ])
        for known in [False, True, True, False]:
            phase_errors = np.random.uniform(-np.pi, np.pi, (ula.size, ))
            ula_perturbed = ula_perturbed.get_perturbed_copy([
                PhaseErrors(phase_errors, known)
            ])
            # The gain errors should remain there.
            self.assertTrue(ula_perturbed.has_perturbation(GainErrors))
            self.assertTrue(ula_perturbed.is_perturbation_known(GainErrors))
            npt.assert_allclose(
                ula_perturbed.get_perturbation_params(GainErrors),
                gain_errors
            )
            # The phase errors should be updated.
            self.assertTrue(ula_perturbed.has_perturbation(PhaseErrors))
            self.assertEqual(ula_perturbed.is_perturbation_known(PhaseErrors), known)
            npt.assert_allclose(
                ula_perturbed.get_perturbation_params(PhaseErrors),
                phase_errors
            )

if __name__ == '__main__':
    unittest.main()
