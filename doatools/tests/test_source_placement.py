import unittest
import numpy as np
import numpy.testing as npt
from doatools.model.sources import FarField1DSourcePlacement, \
                                   FarField2DSourcePlacement, \
                                   NearField2DSourcePlacement

class TestFarField1D(unittest.TestCase):

    def setUp(self):
        self.wavelength = 1

    def test_basics(self):
        n_sources = 10
        locations = np.linspace(-60, 60, n_sources)
        sources = FarField1DSourcePlacement(locations, 'deg')
        self.assertTrue(sources.is_far_field)
        self.assertEqual(sources.size, n_sources)
        self.assertEqual(sources.units, ('deg',))
        npt.assert_array_equal(sources.locations, locations)
        # Test indexing
        for i in range(sources.size):
            self.assertEqual(sources[i], locations[i])
        # Test slicing
        sources_subset = sources[:5]
        self.assertEqual(sources_subset.size, 5)
        self.assertEqual(sources_subset.units, ('deg',))
        npt.assert_array_equal(sources_subset.locations, locations[:5])
        # Slicing should return a copy
        locations_copy = locations.copy()
        sources_subset = sources[:5]
        sources_subset.locations[0] = -70
        npt.assert_array_equal(sources.locations, locations_copy)
        sources_subset = sources[[0, 1, 2]]
        sources_subset.locations[0] = -70
        npt.assert_array_equal(sources.locations, locations_copy)
    
    def test_unit_conversion(self):
        locations_rad = np.linspace(-np.pi/3, np.pi/4, 5)
        location_sets = {
            'rad': locations_rad,
            'deg': np.rad2deg(locations_rad),
            'sin': np.sin(locations_rad)
        }
        for from_unit, loc_from in location_sets.items():
            for to_unit, loc_to in location_sets.items():
                sources = FarField1DSourcePlacement(loc_from, unit=from_unit)
                sources_converted = sources.as_unit(to_unit)
                self.assertEqual(sources_converted.units, (to_unit,))
                npt.assert_allclose(sources_converted.locations, loc_to)

    def test_phase_delay(self):
        # 1D array
        sensor_locations_1 = np.array([0, 1, 2]).reshape((-1, 1))
        # Unit: 'rad' or 'deg'
        # Same locations in different units
        sources_rad = FarField1DSourcePlacement(np.linspace(-np.pi/3, np.pi/4, 5))
        sources_deg = FarField1DSourcePlacement(np.linspace(-60, 45, 5), 'deg')
        for sources in [sources_rad, sources_deg]: 
            # 1D array
            D1, DD1 = sources.phase_delay_matrix(sensor_locations_1, self.wavelength, True)
            D1_expected = np.array([
                [  0.000000,  0.000000,  0.000000, 0.000000, 0.000000],
                [ -5.441398, -3.490751, -0.820120, 2.019664, 4.442883],
                [-10.882796, -6.981501, -1.640241, 4.039327, 8.885766]
            ])
            DD1_expected = np.array([
                [0.000000,  0.000000,  0.000000,  0.000000, 0.000000],
                [3.141593,  5.224278,  6.229432,  5.949737, 4.442883],
                [6.283185, 10.448555, 12.458864, 11.899475, 8.885766]
            ])
            if sources.units[0] == 'deg':
                DD1_expected *= np.pi / 180.0
            npt.assert_allclose(D1, D1_expected, rtol=1e-6)
            npt.assert_allclose(DD1, DD1_expected, rtol=1e-6)
        # Unit: 'sin'
        sources_sin = FarField1DSourcePlacement(np.linspace(-0.4, 0.4, 5), 'sin')
        # 1D array
        D1, DD1 = sources_sin.phase_delay_matrix(sensor_locations_1, self.wavelength, True)
        D1_expected = np.array([
            [ 0.000000,  0.000000, 0.000000, 0.000000, 0.000000],
            [-2.513274, -1.256637, 0.000000, 1.256637, 2.513274],
            [-5.026548, -2.513274, 0.000000, 2.513274, 5.026548]
        ])
        DD1_expected = np.array([
            [ 0.000000,  0.000000,  0.000000,  0.000000,  0.000000],
            [ 6.283185,  6.283185,  6.283185,  6.283185,  6.283185],
            [12.566371, 12.566371, 12.566371, 12.566371, 12.566371]
        ])
        npt.assert_allclose(D1, D1_expected, rtol=1e-6)
        npt.assert_allclose(DD1, DD1_expected, rtol=1e-6)

    def test_spherical_coords(self):
        m = 5
        k = 6
        source_locations = np.linspace(-60, 60, k)
        sensor_locations = np.linspace(0, m - 1, m).reshape((-1, 1))
        sources = FarField1DSourcePlacement(source_locations, 'deg')
        r, az, el = sources.calc_spherical_coords(sensor_locations)
        r_expected = np.full((m, k), np.inf)
        az_expected = np.tile(np.pi/2 - np.deg2rad(source_locations), (m, 1))
        el_expected = np.zeros((m, k))
        npt.assert_allclose(r, r_expected)
        npt.assert_allclose(az, az_expected)
        npt.assert_allclose(el, el_expected)

class TestFarField2D(unittest.TestCase):

    def setUp(self):
        self.wavelength = 1

    def test_basics(self):
        locations = np.array([
            [0, 30], [0, 50], [135, 35], [150, 70], [-120, 60]
        ], dtype=np.float_)
        sources = FarField2DSourcePlacement(locations, 'deg')
        self.assertTrue(sources.is_far_field)
        self.assertEqual(sources.size, len(locations))
        self.assertEqual(sources.units, ('deg', 'deg'))
        npt.assert_array_equal(sources.locations, locations)
        # Test indexing
        for i in range(sources.size):
            npt.assert_array_equal(sources[i], locations[i])
        # Test slicing
        sources_subset = sources[-2:]
        self.assertEqual(sources_subset.size, 2)
        self.assertEqual(sources_subset.units, ('deg', 'deg'))
        npt.assert_array_equal(sources_subset.locations, locations[-2:])

    def test_unit_conversion(self):
        locations = np.array([[-45.0, 20.0], [15.0, 30.0], [75.0, 80.0]])
        sources_deg = FarField2DSourcePlacement(locations, unit='deg')
        sources_rad = sources_deg.as_unit('rad')
        npt.assert_allclose(sources_rad.locations, np.deg2rad(locations))
        self.assertEqual(sources_rad.units, ('rad', 'rad'))
        npt.assert_allclose(sources_rad.as_unit('deg').locations, locations)

    def test_phase_delay(self):
        sources = FarField2DSourcePlacement(np.array([
            [ 27.0, 84.0],
            [-92.0, 68.0],
            [ 37.0, 17.0]
        ]), unit='deg')
        # 1D array
        sensor_locations_1 = np.array([0.0, 0.5, 1.0, 1.5]).reshape((-1, 1))
        D1_expected = np.array([
            [        0.0,          0.0,       0.0],
            [2.925939e-1, -4.107187e-2,  2.399357],
            [5.851879e-1, -8.214374e-2,  4.798713],
            [8.777818e-1, -1.232156e-1,  7.198070]
        ])
        D1 = sources.phase_delay_matrix(sensor_locations_1, self.wavelength)
        npt.assert_allclose(D1, D1_expected, rtol=1e-6, atol=1e-8)
        # 2D array
        sensor_locations_2 = np.array([
            [0.0, 0.0],
            [0.0, 0.5],
            [0.5, 0.0],
            [0.5, 0.5]
        ])
        D2 = sources.phase_delay_matrix(sensor_locations_2, self.wavelength)
        D2_expected = np.array([
            [        0.0,          0.0,      0.0],
            [1.490841e-1, -1.176144e+0, 1.808045],
            [2.925939e-1, -4.107187e-2, 2.399357],
            [4.416780e-1, -1.217216e+0, 4.207402]
        ])
        npt.assert_allclose(D2, D2_expected, rtol=1e-6, atol=1e-8)
        # 3D array
        sensor_locations_3 = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.5, 0.5, 0.5]
        ])
        D3 = sources.phase_delay_matrix(sensor_locations_3, self.wavelength)
        D3_expected = np.array([
            [0.000000e+0,  0.000000e+0, 0.000000e+0],
            [2.925939e-1, -4.107187e-2, 2.399357e+0],
            [1.490841e-1, -1.176144e+0, 1.808045e+0],
            [4.416780e-1, -1.217216e+0, 4.207402e+0],
            [3.124383e+0,  2.912834e+0, 9.185128e-1],
            [3.416977e+0,  2.871762e+0, 3.317869e+0],
            [3.273467e+0,  1.736690e+0, 2.726558e+0],
            [3.566061e+0,  1.695618e+0, 5.125914e+0]
        ])
        npt.assert_allclose(D3, D3_expected, rtol=1e-6, atol=1e-8)

    def test_spherical_coords(self):
        source_locations = np.array([
            [-np.pi/2, np.pi/9],
            [np.pi/3, np.pi/6],
            [np.pi/4, -np.pi/3]
        ])
        ref_locations = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0]
        ])
        m, k = ref_locations.shape[0], source_locations.shape[0]
        sources = FarField2DSourcePlacement(source_locations)
        r, az, el = sources.calc_spherical_coords(ref_locations)
        npt.assert_array_equal(r, np.full((m, k), np.inf))
        npt.assert_allclose(az, np.tile(source_locations[:, 0], (m, 1)))
        npt.assert_allclose(el, np.tile(source_locations[:, 1], (m, 1)))

class TestNearField2D(unittest.TestCase):

    def setUp(self):
        self.wavelength = 1.0
        
    def test_basics(self):
        source_locations = np.array([
            [100.0, 20.0],
            [-30.0, 40.0],
            [ 70.0, 24.0]]
        )
        sources = NearField2DSourcePlacement(source_locations)
        self.assertFalse(sources.is_far_field)
        self.assertEqual(sources.size, source_locations.shape[0])
        self.assertEqual(sources.units, ('m', 'm'))
        npt.assert_allclose(sources.locations, source_locations)

    def test_phase_delay(self):
        source_locations = np.array([
            [30.0, 40.0],
            [50.0, -120.0],
            [80.0, 60.0]
        ])
        sources = NearField2DSourcePlacement(source_locations)
        # This is a linear array.
        # We pad zeros to test alignment of different number of dimensions.
        sensor_locations = np.array([0.0, 0.5, 1.0, 1.5])
        # The phase delay matrix should remain the same.
        D_expected = np.array([
            [0.000000, 0.000000, 0.000000],
            [1.874842, 1.203149, 2.510435],
            [3.729213, 2.395958, 5.015147],
            [5.562744, 3.578379, 7.514067]
        ])
        for d in [1, 2, 3]:
            cur_sensor_locations = np.zeros((sensor_locations.size, d))
            cur_sensor_locations[:, 0] = sensor_locations
            D = sources.phase_delay_matrix(cur_sensor_locations, self.wavelength)
            npt.assert_allclose(D, D_expected, rtol=1e-6, atol=1e-8)

    def test_spherical_coords(self):
        source_locations = np.array([
            [10.0, 10.0],
            [20.0, 20.0],
            [-10.0, 10.0]
        ])
        sources = NearField2DSourcePlacement(source_locations)
        # The reference locations are on the x-axis.
        # We pad zeros to test alignment of different number of dimensions.
        # The spherical coordinates should remain the same.
        ref_locations = np.array([-10.0, 0.0, 10.0, 20.0])
        r_expected = np.array([
            [22.360680, 36.055513, 10.000000],
            [14.142136, 28.284271, 14.142136],
            [10.000000, 22.360680, 22.360680],
            [14.142136, 20.000000, 31.622777]
        ])
        az_expected = np.array([
            [0.463648, 0.588003, 1.570796],
            [0.785398, 0.785398, 2.356194],
            [1.570796, 1.107149, 2.677945],
            [2.356194, 1.570796, 2.819842]
        ])
        el_expected = np.zeros((ref_locations.size, sources.size))
        for d in [1, 2, 3]:
            cur_ref_locations = np.zeros((ref_locations.size, d))
            cur_ref_locations[:, 0] = ref_locations
            r, az, el = sources.calc_spherical_coords(cur_ref_locations)
            npt.assert_allclose(r, r_expected, rtol=1e-6, atol=1e-8)
            npt.assert_allclose(az, az_expected, rtol=1e-6, atol=1e-8)
            npt.assert_allclose(el, el_expected, rtol=1e-6, atol=1e-8)

if __name__ == '__main__':
    unittest.main()

