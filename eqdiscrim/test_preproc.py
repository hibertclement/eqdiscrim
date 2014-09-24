import unittest
import numpy as np
from preproc import latlon_to_xy, xy_to_latlon
from preproc import dist_to_n_closest_stations

def suite():

    suite = unittest.TestSuite()
    suite.addTest(GeoPreprocTests('test_latlon_conversion'))
    suite.addTest(GeoPreprocTests('test_dist_to_nth_station'))
    return suite


class GeoPreprocTests(unittest.TestCase):
    
    def test_latlon_conversion(self):

        nval = 100
        X = np.random.rand(nval,5)
        names = list(["ID", "LAT", "LON", "BLA1", "BLA2"])
        # make the second colum French latitudes
        ilat = 1
        minlat = 42.0
        maxlat = 50.0
        X[:, ilat] = X[:, ilat] * (maxlat - minlat) + minlat
        # make the third colum French longitudes
        ilon = 2
        minlon = -6.0
        maxlon = 10.0
        X[:, ilon] = X[:, ilon] * (maxlon - minlon) + minlon

        X_xy, names_xy = latlon_to_xy(X, names, ilat, ilon)

        self.assertEqual(names_xy[ilat], 'Y')
        self.assertEqual(names_xy[ilon], 'X')

        new_X, new_names = xy_to_latlon(X_xy, names_xy, ilon, ilat)

        np.testing.assert_array_almost_equal(X, new_X)

    def test_dist_to_nth_station(self):

        nval = 100
        nst = 10
        X_xy = np.random.rand(nval, 2)
        S_xy = np.random.rand(nst, 2)

        d = dist_to_n_closest_stations(X_xy, S_xy, 3)

        ival = np.random.randint(nval)

        self.assertTrue(d[ival, 0] < d[ival, 1])
        self.assertTrue(d[ival, 1] < d[ival, 2])

if __name__ == '__main__':

    unittest.TextTestRunner(verbosity=2).run(suite())
