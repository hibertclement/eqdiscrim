import unittest
import numpy as np
from datetime import datetime
from preproc import latlon_to_xy, xy_to_latlon
from preproc import dist_to_n_closest_stations

def suite():

    suite = unittest.TestSuite()
    suite.addTest(GeoPreprocTests('test_latlon_conversion'))
    suite.addTest(GeoPreprocTests('test_dist_to_nth_station'))
    suite.addTest(GeoPreprocTests('test_dist_to_nth_station_pathological'))
    suite.addTest(GeoPreprocTests('test_dist_to_nth_station_timing'))
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

        self.assertTrue(d[ival, 0] <= d[ival, 1])
        self.assertTrue(d[ival, 1] <= d[ival, 2])

    def test_dist_to_nth_station_pathological(self):

        nval = 100
        nst = 2
        X_xy = np.random.rand(nval, 2)
        S_xy = np.random.rand(nst, 2)

        d = dist_to_n_closest_stations(X_xy, S_xy, 3)

        ival = np.random.randint(nval)

        self.assertTrue(d[ival, 0] <= d[ival, 1])
        self.assertTrue(d[ival, 1] <= d[ival, 2])

    def test_dist_to_nth_station_timing(self):

        nval = 100
        nst = 10
        X_xy = np.random.rand(nval, 2)
        S_xy = np.random.rand(nst, 2)

        # make some random day vectors
        rand_eday = np.random.randint(1, 30, nval)
        rand_sdays = np.random.randint(1, 30, (nst, 2))

        # set up emtpy time vectors
        etimes = np.empty(nval, dtype=object)
        stimes = np.empty((nst, 2), dtype=object)

        # populate time vectors
        for i in xrange(nval):
            etimes[i] = datetime(2014, 6, rand_eday[i], 0, 0, 0, 0)
        for i in xrange(nst):
            # the station end times can be before the start times to simulate
            # pathological cases
            stimes[i, 0] = datetime(2014, 6, rand_sdays[i, 0], 0, 0, 0, 0)
            stimes[i, 1] = datetime(2014, 6, rand_sdays[i, 1], 0, 0, 0, 0)

        X = np.hstack((X_xy, etimes.reshape(nval, 1)))
        S = np.hstack((S_xy, stimes))

        d = dist_to_n_closest_stations(X, S, 3, timing=True)

        # get the number of actual events that can be used
        nval, nd = d.shape
        ival = np.random.randint(nval)

        self.assertTrue(d[ival, 0] <= d[ival, 1])
        self.assertTrue(d[ival, 1] <= d[ival, 2])


if __name__ == '__main__':

    unittest.TextTestRunner(verbosity=2).run(suite())
