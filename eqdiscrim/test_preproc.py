import unittest
import numpy as np
from datetime import datetime
from preproc import latlon_to_xy, xy_to_latlon
from preproc import dist_to_n_closest_stations
from preproc import n_stations_per_year
from preproc import GutenbergRichter
from preproc import toHourFraction, toWeekdayFraction


def suite():

    suite = unittest.TestSuite()
    suite.addTest(GeoPreprocTests('test_latlon_conversion'))
    suite.addTest(GeoPreprocTests('test_dist_to_nth_station'))
    suite.addTest(GeoPreprocTests('test_dist_to_nth_station_pathological'))
    suite.addTest(GeoPreprocTests('test_dist_to_nth_station_timing'))
    suite.addTest(GeoPreprocTests('test_n_stations_per_year'))
    suite.addTest(GeoPreprocTests('test_GutenbergRichter'))
    suite.addTest(GeoPreprocTests('test_timeFractions'))
    return suite


class GeoPreprocTests(unittest.TestCase):

    def test_latlon_conversion(self):

        nval = 100
        X = np.random.rand(nval, 5)
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

    def test_n_stations_per_year(self):

        nst = 500

        # make some random year vectors
        rand_start_years = np.random.randint(1960, 2015, (nst, 1))
        rand_duration_years = np.random.randint(0, 100, (nst, 1))
        rand_end_years = rand_start_years + rand_duration_years

        # stack
        years_start_end = np.hstack((rand_start_years.reshape(nst, 1),
                                     rand_end_years.reshape(nst, 1)))

        # transform to datetime
        S = np.empty((nst, 2), dtype=object)

        for i in xrange(nst):
            S[i, 0] = datetime(years_start_end[i, 0], 1, 1, 0, 0, 0)
            S[i, 1] = datetime(years_start_end[i, 1], 1, 1, 0, 0, 0)

        # get count for a fixed year
        year_count = n_stations_per_year(S, 1970, 1975)
        n_bef_1970 = np.sum(rand_start_years <= 1970)

        self.assertTrue(n_bef_1970 >= year_count[0, 1])

    def test_GutenbergRichter(self):

        # make a uniform distribution as a function of magnitude
        N0 = 10     # number of events in largest magnitude
        mags = np.arange(0, 5, 0.1)     # magnitude windows
        nmags = len(mags)-1
        Ntmp = (np.arange(nmags)+1)*N0
        N = Ntmp[::-1]
        log10N = np.log10(N)
        magnitudes = np.empty(nmags*N0, dtype=np.float)
        for i in xrange(nmags):
            magnitudes[i*N0:(i+1)*N0] = mags[i]+0.01

        # call GR function
        log10N_GR, mags = GutenbergRichter(magnitudes, 0, 5, 0.1)

        np.testing.assert_array_almost_equal(log10N, log10N_GR, 5)

    def test_timeFractions(self):

        d = datetime(2000, 6, 1, 3, 15, 0, 0)
        hf_expected = 3.25
        hf = toHourFraction(d)
        self.assertAlmostEqual(hf_expected, hf)

        d = datetime(2000, 6, 1, 6, 0, 0, 0)
        wf_expected = 4.25
        wf = toWeekdayFraction(d)
        self.assertAlmostEqual(wf_expected, wf)
        


if __name__ == '__main__':

    unittest.TextTestRunner(verbosity=2).run(suite())
