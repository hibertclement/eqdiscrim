import unittest
import numpy as np
from cat_io.sihex_io import read_sihex_xls
from cat_io.renass_io import read_renass, read_stations_fr
from datetime import datetime, timedelta
from dateutil import tz

utc = tz.gettz('UTC')


def suite():
    suite = unittest.TestSuite()
    #suite.addTest(IoReadTests('test_read_sihex_xls'))
    #suite.addTest(IoReadTests('test_read_renass'))
    suite.addTest(IoReadTests('test_read_stations_fr'))
    return suite


class IoReadTests(unittest.TestCase):

    def test_read_sihex_xls(self):

        X, y, names = read_sihex_xls()

        nev, natt = X.shape
        self.assertEqual(nev, 50568)
        self.assertEqual(natt, 7)
        self.assertEqual(y[0], 'ke')
        self.assertEqual(names[1], 'OTIME')

        test_otime = datetime(1962, 3, 7, 1, 25, 39, np.int(0.7 * 1e6), utc)
        self.assertEqual(test_otime - X[3, 1], timedelta(0))

    def test_read_renass(self):
        
        stations, names = read_renass()
        nsta, natt = stations.shape
        self.assertEqual(nsta, 81)
        self.assertEqual(natt, 4)
        self.assertEqual(stations[1, 1], 43.0858)

    def test_read_stations_fr(self):
        
        stations, names = read_stations_fr()
        nsta, natt = stations.shape
        self.assertEqual(nsta, 306)
        self.assertEqual(natt, 6)
        self.assertEqual(stations[1, 1], 47.268)

        test_time = datetime(1962, 1, 22, 6, 50, 55, 0, utc)
        self.assertEqual(test_time - stations[1, 4], timedelta(0))
        test_time = datetime(2009, 12, 31, 11, 10, 22, 0, utc)
        self.assertEqual(test_time - stations[1, 5], timedelta(0))

if __name__ == '__main__':

    unittest.TextTestRunner(verbosity=2).run(suite())
