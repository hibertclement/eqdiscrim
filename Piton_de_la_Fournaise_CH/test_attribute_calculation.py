import unittest, os
import numpy as np
import pandas as pd
from obspy.core import UTCDateTime

import data_io as io

datadir = 'test_data'

def suite():
    suite = unittest.TestSuite()
    suite.addTest(IOTests('test_read_catalog'))
    suite.addTest(IOTests('test_read_and_cut_events'))

    return suite


class IOTests(unittest.TestCase):

    def test_read_catalog(self):

        fname = os.path.join(datadir, 'catalog_test_version.xls')
        cat = io.read_catalog(fname)

        nr, nc = cat.shape
        self.assertEqual(nr, 16)
        self.assertAlmostEqual(cat[0, nc-1],3.96782204729149)

    def test_read_and_cut_events(self):

        cat_fname = os.path.join(datadir, 'catalog_test_version.xls')
        cat = io.read_catalog(cat_fname)
        data_regex = os.path.join(datadir, 'YA*')

        starttime = UTCDateTime('20091027T072417.600')
        onset = starttime + 0.68
        endtime = starttime + 4.0

        st_list = io.read_and_cut_events(cat, data_regex)

        st = st_list[0]
        tr = st[0]

        self.assertAlmostEqual(tr.stats.sampling_rate, 100.0)
        self.assertAlmostEqual(tr.stats.starttime - onset, 0.0, 2)
        self.assertAlmostEqual(tr.stats.endtime - endtime, 0.0, 2)

        


if __name__ == '__main__':

    unittest.TextTestRunner(verbosity=2).run(suite())
