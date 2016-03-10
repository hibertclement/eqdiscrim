import unittest, os
import numpy as np
import pandas as pd
from obspy.core import UTCDateTime, read
from obspy.signal.filter import bandpass
from scipy.io import loadmat
from scipy.signal import butter, filtfilt

import data_io as io
import attributes as at

datadir = 'test_data'

def suite():
    suite = unittest.TestSuite()
    suite.addTest(IOTests('test_read_catalog'))
    suite.addTest(IOTests('test_read_and_cut_events'))
    suite.addTest(AttributeTests('test_filter_coefs'))
    suite.addTest(AttributeTests('test_filtering_data'))
    suite.addTest(AttributeTests('test_attribute_values_0_11'))
    suite.addTest(AttributeTests('test_attribute_values_12_16'))
    suite.addTest(AttributeTests('test_attribute_values_17_21'))
    suite.addTest(AttributeTests('test_attribute_values_22'))

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


class AttributeTests(unittest.TestCase):

    def test_filter_coefs(self):

        mat_fname = os.path.join(datadir, 'filter_coef.mat')
        mat = loadmat(mat_fname)

        Fb_mat = mat['Fa']
        Fa_mat = mat['Fb']

        NyF = 100.0 / 2.0
        FFI = 0.1
        FFE = 1.0

        Fb_py, Fa_py = butter(2, [FFI/NyF, FFE/NyF], 'bandpass')

        for i in xrange(len(Fb_py)):
            self.assertAlmostEqual(Fb_mat[0, i], Fb_py[i])
        for i in xrange(len(Fa_py)):
            self.assertAlmostEqual(Fa_mat[0, i], Fa_py[i])

    def test_filtering_data(self):

        mat_fname = os.path.join(datadir, 'filtered_data.mat')
        mat = loadmat(mat_fname)
        filt_data_mat = mat['filt_data']

        NyF = 100.0 / 2.0
        FFI = 0.1
        FFE = 1.0
        Fb, Fa = butter(2, [FFI/NyF, FFE/NyF], btype='bandpass')

        st = read(os.path.join(datadir, 'events', 'event_01*HHZ*SAC'))
        tr = st[0].copy()
        filt_data_py = filtfilt(Fb, Fa, st[0].data)
        tr.filter('bandpass', freqmin=FFI, freqmax=FFE, corners=2, zerophase=True)

        self.assertAlmostEqual(len(filt_data_mat), len(tr.data))
        self.assertAlmostEqual(max(filt_data_mat)[0], max(tr.data))
        #self.assertAlmostEqual(len(filt_data_mat), len(filt_data_py))
        #self.assertAlmostEqual(max(filt_data_mat)[0], max(filt_data_py))
        
    def setUp(self):

       # load attributes for first event
        mat_fname = os.path.join(datadir, 'event01.mat')
        mat = loadmat(mat_fname)
        self.mat_array = mat['att']

        # get attributes for first event en python
        st = read(os.path.join(datadir, 'events', 'event_01*SAC'))

        self.py_array = at.calculate_all_attributes(st)


    def test_attribute_values_0_11(self):

         # start assertions
        self.assertSequenceEqual(self.mat_array.shape, self.py_array.shape)

        for i in xrange(12):
            self.assertAlmostEqual(self.mat_array[0, i], self.py_array[0, i], 5)

    def test_attribute_values_12_16(self):

        for i in [12, 13, 14, 15, 16]:
            self.assertAlmostEqual(self.mat_array[0, i], self.py_array[0, i], 5)

    def test_attribute_values_17_21(self):

        for i in [17, 18, 19, 20, 21]:
            self.assertAlmostEqual(self.mat_array[0, i], self.py_array[0, i], 5)

    def test_attribute_values_22(self):

        self.assertAlmostEqual(self.mat_array[0, 22], self.py_array[0, 22], 5)


        


if __name__ == '__main__':

    unittest.TextTestRunner(verbosity=2).run(suite())
