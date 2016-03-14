import unittest, os
import numpy as np
import pandas as pd
from obspy.core import UTCDateTime, read
from obspy.signal.filter import bandpass
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, lfilter, spectrogram

import data_io as io
import attributes as at

datadir = 'test_data'

def suite():
    suite = unittest.TestSuite()
    suite.addTest(IOTests('test_read_catalog'))
    suite.addTest(IOTests('test_read_and_cut_events'))
    suite.addTest(PythonMatlabTests('test_filter_coefs'))
    suite.addTest(PythonMatlabTests('test_filtering_data'))
    suite.addTest(PythonMatlabTests('test_spectrogram_calc'))
    suite.addTest(AttributeTests('test_polarization_synth'))
    suite.addTest(AttributeTests('test_attribute_values_0_11'))
    suite.addTest(AttributeTests('test_attribute_values_12_16'))
    suite.addTest(AttributeTests('test_attribute_values_17_21'))
    suite.addTest(AttributeTests('test_attribute_values_22'))
    suite.addTest(AttributeTests('test_attribute_values_23_32'))
    suite.addTest(AttributeTests('test_attribute_values_33_36'))
    suite.addTest(AttributeTests('test_attribute_values_37_39'))
    suite.addTest(AttributeTests('test_attribute_values_57_60'))

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

        for i in np.arange(12, 17):
            self.assertAlmostEqual(self.mat_array[0, i], self.py_array[0, i], 5)

    def test_attribute_values_17_21(self):

        for i in np.arange(17, 22):
            self.assertAlmostEqual(self.mat_array[0, i], self.py_array[0, i], 5)

    def test_attribute_values_22(self):

        self.assertAlmostEqual(self.mat_array[0, 22], self.py_array[0, 22], 5)

    def test_attribute_values_23_32(self):

        for i in np.arange(23, 33):
            self.assertAlmostEqual(self.mat_array[0, i], self.py_array[0, i], 5)

    def test_attribute_values_33_36(self):

        for i in np.arange(33, 37):
            self.assertAlmostEqual(self.mat_array[0, i], self.py_array[0, i], 5)

    def test_attribute_values_37_39(self):

        for i in np.arange(37, 40):
            self.assertAlmostEqual(self.mat_array[0, i], self.py_array[0, i], 5)

    def test_attribute_values_57_60(self):

        for i in np.arange(57, 61):
            self.assertAlmostEqual(self.mat_array[0, i], self.py_array[0, i], 5)

    def test_polarization_synth(self):

        fname = os.path.join(datadir, 'pol*.sac')
        st = read(fname)
        st.sort(keys=['channel'], reverse=True)

        env = at.envelope(st)
        rectilinP, azimuthP, dipP, Plani = at.get_polarization_stuff(st, env)

        mat_fname = os.path.join(datadir, 'pol_test.mat')
        mat = loadmat(mat_fname)

        self.assertAlmostEqual(rectilinP, mat['rectilinP'][0][0])
        self.assertAlmostEqual(azimuthP, mat['azimuthP'][0][0])
        self.assertAlmostEqual(dipP, mat['dipP'][0][0])
        self.assertAlmostEqual(Plani, mat['Plani'][0][0])


class PythonMatlabTests(unittest.TestCase):

    def test_filter_coefs(self):

        mat_fname = os.path.join(datadir, 'filter_coef.mat')
        mat = loadmat(mat_fname)

        Fb_mat = mat['Fb'].flatten()
        Fa_mat = mat['Fa'].flatten()

        NyF = float(mat['NyF'].flatten()[0])
        FFI = float(mat['FFI'].flatten()[0])
        FFE = float(mat['FFE'].flatten()[0])

        Fb_py, Fa_py = butter(2, [FFI/NyF, FFE/NyF], 'bandpass')

        for i in xrange(len(Fb_py)):
            self.assertAlmostEqual(Fb_mat[i], Fb_py[i])
        for i in xrange(len(Fa_py)):
            self.assertAlmostEqual(Fa_mat[i], Fa_py[i])

    def test_filtering_data(self):

        # load filter coefficients
        mat_fname = os.path.join(datadir, 'filter_coef.mat')
        mat = loadmat(mat_fname)

        NyF = float(mat['NyF'].flatten()[0])
        FFI = float(mat['FFI'].flatten()[0])
        FFE = float(mat['FFE'].flatten()[0])

        Fb, Fa = butter(2, [FFI/NyF, FFE/NyF], 'bandpass')

        # load filtered data
        mat_fname = os.path.join(datadir, 'filtered_data.mat')
        mat = loadmat(mat_fname)

        mat_fdata_01 = mat['fdata_01'].flatten()
        mat_fdata_02 = mat['fdata_02'].flatten()
        mat_fdata_03 = mat['fdata_03'].flatten()
        mat_fdata_04 = mat['fdata_04'].flatten()
        mat_fdata_05 = mat['fdata_05'].flatten()
        mat_fdata_06 = mat['fdata_06'].flatten()

        # synthetic (clean) data
        st = read(os.path.join(datadir, 'IU*MXZ*sac'))
        tr = st[0].copy()

        fdata_01 = lfilter(Fb, Fa, tr)
        fdata_02 = lfilter(Fb, Fa, fdata_01[::-1])
        fdata_02 = fdata_02[::-1]
        #fdata_03 = filtfilt(Fb, Fa, tr)
        fdata_03 = at.l2filter(Fb, Fa, tr)

        # real (dirty) data
        st = read(os.path.join(datadir, 'events', 'event_01*HHZ*SAC'))
        tr = st[0].copy()

        fdata_04 = lfilter(Fb, Fa, tr)
        fdata_05 = lfilter(Fb, Fa, fdata_04[::-1])
        fdata_05 = fdata_05[::-1]
        #fdata_06 = filtfilt(Fb, Fa, tr)
        fdata_06 = at.l2filter(Fb, Fa, tr)

        # check lengths
        self.assertEqual(len(mat_fdata_01), len(fdata_01))

        # check synthetic data
        mat_sum_diff = np.sum(np.abs(mat_fdata_03 - mat_fdata_02))
        py_sum_diff = np.sum(np.abs(fdata_03 - fdata_02))

        self.assertAlmostEqual(max(mat_fdata_01), max(fdata_01))
        self.assertAlmostEqual(max(mat_fdata_02), max(fdata_02))
        self.assertAlmostEqual(max(mat_fdata_03), max(fdata_03))
        self.assertAlmostEqual(mat_fdata_01[0], fdata_01[0])
        self.assertAlmostEqual(mat_fdata_02[0], fdata_02[0])
        self.assertAlmostEqual(mat_fdata_03[0], fdata_03[0])
        # check difference between two times one-pass and two-pass
        self.assertAlmostEqual(mat_sum_diff, 0.0)
        self.assertAlmostEqual(py_sum_diff, 0.0)
        self.assertAlmostEqual(mat_sum_diff, py_sum_diff)


        # check real data
        mat_sum_diff = np.sum(np.abs(mat_fdata_06 - mat_fdata_05))
        py_sum_diff = np.sum(np.abs(fdata_06 - fdata_05))

        self.assertAlmostEqual(max(mat_fdata_04), max(fdata_04))
        self.assertAlmostEqual(max(mat_fdata_05), max(fdata_05))
        self.assertAlmostEqual(max(mat_fdata_06), max(fdata_06))
        self.assertAlmostEqual(mat_fdata_04[0], fdata_04[0])
        self.assertAlmostEqual(mat_fdata_05[0], fdata_05[0])
        self.assertAlmostEqual(mat_fdata_06[0], fdata_06[0])
        # check difference between two times one-pass and two-pass
        self.assertAlmostEqual(mat_sum_diff, 0.0)
        self.assertAlmostEqual(py_sum_diff, 0.0)
        self.assertAlmostEqual(mat_sum_diff, py_sum_diff)

        # test conclusion : use two single-pass filters for 2-pass filtering
        # in both python and matlab
 
    def test_spectrogram_calc(self):

        fname = os.path.join(datadir, 'IU*MXZ*sac')
        st = read(fname)
        tr = st[0]

        mat_fname = os.path.join(datadir, 'spec_test.mat')
        mat = loadmat(mat_fname)
        mat_spec = mat['spec']
        mat_smooth_spec = mat['smooth_spec']
        mat_f = mat['F'].flatten()
        mat_t = mat['T'].flatten()
        mat_fft = mat['FFT'].flatten()

        sps = 4.0
        npts = tr.stats.npts
        n = at.nextpow2(2*npts-1)
        n2 = at.nextpow2(2*200-1)

        fft = np.fft.fft(st[0].data, n)
        f, t, spec = spectrogram(st[0].data, fs=4.0, window='hamming',
                                 nperseg=200, nfft=n2, noverlap=90,
                                 scaling='density')

        nf = len(fft)
        i_f = np.random.randint(0, nf)
        self.assertEqual(len(mat_fft), len(fft))
        self.assertAlmostEqual(mat_fft[i_f], fft[i_f])

        nr, nc = spec.shape
        ir = np.random.randint(0, nr)
        ic = np.random.randint(0, nc)

        self.assertEqual(len(mat_t), len(t))
        self.assertEqual(len(mat_f), len(f))
        self.assertAlmostEqual(mat_f[ir], f[ir])
        self.assertAlmostEqual(mat_t[ic], t[ic])
        self.assertSequenceEqual(mat_spec.shape, spec.shape)
        for i in xrange(nc):
            self.assertAlmostEqual(mat_spec[ir, i], spec[ir, i])


if __name__ == '__main__':

    unittest.TextTestRunner(verbosity=2).run(suite())
