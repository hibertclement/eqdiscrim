import unittest
import pytest
import os
import socket
import tempfile
import numpy as np
import synthetics as syn
import eqdiscrim_io as io
from obspy import UTCDateTime

hostname = socket.gethostname()


def suite():
    suite = unittest.TestSuite()
    suite.addTest(SimpleTests('test_OVPF_arclink'))
    suite.addTest(SimpleTests('test_OVPF_arclink_locid_wild'))
    suite.addTest(SimpleTests('test_OVPF_dump'))
    suite.addTest(SimpleTests('test_read_OVPF_dump'))
    suite.addTest(SimpleTests('test_combinations'))

    return suite
    
    
class SimpleTests(unittest.TestCase):
    
        
    @unittest.skipUnless(hostname.startswith('pitonpapangue'), 'requires pitonpapangue')
    def test_OVPF_arclink(self):

        starttime = UTCDateTime(2016, 1, 1, 0, 0, 0)
        endtime = starttime + 120
        st = io.get_OVPF_arclink_data("PF", "PRO", "00", "?H?", starttime, endtime)

        hz = 100
        npts = 120 * hz + 1 

        self.assertEqual(st[0].stats.npts, npts)

    @unittest.skipUnless(hostname.startswith('pitonpapangue'), 'requires pitonpapangue')
    def test_OVPF_arclink_locid_wild(self):

        starttime = UTCDateTime(2016, 1, 1, 0, 0, 0)
        endtime = starttime + 120
        st1 = io.get_OVPF_arclink_data("PF", "PRO", "00", "?H?", starttime, endtime)
        st2 = io.get_OVPF_arclink_data("PF", "PRO", "*", "?H?", starttime, endtime)

        hz = 100
        npts = 120 * hz + 1 

        self.assertEqual(max(st1[0].data), max(st2[0].data))


    @unittest.skipUnless(hostname.startswith('pitonpapangue'), 'requires pitonpapangue')
    def test_OVPF_dump(self):

        starttime = UTCDateTime(2016, 1, 1, 0, 0, 0)
        endtime = starttime + 3*3600

        fname = "test_mc3.csv"

        thepage = io.get_OVPF_MC3_dump_file(starttime, endtime, fname)
        os.unlink(fname)
        
        self.assertEqual(len(thepage), 36374)


    @unittest.skipUnless(hostname.startswith('pitonpapangue'), 'requires pitonpapangue')
    def test_read_OVPF_dump(self):

        starttime = UTCDateTime(2016, 1, 1, 0, 0, 0)
        endtime = starttime + 3*3600

        fname = "test_mc3.csv"

        io.get_OVPF_MC3_dump_file(starttime, endtime, fname)
        df = io.read_MC3_dump_file(fname)

        self.assertEqual(len(df), 594)

    def test_combinations(self):
        stations = ["BOR", "RVL", "CSS"]

        comb_list = io.get_station_combinations(stations)
        self.assertEqual(len(comb_list), 4)


@pytest.fixture()
def cm_dict(request):
    cm = np.diag([1, 1, 0.3, 1, 1])
    cm [3, 1] = 0.7

    labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    score_mean = 0.95
    score_2stdev  = 0.05

    return {'cm' : cm, 'labels' : labels, 'score_mean' : score_mean,
            'score_2stdev' : score_2stdev}

@pytest.fixture()
def tmp_file(request):
    f_, fname = tempfile.mkstemp()
    def fin():
        os.unlink(fname)
    request.addfinalizer(fin)
    return fname

    
def test_pickle_io(cm_dict, tmp_file):

    io.dump(cm_dict, tmp_file)
    cm_dict1 = io.load(tmp_file)

    for key in cm_dict.keys():
        assert key in cm_dict1.keys()


if __name__ == '__main__':

    unittest.TextTestRunner(verbosity=2).run(suite())


