import unittest
import os
import numpy as np
import synthetics as syn
import eqdiscrim_io as io
from obspy import UTCDateTime



def suite():
    suite = unittest.TestSuite()
    suite.addTest(SimpleTests('test_OVPF_arclink'))
    suite.addTest(SimpleTests('test_OVPF_arclink_locid_wild'))
    suite.addTest(SimpleTests('test_OVPF_dump'))
    suite.addTest(SimpleTests('test_read_OVPF_dump'))

    return suite
    
    
class SimpleTests(unittest.TestCase):
    
        
    def test_OVPF_arclink(self):

        starttime = UTCDateTime(2016, 1, 1, 0, 0, 0)
        endtime = starttime + 120
        st = io.get_OVPF_arclink_data("PF", "PRO", "00", "?H?", starttime, endtime)

        hz = 100
        npts = 120 * hz + 1 

        self.assertEqual(st[0].stats.npts, npts)

    def test_OVPF_arclink_locid_wild(self):

        starttime = UTCDateTime(2016, 1, 1, 0, 0, 0)
        endtime = starttime + 120
        st1 = io.get_OVPF_arclink_data("PF", "PRO", "00", "?H?", starttime, endtime)
        st2 = io.get_OVPF_arclink_data("PF", "PRO", "*", "?H?", starttime, endtime)

        hz = 100
        npts = 120 * hz + 1 

        self.assertEqual(max(st1[0].data), max(st2[0].data))


    def test_OVPF_dump(self):

        starttime = UTCDateTime(2016, 1, 1, 0, 0, 0)
        endtime = starttime + 3*3600

        fname = "test_mc3.csv"

        thepage = io.get_OVPF_MC3_dump_file(starttime, endtime, fname)
        os.unlink(fname)
        
        self.assertEqual(len(thepage), 36374)


    def test_read_OVPF_dump(self):

        starttime = UTCDateTime(2016, 1, 1, 0, 0, 0)
        endtime = starttime + 3*3600

        fname = "test_mc3.csv"

        io.get_OVPF_MC3_dump_file(starttime, endtime, fname)
        df = io.read_MC3_dump_file(fname)
        os.unlink(fname)

        self.assertEqual(len(df), 594)


if __name__ == '__main__':

    unittest.TextTestRunner(verbosity=2).run(suite())
