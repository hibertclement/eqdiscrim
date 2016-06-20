import unittest
import numpy as np
import synthetics as syn
import eqdiscrim_io as io
from obspy import UTCDateTime



def suite():
    suite = unittest.TestSuite()
    suite.addTest(SimpleTests('test_OVPF_arclink'))

    return suite
    
    
class SimpleTests(unittest.TestCase):
    
        
    def test_OVPF_arclink(self):

        starttime = UTCDateTime(2016, 1, 1, 0, 0, 0)
        endtime = starttime + 120
        st = io.get_OVPF_arclink_data("PF", "PRO", "00", "HH?", starttime, endtime)

        hz = 100
        npts = 120 * hz + 1 

        self.assertEqual(st[0].stats.npts, npts)




if __name__ == '__main__':

    unittest.TextTestRunner(verbosity=2).run(suite())
