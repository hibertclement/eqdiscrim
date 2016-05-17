import unittest
import numpy as np
import synthetics as syn



def suite():
    suite = unittest.TestSuite()
    suite.addTest(SimpleTests('test_npts'))
    suite.addTest(SimpleTests('test_duration'))
    suite.addTest(SimpleTests('test_max_triangle'))

    return suite
    
    
class SimpleTests(unittest.TestCase):
    
    def setUp(self):
        self.npts=12345
        self.dt = 0.01
        self.amp = 12345
        self.tr = syn.gen_gaussian_noise(self.dt, self.npts, self.amp)
        
    def test_npts(self):
        self.assertEqual(self.tr.stats.npts, self.npts)
    
    def test_duration(self):
        self.assertEqual(self.tr.stats.endtime - self.tr.stats.starttime, 
                         self.dt*(self.npts-1))

    def test_max_triangle(self):
        imax_fraction = 0.2
        max_amp = 54321
        tr, modulation = syn.modulate_trace_triangle(self.tr, imax_fraction, 
                                                     max_amp)
        self.assertTrue(max(modulation)< max_amp)
        self.assertAlmostEqual(imax_fraction,
                               np.argmax(modulation) / float(len(modulation)))



if __name__ == '__main__':

    unittest.TextTestRunner(verbosity=2).run(suite())