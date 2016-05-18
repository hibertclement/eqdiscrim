import unittest
import numpy as np
import synthetics as syn
import signal_proc as sp


def suite():
    suite = unittest.TestSuite()
    suite.addTest(SimpleTests('test_smooth'))

    return suite
    
    
class SimpleTests(unittest.TestCase):
    
    def setUp(self):
        self.npts = 3000
        self.dt = 0.01
        self.poly_coefs = [0.05, -0.75, 0.1, 1.0]
        self.tr = syn.gen_sweep_poly(self.dt, self.npts, self.poly_coefs)
        
    def test_smooth(self):
        imax_fraction = 0.2
        max_amp = 300
        tr, modulation = syn.modulate_trace_triangle(self.tr, imax_fraction, 
                                                     max_amp)
        smooth_modulation = sp.smooth(modulation)
        self.assertEqual(len(modulation), len(smooth_modulation))
        self.assertEqual(np.argmax(modulation), np.argmax(smooth_modulation))



if __name__ == '__main__':

    unittest.TextTestRunner(verbosity=2).run(suite())
