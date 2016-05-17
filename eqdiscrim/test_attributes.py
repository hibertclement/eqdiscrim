import unittest
import synthetics as syn
import attributes as att



def suite():
    suite = unittest.TestSuite()
    suite.addTest(SimpleTests('test_asdec'))

    return suite
    
    
class SimpleTests(unittest.TestCase):
    
    def setUp(self):
        self.npts=12345
        self.dt = 0.01
        self.amp = 12345
        self.tr = syn.gen_gaussian_noise(self.dt, self.npts, self.amp)
        
    def test_asdec(self):
        imax_fraction = 0.2
        max_amp = 1
        tr, modulation = syn.modulate_trace_triangle(self.tr, imax_fraction, 
                                                     max_amp)
        env = att.envelope(tr)
        AsDec = att.get_AsDec(tr, env)
        self.assertAlmostEqual(AsDec, imax_fraction / (1-imax_fraction), 1)



if __name__ == '__main__':

    unittest.TextTestRunner(verbosity=2).run(suite())