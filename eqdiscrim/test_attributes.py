import unittest
import synthetics as syn
import attributes as att



def suite():
    suite = unittest.TestSuite()
    suite.addTest(SimpleTests('test_asdec'))

    return suite
    
    
class SimpleTests(unittest.TestCase):
    
    def setUp(self):
        self.npts = 3000
        self.dt = 0.01
        self.poly_coefs = [0.05, -0.75, 0.1, 1.0]
        self.tr = syn.gen_sweep_poly(self.dt, self.npts, self.poly_coefs)
        
    def test_asdec(self):
        imax_fraction = 0.2
        max_amp = 300
        tr, modulation = syn.modulate_trace_triangle(self.tr, imax_fraction, 
                                                     max_amp)
        AsDec = att.get_AsDec(tr)
        self.assertAlmostEqual(AsDec, imax_fraction / (1-imax_fraction), 2)



if __name__ == '__main__':

    unittest.TextTestRunner(verbosity=2).run(suite())
