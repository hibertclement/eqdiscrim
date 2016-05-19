import unittest
import numpy as np
import synthetics as syn
import attributes as att



def suite():
    suite = unittest.TestSuite()
    suite.addTest(SimpleTests('test_asdec'))
    suite.addTest(SimpleTests('test_duration'))
    suite.addTest(SimpleTests('test_rapps'))
    suite.addTest(HigherStatsTests('test_signal_kurtosis'))
    suite.addTest(HigherStatsTests('test_signal_skew'))
    suite.addTest(AutoCorTests('test_peaks'))

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

    def test_duration(self):
        Duration = att.get_Duration(self.tr)
        self.assertAlmostEqual(Duration, (self.npts-1) * self.dt)

    def test_rapps(self):
        imax_fraction = 0.5
        max_amp = 300
        tr, modulation = syn.modulate_trace_triangle(self.tr, imax_fraction, 
                                                     max_amp)
        RappMaxMean, RappMaxMedian, RappMaxStd = att.get_RappStuff(tr)
        self.assertTrue(RappMaxMean > 2)
        self.assertAlmostEqual(RappMaxMedian, RappMaxMean, 1)


class HigherStatsTests(unittest.TestCase):

    def setUp(self):
        self.npts = 300000
        self.dt = 0.01
        self.amp = 100.0
        self.tr = syn.gen_gaussian_noise(self.dt, self.npts, self.amp)
        self.KurtoEnv, self.KurtoSig, self.SkewnessEnv, self.SkewnessSig = \
            att.get_KurtoSkew(self.tr)

    def test_signal_kurtosis(self):
        self.assertAlmostEqual(self.KurtoSig, 3.0, 1)

    def test_signal_skew(self):
        self.assertAlmostEqual(self.SkewnessSig, 0.0, 1)


class AutoCorTests(unittest.TestCase):

    def setUp(self):
        self.npts = 3000
        self.dt = 0.01
        self.sig_npts = 100
        self.sig_amp = 5
        self.n_sig = 3
        self.tr = syn.gen_repeat_signal(self.dt, self.npts, self.sig_npts,
                                        self.sig_amp, self.n_sig)
        self.CorrPeakNumber, self.INTRATIO = att.get_CorrStuff(self.tr)

    def test_peaks(self):
        print self.INTRATIO
        self.assertEqual(self.CorrPeakNumber, 3)

if __name__ == '__main__':

    unittest.TextTestRunner(verbosity=2).run(suite())
