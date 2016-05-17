import unittest
from seis_proc import extract_events
from cat_io import read_ovpf_cat


def suite():
    suite = unittest.TestSuite()
    suite.addTest(SeisTests('test_extract_events'))
    return suite

class SeisTests(unittest.TestCase):

    def test_extract_events(self):
        ev_id = 2
        cat = read_ovpf_cat('MC3_dump.csv')
        tr = extract_events([cat[ev_id, 0]], [cat[ev_id, 1]])

        sr = tr.stats.sampling_rate
        npts = cat[ev_id, 1]*sr+1

        self.assertEqual(npts, tr.stats.npts)


if __name__ == '__main__':

    unittest.TextTestRunner(verbosity=2).run(suite())
