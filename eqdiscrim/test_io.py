import unittest


def suite():
    suite = unittest.TestSuite()
    suite.addTest(IoReadTests('test_read_sihex_xls'))
    return suite


class IoReadTests(unittest.TestCase):

    def test_read_sihex_xls(self):
        from cat_io.sihex_io import read_sihex_xls

        X, y, names = read_sihex_xls()

        nev, natt = X.shape
        self.assertEqual(nev, 50585)
        self.assertEqual(natt, 8)
        self.assertEqual(y[0], 'ke')
        self.assertEqual(names[1], 'DATE')

if __name__ == '__main__':

    unittest.TextTestRunner(verbosity=2).run(suite())
