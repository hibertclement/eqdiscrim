import unittest
from cat_io import read_ovpf_cat

def suite():
    suite = unittest.TestSuite()
    suite.addTest(IoReadTest('test_read_cat'))
    return suite
    
class IoReadTest(unittest.TestCase):
    
    def test_read_cat(self):
        
        cat = read_ovpf_cat('MC3_dump.csv')
        nev, ncol = cat.shape
        self.assertEqual(ncol, 4)
        self.assertEqual(nev, 4082)
        self.assertEqual(cat[1, 1], 5.2)
        
        
if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())
    