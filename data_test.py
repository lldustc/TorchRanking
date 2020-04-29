import data
import unittest

class DataTest(unittest.TestCase):
    def test_read_data(self):
        tqddict=data.read_qdfile("data/train.txt")
        print(tqddict[10])

if __name__=="__main__":
    unittest.main()