import unittest
import numpy as np

from utils.calc_utils import CalculationsUtils as cu


class CallculationsUtilsTests(unittest.TestCase):

    def setUp(self):
        pass

    def test_sum_arrays(self):
        a = np.array([1, 2, 3, 4])
        b = np.array([8, 7, 6, 5, 4])
        c = np.array([1])
        d = np.array([0, 0, 0, 0, 0, 0])

        res = cu.sum_arrays([a, b, c, d])
        self.assertListEqual(res.tolist(), [10, 9, 9, 9, 4, 0])

        a = [0, 2, 3, 4, 1]
        b = [8, 7, 6, 5, 4]
        res = cu.sum_arrays([a, b])
        self.assertListEqual(res.tolist(), [8, 9, 9, 9, 5])
