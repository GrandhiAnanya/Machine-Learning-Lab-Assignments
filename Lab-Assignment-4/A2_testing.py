import unittest
import numpy as np
from A1_4 import minkowski_distance
from A1_7 import dot_product,euclidean_norm
from A1_8_9 import calculate_mean,calculate_std,calculate_variance

class TestLabFunctions(unittest.TestCase):

    def test_minkowski(self):
        A = [1,2]
        B = [4,6]
        self.assertEqual(minkowski_distance(A,B,1),7)
        self.assertEqual(minkowski_distance(A,B,2),5)

    def test_dot_product(self):
        A = np.array([1, 2, 3])
        B = np.array([4, 5, 6])
        self.assertEqual(dot_product(A, B), 32)

    def test_mean(self):
        data = [2,4,6,8]
        self.assertEqual(calculate_mean(data),5)

    def test_variance(self):
        data = [2,4,6,8]
        self.assertEqual(calculate_variance(data),5)

    def test_std(self):
        data = [2,4,6,8]
        self.assertAlmostEqual(calculate_std(data),2.2360679,places=5)

    def test_norm_1(self):
        vector = np.array([3, 4])
        self.assertAlmostEqual(euclidean_norm(vector), 5.0, places=5)


if __name__ == "__main__":
    unittest.main()