import unittest
import numpy as np
from A4 import minkowski_distance
from A7 import dot,length
from A8_A9 import mean_var_sd

class TestLabFunctions(unittest.TestCase):

    def test_minkowski(self):
        A = [1,2]
        B = [4,6]
        self.assertEqual(minkowski_distance(A,B,1),7)
        self.assertEqual(minkowski_distance(A,B,2),5)

    def test_dot_product(self):
        A = np.array([1, 2, 3])
        B = np.array([4, 5, 6])
        self.assertEqual(dot(A, B), 32)

    def test_length(self):
        a = [3, 4]
        b = [5, 12]
        len_a, len_b = length(a, b)
        self.assertEqual(len_a, 5)
        self.assertEqual(len_b, 13)

    def test_mean_var_sd(self):

        X = [
            [1, 2],
            [3, 4]
        ]

        means, vars, sds = mean_var_sd(X)

        self.assertEqual(means, [2.0, 3.0])
        self.assertEqual(vars, [1.0, 1.0])
        self.assertEqual(sds, [1.0, 1.0])

    


if __name__ == "__main__":
    unittest.main()