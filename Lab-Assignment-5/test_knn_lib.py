import unittest
import numpy as np

from A7_knn import (
    distance_calculation,
    sort,
    identify_class,
    knn,
    fit,
    score
)


class TestKNN(unittest.TestCase):

    # ==================================================
    # Test DISTANCE CALCULATION
    # ==================================================

    def test_distance_calculation(self):

        result = distance_calculation(5, 2, 2)

        self.assertEqual(result, 9)


    # ==================================================
    # Test SORT
    # ==================================================

    def test_sort(self):

        distances = {
            0: [5, "A"],
            1: [2, "Z"],
            2: [4, "A"],
            3: [1, "Z"]
        }

        result = sort(distances)

        expected = [3, 1, 2, 0]

        self.assertEqual(result, expected)


    # ==================================================
    # Test IDENTIFY CLASS
    # ==================================================

    def test_identify_class(self):

        # Indices sorted according to distance
        sorted_indices = [0, 1, 2, 3]

        y_train = np.array([
            "A",
            "A",
            "Z",
            "Z"
        ])

        result = identify_class(
            sorted_indices,
            y_train,
            3
        )

        self.assertEqual(result, "A")


    # ==================================================
    # Test KNN
    # ==================================================

    def test_knn(self):

        X_train = np.array([
            [1, 1],
            [1, 2],
            [8, 8],
            [9, 9]
        ])

        y_train = np.array([
            "A",
            "A",
            "Z",
            "Z"
        ])

        X_test = np.array([
            [1, 1.5]
        ])

        result = knn(
            X_test,
            X_train,
            y_train,
            3
        )

        self.assertEqual(
            result[0],
            "A"
        )


    # ==================================================
    # Test FIT
    # ==================================================

    def test_fit(self):

        X_train = np.array([
            [1, 1],
            [2, 2]
        ])

        y_train = np.array([
            "A",
            "Z"
        ])

        result = fit(
            X_train,
            y_train,
            3
        )

        # Current fit() does not return anything
        self.assertIsNone(result)


    # ==================================================
    # Test SCORE
    # ==================================================

    def test_score(self):

        classes = {
            0: "A",
            1: "A",
            2: "Z",
            3: "A"
        }

        y_test = np.array([
            "A",
            "A",
            "Z",
            "Z"
        ])

        result = score(
            classes,
            y_test
        )

        self.assertEqual(result, 0.75)


if __name__ == "__main__":
    unittest.main()