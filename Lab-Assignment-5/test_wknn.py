import unittest
import numpy as np

from A8_wknn import (
    distance_calculation,
    sort,
    identify_class,
    wknn,
    score
)


class TestWeightedKNN(unittest.TestCase):

    # --------------------------------------------------
    # Test 1: Distance calculation
    # --------------------------------------------------

    def test_distance_calculation(self):

        result = distance_calculation(3, 7, 2)

        expected = 16

        self.assertEqual(result, expected)


    # --------------------------------------------------
    # Test 2: Sorting
    # --------------------------------------------------

    def test_sort(self):

        distances = {
            0: [5, "A", 0.2],
            1: [2, "Z", 0.5],
            2: [8, "A", 0.125],
            3: [1, "Z", 1.0]
        }

        result = sort(distances)

        expected = [3, 1, 0, 2]

        self.assertEqual(result, expected)


    # --------------------------------------------------
    # Test 3: Weighted class identification
    # --------------------------------------------------

    def test_weighted_class(self):

        distances = {
            0: [1, "A", 1.0],
            1: [5, "Z", 0.2],
            2: [6, "Z", 1 / 6]
        }

        sorted_indices = [0, 1, 2]

        y_train = np.array([
            "A",
            "Z",
            "Z"
        ])

        result = identify_class(
            sorted_indices,
            distances,
            y_train,
            3
        )

        # A has weight 1.0
        # Z has weight 0.2 + 0.1667
        # Therefore A wins.
        expected = "A"

        self.assertEqual(result, expected)


    # --------------------------------------------------
    # Test 4: Weighted KNN prediction
    # --------------------------------------------------

    def test_wknn(self):

        X_train = np.array([
            [1, 1],
            [2, 2],
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

        result = wknn(
            X_test,
            X_train,
            y_train,
            3
        )

        expected = {
            0: "A"
        }

        self.assertEqual(result, expected)


    # --------------------------------------------------
    # Test 5: Score
    # --------------------------------------------------

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

        expected = 0.75

        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()