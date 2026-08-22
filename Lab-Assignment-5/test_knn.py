import unittest
import numpy as np

from A7_knn import (
    distance_calculation,
    sort,
    identify_class,
    knn,
    score
)


class TestKNN(unittest.TestCase):

    # --------------------------------------------------
    # Test 1: Distance Calculation
    # --------------------------------------------------

    def test_distance_calculation(self):

        result = distance_calculation(3, 7, 2)

        expected = 16

        self.assertEqual(result, expected)


    # --------------------------------------------------
    # Test 2: Distance when points are identical
    # --------------------------------------------------

    def test_distance_zero(self):

        result = distance_calculation(5, 5, 2)

        expected = 0

        self.assertEqual(result, expected)


    # --------------------------------------------------
    # Test 3: Sorting
    # --------------------------------------------------

    def test_sort(self):

        distances = {
            0: [5, "A"],
            1: [2, "Z"],
            2: [8, "A"],
            3: [1, "Z"]
        }

        result = sort(distances)

        expected = [3, 1, 0, 2]

        self.assertEqual(result, expected)


    # --------------------------------------------------
    # Test 4: Sorting already sorted data
    # --------------------------------------------------

    def test_sort_sorted_data(self):

        distances = {
            0: [1, "A"],
            1: [2, "Z"],
            2: [3, "A"]
        }

        result = sort(distances)

        expected = [0, 1, 2]

        self.assertEqual(result, expected)


    # --------------------------------------------------
    # Test 5: Class identification
    # --------------------------------------------------

    def test_identify_class(self):

        # Sorted training indices
        sorted_indices = [0, 1, 2]

        y_train = np.array([
            "A",
            "A",
            "Z"
        ])

        result = identify_class(
            sorted_indices,
            y_train,
            3
        )

        expected = "A"

        self.assertEqual(result, expected)


    # --------------------------------------------------
    # Test 6: Class tie
    # --------------------------------------------------

    def test_identify_class_tie(self):

        sorted_indices = [0, 1]

        y_train = np.array([
            "A",
            "Z"
        ])

        result = identify_class(
            sorted_indices,
            y_train,
            2
        )

        # According to your tie-breaking mechanism,
        # the closest neighbor is selected.
        expected = "A"

        self.assertEqual(result, expected)


    # --------------------------------------------------
    # Test 7: KNN prediction
    # --------------------------------------------------

    def test_knn(self):

        X_train = np.array([
            [1, 1],
            [1, 2],
            [8, 8],
            [9, 8]
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

        expected = {
            0: "A"
        }

        self.assertEqual(result, expected)


    # --------------------------------------------------
    # Test 8: Score
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