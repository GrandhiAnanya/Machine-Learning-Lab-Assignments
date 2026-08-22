import unittest
import numpy as np

from wknn import (
    calculate_distance,
    calculate_all_distances,
    sort_distances,
    identify_neighbors,
    classify_weighted
)


class TestWeightedKNN(unittest.TestCase):

    # ==================================================
    # 1. Euclidean Distance
    # ==================================================

    def test_euclidean_distance(self):

        test_sample = np.array([1, 2])
        training_sample = np.array([4, 6])

        result = calculate_distance(
            test_sample,
            training_sample,
            "euclidean"
        )

        self.assertAlmostEqual(
            result,
            5.0
        )


    # ==================================================
    # 2. Manhattan Distance
    # ==================================================

    def test_manhattan_distance(self):

        test_sample = np.array([1, 2])
        training_sample = np.array([4, 6])

        result = calculate_distance(
            test_sample,
            training_sample,
            "manhattan"
        )

        self.assertEqual(
            result,
            7
        )


    # ==================================================
    # 3. Invalid Distance
    # ==================================================

    def test_invalid_distance(self):

        with self.assertRaises(ValueError):

            calculate_distance(
                np.array([1, 2]),
                np.array([3, 4]),
                "invalid"
            )


    # ==================================================
    # 4. Calculate All Distances
    # ==================================================

    def test_calculate_all_distances(self):

        X_train = np.array([
            [0, 0],
            [3, 4],
            [6, 8]
        ])

        test_sample = np.array([0, 0])

        result = calculate_all_distances(
            X_train,
            test_sample
        )

        expected = [
            (0.0, 0),
            (5.0, 1),
            (10.0, 2)
        ]

        self.assertEqual(
            result,
            expected
        )


    # ==================================================
    # 5. Bubble Sort
    # ==================================================

    def test_bubble_sort(self):

        distances = [
            (5, 0),
            (2, 1),
            (8, 2),
            (1, 3)
        ]

        result = sort_distances(
            distances,
            "bubble"
        )

        expected = [
            (1, 3),
            (2, 1),
            (5, 0),
            (8, 2)
        ]

        self.assertEqual(
            result,
            expected
        )


    # ==================================================
    # 6. Selection Sort
    # ==================================================

    def test_selection_sort(self):

        distances = [
            (5, 0),
            (2, 1),
            (8, 2),
            (1, 3)
        ]

        result = sort_distances(
            distances,
            "selection"
        )

        expected = [
            (1, 3),
            (2, 1),
            (5, 0),
            (8, 2)
        ]

        self.assertEqual(
            result,
            expected
        )


    # ==================================================
    # 7. Insertion Sort
    # ==================================================

    def test_insertion_sort(self):

        distances = [
            (5, 0),
            (2, 1),
            (8, 2),
            (1, 3)
        ]

        result = sort_distances(
            distances,
            "insertion"
        )

        expected = [
            (1, 3),
            (2, 1),
            (5, 0),
            (8, 2)
        ]

        self.assertEqual(
            result,
            expected
        )


    # ==================================================
    # 8. Identify Neighbors
    # ==================================================

    def test_identify_neighbors(self):

        sorted_distances = [
            (1.0, 0),
            (2.0, 1),
            (4.0, 2),
            (8.0, 3)
        ]

        y_train = np.array([
            "A",
            "A",
            "Z",
            "Z"
        ])

        result = identify_neighbors(
            sorted_distances,
            3,
            y_train
        )

        expected = [
            (1.0, 0),
            (2.0, 1),
            (4.0, 2)
        ]

        self.assertEqual(
            result,
            expected
        )


    # ==================================================
    # 9. Weighted Classification
    # ==================================================

    def test_weighted_classification(self):

        neighbors = [
            (1.0, 0),
            (5.0, 1),
            (6.0, 2)
        ]

        y_train = np.array([
            "A",
            "Z",
            "Z"
        ])

        predicted_class, class_weights = classify_weighted(
            neighbors,
            y_train
        )

        # A weight = 1 / 1 = 1
        # Z weight = 1 / 5 + 1 / 6
        # A should win

        self.assertEqual(
            predicted_class,
            "A"
        )


    # ==================================================
    # 10. Weighted KNN Tie
    # ==================================================

    def test_weighted_tie(self):

        neighbors = [
            (1.0, 0),
            (1.0, 1)
        ]

        y_train = np.array([
            "A",
            "Z"
        ])

        predicted_class, class_weights = classify_weighted(
            neighbors,
            y_train
        )

        # Equal weights.
        # Closest neighbor occurs first -> A
        self.assertEqual(
            predicted_class,
            "A"
        )


if __name__ == "__main__":
    unittest.main()