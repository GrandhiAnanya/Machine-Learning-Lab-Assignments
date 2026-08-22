import unittest
import numpy as np
import pandas as pd

from knn import (
    calculate_distance,
    calculate_all_distances,
    sort_distances,
    identify_neighbors,
    classify
)


class TestKNN(unittest.TestCase):

    # ==================================================
    # 1. Test Euclidean Distance
    # ==================================================

    def test_euclidean_distance(self):

        test_sample = np.array([1, 2])
        training_sample = np.array([4, 6])

        result = calculate_distance(
            test_sample,
            training_sample,
            "euclidean"
        )

        expected = 5.0

        self.assertAlmostEqual(
            result,
            expected
        )


    # ==================================================
    # 2. Test Manhattan Distance
    # ==================================================

    def test_manhattan_distance(self):

        test_sample = np.array([1, 2])
        training_sample = np.array([4, 6])

        result = calculate_distance(
            test_sample,
            training_sample,
            "manhattan"
        )

        expected = 7

        self.assertEqual(
            result,
            expected
        )


    # ==================================================
    # 3. Test Invalid Distance Metric
    # ==================================================

    def test_invalid_distance_metric(self):

        test_sample = np.array([1, 2])
        training_sample = np.array([4, 6])

        with self.assertRaises(ValueError):

            calculate_distance(
                test_sample,
                training_sample,
                "invalid"
            )


    # ==================================================
    # 4. Test Distance = 0
    # ==================================================

    def test_same_points(self):

        sample = np.array([5, 5])

        result = calculate_distance(
            sample,
            sample,
            "euclidean"
        )

        self.assertEqual(
            result,
            0
        )


    # ==================================================
    # 5. Test Calculate All Distances
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
    # 6. Test Bubble Sort
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
    # 7. Test Selection Sort
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
    # 8. Test Insertion Sort
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
    # 9. Test Equal Distance Tie
    # ==================================================

    def test_equal_distance_sort(self):

        distances = [
            (2, 3),
            (2, 1),
            (1, 2)
        ]

        result = sort_distances(
            distances,
            "bubble"
        )

        expected = [
            (1, 2),
            (2, 1),
            (2, 3)
        ]

        self.assertEqual(
            result,
            expected
        )


    # ==================================================
    # 10. Test Invalid Sorting Algorithm
    # ==================================================

    def test_invalid_sorting_algorithm(self):

        distances = [
            (5, 0),
            (2, 1)
        ]

        with self.assertRaises(ValueError):

            sort_distances(
                distances,
                "invalid"
            )


    # ==================================================
    # 11. Test Identify Neighbors
    # ==================================================

    def test_identify_neighbors(self):

        sorted_distances = [
            (1, 3),
            (2, 1),
            (4, 0),
            (8, 2)
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
            (1, 3),
            (2, 1),
            (4, 0)
        ]

        self.assertEqual(
            result,
            expected
        )


    # ==================================================
    # 12. Test Normal KNN Classification
    # ==================================================

    def test_classify(self):

        neighbors = [
            (1.0, 0),
            (2.0, 1),
            (3.0, 2)
        ]

        y_train = np.array([
            "A",
            "A",
            "Z"
        ])

        predicted_class, class_counts = classify(
            neighbors,
            y_train
        )

        self.assertEqual(
            predicted_class,
            "A"
        )

        self.assertEqual(
            class_counts["A"],
            2
        )

        self.assertEqual(
            class_counts["Z"],
            1
        )


    # ==================================================
    # 13. Test Classification Tie
    # ==================================================

    def test_classification_tie(self):

        neighbors = [
            (1.0, 0),
            (2.0, 1)
        ]

        y_train = np.array([
            "A",
            "Z"
        ])

        predicted_class, class_counts = classify(
            neighbors,
            y_train
        )

        # Closest neighbor belongs to A
        self.assertEqual(
            predicted_class,
            "A"
        )


if __name__ == "__main__":
    unittest.main()