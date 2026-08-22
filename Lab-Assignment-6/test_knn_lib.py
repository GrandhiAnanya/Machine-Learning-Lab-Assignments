import unittest
import numpy as np

from knn_lib import Fit, predict,score


class TestKNNLibraryFunctions(unittest.TestCase):

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

        X_fitted, y_fitted = Fit(
            X_train,
            y_train
        )

        np.testing.assert_array_equal(
            X_fitted,
            X_train
        )

        np.testing.assert_array_equal(
            y_fitted,
            y_train
        )


    # ==================================================
    # Test PREDICT
    # ==================================================
    def test_predict(self):

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

        X_fitted, y_fitted = Fit(
            X_train,
            y_train
        )

        predictions = predict(
            X_fitted,
            y_fitted,
            X_test,
            3
        )

        self.assertEqual(
            predictions[0],
            "A"
        )
        
    def test_score(self):

     predictions = [
        "A",
        "A",
        "Z",
        "A"
     ]

     y_test = np.array([
        "A",
        "A",
        "Z",
        "Z"
     ])

     result = score(
        predictions,
        y_test
     )

     self.assertEqual(
        result,
        0.75
     )
     

if __name__ == "__main__":
    unittest.main()