import unittest
import pandas as pd
import numpy as np

from knn_classifier_A1 import (
    label_encoding,
    one_hot_encoding,
    fill_values
)


class TestPreprocessing(unittest.TestCase):

    # --------------------------------------------------
    # Test 1: Label Encoding
    # --------------------------------------------------

    def test_label_encoding(self):

        column = pd.Series(
            ["A", "B", "A", "C"]
        )

        encoded, mapping = label_encoding(column)

        # Same category must get same value
        self.assertEqual(
            encoded.iloc[0],
            encoded.iloc[2]
        )

        # Different categories must have
        # different encoded values
        self.assertNotEqual(
            encoded.iloc[0],
            encoded.iloc[1]
        )

        self.assertEqual(
            len(mapping),
            3
        )


    # --------------------------------------------------
    # Test 2: One Hot Encoding
    # --------------------------------------------------

    def test_one_hot_encoding(self):

        column = pd.Series(
            ["A", "B", "A"],
            name="Class"
        )

        result = one_hot_encoding(column)

        # Two unique categories
        self.assertEqual(
            result.shape[1],
            2
        )

        # Correct column names
        self.assertIn(
            "Class_A",
            result.columns
        )

        self.assertIn(
            "Class_B",
            result.columns
        )


    # --------------------------------------------------
    # Test 3: Missing value imputation
    # --------------------------------------------------

    def test_fill_values(self):

        df = pd.DataFrame({
            "feature1": [10, 20, np.nan, 40],
            "feature2": [1, np.nan, 3, 4]
        })

        numeric_columns = [
            "feature1",
            "feature2"
        ]

        fill_values(
            numeric_columns,
            df
        )

        # No missing values should remain
        self.assertEqual(
            df["feature1"].isnull().sum(),
            0
        )

        self.assertEqual(
            df["feature2"].isnull().sum(),
            0
        )


if __name__ == "__main__":
    unittest.main()