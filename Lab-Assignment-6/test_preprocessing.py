import unittest
import pandas as pd
import numpy as np

from knn import encoding, imputation


class TestPreprocessingFunctions(unittest.TestCase):

    # ==================================================
    # Test ENCODING
    # ==================================================

    def test_encoding(self):

        df = pd.DataFrame({
            "gender": ["Male", "Female", "Male", "Female"],
            "color": ["Red", "Blue", "Green", "Red"],
            "age": [20, 21, 22, 23],
            "person_id": ["A", "Z", "A", "Z"]
        })
        df["person_id"] = df["person_id"].astype(object)

        X, y = encoding(df)

        # Check that binary categorical column is encoded
        self.assertTrue(
            pd.api.types.is_numeric_dtype(X["gender"])
        )

        # Check that target is encoded
        self.assertTrue(
            pd.api.types.is_numeric_dtype(y)
        )

        # Check number of rows remains the same
        self.assertEqual(
            len(X),
            len(df)
        )

        # Check target length
        self.assertEqual(
            len(y),
            len(df)
        )


    # ==================================================
    # Test IMPUTATION
    # ==================================================

    def test_imputation(self):

        df = pd.DataFrame({
        "gender": ["Male", "Female", "Male", "Female"],
        "color": ["Red", "Blue", "Green", "Red"],
        "age": [20, 21, 22, 23],
        "person_id": ["A", "Z", "A", "Z"]
         })

        df["person_id"] = df["person_id"].astype(object)

        result = imputation(df)

        # Check that no missing values remain
        self.assertEqual(
            result.isnull().sum().sum(),
            0
        )

        # Check number of rows remains unchanged
        self.assertEqual(
            len(result),
            len(df)
        )

        # Check number of columns remains unchanged
        self.assertEqual(
            len(result.columns),
            len(df.columns)
        )


if __name__ == "__main__":
    unittest.main()