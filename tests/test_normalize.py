import numpy as np
import pandas as pd
import pytest

import msreport.normalize


# Missing tests:
# Test fit with pseudo reference -> fit()
# Test fit with paired samples -> fit()
# Test is_fitted True or False -> is_fitted()
# Test get_fits -> get_fits()


class TestFixedValueNormalizer:
    @pytest.fixture(autouse=True)
    def _init_table(self):
        self.table = pd.DataFrame(
            {
                "A": [9, 10, 11],
                "B": [4, 5, 6],
                "C": [2, 3, 4],
            }
        )

    def test_transform_is_applied_correctly(self):
        normalizer = msreport.normalize.FixedValueNormalizer(
            center_function=np.median, comparison="reference"
        )
        # Normalize all columns to column "B"
        normalizer._sample_fits = {"A": 5, "B": 0, "C": -2}
        normalized_table = normalizer.transform(self.table)
        for column in normalized_table.columns:
            np.testing.assert_array_equal(normalized_table[column], self.table["B"])


class TestValueDependentNormalizer:
    @pytest.fixture(autouse=True)
    def _init_table(self):
        self.table = pd.DataFrame(
            {
                "A": [9, 10, 11],
                "B": [4, 5, 6],
                "C": [2, 3, 4],
            }
        )

    def test_transform_is_applied_correctly_with_interpolation_of_fits(self):
        normalizer = msreport.normalize.ValueDependentNormalizer(lambda x: x)
        # Normalize all values to become zero
        normalizer._sample_fits = {
            "A": ([9, 9], [11, 11]),
            "B": ([4, 4], [6, 6]),
            "C": ([2, 2], [4, 4]),
        }
        normalized_table = normalizer.transform(self.table)
        np.testing.assert_array_equal(normalized_table, self.table - self.table)
