import numpy as np
import pandas as pd
import pytest

import msreport.normalize


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
        self.test_fits = {"A": 5, "B": 0, "C": -2}
        self.test_normalizer = msreport.normalize.FixedValueNormalizer(
            center_function=lambda x: x, comparison="reference"
        )

    def test_is_fitted_false(self):
        assert not self.test_normalizer.is_fitted()

    def test_is_fitted_true(self):
        self.test_normalizer._sample_fits = self.test_fits
        assert self.test_normalizer.is_fitted()

    def test_get_fits(self):
        self.test_normalizer._sample_fits = self.test_fits
        assert self.test_normalizer.get_fits() == self.test_fits

    def test_transform_is_applied_correctly(self):
        # Normalize all columns to column "B" with self.test_fits
        self.test_normalizer._sample_fits = self.test_fits
        normalized_table = self.test_normalizer.transform(self.table)
        for column in normalized_table.columns:
            np.testing.assert_array_equal(normalized_table[column], self.table["B"])

    # Missing tests:
    # Test fit with pseudo reference -> fit()
    # Test fit with paired samples -> fit()


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
        self.test_fits = {
            "A": ([9, 9], [11, 11]),
            "B": ([4, 4], [6, 6]),
            "C": ([2, 2], [4, 4]),
        }
        self.test_normalizer = msreport.normalize.ValueDependentNormalizer(lambda x: x)

    def test_is_fitted_false(self):
        assert not self.test_normalizer.is_fitted()

    def test_is_fitted_true(self):
        self.test_normalizer._sample_fits = self.test_fits
        assert self.test_normalizer.is_fitted()

    def test_get_fits(self):
        self.test_normalizer._sample_fits = self.test_fits
        assert self.test_normalizer.get_fits() == self.test_fits

    def test_transform_is_applied_correctly_with_interpolation_of_fits(self):
        # Normalize all values to become zero with self.test_fits
        self.test_normalizer._sample_fits = self.test_fits
        normalized_table = self.test_normalizer.transform(self.table)
        np.testing.assert_array_equal(normalized_table, self.table - self.table)

    # Missing tests:
    # Test fit with pseudo reference -> fit()
