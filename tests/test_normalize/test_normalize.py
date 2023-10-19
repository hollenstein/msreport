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

    def test_fitting_with_pseudo_reference(self):
        center_func = np.mean
        normalizer = msreport.normalize.FixedValueNormalizer(
            center_function=center_func, comparison="reference"
        ).fit(self.table)
        expected_fits = {
            c: center_func(self.table[c] - self.table.mean(axis=1)) for c in self.table
        }
        assert normalizer.get_fits() == expected_fits

    def test_fitting_with_paired_samples(self):
        center_func = np.mean
        normalizer = msreport.normalize.FixedValueNormalizer(
            center_function=center_func, comparison="paired"
        ).fit(self.table)

        expected_fits = np.array(
            [center_func(self.table[c] - self.table.mean(axis=1)) for c in self.table]
        )
        observed_fits = np.array([normalizer.get_fits()[c] for c in self.table])

        np.testing.assert_allclose(observed_fits, expected_fits, rtol=1e-07, atol=1e-07)


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

    def test_fitting_with_pseudo_reference(self):
        center_func = lambda x, y: (np.array(x), np.array(x) - np.array(y))
        normalizer = msreport.normalize.ValueDependentNormalizer(
            fit_function=center_func,
        ).fit(self.table)
        expected_fits = {
            c: center_func(self.table[c], self.table.mean(axis=1)) for c in self.table
        }
        observed_fits = normalizer.get_fits()
        for column in self.table:
            np.testing.assert_array_equal(observed_fits[column], expected_fits[column])  # fmt: skip


class TestNormalizers:
    def test_median_normalizer(self):
        table = pd.DataFrame(
            {
                "A": [10, 11, 12, 13, 14, 15, 30, 30, 30],
                "B": [11, 12, 13, 14, 15, 16, 60, 60, 60],
            }
        )
        normalizer = msreport.normalize.MedianNormalizer().fit(table)
        fitted_A = normalizer.get_fits()["A"]
        expected_A = -0.5
        np.testing.assert_allclose(fitted_A, expected_A, rtol=1e-07, atol=1e-07)

    def test_mode_normalizer(self):
        table = pd.DataFrame(
            {
                "A": [10, 11, 12, 13, 14, 15, 30, 30, 30],
                "B": [11, 12, 13, 14, 15, 16, 60, 60, 60],
            }
        )
        normalizer = msreport.normalize.ModeNormalizer().fit(table)
        fitted_A = normalizer.get_fits()["A"]
        expected_A = -0.5608569864240109
        np.testing.assert_allclose(fitted_A, expected_A, rtol=1e-07, atol=1e-07)


class TestZscoreScaler:
    def test_is_always_fitted(self):
        scaler = msreport.normalize.ZscoreScaler()
        assert scaler.is_fitted()

    def test_after_transform_row_mean_is_zero_std_is_one(self):
        scaler = msreport.normalize.ZscoreScaler()
        table = pd.DataFrame({"A": [10, 11], "B": [11, 12], "C": [12, 13]})
        table_scaled = scaler.transform(table)
        assert np.allclose(table_scaled.mean(axis=1), 0)
        assert np.allclose(table_scaled.std(axis=1, ddof=0), 1)

    def test_after_transform_only_with_mean_row_mean_is_zero_std_is_not_one(self):
        scaler = msreport.normalize.ZscoreScaler(with_mean=True, with_std=False)
        table = pd.DataFrame({"A": [10, 11], "B": [11, 12], "C": [12, 13]})
        table_scaled = scaler.transform(table)
        assert np.allclose(table_scaled.mean(axis=1), 0)
        assert not np.allclose(table_scaled.std(axis=1, ddof=0), 1)

    def test_after_transform_only_with_std_row_mean_is_not_zero_std_is_one(self):
        scaler = msreport.normalize.ZscoreScaler(with_mean=False, with_std=True)
        table = pd.DataFrame({"A": [10, 11], "B": [11, 12], "C": [12, 13]})
        table_scaled = scaler.transform(table)
        assert not np.allclose(table_scaled.mean(axis=1), 0)
        assert np.allclose(table_scaled.std(axis=1, ddof=0), 1)

    def test_nan_values_are_ignored_during_scaling(self):
        scaler = msreport.normalize.ZscoreScaler()
        table = pd.DataFrame(
            {"A": [11, np.nan], "B": [11, np.nan], "C": [12, 11], "D": [12, 12]}
        )
        table_scaled = scaler.transform(table)
        assert np.allclose(table_scaled.mean(axis=1), 0)
        assert np.allclose(table_scaled.std(axis=1, ddof=0), 1)
        # Check explicitly that NaN values are ignored and preserved
        assert np.allclose(
            table_scaled["A"].to_numpy(), np.array([-1.0, np.nan]), equal_nan=True
        )
        assert np.allclose(
            table_scaled["D"].to_numpy(), np.array([1.0, 1.0]), equal_nan=True
        )
