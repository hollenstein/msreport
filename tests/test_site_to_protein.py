import numpy as np
import pandas as pd
import pytest

from msreport.normalize import CategoricalNormalizer
from msreport.errors import NotFittedError


@pytest.fixture(autouse=True)
def reference_table():
    reference_table = pd.DataFrame(
        {
            "Category": ["A", "B", "C"],
            "Sample 1": [1.0, 0.0, 0.0],
            "Sample 2": [-1.0, 0.0, 10.0],
        },
    )
    return reference_table


@pytest.fixture(autouse=True)
def sample_table():
    sample_table = pd.DataFrame(
        {
            "Category": ["A", "C", "A", "A"],
            "Sample 1": [0.0, 0.0, 11.0, 0.0],
            "Sample 2": [0.0, 0.0, 9.0, 0.0],
        },
    )
    return sample_table


@pytest.fixture(autouse=True)
def transformed_sample_table():
    sample_table = pd.DataFrame(
        {
            "Category": ["A", "C", "A", "A"],
            "Sample 1": [-1.0, 0.0, 10.0, -1.0],
            "Sample 2": [1.0, -10.0, 10.0, 1.0],
        },
    )
    return sample_table


class TestFittingOfNormalizer:
    def test_is_not_fitted_when_created(self):
        normalizer = CategoricalNormalizer("Category")
        assert not normalizer.is_fitted()

    def test_is_fitted_after_fitting(self, reference_table):
        normalizer = CategoricalNormalizer("Category")
        normalizer.fit(reference_table)
        assert normalizer.is_fitted()

    def test_fitting_calculates_correct_fits(self, reference_table):
        normalizer = CategoricalNormalizer("Category")
        normalizer.fit(reference_table)
        pd.testing.assert_frame_equal(
            normalizer.get_fits(), reference_table.set_index("Category")
        )

    def test_fitting_raises_key_error_when_input_table_does_not_contain_the_category_column(self, reference_table):  # fmt: skip
        normalizer = CategoricalNormalizer("Not present column")
        with pytest.raises(KeyError):
            normalizer.fit(reference_table)

    def test_fitting_raises_value_error_when_input_table_contains_nan(self, reference_table):  # fmt: skip
        reference_table.at[1, "Sample 1"] = np.nan
        normalizer = CategoricalNormalizer("Category")
        with pytest.raises(ValueError):
            normalizer.fit(reference_table)


class TestNormalizerTransform:
    def test_transform_on_unfitted_normalizer_raises_NotFittedError(self, reference_table):  # fmt: skip
        normalizer = CategoricalNormalizer("Category")
        with pytest.raises(NotFittedError):
            normalizer.transform(reference_table)

    def test_transform_of_table_with_columns_not_present_in_fits_raises_key_error(self, reference_table):  # fmt: skip
        normalizer = CategoricalNormalizer("Category").fit(reference_table)

        table_to_transform = reference_table.rename(columns={reference_table.columns[-1]: "Test"})  # fmt: skip
        with pytest.raises(KeyError):
            normalizer.transform(table_to_transform)

    def test_transform_on_reference_table_does_not_throw_an_error(self, reference_table):  # fmt: skip
        normalizer = CategoricalNormalizer("Category").fit(reference_table)
        normalizer.transform(reference_table)

    def test_transform_does_not_modify_input_table(self, reference_table):
        copy_of_reference_table = reference_table.copy()
        normalizer = CategoricalNormalizer("Category").fit(reference_table)
        normalizer.transform(reference_table)
        pd.testing.assert_frame_equal(copy_of_reference_table, reference_table)

    def test_transform_on_reference_table_sets_all_values_to_zero(self, reference_table):  # fmt: skip
        normalizer = CategoricalNormalizer("Category").fit(reference_table)

        transformed_table = normalizer.transform(reference_table)
        sample_columns = [col for col in reference_table.columns if col != "Category"]
        assert (transformed_table[sample_columns] != 0).values.sum() == 0

    def test_transform_on_reference_table_results_in_equal_values_between_samples(self, reference_table):  # fmt: skip
        normalizer = CategoricalNormalizer("Category").fit(reference_table)
        transformed_table = normalizer.transform(reference_table)

        sample_columns = [col for col in reference_table.columns if col != "Category"]
        np.testing.assert_allclose(
            transformed_table[sample_columns[0]], transformed_table[sample_columns[1]]
        )

    def test_transform_correctly_applied_to_sample_table(self, reference_table, sample_table, transformed_sample_table):  # fmt: skip
        # This also tests that the transform is correctly applied to a table with only a subset of the categories
        normalizer = CategoricalNormalizer("Category").fit(reference_table)

        transformed_table = normalizer.transform(sample_table)
        pd.testing.assert_frame_equal(transformed_table, transformed_sample_table)

    def test_transform_correctly_applied_to_table_with_subset_of_the_fitted_columns(self, reference_table, sample_table, transformed_sample_table):  # fmt: skip
        normalizer = CategoricalNormalizer("Category").fit(reference_table)

        sample_table.drop(columns="Sample 2", inplace=True)
        transformed_sample_table.drop(columns="Sample 2", inplace=True)

        transformed_table = normalizer.transform(sample_table)
        pd.testing.assert_frame_equal(transformed_table, transformed_sample_table)
