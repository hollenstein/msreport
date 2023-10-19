from attr import ib
import numpy as np
import pandas as pd
import pytest

import msreport.qtable
import msreport.analyze


@pytest.fixture
def example_data():
    design = pd.DataFrame(
        [
            ("Sample_A1", "Experiment_A", "1"),
            ("Sample_A2", "Experiment_A", "1"),
            ("Sample_B1", "Experiment_B", "1"),
            ("Sample_B2", "Experiment_B", "1"),
        ],
        columns=["Sample", "Experiment", "Replicate"],
    )
    data = pd.DataFrame(
        {
            "Total peptides": [2, 1, 2],
            "Representative protein": ["A", "B", "C"],
            "Intensity Sample_A1": [10, np.nan, 10.3],
            "Intensity Sample_A2": [10, np.nan, 10.3],
            "Intensity Sample_B1": [11, np.nan, np.nan],
            "Intensity Sample_B2": [15, np.nan, 10.3],
            "Mean Experiment_A": [10, np.nan, 10.3],  # <- Adjust to Sample_A1/A2
            "Mean Experiment_B": [13, np.nan, 10.3],  # <- Adjust to Sample_A1/A2
            "Ratio [log2]": [-3, np.nan, 0],  # <- Experiment_A/Experiment_B
            "Average expression": [11.5, np.nan, 10.3],  # <- Experiment_A/Experiment_B
            "iBAQ peptides": [2, -1, np.nan],
        }
    )
    missing_values = pd.DataFrame(
        {
            "Missing total": [0, 4, 1],
            "Missing Experiment_A": [0, 2, 0],
            "Missing Experiment_B": [0, 2, 1],
        }
    )

    example_data = {"data": data, "design": design, "missing_values": missing_values}
    return example_data


@pytest.fixture
def example_qtable(example_data):
    qtable = msreport.qtable.Qtable(example_data["data"], design=example_data["design"])
    qtable.set_expression_by_tag("Intensity")
    return qtable


class MockImputer:
    def fit(self, table: pd.DataFrame):
        return self

    def is_fitted(self):
        return True

    def transform(self, table: pd.DataFrame):
        _table = table.copy()
        for column in _table.columns:
            column_data = np.array(_table[column], dtype=float)
            column_data[~np.isfinite(column_data)] = 1
            _table[column] = column_data
        return _table


class MockNormalizer:
    def __init__(self):
        self._is_fitted = False
        self.shift = 0

    def fit(self, table: pd.DataFrame):
        self._is_fitted = True
        self.shift = np.nanmean(table)
        return self

    def is_fitted(self):
        return self._is_fitted

    def transform(self, table: pd.DataFrame):
        _table = table.copy()
        for column in _table.columns:
            column_data = np.array(_table[column], dtype=float)
            column_data[np.isfinite(column_data)] += self.shift
            _table[column] = column_data
        return _table


class TestValidateProteins:
    @pytest.fixture(autouse=True)
    def _init_qtable(self, example_qtable):
        self.qtable = example_qtable

    def test_valid_column_is_added(self):
        self.qtable.data = self.qtable.data.drop(columns="Valid")
        msreport.analyze.validate_proteins(self.qtable, remove_contaminants=False)
        data_columns = self.qtable.data.columns.to_list()
        assert "Valid" in data_columns

    @pytest.mark.parametrize(
        "min_peptides, expected_valid", [(0, 3), (1, 3), (2, 2), (3, 0)]
    )
    def test_validate_with_min_peptides(self, min_peptides, expected_valid):
        msreport.analyze.validate_proteins(
            self.qtable, remove_contaminants=False, min_peptides=min_peptides
        )
        assert expected_valid == self.qtable.data["Valid"].sum()


class TestNormalizeExpression:
    def test_normalization_with_fitted_normalizer(self, example_qtable):
        shift = 1
        normalizer = MockNormalizer().fit(example_qtable.make_expression_table())
        normalizer.shift = shift

        expr_before = example_qtable.make_expression_table()

        msreport.analyze.normalize_expression(
            example_qtable, normalizer, exclude_invalid=False
        )
        expr_after = example_qtable.make_expression_table()
        np.testing.assert_array_equal(expr_after, expr_before + shift)

    def test_correct_fitting_with_not_fitted_normalizer_and_exclude_invalid_false(
        self, example_qtable
    ):
        invalid_mask = example_qtable.data["Representative protein"] == "C"
        example_qtable.data.loc[invalid_mask, "Valid"] = False

        expr_table = example_qtable.make_expression_table()
        normalizer = MockNormalizer()
        msreport.analyze.normalize_expression(
            example_qtable, normalizer, exclude_invalid=False
        )

        assert normalizer.shift == np.nanmean(expr_table)

    def test_correct_fitting_with_not_fitted_normalizer_and_exclude_invalid_True(
        self, example_qtable
    ):
        invalid_mask = example_qtable.data["Representative protein"] == "C"
        example_qtable.data.loc[invalid_mask, "Valid"] = False

        expr_table = example_qtable.make_expression_table()
        normalizer = MockNormalizer()
        msreport.analyze.normalize_expression(
            example_qtable, normalizer, exclude_invalid=True
        )

        assert normalizer.shift == np.nanmean(expr_table[example_qtable["Valid"]])


class TestImputeMissingValues:
    @pytest.fixture(autouse=True)
    def _init_imputer(self, example_qtable):
        self.imputer = MockImputer()

    def test_all_entries_are_imputed_with_exclude_invalid_false(self, example_qtable):
        invalid_mask = example_qtable.data["Representative protein"] == "C"
        example_qtable.data.loc[invalid_mask, "Valid"] = False

        msreport.analyze.impute_missing_values(
            example_qtable, self.imputer, exclude_invalid=False
        )

        expr_table = example_qtable.make_expression_table()
        number_missing_values = expr_table.isna().sum().sum()
        assert number_missing_values == 0

    def test_valid_are_imputed_with_exclude_invalid_true(self, example_qtable):
        invalid_mask = example_qtable.data["Representative protein"] == "C"
        example_qtable.data.loc[invalid_mask, "Valid"] = False

        msreport.analyze.impute_missing_values(
            example_qtable, self.imputer, exclude_invalid=False
        )
        expr_table = example_qtable.make_expression_table(exclude_invalid=True)

        number_missing_values = expr_table.isna().sum().sum()
        assert number_missing_values == 0

    def test_invalid_are_not_imputed_with_exclude_invalid_true(self, example_qtable):
        invalid_mask = example_qtable.data["Representative protein"] == "C"
        example_qtable.data.loc[invalid_mask, "Valid"] = False

        table_before = example_qtable.make_expression_table(features=["Valid"])
        number_missing_values_of_invalid_before_imputation = (
            table_before[invalid_mask].isna().sum().sum()
        )
        assert number_missing_values_of_invalid_before_imputation > 0

        msreport.analyze.impute_missing_values(
            example_qtable, self.imputer, exclude_invalid=True
        )
        table_after = example_qtable.make_expression_table(features=["Valid"])

        expr_cols = table_before.columns.drop("Valid")
        invalid_before_imputation = table_before.loc[~table_before["Valid"], expr_cols]
        invalid_after_imputation = table_after.loc[~table_after["Valid"], expr_cols]
        assert invalid_after_imputation.equals(invalid_before_imputation)


def test_calculate_experiment_means(example_data, example_qtable):
    msreport.analyze.calculate_experiment_means(example_qtable)

    experiments = example_qtable.get_experiments()
    qtable_columns = example_qtable.data.columns.to_list()
    assert all([f"Expression {e}" in qtable_columns for e in experiments])
    assert all(
        [f"Expression {e}" in example_qtable._expression_features for e in experiments]
    )
    assert np.allclose(
        example_qtable.data["Expression Experiment_B"],
        example_data["data"]["Mean Experiment_B"],
        equal_nan=True,
    )


class TestCalculateMultiGroupComparison:
    def test_with_one_group(self, example_data, example_qtable):
        experiment_pairs = [("Experiment_A", "Experiment_B")]
        msreport.analyze.calculate_multi_group_comparison(
            example_qtable, experiment_pairs, exclude_invalid=False
        )

        exp1, exp2 = experiment_pairs[0]
        qtable_columns = example_qtable.data.columns.to_list()
        for column_tag in ["Average expression", "Ratio [log2]"]:
            assert f"{column_tag} {exp1} vs {exp2}" in qtable_columns
            assert np.allclose(
                example_qtable.data[f"{column_tag} {exp1} vs {exp2}"],
                example_data["data"][column_tag],
                equal_nan=True,
            )


def test_two_group_comparison(example_data, example_qtable):
    experiment_pair = ["Experiment_A", "Experiment_B"]
    exp1, exp2 = experiment_pair
    msreport.analyze.two_group_comparison(example_qtable, experiment_pair)

    qtable_columns = example_qtable.data.columns.to_list()
    for column_tag in ["Average expression", "Ratio [log2]"]:
        assert f"{column_tag} {exp1} vs {exp2}" in qtable_columns
        assert np.allclose(
            example_qtable.data[f"{column_tag} {exp1} vs {exp2}"],
            example_data["data"][column_tag],
            equal_nan=True,
        )


class TestCreateSiteToProteinNormalizer:
    def test_correct_index_set_in_fits_table(self, example_qtable):
        normalizer = msreport.analyze.create_site_to_protein_normalizer(example_qtable, category_column="Representative protein")  # fmt: skip
        assert normalizer.get_fits().index.name == "Representative protein"

    def test_correct_samples_present_in_fits_table(self, example_qtable):
        normalizer = msreport.analyze.create_site_to_protein_normalizer(example_qtable, category_column="Representative protein")  # fmt: skip
        assert sorted(normalizer.get_fits().columns) == sorted(example_qtable.get_samples())  # fmt: skip

    def test_only_categories_from_reference_table_are_in_fits(self, example_qtable):
        normalizer = msreport.analyze.create_site_to_protein_normalizer(example_qtable, category_column="Representative protein")  # fmt: skip
        assert normalizer.get_fits().index.isin(example_qtable["Representative protein"]).all()  # fmt: skip

    def test_fits_contains_rows(self, example_qtable):
        normalizer = msreport.analyze.create_site_to_protein_normalizer(example_qtable, category_column="Representative protein")  # fmt: skip
        assert normalizer.get_fits().shape[0] > 0


class TestCreateIbaqTransformer:
    @pytest.fixture(autouse=True)
    def _init_normalizer_and_qtable(self, example_qtable):
        self.qtable = example_qtable
        self.normalizer = msreport.analyze.create_ibaq_transformer(
            self.qtable,
            category_column="Representative protein",
            ibaq_column="iBAQ peptides",
        )

    def test_correct_index_set_in_fits_table(self):
        assert self.normalizer.get_fits().index.name == "Representative protein"

    def test_correct_samples_present_in_fits_table(self):
        assert sorted(self.normalizer.get_fits().columns) == sorted(self.qtable.get_samples())  # fmt: skip

    def test_only_categories_from_reference_table_are_in_fits(self):
        assert self.normalizer.get_fits().index.isin(self.qtable["Representative protein"]).all()  # fmt: skip

    def test_all_ibaq_counts_are_numbers_greater_or_equal_than_zero(self):
        assert np.isfinite(self.normalizer.get_fits().values).all()
        assert (self.normalizer.get_fits().values >= 0).all()

    def test_ibaq_counts_have_been_log2_transformed(self):
        ibaq_peptides = [1, 2, 4]
        self.qtable["iBAQ peptides"] = ibaq_peptides
        normalizer = msreport.analyze.create_ibaq_transformer(
            self.qtable, category_column="Representative protein", ibaq_column="iBAQ peptides",  # fmt: skip
        )
        for sample in self.qtable.get_samples():
            assert np.allclose(normalizer.get_fits()[sample], np.log2(ibaq_peptides))


class TestNormalizeExpressionByCategory:
    @pytest.fixture(autouse=True)
    def _init_normalizer(self):
        class MockCategoricalNormalizer:
            def fit(self, _table: pd.DataFrame):
                return self

            def is_fitted(self):
                return True

            def transform(self, table: pd.DataFrame):
                table = table.copy()
                table.loc[:, :] = 0
                return table

            def get_category_column(self):
                return "Representative protein"

        self.normalizer = MockCategoricalNormalizer()

    def test_raises_key_error_when_category_column_is_absent(self, example_qtable):
        example_qtable.data.rename(columns={"Representative protein": "Absent column"}, inplace=True)  # fmt: skip
        with pytest.raises(KeyError):
            msreport.analyze.normalize_expression_by_category(example_qtable, self.normalizer)  # fmt: skip

    def test_expression_values_have_been_set_to_zero_by_mock_normalizer(self, example_qtable):  # fmt: skip
        msreport.analyze.normalize_expression_by_category(example_qtable, self.normalizer)  # fmt: skip
        assert example_qtable.make_expression_table().eq(0).values.all()
