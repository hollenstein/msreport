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


class TestImputeMissingValues:
    def test_all_entries_are_imputed_with_exclude_invalid_false(self, example_qtable):
        invalid_mask = example_qtable.data["Representative protein"] == "C"
        example_qtable.data.loc[invalid_mask, "Valid"] = False

        msreport.analyze.impute_missing_values(example_qtable, exclude_invalid=False)

        expr_table = example_qtable.make_expression_table()
        number_missing_values = expr_table.isna().sum().sum()
        assert number_missing_values == 0

    def test_valid_are_imputed_with_exclude_invalid_true(self, example_qtable):
        invalid_mask = example_qtable.data["Representative protein"] == "C"
        example_qtable.data.loc[invalid_mask, "Valid"] = False

        msreport.analyze.impute_missing_values(example_qtable, exclude_invalid=False)
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

        msreport.analyze.impute_missing_values(example_qtable, exclude_invalid=True)
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
