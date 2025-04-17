import numpy as np
import pandas as pd
import pytest

import msreport.plot
import msreport.qtable


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
            "Representative protein": ["A", "B", "C"],
            "Intensity Sample_A1": [10, 11, 10.3],
            "Intensity Sample_A2": [10, np.nan, 10.3],
            "Intensity Sample_B1": [11, 11, np.nan],
            "Intensity Sample_B2": [15, np.nan, 10.3],
            "Expression Experiment_A": [10, 11, 10.3],  # <- Adjust to Sample_A1/A2
            "Expression Experiment_B": [13, 11, 10.3],  # <- Adjust to Sample_A1/A2
            "Ratio [log2] Experiment_A vs Experiment_B": [-3, np.nan, 0],
            "P-value Experiment_A vs Experiment_B": [0.0001, np.nan, 0.1],
            "Average expression Experiment_A vs Experiment_B": [11.5, 11, 10.3],
            "Valid": [True, False, True],
        }
    )
    missing_values = pd.DataFrame(
        {
            "Missing total": [0, 4, 1],
            "Missing Experiment_A": [0, 2, 0],
            "Missing Experiment_B": [0, 2, 1],
            "Events total": [4, 0, 3],
            "Events Experiment_A": [2, 0, 2],
            "Events Experiment_B": [2, 0, 1],
        }
    )
    data = data.join(missing_values)
    example_data = {"data": data, "design": design}
    return example_data


@pytest.fixture
def example_qtable(example_data):
    qtable = msreport.qtable.Qtable(example_data["data"], design=example_data["design"])
    qtable.set_expression_by_tag("Intensity")
    return qtable


class TestVolcanoMa:
    @pytest.fixture(autouse=True)
    def _init_qtable(self, example_qtable):
        self.qtable = example_qtable

    def test_no_error_without_missing_values_due_to_exclude_invalid(self):
        fig, axes = msreport.plot.volcano_ma(
            self.qtable,
            ["Experiment_A", "Experiment_B"],
            comparison_tag=" vs ",
            pvalue_tag="P-value",
            special_proteins=["A", "B", "C"],
            exclude_invalid=True,
        )

    def test_no_error_with_missing_values_but_no_special_protein_labeling(self):
        fig, axes = msreport.plot.volcano_ma(
            self.qtable,
            ["Experiment_A", "Experiment_B"],
            comparison_tag=" vs ",
            pvalue_tag="P-value",
            exclude_invalid=False,
        )

    def test_no_error_with_missing_values_of_special_proteins(self):
        fig, axes = msreport.plot.volcano_ma(
            self.qtable,
            ["Experiment_A", "Experiment_B"],
            comparison_tag=" vs ",
            pvalue_tag="P-value",
            special_proteins=["A", "B", "C"],
            exclude_invalid=False,
        )

    def test_no_error_when_annotation_column_does_not_exist(self):
        fig, axes = msreport.plot.volcano_ma(
            self.qtable,
            ["Experiment_A", "Experiment_B"],
            comparison_tag=" vs ",
            pvalue_tag="P-value",
            special_proteins=["A", "B", "C"],
            exclude_invalid=False,
            annotation_column="Nonexistent_Column",
        )

    def test_qtable_id_column_is_used_for_matching_special_proteins(self, example_data):
        example_data["data"].rename(columns={"Representative protein": "ID"}, inplace=True)  # fmt: skip
        qtable = msreport.qtable.Qtable(example_data["data"], design=example_data["design"], id_column="ID")  # fmt: skip
        fig, axes = msreport.plot.volcano_ma(
            qtable,
            ["Experiment_A", "Experiment_B"],
            comparison_tag=" vs ",
            pvalue_tag="P-value",
            special_proteins=["A", "B", "C"],
            exclude_invalid=False,
        )


class TestExpressionComparison:
    @pytest.fixture(autouse=True)
    def _init_qtable(self, example_qtable):
        self.qtable = example_qtable

    def test_no_error_without_missing_values_due_to_exclude_invalid(self):
        fig, axes = msreport.plot.expression_comparison(
            self.qtable,
            ["Experiment_A", "Experiment_B"],
            comparison_tag=" vs ",
            special_proteins=["A", "B", "C"],
            exclude_invalid=True,
        )

    def test_no_error_with_missing_values_but_no_special_protein_labeling(self):
        fig, axes = msreport.plot.expression_comparison(
            self.qtable,
            ["Experiment_A", "Experiment_B"],
            comparison_tag=" vs ",
            exclude_invalid=False,
        )

    def test_no_error_with_missing_values_of_special_proteins(self):
        fig, axes = msreport.plot.expression_comparison(
            self.qtable,
            ["Experiment_A", "Experiment_B"],
            comparison_tag=" vs ",
            special_proteins=["A", "B", "C"],
            exclude_invalid=False,
        )
