import numpy as np
import pandas as pd
import pytest

import msreport.aggregate.summarize as SUMMARIZE


@pytest.fixture
def example_data():
    table = pd.DataFrame(
        {
            "ID": ["A", "A", "B", "C", "C", "C"],
            "Peptide sequence": ["AAA", "AAA", "BBB", "CCA", "CCB", "CCB"],
            "Quant S1": [1, 1, 1, 1, 1, 2.5],
            "Quant S2": [2, 2, 2, 2, 2, 2],
        }
    )
    example_data = {"table": table, "groub by": "ID", "samples": ["S1", "S2"]}
    return example_data


class TestCountUnique:
    @pytest.fixture(autouse=True)
    def _init_data(self, example_data):
        self.table = example_data["table"]
        self.group_by = example_data["groub by"]

    def test_count_unique_peptides(self):
        unique_peptides = SUMMARIZE.count_unique(
            self.table,
            self.group_by,
            input_column="Peptide sequence",
            output_column="Unique",
            is_sorted=False,
        )
        expected_result = pd.DataFrame(data={"Unique": [1, 1, 2]}, index=["A", "B", "C"])  # fmt: skip
        pd.testing.assert_frame_equal(unique_peptides, expected_result, check_dtype=False)  # fmt: skip


class TestJoinUnique:
    @pytest.fixture(autouse=True)
    def _init_data(self, example_data):
        self.table = example_data["table"]
        self.group_by = example_data["groub by"]

    def test_join_unique(self):
        sequences = SUMMARIZE.join_unique(
            self.table,
            self.group_by,
            input_column="Peptide sequence",
            output_column="Unique",
            is_sorted=False,
        )
        expected_result = pd.DataFrame(data={"Unique": ["AAA", "BBB", "CCA;CCB"]}, index=["A", "B", "C"])  # fmt: skip
        pd.testing.assert_frame_equal(sequences, expected_result, check_dtype=False)


class TestSumColumns:
    @pytest.fixture(autouse=True)
    def _init_data(self, example_data):
        self.table = example_data["table"]
        self.group_by = example_data["groub by"]
        self.samples = example_data["samples"]

    def test_correct_calculation_without_output_tag(self):
        summed_columns = SUMMARIZE.sum_columns(
            self.table,
            self.group_by,
            self.samples,
            input_tag="Quant",
            is_sorted=False,
        )
        expected_result = pd.DataFrame(
            data={"Quant S1": [2, 1, 4.5], "Quant S2": [4, 2, 6]}, index=["A", "B", "C"]
        )
        pd.testing.assert_frame_equal(summed_columns, expected_result, check_dtype=False)  # fmt: skip

    def test_with_output_tag(self):
        summed_columns = SUMMARIZE.sum_columns(
            self.table,
            self.group_by,
            self.samples,
            input_tag="Quant",
            output_tag="OUT",
            is_sorted=False,
        )
        assert summed_columns.columns.tolist() == ["OUT S1", "OUT S2"]


class TestSumColumnsMaxlfq:
    @pytest.fixture(autouse=True)
    def _init_data(self, example_data):
        self.table = example_data["table"]
        self.group_by = example_data["groub by"]
        self.samples = example_data["samples"]

    def test_correct_calculation_without_output_tag(self):
        summed_columns = SUMMARIZE.sum_columns_maxlfq(
            self.table,
            self.group_by,
            self.samples,
            input_tag="Quant",
            is_sorted=False,
        )
        expected_result = pd.DataFrame(
            data={"Quant S1": [2, 1, 3.5], "Quant S2": [4, 2, 7]}, index=["A", "B", "C"]
        )
        pd.testing.assert_frame_equal(summed_columns, expected_result, check_dtype=False)  # fmt: skip

    def test_with_output_tag(self):
        summed_columns = SUMMARIZE.sum_columns_maxlfq(
            self.table,
            self.group_by,
            self.samples,
            input_tag="Quant",
            output_tag="OUT",
            is_sorted=False,
        )
        assert summed_columns.columns.tolist() == ["OUT S1", "OUT S2"]


class TestApplyAggregationToUniqueGroups:
    @pytest.fixture(autouse=True)
    def _init_table(self):
        self.table = pd.DataFrame(
            {
                "ID": ["A", "A", "B", "C", "C", "C"],
                "column1": [1, 1, 1, 1, 1, 1],
                "column2": [2, 2, 2, 2, 2, 2],
            }
        )
        self.column1_group_sums = np.array([2, 1, 3])
        self.column2_group_sums = np.array([4, 2, 6])
        self.both_column_group_sums = np.array(
            [self.column1_group_sums, self.column2_group_sums]
        ).transpose()
        self.group_by = "ID"

    def test_aggregation_with_single_column(self):
        aggregation, groups = SUMMARIZE.aggregate_unique_groups(
            table=self.table,
            group_by=self.group_by,
            columns_to_aggregate="column1",
            condenser=np.sum,
            is_sorted=True,
        )
        np.testing.assert_array_equal(self.column1_group_sums, aggregation)

    def test_aggregation_with_two_columns(self):
        aggregation, groups = SUMMARIZE.aggregate_unique_groups(
            table=self.table,
            group_by=self.group_by,
            columns_to_aggregate=["column1", "column2"],
            condenser=lambda x: np.sum(x, axis=0),
            is_sorted=True,
        )
        np.testing.assert_array_equal(self.both_column_group_sums, aggregation)


class TestPrepareGroupingIndices:
    @pytest.fixture(autouse=True)
    def _init_table(self):
        self.sorted_values = np.array(["A", "A", "B", "B", "B", "C", "D"])
        self.scrambled_values = self.sorted_values[[2, 1, 5, 0, 3, 4, 6]]

        self.sorted_table = pd.DataFrame({"ID": self.sorted_values})
        self.scrambled_table = pd.DataFrame({"ID": self.scrambled_values})

        self.group_by = "ID"
        self.expected_indices = np.array([0, 2, 5, 6])

    def test_correct_indices_with_is_sorted_true(self):
        start_indices, groups, table = SUMMARIZE._prepare_grouping_indices(
            self.sorted_table, self.group_by, is_sorted=True
        )
        np.testing.assert_array_equal(start_indices, self.expected_indices)

    def test_correct_group_names_with_is_sorted_true(self):
        start_indices, groups, table = SUMMARIZE._prepare_grouping_indices(
            self.sorted_table, self.group_by, is_sorted=True
        )
        np.testing.assert_array_equal(groups, np.unique(self.sorted_values))

    def test_table_sorted_properly_with_is_sorted_false(self):
        start_indices, groups, table = SUMMARIZE._prepare_grouping_indices(
            self.scrambled_table, self.group_by, is_sorted=False
        )
        np.testing.assert_array_equal(table, self.sorted_table)
