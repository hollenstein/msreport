import numpy as np
import pandas as pd
import pytest

import msreport.aggregate.summarize as SUMMARIZE


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
            aggregate_columns="column1",
            condenser=np.sum,
            is_sorted=True,
        )
        np.testing.assert_array_equal(self.column1_group_sums, aggregation)

    def test_aggregation_with_two_columns(self):
        aggregation, groups = SUMMARIZE.aggregate_unique_groups(
            table=self.table,
            group_by=self.group_by,
            aggregate_columns=["column1", "column2"],
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
