import numpy as np
import pandas as pd
import pytest

import msreport.aggregate


class TestUniqueValues:
    @pytest.fixture(autouse=True)
    def _init_table(self):
        self.table = pd.DataFrame(
            {
                "ID": ["A", "A", "B", "C", "C", "C"],
                "column1": [1, 2, 3, 1, 1, 1],
                "column2": [2, 2, 2, 2, 2, 2],
            }
        )

    def test_unique_values(self):
        # msreport.aggregate.unique_values(table, group_by, column, is_sorted)
        import warnings

        warnings.warn(NotImplementedError)


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
        aggregation, groups = msreport.aggregate._apply_aggregation_to_unique_groups(
            table=self.table,
            group_by=self.group_by,
            aggregate_columns="column1",
            aggregate_function=np.sum,
            is_sorted=True,
        )
        np.testing.assert_array_equal(self.column1_group_sums, aggregation)

    def test_aggregation_with_two_columns(self):
        aggregation, groups = msreport.aggregate._apply_aggregation_to_unique_groups(
            table=self.table,
            group_by=self.group_by,
            aggregate_columns=["column1", "column2"],
            aggregate_function=lambda x: np.sum(x, axis=0),
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
        start_indices, groups, table = msreport.aggregate._prepare_grouping_indices(
            self.sorted_table, self.group_by, is_sorted=True
        )
        np.testing.assert_array_equal(start_indices, self.expected_indices)

    def test_correct_group_names_with_is_sorted_true(self):
        start_indices, groups, table = msreport.aggregate._prepare_grouping_indices(
            self.sorted_table, self.group_by, is_sorted=True
        )
        np.testing.assert_array_equal(groups, np.unique(self.sorted_values))

    def test_table_sorted_properly_with_is_sorted_false(self):
        start_indices, groups, table = msreport.aggregate._prepare_grouping_indices(
            self.scrambled_table, self.group_by, is_sorted=False
        )
        np.testing.assert_array_equal(table, self.sorted_table)


class TestAggfuncJoinStr:
    @pytest.mark.parametrize(
        "array, expected_result",
        [
            (np.array(["a", "b", "b"]), "a;b;b"),  # Strings
            (np.array([1, 3, 2]), "1;2;3"),  # Integers
            (np.array([2, 1, 2]), "1;2;2"),  # Integers
            (np.array([1, "a", 2]), "1;2;a"),  # Strings and integers mixed
            (np.array([1, "a", np.nan]), "1;a;nan"),  # With np.nan
        ],
    )
    def test_one_column_with_default_separator(self, array, expected_result):
        result = msreport.aggregate._aggfunc_join_str(array)
        assert result == expected_result

    def test_with_specified_separator(self):
        array = np.array(["a", "b", "b"])
        expected_result = "a:::b:::b"
        result = msreport.aggregate._aggfunc_join_str(array, sep=":::")
        assert result == expected_result

    @pytest.mark.parametrize(
        "array, expected_result",
        [
            (np.array([["a", "b"], ["c", "d"], ["a", "b"]]), "a;a;b;b;c;d"),
            (np.array([[1, 2], ["a", "b"]]), "1;2;a;b"),
        ],
    )
    def test_with_multiple_columns(self, array, expected_result):
        result = msreport.aggregate._aggfunc_join_str(array)
        assert result == expected_result


class TestAggfuncJoinStrPerColumn:
    @pytest.mark.parametrize(
        "array, expected_result",
        [
            (
                np.array([["a", "A"], ["b", "B"], ["c", "C"]]),
                np.array(["a;b;c", "A;B;C"]),
            ),
            (
                np.array([["a", "A", 1], ["b", "B", 2]]),
                np.array(["a;b", "A;B", "1;2"]),
            ),
            (
                np.array([["a"]]),
                np.array(["a"]),
            ),
            (
                np.array(["a"]),
                np.array(["a"]),
            ),
            (
                np.array([["a", "A"], ["a", 1], ["b", 1]]),
                np.array(["a;a;b", "1;1;A"]),
            ),
        ],
    )
    def test_various_inputs(self, array, expected_result):
        result = msreport.aggregate._aggfunc_join_str_per_column(array)
        np.testing.assert_array_equal(result, expected_result)


class TestAggfuncJoinUniqueStr:
    @pytest.mark.parametrize(
        "array, expected_result",
        [
            (np.array(["a", "b", "c"]), "a;b;c"),  # Strings
            (np.array(["c", "a", "b"]), "a;b;c"),  # Sort outcome
            (np.array(["a", "a", "b"]), "a;b"),  # With repeats
            (np.array([1, 3, 2]), "1;2;3"),  # Integers
            (np.array([1, "a", 2]), "1;2;a"),  # Strings and integers mixed
            (np.array([1, "a", np.nan]), "1;a;nan"),  # With np.nan
        ],
    )
    def test_one_column_with_default_separator(self, array, expected_result):
        result = msreport.aggregate._aggfunc_join_unique_str(array)
        assert result == expected_result

    def test_with_specified_separator(self):
        array = np.array(["a", "b", "c"])
        expected_result = "a:::b:::c"
        result = msreport.aggregate._aggfunc_join_unique_str(array, sep=":::")
        assert result == expected_result

    @pytest.mark.parametrize(
        "array, expected_result",
        [
            (np.array([["a", "b"], ["c", "d"], ["e", "f"]]), "a;b;c;d;e;f"),
            (np.array([["a", "b"], ["a", "c"]]), "a;b;c"),
            (np.array([[1, 2], ["a", "b"]]), "1;2;a;b"),
        ],
    )
    def test_with_multiple_columns(self, array, expected_result):
        result = msreport.aggregate._aggfunc_join_unique_str(array)
        assert result == expected_result


class TestAggfuncJoinUniqueStrPerColumn:
    @pytest.mark.parametrize(
        "array, expected_result",
        [
            (
                np.array([["a", "A"], ["b", "B"], ["c", "C"]]),
                np.array(["a;b;c", "A;B;C"]),
            ),
            (
                np.array([["a", "A", 1], ["a", "B", 2]]),
                np.array(["a", "A;B", "1;2"]),
            ),
            (
                np.array([["a"]]),
                np.array(["a"]),
            ),
            (
                np.array(["a"]),
                np.array(["a"]),
            ),
            (
                np.array([["a", "A"], ["a", 1], ["b", 1]]),
                np.array(["a;b", "1;A"]),
            ),
        ],
    )
    def test_various_inputs(self, array, expected_result):
        result = msreport.aggregate._aggfunc_join_unique_str_per_column(array)
        np.testing.assert_array_equal(result, expected_result)


class TestAggfuncSum:
    @pytest.mark.parametrize(
        "array, expected_result",
        [
            (np.array([1, 2, 3]), 6),
            (np.array([-1, 0, 1]), 0),
            (np.array([np.nan, np.nan, np.nan]), np.nan),
            (np.array([np.nan, 1, 2]), 3),
        ],
    )
    def test_with_single_column(self, array, expected_result):
        result = msreport.aggregate._aggfunc_sum(array)
        np.testing.assert_equal(result, expected_result)

    @pytest.mark.parametrize(
        "array, expected_result",
        [
            (np.array([[1, 10], [2, 20]]), 33),
            (np.array([[1, 10], [2, np.nan]]), 13),
            (np.array([[1, np.nan], [2, np.nan]]), 3),
            (np.array([[np.nan, np.nan], [np.nan, np.nan]]), np.nan),
        ],
    )
    def test_with_multiple_columns(self, array, expected_result):
        result = msreport.aggregate._aggfunc_sum(array)
        np.testing.assert_equal(result, expected_result)


class TestAggfuncSumPerColumn:
    @pytest.mark.parametrize(
        "array, expected_result",
        [
            (np.array([[1, 10], [2, 20]]), np.array([3, 30])),
            (np.array([[1, 10], [2, np.nan]]), np.array([3, 10])),
            (np.array([[1, np.nan], [2, np.nan]]), np.array([3, np.nan])),
            (np.array([[np.nan, np.nan], [np.nan, np.nan]]), np.array([np.nan, np.nan])),  # fmt: skip
        ],
    )
    def test_with_multiple_columns(self, array, expected_result):
        result = msreport.aggregate._aggfunc_sum_per_column(array)
        np.testing.assert_equal(result, expected_result)


class TestAggfuncMax:
    @pytest.mark.parametrize(
        "array, expected_result",
        [
            (np.array([1, 2, 3]), 3),
            (np.array([-1, 0, 1]), 1),
            (np.array([np.nan, np.nan, np.nan]), np.nan),
            (np.array([np.nan, 1, 2]), 2),
        ],
    )
    def test_with_single_column(self, array, expected_result):
        result = msreport.aggregate._aggfunc_max(array)
        np.testing.assert_equal(result, expected_result)

    @pytest.mark.parametrize(
        "array, expected_result",
        [
            (np.array([[1, 10], [2, 20]]), 20),
            (np.array([[1, 10], [2, np.nan]]), 10),
            (np.array([[1, np.nan], [2, np.nan]]), 2),
            (np.array([[np.nan, np.nan], [np.nan, np.nan]]), np.nan),
        ],
    )
    def test_with_multiple_columns(self, array, expected_result):
        result = msreport.aggregate._aggfunc_max(array)
        np.testing.assert_equal(result, expected_result)


class TestAggfuncMaxPerColumn:
    @pytest.mark.parametrize(
        "array, expected_result",
        [
            (np.array([[1, 10], [2, 20]]), np.array([2, 20])),
            (np.array([[1, 10], [2, np.nan]]), np.array([2, 10])),
            (np.array([[1, np.nan], [2, np.nan]]), np.array([2, np.nan])),
            (np.array([[np.nan, np.nan], [np.nan, np.nan]]), np.array([np.nan, np.nan])),  # fmt: skip
        ],
    )
    def test_with_multiple_columns(self, array, expected_result):
        result = msreport.aggregate._aggfunc_max_per_column(array)
        np.testing.assert_equal(result, expected_result)


class TestAggfuncMin:
    @pytest.mark.parametrize(
        "array, expected_result",
        [
            (np.array([1, 2, 3]), 1),
            (np.array([-1, 0, 1]), -1),
            (np.array([np.nan, np.nan, np.nan]), np.nan),
            (np.array([np.nan, 1, 2]), 1),
        ],
    )
    def test_with_single_column(self, array, expected_result):
        result = msreport.aggregate._aggfunc_min(array)
        np.testing.assert_equal(result, expected_result)

    @pytest.mark.parametrize(
        "array, expected_result",
        [
            (np.array([[1, 10], [2, 20]]), 1),
            (np.array([[1, 10], [2, np.nan]]), 1),
            (np.array([[1, np.nan], [2, np.nan]]), 1),
            (np.array([[np.nan, np.nan], [np.nan, np.nan]]), np.nan),
        ],
    )
    def test_with_multiple_columns(self, array, expected_result):
        result = msreport.aggregate._aggfunc_min(array)
        np.testing.assert_equal(result, expected_result)


class TestAggfuncMinPerColumn:
    @pytest.mark.parametrize(
        "array, expected_result",
        [
            (np.array([[1, 10], [2, 20]]), np.array([1, 10])),
            (np.array([[1, 10], [2, np.nan]]), np.array([1, 10])),
            (np.array([[1, np.nan], [2, np.nan]]), np.array([1, np.nan])),
            (np.array([[np.nan, np.nan], [np.nan, np.nan]]), np.array([np.nan, np.nan])),  # fmt: skip
        ],
    )
    def test_with_multiple_columns(self, array, expected_result):
        result = msreport.aggregate._aggfunc_min_per_column(array)
        np.testing.assert_equal(result, expected_result)


class TestAggfuncCountUnique:
    @pytest.mark.parametrize(
        "array, expected_result",
        [
            (np.array(["a", "b", "c"]), 3),
            (np.array(["a", "a", "b"]), 2),
            (np.array(["a", "a", "a"]), 1),
            (np.array(["a", "b", ""]), 2),
            (np.array([1, 2, 3]), 3),
            (np.array([1, 2, np.nan]), 2),
            (np.array([np.nan, np.nan, np.nan]), 0),
            (np.array([]), 0),
        ],
    )
    def test_with_single_column(self, array, expected_result):
        result = msreport.aggregate._aggfunc_count_unique(array)
        np.testing.assert_equal(result, expected_result)

    @pytest.mark.parametrize(
        "array, expected_result",
        [
            (np.array([["a", "A"], ["b", "B"]]), 4),
            (np.array([["a", "A"], ["a", "A"]]), 2),
            (np.array([["a", "a"], ["a", "A"]]), 2),
            (np.array([[], []]), 0),
        ],
    )
    def test_with_multiple_columns(self, array, expected_result):
        result = msreport.aggregate._aggfunc_count_unique(array)
        np.testing.assert_equal(result, expected_result)


class TestAggfuncCountUniquePerColumn:
    @pytest.mark.parametrize(
        "array, expected_result",
        [
            (np.array([["a", "A"], ["b", "B"]]), np.array([2, 2])),
            (np.array([["a", "A"], ["a", "A"]]), np.array([1, 1])),
            (np.array([["a", "A"], ["a", "B"]]), np.array([1, 2])),
            (np.array([[], []]), np.array([0, 0])),
        ],
    )
    def test_with_multiple_columns(self, array, expected_result):
        result = msreport.aggregate._aggfunc_count_unique_per_column(array)
        np.testing.assert_equal(result, expected_result)
