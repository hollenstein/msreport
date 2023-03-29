from typing import Callable, Iterable, Union

import numpy as np
import pandas as pd


def unique_values(
    table: pd.DataFrame, group_by: str, column: str, is_sorted: bool = False
) -> pd.DataFrame:
    """TODO"""
    raise NotImplementedError
    aggregation, groups = _apply_aggregation_to_unique_groups(
        table, group_by, column, _aggfunc_join_unique_str, is_sorted
    )
    data = {group_by: groups, column: aggregation}
    return pd.DataFrame(data).set_index(group_by)


def _aggfunc_join_unique_str(array: np.ndarray, sep: str = ";") -> str:
    """Returns a joined string of unique sorted values from the array."""
    return sep.join(sorted([str(i) for i in set(array.flatten())]))


def _aggfunc_join_unique_str_per_column(
    array: np.ndarray, sep: str = ";"
) -> np.ndarray:
    """Returns for each column a joined strings of unique sorted values."""
    return np.array([_aggfunc_join_unique_str(i) for i in array.transpose()])


def _apply_aggregation_to_unique_groups(
    table: pd.DataFrame,
    group_by: str,
    aggregate_columns: Union[str, Iterable],
    aggregate_function: Callable,
    is_sorted: bool,
) -> np.ndarray:
    """Applies an aggregation function to unique groups from a table.

    Args:
        table: Dataframe used for generating groups, which will be aggregated.
        group_by: Column used to determine unique groups.
        aggregate_columns: Columns or list of columns, which will be passed to the
            aggregation function.
        aggregate_function: Function that is applied to group values for generating the
            aggregation result. If multiple columns are specified for aggregation,
            the input array for the aggregate_function is two dimensional, with the
            first dimension corresponding to rows and the second to the column. E.g. an
            array with 3 rows and 2 columns: np.array([[1, 'a'], [2, 'b'], [3, 'c']])
        is_sorted: If True the table is expected to be sorted with 'group_by'.

    Returns:
        An array containing the aggregated values and the corresponding list of unique
        group_names. The length of the second array dimension is equal to the number of
        specified columns.
    """
    group_start_indices, group_names, table = _prepare_grouping_indices(
        table, group_by, is_sorted
    )
    array = table[aggregate_columns].to_numpy()
    aggregation_result = np.array(
        [aggregate_function(i) for i in np.split(array, group_start_indices[1:])]
    )
    return aggregation_result, group_names


def _prepare_grouping_indices(
    table: pd.DataFrame, group_by: str, is_sorted: bool
) -> (pd.DataFrame, Iterable[str], Iterable[int], Iterable[int]):
    """Prepares group start indices and names from a sorted dataframe.

    Args:
        table: Dataframe used for generating groups.
        group_by: Column used to determine unique groups.
        is_sorted: If True the table is expected to be already sorted with 'group_by'.

    Returns:
        table, group_names, group_start_indices
    """
    if not is_sorted:
        table = table.sort_values(by=group_by)
    group_names, group_start_indices, group_lengths = np.unique(
        table[group_by], return_counts=True, return_index=True
    )
    return group_start_indices, group_names, table
