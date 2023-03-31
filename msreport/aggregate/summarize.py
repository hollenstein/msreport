from typing import Callable, Iterable, Union

import numpy as np
import pandas as pd


def aggregate_unique_groups(
    table: pd.DataFrame,
    group_by: str,
    aggregate_columns: Union[str, Iterable],
    condenser: Callable,
    is_sorted: bool,
) -> np.ndarray:
    """Aggregates a table by applying a condenser function to unique groups.

    The function returns two arrays containing the aggregated values and the
    corresponding group names. This function can be used to summarize data from an ion
    table to a peptide, protein or modification table. Suitable condenser functions
    can be found in the module msreport.aggregate.condense

    Args:
        table: Dataframe used for generating groups, which will be aggregated.
        group_by: Column used to determine unique groups.
        aggregate_columns: Columns or list of columns, which will be passed to the
            condenser function.
        condenser: Function that is applied to each group for generating the
            aggregation result. If multiple columns are specified for aggregation,
            the input array for the condenser function is two dimensional, with the
            first dimension corresponding to rows and the second to the column. E.g. an
            array with 3 rows and 2 columns: np.array([[1, 'a'], [2, 'b'], [3, 'c']])
        is_sorted: Indicates whether the table is already sorted with respect to the
            'group_by' column.

    Returns:
        Two numpy arrays, the first array contains the aggregation results of each each
        unique group and the second array contains the correpsonding group names.
    """
    group_start_indices, group_names, table = _prepare_grouping_indices(
        table, group_by, is_sorted
    )
    array = table[aggregate_columns].to_numpy()
    aggregation_result = np.array(
        [condenser(i) for i in np.split(array, group_start_indices[1:])]
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
