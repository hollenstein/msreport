from typing import Callable, Iterable, Optional, Union

import numpy as np
import pandas as pd

import msreport.aggregate.condense as CONDENSE
from msreport.helper import find_sample_columns


def count_unique(
    table: pd.DataFrame,
    group_by: str,
    input_column: Union[str, Iterable],
    output_column: str = "Unique counts",
    is_sorted: bool = False,
) -> pd.DataFrame:
    """Counts unique values per group

    Note that empty strings and np.nan do not contribute to the unique value count.

    Args:
        table: Dataframe used for generating groups, which will be aggregated.
        group_by: Column used to determine unique groups.
        input_column: Column or list of columns, which will be used for counting the
            number of unique values per group.
        output_column: Optional, allows specifying an alternative column name for the
            returned dataframe. Default is "Unique counts".
        is_sorted: Indicates whether the table is already sorted with respect to the
            'group_by' column.

    Returns:
        A dataframe with unique 'group_by' values as index and a unique counts column
        containing the number of unique counts per group.
    """
    aggregation, groups = aggregate_unique_groups(
        table, group_by, input_column, CONDENSE.count_unique, is_sorted
    )
    return pd.DataFrame(columns=[output_column], data=aggregation, index=groups)


def join_unique(
    table: pd.DataFrame,
    group_by: str,
    input_column: Union[str, Iterable],
    output_column: str = "Unique values",
    sep: str = ";",
    is_sorted: bool = False,
) -> pd.DataFrame:
    """Per group concatenates unique values as a string with a separter.

    Args:
        table: Dataframe used for generating groups, which will be aggregated.
        group_by: Column used to determine unique groups.
        input_column: Column or list of columns, which will be used for defining the
            unique values.
        output_column: Optional, allows specifying an alternative column name for the
            returned dataframe. Default is "Unique values".
        sep: String that is used as a separator between unique values.
        is_sorted: Indicates whether the table is already sorted with respect to the
            'group_by' column.

    Returns:
        A dataframe with unique 'group_by' values as index and a unique values column
        containing the joined unique values per group. Unique values are sorted and
        joined with the specified separator.
    """
    aggregation, groups = aggregate_unique_groups(
        table,
        group_by,
        input_column,
        lambda x: CONDENSE.join_unique_str(x, sep=sep),
        is_sorted,
    )
    return pd.DataFrame(columns=[output_column], data=aggregation, index=groups)


def sum_columns(
    table: pd.DataFrame,
    group_by: str,
    samples: Iterable[str],
    input_tag: str,
    output_tag: Optional[str] = None,
    is_sorted: bool = False,
) -> pd.DataFrame:
    """Sums sample column values per group.

    Args:
        table: Dataframe used for generating groups, which will be aggregated.
        group_by: Column used to determine unique groups.
        samples: List of sample names that appear in columns of the table.
        input_tag: Substring of column names, which is used together with the sample
            names to determine the columns that will be summarized.
        output_tag: Optional, if specified the 'input_tag' is replaced by the
            'output_tag' in the columns of the returned dataframe.
        is_sorted: Indicates whether the table is already sorted with respect to the
            'group_by' column.

    Returns:
        A dataframe with unique 'group_by' values as index and one column per sample.
        The columns contain the summed group values per sample.
    """
    output_tag = input_tag if output_tag is None else output_tag
    columns = find_sample_columns(table, input_tag, samples)
    aggregation, groups = aggregate_unique_groups(
        table, group_by, columns, CONDENSE.sum_per_column, is_sorted
    )
    output_columns = [column.replace(input_tag, output_tag) for column in columns]
    return pd.DataFrame(columns=output_columns, data=aggregation, index=groups)


def sum_columns_maxlfq(
    table: pd.DataFrame,
    group_by: str,
    samples: Iterable[str],
    input_tag: str,
    output_tag: Optional[str] = None,
    is_sorted: bool = False,
) -> pd.DataFrame:
    """Sums sample column values per group using the MaxLFQ approach.

    Performs per group a least squares regression of pair-wise median ratios to
    calculate estimated abundance profiles. These profiles are then scaled based on the
    intensity values such that the columns with finite profile values are used and the
    sum of the scaled profiles matches the sum of the input array.

    Args:
        table: Dataframe used for generating groups, which will be aggregated.
        group_by: Column used to determine unique groups.
        samples: List of sample names that appear in columns of the table.
        input_tag: Substring of column names, which is used together with the sample
            names to determine the columns that will be summarized.
        output_tag: Optional, if specified the 'input_tag' is replaced by the
            'output_tag' in the columns of the returned dataframe.
        is_sorted: Indicates whether the table is already sorted with respect to the
            'group_by' column.

    Returns:
        A dataframe with unique 'group_by' values as index and one column per sample.
        The columns contain the summed group values per sample.
    """
    output_tag = input_tag if output_tag is None else output_tag
    columns = find_sample_columns(table, input_tag, samples)
    aggregation, groups = aggregate_unique_groups(
        table, group_by, columns, CONDENSE.sum_by_median_ratio_regression, is_sorted
    )
    output_columns = [column.replace(input_tag, output_tag) for column in columns]
    return pd.DataFrame(columns=output_columns, data=aggregation, index=groups)


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
        aggregate_columns: Column or list of columns, which will be passed to the
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
        group_start_indices, group_names, table
    """
    if not is_sorted:
        table = table.sort_values(by=group_by)
    group_names, group_start_indices, group_lengths = np.unique(
        table[group_by], return_counts=True, return_index=True
    )
    return group_start_indices, group_names, table
