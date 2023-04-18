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
    """Aggregates column(s) by counting unique values for each unique group.

    Note that empty strings and np.nan do not contribute to the unique value count.

    Args:
        table: The input DataFrame used for aggregating on unique groups.
        group_by: The name of the column used to determine unique groups for
            aggregation.
        input_column: A column or a list of columns, whose unique values will be counted
            for each unique group during aggregation.
        output_column: The name of the column containing the aggregation results. By
            default "Unique values" is used as the name of the output column.
        is_sorted: Indicates whether the input dataframe is already sorted with respect
            to the 'group_by' column.

    Returns:
        A dataframe with unique 'group_by' values as index and a unique counts column
        containing the number of unique counts per group.

    Example:
        >>> table = pd.DataFrame(
        ...     {
        ...         "ID": ["A", "A", "B", "C", "C", "C"],
        ...         "Peptide sequence": ["a", "a", "b", "c1", "c2", "c2"],
        ...     }
        ... )
        >>> count_unique(table, group_by="ID", input_column="Peptide sequence")
           Unique counts
        A              1
        B              1
        C              2
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
    """Aggregates column(s) by concatenating unique values for each unique group.

    Note that empty strings and np.nan do not contribute to the unique value count.

    Args:
        table: The input DataFrame used for aggregating on unique groups.
        group_by: The name of the column used to determine unique groups for
            aggregation.
        input_column: A column or a list of columns, whose unique values will be joined
            into a single string for each unique group
        output_column: The name of the column containing the aggregation results. By
            default "Unique values" is used as the name of the output column.
        sep: The separator string used to join multiple unique values together. Default
            is ";".
        is_sorted: Indicates whether the input dataframe is already sorted with respect
            to the 'group_by' column.

    Returns:
        A dataframe with unique 'group_by' values as index and a unique values column
        containing the joined unique values per group. Unique values are sorted and
        joined with the specified separator.

    Example:
        >>> table = pd.DataFrame(
        ...     {
        ...         "ID": ["A", "A", "B", "C", "C", "C"],
        ...         "Peptide sequence": ["a", "", "b", "c1", "c2", "c2"],
        ...     }
        ... )
        >>> join_unique(table, group_by="ID", input_column="Peptide sequence")
          Unique values
        A             a
        B             b
        C         c1;c2
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
    """Aggregates column(s) by summing up values for each unique group.

    Args:
        table: The input DataFrame used for aggregating on unique groups.
        group_by: The name of the column used to determine unique groups for
            aggregation.
        samples: List of sample names that appear in columns of the table as substrings.
        input_tag: Substring of column names, which is used together with the sample
            names to determine the columns whose values will be summarized for each
            unique group.
        output_tag: Optional, allows changing the ouptut column names by replacing the
            'input_tag' with the 'output_tag'. If not specified the names of the columns
            that were used for aggregation will be used in the returned dataframe.
        is_sorted: Indicates whether the input dataframe is already sorted with respect
            to the 'group_by' column.

    Returns:
        A dataframe with unique 'group_by' values as index and one column per sample.
        The columns contain the summed group values per sample.

    Example:
        >>> table = pd.DataFrame(
        ...     {
        ...         "ID": ["A", "A", "B", "C", "C", "C"],
        ...         "Col S1": [1, 1, 1, 1, 1, 1],
        ...         "Col S2": [2, 2, 2, 2, 2, 2],
        ...     }
        ... )
        >>> sum_columns(table, "ID", samples=["S1", "S2"], input_tag="Col")
           Col S1  Col S2
        A       2       4
        B       1       2
        C       3       6
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
    """Aggregates column(s) by applying the MaxLFQ summation approach to unique group.

    This function estimates abundance profiles from sample columns using pairwise median
    ratios and least square regression. It then selects abundance profiles with finite
    values and the corresponding input columns and scales the abundance profiles so that
    their total sum is equal to the total sum of the corresponding input columns.

    Args:
        table: The input DataFrame used for aggregating on unique groups.
        group_by: The name of the column used to determine unique groups for
            aggregation.
        samples: List of sample names that appear in columns of the table as substrings.
        input_tag: Substring of column names, which is used together with the sample
            names to determine the columns whose values will be summarized for each
            unique group.
        output_tag: Optional, allows changing the ouptut column names by replacing the
            'input_tag' with the 'output_tag'. If not specified the names of the columns
            that were used for aggregation will be used in the returned dataframe.
        is_sorted: Indicates whether the input dataframe is already sorted with respect
            to the 'group_by' column.

    Returns:
        A dataframe with unique 'group_by' values as index and one column per sample.
        The columns contain the summed group values per sample.

    Example:
        >>> table = pd.DataFrame(
        ...     {
        ...         "ID": ["A", "A", "B", "C", "C", "C"],
        ...         "Col S1": [1, 1, 1, 1, 1, 1],
        ...         "Col S2": [2, 2, 2, 2, 2, 2],
        ...     }
        ... )
        >>> sum_columns_maxlfq(table, "ID", samples=["S1", "S2"], input_tag="Col")
           Col S1  Col S2
        A     2.0     4.0
        B     1.0     2.0
        C     3.0     6.0
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
    columns_to_aggregate: Union[str, Iterable],
    condenser: Callable,
    is_sorted: bool,
) -> (np.ndarray, np.ndarray):
    """Aggregates column(s) by applying a condenser function to unique groups.

    The function returns two arrays containing the aggregated values and the
    corresponding group names. This function can be used for example to summarize data
    from an ion table to a peptide, protein or modification table. Suitable condenser
    functions can be found in the module msreport.aggregate.condense

    Args:
        table: The input dataframe used for aggregating on unique groups.
        group_by: The name of the column used to determine unique groups for
            aggregation.
        columns_to_aggregate: A column or a list of columns, which will be passed to the
            condenser function for applying an aggregation to each unique group.
        condenser: Function that is applied to each group for generating the
            aggregation result. If multiple columns are specified for aggregation,
            the input array for the condenser function will be two dimensional, with the
            first dimension corresponding to rows and the second to the column. E.g. an
            array with 3 rows and 2 columns: np.array([[1, 'a'], [2, 'b'], [3, 'c']])
        is_sorted: Indicates whether the input dataframe is already sorted with respect
            to the 'group_by' column.

    Returns:
        Two numpy arrays, the first array contains the aggregation results of each each
        unique group and the second array contains the correpsonding group names.
    """
    group_start_indices, group_names, table = _prepare_grouping_indices(
        table, group_by, is_sorted
    )
    array = table[columns_to_aggregate].to_numpy()
    aggregation_result = np.array(
        [condenser(i) for i in np.split(array, group_start_indices[1:])]
    )
    return aggregation_result, group_names


def _prepare_grouping_indices(
    table: pd.DataFrame, group_by: str, is_sorted: bool
) -> (np.ndarray, np.ndarray, pd.DataFrame):
    """Prepares start indices and names of unique groups from a sorted dataframe.

    Args:
        table: The input DataFrame used for generating unique groups.
        group_by: The name of the column used to determine unique groups.
        is_sorted: If True, the input DataFrame is assumed to be already sorted with
            respected to the 'group_by' column. Ohterwise, the input DataFrame is sorted
            by the 'group_by' column and the sorted DataFrame is returned.

    Returns:
        A tuple containing the following three elements:
        - A numpy array containing the start indices of each unique group
        - A numpy array containing the names of each unique group
        - The input DataFrame sorted by the 'group_by' column, if it was not already
          sorted.
    """
    if not is_sorted:
        table = table.sort_values(by=group_by)
    group_names, group_start_indices, group_lengths = np.unique(
        table[group_by], return_counts=True, return_index=True
    )
    return group_start_indices, group_names, table
