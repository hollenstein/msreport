import itertools
import numpy as np
import pandas as pd
from typing import Iterable


def find_columns(df: pd.DataFrame, substring: str) -> list[str]:
    """ Returns a list column names containing the substring """
    matched_columns = [substring in col for col in df.columns]
    matched_column_names = np.array(df.columns)[matched_columns].tolist()
    return matched_column_names


def gaussian_imputation(table: pd.DataFrame, median_downshift: float,
                        std_width: float) -> pd.DataFrame:
    """ Imput missing values by drawing values from a normal distribution.

    Imputation is performed column wise, and the parameters for the normal
    distribution are calculated independently for each column.

    Attributes:
        table: table containing missing values that will be replaced
        median_downshift: number of standard deviations the median of the
            measured values is downshifted for the normal distribution
        std_width: width of the normal distribution relative to the
            standard deviation of the measured values

    Returns:
        A new pandas.DataFrame containing imputed values
    """
    imputed_table = table.copy()
    for column in imputed_table:
        median = np.nanmedian(imputed_table[column])
        std = np.nanstd(imputed_table[column])

        mu = median - (std * median_downshift)
        sigma = std * std_width
        missing_values = imputed_table[column].isnull()
        num_missing_values = missing_values.sum()
        imputed_values = np.random.normal(mu, sigma, num_missing_values)

        imputed_table.loc[missing_values, column] = imputed_values
    return imputed_table


def solve_ratio_matrix(matrix: np.ndarray) -> np.ndarray:
    """ Solve a square matrix containing pair wise ratios. """
    # Not tested #
    assert matrix.shape[0] == matrix.shape[1]
    num_groups = matrix.shape[0]
    group_combinations = list(itertools.combinations(range(num_groups), 2))
    num_combinations = len(group_combinations)

    coefficient_matrix = np.zeros((num_combinations, num_groups))
    dependent_variables = np.empty(num_combinations)

    for variable_position, (i, j) in enumerate(group_combinations):
        ratio_ij = matrix[i, j]
        coefficient_matrix[variable_position, i] = 1
        coefficient_matrix[variable_position, j] = -1
        dependent_variables[variable_position] = ratio_ij

    x, resid, rank, s = np.linalg.lstsq(
        coefficient_matrix, dependent_variables, rcond=None
    )
    return x


import scipy.stats
def mode(values: Iterable) -> float:
    # Not tested #
    kde = scipy.stats.gaussian_kde(values)
    sampling_dist = 0.001
    sampling_num = int(np.ceil((max(values) - min(values)) / sampling_dist))
    sampling_points = np.linspace(min(values), max(values), sampling_num)

    mode = sampling_points[np.argmax(kde(sampling_points))]
    return mode


def guess_design(table, tag):
    """ Extract sample names and experiments from intensity columns. """
    # Not tested #
    sample_entries = []
    for column in find_columns(table, tag):
        sample = column.replace(tag, '').strip()
        experiment = '_'.join(sample.split('_')[:-1])
        sample_entries.append([sample, experiment])
    design = pd.DataFrame(sample_entries, columns=['Sample', 'Experiment'])
    return design
