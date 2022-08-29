import itertools
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.stats
import scipy.optimize

import maspy.peptidemethods


def gaussian_imputation(
    table: pd.DataFrame, median_downshift: float, std_width: float
) -> pd.DataFrame:
    """Impute missing values by drawing values from a normal distribution.

    Imputation is performed column wise, and the parameters for the normal
    distribution are calculated independently for each column.

    Args:
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
    """Solve a square matrix containing pair wise log ratios."""
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


def mode(values: Iterable) -> float:
    """Calculate the mode by using kernel-density estimation."""
    median = np.median(values)
    bounds = (median - 1.5, median + 1.5)
    kde = scipy.stats.gaussian_kde(values)
    optimize_result = scipy.optimize.minimize_scalar(
        lambda x: -kde(x)[0], method="Bounded", bounds=bounds
    )
    mode = optimize_result.x
    # Maybe add fallback function if optimize was not successful
    return mode


def calculate_tryptic_ibaq_peptides(protein_sequence: str) -> int:
    cleavage_rule = "[KR]"
    missed_cleavage = 0
    min_length = 7
    max_length = 30

    digestion_products = maspy.peptidemethods.digestInSilico(
        protein_sequence,
        removeNtermM=False,
        cleavageRule=cleavage_rule,
        missedCleavage=missed_cleavage,
        minLength=min_length,
        maxLength=max_length,
    )
    unique_peptide_sequences = set([p for p, i in digestion_products])
    return len(unique_peptide_sequences)


def make_coverage_mask(
    protein_length: int, peptide_positions: list[(int, int)]
) -> np.array:
    coverage_mask = np.zeros(protein_length, dtype="bool")
    for start, end in peptide_positions:
        coverage_mask[start - 1 : end] = True
    return coverage_mask


def calculate_sequence_coverage(
    protein_length: int, peptide_positions: list[(int, int)], ndigits: int = 1
) -> np.array:
    coverage_mask = make_coverage_mask(protein_length, peptide_positions)
    coverage = round(coverage_mask.sum() / protein_length * 100, ndigits)
    return coverage
