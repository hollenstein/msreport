import itertools
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import scipy.stats
import scipy.optimize

import pyteomics.parser


def gaussian_imputation(
    table: pd.DataFrame,
    median_downshift: float,
    std_width: float,
    column_wise: bool,
    seed: Optional[float] = None,
) -> pd.DataFrame:
    """Imputes missing values in a table by drawing values from a normal distribution.

    Missing values are imputed by drawing random numbers for a gaussian distribution.
    Sigma and mu of this distribution are calculated by adjusting the standard deviation
    and median of the observed values.

    Args:
        table: Table containing missing values that will be imputed.
        median_downshift: Times of standard deviations the observed median is
            downshifted for calulating mu of the normal distribution.
        std_width: Factor for adjusting the standard deviation of the observed values
            to obtain sigma of the normal distribution.
        column_wise: Specifies whether imputation is performed for each column
            separately or on the whole table together. Also affects if mu and sigma are
            calculated for each column separately or for the whole table.
        seed: Optional, allows specifying a number for initializing the random number
            generator. Using the same seed for the same input table will generate the
            same set of imputed values each time. Default is None, which results in
            different imputed values being generated each time.

    Returns:
        Copy of the table containing imputed values.
    """
    np.random.seed(seed)
    imputed_table = table.copy()
    if not column_wise:
        median = np.nanmedian(imputed_table)
        std = np.nanstd(imputed_table)
    for column in imputed_table:
        if column_wise:
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
    """Solves a square matrix containing pair wise log ratios.

    Args:
        matrix: A two-dimensional array with equal length in both dimensions.

    Returns:
        An array containing the least-squares solution.
    """
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
    """Calculate the mode by using kernel-density estimation.

    Args:
        values: Sequence of values for which the mode will be estimated.

    Returns:
        The estimated mode.
    """
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
    """Calculates the number of tryptic iBAQ peptides.

    The number of iBAQ peptides is calculated as the number of tryptic peptides with a
    length between 7 and 30 amino acids. Multiple peptides with the same sequence are
    counted multiple times.

    Args:
        protein_sequence: Amino acid sequence of a protein.

    Returns:
        Number of tryptic iBAQ peptides for the given protein sequence.
    """
    cleavage_rule = "[KR]"
    missed_cleavage = 0
    min_length = 7
    max_length = 30

    digestion_products = pyteomics.parser.icleave(
        protein_sequence,
        cleavage_rule,
        missed_cleavages=missed_cleavage,
        min_length=min_length,
        max_length=max_length,
        regex=True,
    )
    ibaq_peptides = [sequence for index, sequence in digestion_products]
    return len(ibaq_peptides)


def make_coverage_mask(
    protein_length: int, peptide_positions: list[(int, int)]
) -> np.array:
    """Returns a Boolean array with True for positions present in 'peptide_positions'.

    Args:
        protein_length: The number of amino acids in the protein sequence.
        peptide_positions: List of peptide start and end positions.

    Returns:
        A 1-dimensional Boolean array with length equal to 'protein_length'.
    """
    coverage_mask = np.zeros(protein_length, dtype="bool")
    for start, end in peptide_positions:
        coverage_mask[start - 1 : end] = True
    return coverage_mask


def calculate_sequence_coverage(
    protein_length: int, peptide_positions: list[(int, int)], ndigits: int = 1
) -> np.array:
    """Calculates the protein sequence coverage given a list of peptide positions.

    Args:
        protein_length: The number of amino acids in the protein sequence.
        peptide_positions: List of peptide start and end positions.
        ndigits: Optional, number of decimal places for rounding the sequence coverage.

    Returns:
        Sequence coverage in percent, with values ranging from 0 to 100.
    """
    coverage_mask = make_coverage_mask(protein_length, peptide_positions)
    coverage = round(coverage_mask.sum() / protein_length * 100, ndigits)
    return coverage
