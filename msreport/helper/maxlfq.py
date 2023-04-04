import itertools
import warnings

import numpy as np
import sklearn.linear_model


def calculate_pairwise_ratio_matrix(
    array: np.ndarray, log_transformed: bool = False
) -> np.ndarray:
    """Calculates a pairwise ratio matrix from an intensity array.

    Args:
        array: A two-dimensional array, with the first dimension corresponding to rows
            and the second dimension to columns.
        log_transformed: If true, the 'array' contains log transformed intensity values.
            Otherwise the 'array' is log2 transformed before calculating the ratios.

    Returns:
        A three dimensional np.array, containing pair-wise ratios. With the shape
        (Number rows in 'array', number columns in 'array', number columns in 'array')
    """
    if np.issubdtype(array.dtype, np.integer):
        log_array = array.astype(float)
    else:
        log_array = array.copy()

    if not log_transformed:
        log_array[log_array == 0] = np.nan
        log_array = np.log2(log_array)

    ratio_matrix = log_array[:, :, None] - log_array[:, None, :]
    ratio_matrix[~np.isfinite(ratio_matrix)] = np.nan
    return ratio_matrix


def calculate_pairwise_median_ratio_matrix(
    array: np.ndarray, log_transformed: bool = False
) -> np.ndarray:
    """Calculates a pairwise median ratio matrix from an intensity array.

    Args:
        array: A two-dimensional array, with the first dimension corresponding to rows
            and the second dimension to columns.
        log_transformed: If true, the 'array' contains log transformed intensity values.
            Otherwise the 'array' is log2 transformed before calculating the ratios.

    Returns:
        A three dimensional np.array, containing pair-wise median ratios. With the shape
        (number columns in 'array', number columns in 'array').
    """
    log_array = np.array(array)
    if not log_transformed:
        log_array[log_array == 0] = np.nan
        log_array = np.log2(log_array)

    num_cols = log_array.shape[1]
    ratio_marix = np.full((num_cols, num_cols), fill_value=np.nan)
    ratio_marix = np.zeros((num_cols, num_cols))
    for i, j in itertools.combinations(range(num_cols), 2):
        ratios = log_array[:, i] - log_array[:, j]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median_ratio = np.nanmedian(ratios[np.isfinite(ratios)])
        ratio_marix[i, j] = median_ratio

    # Generate a full, mirrowed matrix where the lower triangle is upper triangle * -1
    ratio_marix = ratio_marix - ratio_marix.T - np.diag(np.diag(ratio_marix))
    return ratio_marix


def prepare_coefficient_matrix(
    ratio_matrix: np.ndarray,
) -> (np.ndarray, np.ndarray, np.ndarray):
    """Generates a coefficient matrix from a ratio matrix.

    Args:
        ratio_matrix: Two- or three-dimensional numpy array containing one or multiple
            pair-wise ratio matrices.

    Returns:
        Tuple containing the coefficient matrix, ratio array, and initial rows array.
    """
    # TODO: Update docstring!
    if len(ratio_matrix.shape) == 2:
        result = _coefficients_from_single_row_matrix(ratio_matrix)
    else:
        result = _coefficients_from_multi_row_matrix(ratio_matrix)
    coef_matrix, ratio_array, initial_rows = result

    return coef_matrix, ratio_array, initial_rows


def calculate_lstsq_profiles(coef_matrix: np.ndarray, ratio_array: np.ndarray):
    """Calculates log ratio profiles with least-squares.

    Args:
        coef_matrix: Two-dimensional numpy array representing the coefficients.
        ratio_array: One-dimensional numpy array representing the ratios.

    Returns:
        One-dimensional numpy array representing the estimated least-squares profile.
    """
    # TODO: Update docstring!
    # TODO: Not tested!
    finite_rows = np.isfinite(ratio_array)
    coef_matrix = coef_matrix[finite_rows]
    ratio_array = ratio_array[finite_rows]

    absent_coef = np.abs(coef_matrix).sum(axis=0) == 0
    coef_estimates, resid, rank, s = np.linalg.lstsq(
        coef_matrix[:, ~absent_coef], ratio_array, rcond=None
    )
    log_profile = np.zeros(coef_matrix.shape[1])
    log_profile[absent_coef] = np.nan
    log_profile[~absent_coef] = coef_estimates
    return log_profile


def _coefficients_from_single_row_matrix(ratio_matrix):
    """Calculates coefficients, ratios, and initial rows for a single row matrix.

    Args:
        ratio_matrix: Two-dimensional numpy array containing a pair-wise ratio matrix.

    Returns:
        Tuple containing the coefficient matrix, ratio array, and initial rows array.
    """
    # TODO: Update docstring!
    num_coef = ratio_matrix.shape[1]
    coef_combinations = list(itertools.combinations(range(num_coef), 2))
    num_coef_combinations = len(coef_combinations)

    coef_matrix = np.zeros((num_coef_combinations, num_coef))
    ratio_array = np.zeros(num_coef_combinations)
    initial_rows = np.zeros(num_coef_combinations, dtype=int)

    for variable_position, (i, j) in enumerate(coef_combinations):
        ratio_ij = ratio_matrix[i, j]
        coef_matrix[variable_position, i] = 1
        coef_matrix[variable_position, j] = -1
        ratio_array[variable_position] = ratio_ij
    return coef_matrix, ratio_array, initial_rows


def _coefficients_from_multi_row_matrix(ratio_matrix):
    """Calculates coefficients, ratios, and initial rows for a multi row matrix.

    Args:
        ratio_matrix: Two-dimensional numpy array representing the ratios.

    Returns:
        Tuple containing the coefficient matrix, ratio array, and initial rows array.
    """
    # TODO: No docstring!
    num_coef = ratio_matrix.shape[1]
    coef_combinations = list(itertools.combinations(range(num_coef), 2))
    num_coef_combinations = len(coef_combinations)
    num_matrices = ratio_matrix.shape[0]
    coef_matrix_rows = num_coef_combinations * num_matrices

    coef_matrix = np.zeros((coef_matrix_rows, num_coef))
    ratio_array = np.zeros(coef_matrix_rows)
    initial_rows = np.zeros(coef_matrix_rows, dtype=int)

    for matrix_position, matrix in enumerate(ratio_matrix):
        for variable_position, (i, j) in enumerate(coef_combinations):
            position = (matrix_position * num_coef_combinations) + variable_position
            ratio_ij = matrix[i, j]
            coef_matrix[position, i] = 1
            coef_matrix[position, j] = -1
            ratio_array[position] = ratio_ij
            initial_rows[position] = matrix_position

    return coef_matrix, ratio_array, initial_rows
