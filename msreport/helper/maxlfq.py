import itertools
import warnings
from typing import Callable

import numpy as np

import msreport.helper


def calculate_pairwise_log_ratio_matrix(
    array: np.ndarray, log_transformed: bool = False
) -> np.ndarray:
    """Calculates a pairwise log ratio matrix from an intensity array.

    Args:
        array: A two-dimensional numpy array, with the first dimension corresponding to
            rows and the second dimension to columns.
        log_transformed: If True, the 'array' is expected to contain log transformed
            intensity values. If False, the array is expected to contain non-transformed
            intensity values, which are log2 transformed for the calculation of ratios.

    Returns:
        A 3-dimensional numpy array, containing pair-wise log ratios. The shape of the
        output array is (n, i, i), where n is the number of rows of the input array and
        i is the number of columns.

    Example:
        >>> array = np.array(
        ...     [
        ...         [4.0, 4.0, 8.0],
        ...         [8.0, 9.0, np.nan],
        ...     ]
        ... )
        >>> calculate_pairwise_log_ratio_matrix(array)
        array([[[ 0.      ,  0.      , -1.      ],
                [ 0.      ,  0.      , -1.      ],
                [ 1.      ,  1.      ,  0.      ]],
        <BLANKLINE>
               [[ 0.      , -0.169925,       nan],
                [ 0.169925,  0.      ,       nan],
                [      nan,       nan,       nan]]])
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


def calculate_pairwise_median_log_ratio_matrix(
    array: np.ndarray, log_transformed: bool = False
) -> np.ndarray:
    """Calculates a pairwise median log ratio matrix from an intensity array.

    Args:
        array: A two-dimensional numpy array, with the first dimension corresponding to
            rows and the second dimension to columns.
        log_transformed: If True, the 'array' is expected to contain log transformed
            intensity values. If False, the array is expected to contain non-transformed
            intensity values, which are log2 transformed for the calculation of ratios.

    Returns:
        A square 2-dimensional numpy array, containing pair-wise log ratios. The shape
        of the output array is (i, i), where i is the number of columns of the input
        array.

    Example:
        >>> array = np.array(
        ...     [
        ...         [4.0, 4.0, 8.0],
        ...         [8.0, 9.0, np.nan],
        ...     ]
        ... )
        >>> calculate_pairwise_median_log_ratio_matrix(array)
        array([[ 0.       , -0.0849625, -1.       ],
               [ 0.0849625,  0.       , -1.       ],
               [ 1.       ,  1.       ,  0.       ]])
    """
    ratio_marix = _calculate_pairwise_centered_log_ratio_matrix(
        array, np.median, log_transformed=log_transformed
    )
    return ratio_marix


def calculate_pairwise_mode_log_ratio_matrix(
    array: np.ndarray, log_transformed: bool = False
) -> np.ndarray:
    """Calculates a pairwise mode ratio matrix from an intensity array.

    Args:
        array: A two-dimensional numpy array, with the first dimension corresponding to
            rows and the second dimension to columns.
        log_transformed: If True, the 'array' is expected to contain log transformed
            intensity values. If False, the array is expected to contain non-transformed
            intensity values, which are log2 transformed for the calculation of ratios.

    Returns:
        A square 2-dimensional numpy array, containing pair-wise ratios. The shape of
        the output array is (i, i), where i is the number of columns of the input array.

    Example:
        >>> array = np.array(
        ...     [
        ...         [4.0, 4.0, 8.0],
        ...         [8.0, 9.0, np.nan],
        ...     ]
        ... )
        >>> calculate_pairwise_mode_log_ratio_matrix(array)
        array([[ 0.       , -0.0849625, -1.       ],
               [ 0.0849625,  0.       , -1.       ],
               [ 1.       ,  1.       ,  0.       ]])
    """
    ratio_marix = _calculate_pairwise_centered_log_ratio_matrix(
        array, msreport.helper.mode, log_transformed=log_transformed
    )
    return ratio_marix


def prepare_coefficient_matrix(
    ratio_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepares coefficients, ratios, and initial row indices from a log ratio matrix.

    Args:
        ratio_matrix: A numpy array containing one or multiple pair-wise ratio matrices.
            Each ratio matrix must be a square array, with a ratio at position (i, j)
            being calculated from an abundance table as 'column i - column j'. Ratios
            should have been calculated by row index Only the upper triangular part of
            the ratio matrix is used to generate the coefficient matrix. If the
            'ratio_matrix' contains multiple ratio matrices, the shape of the array
            has to be (n, i, i), where n is the number of ratio matrices and i is the
            number of rows and columns per ratio matrix. If only one ratio matrix is
            provided, the shape of the array has to be (i, i).

    Returns:
        A tuple containing the following three elements:
        - A coefficent matrix. 2d array with the number of rows corresponding to the
          first dimension of the 'ratio_matrix' and the number of columns to the second
          dimension.
        - Ratios: 1d array containing the ratios from the ratio matrix, each entry
          corresponds to a row in the coefficent matrix.
        - Ratio matrix row indices: 1d array containing row indicies that refer to the
          index of the first dimenstion from the 'ratio_matrix', and thus to row indices
          from the original table that was used to generate 'the ratio_matrix'. If
          the 'ratio_matrix' has only 2 dimensions all values are zero.

    Example:
        >>> ratio_matrix = np.array(
        ...     [
        ...         [0.0, -0.1, -1.0],
        ...         [0.1, 0.0, -1.0],
        ...         [1.0, 1.0, 0.0],
        ...     ]
        ... )
        >>> prepare_coefficient_matrix(ratio_matrix)
        (array([[ 1, -1,  0],
                [ 1,  0, -1],
                [ 0,  1, -1]]),
         array([-0.1, -1. , -1. ]),
         array([0, 0, 0]))

    """
    if len(ratio_matrix.shape) == 2:
        result = _coefficients_from_single_row_matrix(ratio_matrix)
    else:
        result = _coefficients_from_multi_row_matrix(ratio_matrix)
    coef_matrix, ratio_array, initial_rows = result

    return coef_matrix, ratio_array, initial_rows


def log_profiles_by_lstsq(coef_matrix: np.ndarray, ratio_array: np.ndarray):
    """Calculates estimated log abundance profiles by least-squares fitting.

    Args:
        coef_matrix: Two-dimensional numpy array representing the coefficients.
        ratio_array: One-dimensional numpy array representing the ratios, each entry
            corresponds to a row in the coefficent matrix.

    Returns:
        One-dimensional numpy array containing the estimated least-squares profile.

    Example:
        >>> coef_matrix = np.array(
        ...     [
        ...         [1, -1, 0],
        ...         [1, 0, -1],
        ...         [0, 1, -1],
        ...     ]
        ... )
        >>> ratio_array = np.array([-0.1, -1.0, -1.0])
        >>> log_profiles_by_lstsq(coef_matrix, ratio_array)
        array([-0.36666667, -0.3       ,  0.66666667])
    """
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


def _calculate_pairwise_centered_log_ratio_matrix(
    array: np.ndarray, center_function: Callable, log_transformed: bool = False
) -> np.ndarray:
    """Calculates a pairwise, centered log2 ratio matrix from an intensity array.

    Args:
        array: A two-dimensional numpy array, with the first dimension corresponding to
            rows and the second dimension to columns.
        center_function: Function that is applied to the ratios of each pair-wise
            comparison of columns in the input array to calculate the centered ratio.
        log_transformed: If True, the 'array' is expected to contain log transformed
            intensity values. If False, the array is expected to contain non-transformed
            intensity values, which are log2 transformed for the calculation of ratios.

    Returns:
        A square 2-dimensional numpy array, containing pair-wise ratios. The shape of
        the output array is (i, i), where i is the number of columns of the input array.
    """
    # Note: Is currently tested only via the calculate_pairwise_median_log_ratio_matrix
    #       and calculate_pairwise_mode_log_ratio_matrix functions.
    if np.issubdtype(array.dtype, np.integer):
        log_array = array.astype(float)
    else:
        log_array = array.copy()

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
            median_ratio = center_function(ratios[np.isfinite(ratios)])
        ratio_marix[i, j] = median_ratio

    # Generate a full, mirrowed matrix where the lower triangle is upper triangle * -1
    ratio_marix = ratio_marix - ratio_marix.T - np.diag(np.diag(ratio_marix))
    return ratio_marix


def _coefficients_from_single_row_matrix(ratio_matrix):
    """Calculates coefficients, ratios, and initial row indices for a single row matrix.

    Args:
        ratio_matrix: A numpy array containing one single pair-wise ratio matrix. The
            ratio matrix must be a square array, with a ratio at position (i, j) being
            calculated from an abundance table as 'column i - column j'. Only the upper
            triangular part of the ratio matrix is used to generate the coefficient
            matrix.

    Returns:
        A tuple containing the following three elements:
        - A coefficent matrix. 2d array with the number of rows corresponding to the
          first dimension of the 'ratio_matrix' and the number of columns to the second
          dimension.
        - Ratios: 1d array containing the ratios from the ratio matrix, each entry
          corresponds to a row in the coefficent matrix.
        - Ratio matrix row indices: 1d array with equal length as the ratios array,
          containing all zero values. is returned for consistency with the function
          `_coefficients_from_multi_row_matrix`.
    """
    num_coef = ratio_matrix.shape[1]
    coef_combinations = list(itertools.combinations(range(num_coef), 2))
    num_coef_combinations = len(coef_combinations)

    coef_matrix = np.zeros((num_coef_combinations, num_coef), dtype=int)
    ratio_array = np.zeros(num_coef_combinations)
    idx_ratio_matrix_first_dimension = np.zeros(num_coef_combinations, dtype=int)

    for variable_position, (i, j) in enumerate(coef_combinations):
        ratio_ij = ratio_matrix[i, j]
        coef_matrix[variable_position, i] = 1
        coef_matrix[variable_position, j] = -1
        ratio_array[variable_position] = ratio_ij
    return coef_matrix, ratio_array, idx_ratio_matrix_first_dimension


def _coefficients_from_multi_row_matrix(ratio_matrix):
    """Calculates coefficients, ratios, and initial row indices for a multi row matrix.

    Args:
        ratio_matrix: A numpy array containing multiple pair-wise ratio matrices. Each
            ratio matrix must be a square array, with a ratio at position (i, j) being
            calculated from an abundance table as 'column i - column j'. Only the upper
            triangular part of the ratio matrix is used to generate the coefficient
            matrix. The shape of 'ratio_matrix' must be(n, i, i), where n is the number
            of ratio matrices and i is the number of rows and columns per ratio
            matrix.

    Returns:
        A tuple containing the following three elements:
        - A coefficent matrix. 2d array with the number of rows corresponding to the
          first dimension of the 'ratio_matrix' and the number of columns to the second
          and third dimension.
        - Ratios: 1d array containing the ratios from the ratio matrix, each entry
          corresponds to a row in the coefficent matrix.
        - Ratio matrix row indices: 1d array containing row indicies that refer to the
          index of the first dimenstion from the 'ratio_matrix', and thus to row indices
          from the original table that was used to generate 'the ratio_matrix'.
    """
    num_coef = ratio_matrix.shape[1]
    coef_combinations = list(itertools.combinations(range(num_coef), 2))
    num_coef_combinations = len(coef_combinations)
    num_matrices = ratio_matrix.shape[0]
    coef_matrix_rows = num_coef_combinations * num_matrices

    coef_matrix = np.zeros((coef_matrix_rows, num_coef), dtype=int)
    ratio_array = np.zeros(coef_matrix_rows)
    idx_ratio_matrix_first_dimension = np.zeros(coef_matrix_rows, dtype=int)

    for matrix_position, matrix in enumerate(ratio_matrix):
        for variable_position, (i, j) in enumerate(coef_combinations):
            position = (matrix_position * num_coef_combinations) + variable_position
            ratio_ij = matrix[i, j]
            coef_matrix[position, i] = 1
            coef_matrix[position, j] = -1
            ratio_array[position] = ratio_ij
            idx_ratio_matrix_first_dimension[position] = matrix_position

    return coef_matrix, ratio_array, idx_ratio_matrix_first_dimension
