import numpy as np

import msreport.helper.maxlfq as MAXLFQ


def join_str(array: np.ndarray, sep: str = ";") -> str:
    """Returns a joined string of sorted values from the array.

    Note that empty strings or np.nan are not included in the joined string.
    """
    elements = []
    for value in array.flatten():
        if value != "" and not (isinstance(value, float) and np.isnan(value)):
            elements.append(str(value))
    return sep.join(sorted(elements))


def join_str_per_column(array: np.ndarray, sep: str = ";") -> np.ndarray:
    """Returns for each column a joined string of sorted values.

    Note that empty strings or np.nan are not included in the joined string.
    """
    return np.array([join_str(i) for i in array.transpose()])


def join_unique_str(array: np.ndarray, sep: str = ";") -> str:
    """Returns a joined string of unique sorted values from the array."""
    elements = []
    for value in array.flatten():
        if value != "" and not (isinstance(value, float) and np.isnan(value)):
            elements.append(str(value))
    return sep.join(sorted(set(elements)))


def join_unique_str_per_column(array: np.ndarray, sep: str = ";") -> np.ndarray:
    """Returns for each column a joined strings of unique sorted values."""
    return np.array([join_unique_str(i) for i in array.transpose()])


def sum(array: np.ndarray) -> float:
    """Returns sum of values from one or multiple columns.

    Note that if no finite values are present in the array np.nan is returned.
    """
    array = array.flatten()
    if np.isfinite(array).any():
        return np.nansum(array)
    else:
        return np.nan


def sum_per_column(array: np.ndarray) -> np.ndarray:
    """Returns for each column the sum of values.

    Note that if no finite values are present in a column np.nan is returned.
    """
    return np.array([sum(i) for i in array.transpose()])


def maximum(array: np.ndarray) -> float:
    """Returns the highest finitevalue from one or multiple columns."""
    array = array.flatten()
    if np.isfinite(array).any():
        return np.nanmax(array)
    else:
        return np.nan


def maximum_per_column(array: np.ndarray) -> np.ndarray:
    """Returns for each column the highest finite value."""
    return np.array([maximum(i) for i in array.transpose()])


def minimum(array: np.ndarray) -> int:
    """Returns the lowest finite value from one or multiple columns."""
    array = array.flatten()
    if np.isfinite(array).any():
        return np.nanmin(array)
    else:
        return np.nan


def minimum_per_column(array: np.ndarray) -> np.ndarray:
    """Returns for each column the lowest finite value."""
    return np.array([minimum(i) for i in array.transpose()])


def count_unique(array: np.ndarray) -> int:
    """Returns the number of unique values from one or multiple columns.

    Note that empty strings or np.nan are not counted as unique values.
    """
    unique_elements = {
        x for x in array.flatten() if not (isinstance(x, float) and np.isnan(x))
    }
    unique_elements.discard("")

    return len(unique_elements)


def count_unique_per_column(array: np.ndarray) -> np.ndarray:
    """Returns for each column the number of unique values.

    Note that empty strings or np.nan are not counted as unique values.
    """
    if array.size > 0:
        return np.array([count_unique(i) for i in array.transpose()])
    else:
        return np.full(array.shape[0], 0)


def profile_by_median_ratio_regression(array: np.ndarray) -> np.ndarray:
    """Calculates abundance profiles by lstsq regression of pair-wise median ratios.

    The function performs a least squares regression of pair-wise median ratios to
    calculate estimated abundance profiles.

    Args:
        array: A two-dimensional array containing abundance values, with the first
            dimension corresponding to rows and the second dimension to columns.
            Abundance values must not be log transformed.

    Returns:
        An array containing estimated abundance profiles, with length equal to the
        number of columns in the input array.
    """
    ratio_matrix = MAXLFQ.calculate_pairwise_median_log_ratio_matrix(
        array, log_transformed=False
    )
    coef_matrix, ratio_array, initial_rows = MAXLFQ.prepare_coefficient_matrix(
        ratio_matrix
    )
    log_profile = MAXLFQ.log_profiles_by_lstsq(coef_matrix, ratio_array)
    profile = np.power(2, log_profile)
    return profile


def sum_by_median_ratio_regression(array: np.ndarray) -> np.ndarray:
    """Calculates summed abundance by lstsq regression of pair-wise median ratios.

    The function performs a least squares regression of pair-wise median ratios to
    calculate estimated abundance profiles. These profiles are then scaled based on the
    input array such that the columns with finite profile values are used and the sum of
    the scaled profiles matches the sum of the input array.

    Args:
        array: A two-dimensional array containing abundance values, with the first
            dimension corresponding to rows and the second dimension to columns.
            Abundance values must not be log transformed.

    Returns:
        An array containing summed abundance estimates, with length equal to the number
        of columns in the input array.
    """
    profile = profile_by_median_ratio_regression(array)
    scaled_profile = profile
    if np.isfinite(profile).any():
        profile_mask = np.isfinite(profile)
        scaled_profile[profile_mask] = profile[profile_mask] * (
            np.nansum(array[:, profile_mask]) / np.nansum(profile[profile_mask])
        )

    return scaled_profile
