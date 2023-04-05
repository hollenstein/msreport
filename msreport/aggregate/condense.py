import numpy as np

import msreport.helper.maxlfq as MAXLFQ


def join_str(array: np.ndarray, sep: str = ";") -> str:
    """Returns a joined string of sorted values from the array."""
    return sep.join(sorted([str(i) for i in array.flatten()]))


def join_str_per_column(array: np.ndarray, sep: str = ";") -> np.ndarray:
    """Returns for each column a joined string of sorted values."""
    return np.array([join_str(i) for i in array.transpose()])


def join_unique_str(array: np.ndarray, sep: str = ";") -> str:
    """Returns a joined string of unique sorted values from the array."""
    return sep.join(sorted([str(i) for i in set(array.flatten())]))


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
