from __future__ import annotations
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from msreport.errors import NotFittedError


class FixedValueImputer:
    """Imputer for completing missing values with a fixed value.

    Replace missing values using a constant value or with an integer that is smaller
    than the minimum value of each column or smaller than the minimum value of the whole
    array.
    """

    def __init__(
        self, strategy: str, fill_value: Optional[float] = None, local: bool = True
    ):
        """Initializes the FixedValueImputer.

        Args:
            strategy: The imputation strategy.
                - If "constant", replace missing values with 'fill_value'.
                - If "below", replace missing values with an integer that is smaller
                  than the minimal value of the fitted dataframe. Minimal values are
                  calculated per column if 'local' is True, otherwise the minimal value
                  is calculated for all columns.
            fill_value: When strategy is "constant", 'fill_value' is used to replace all
                occurrences of missing_values.
            local: If True, imputation is performed independently per column, otherwise
                the whole dataframe is imputed togeter. Default True.
        """
        self.strategy = strategy
        self.fill_value = fill_value
        self.local = local
        self._sample_fill_values: dict[str, float] = {}

    def fit(self, table: pd.DataFrame) -> FixedValueImputer:
        """Fits the FixedValueImputer.

        Args:
            table: Input Dataframe for generating fill values for each column.

        Returns:
            Returns the fitted FixedValueImputer instance.
        """
        if self.strategy == "constant":
            # if not isinstance(self.fill_value, (float, int)):
            #     raise Excpetion()
            fill_values = {column: self.fill_value for column in table.columns}
        elif self.strategy == "below":
            if self.local:
                fill_values = {}
                for column in table:
                    fill_values[column] = _calculate_integer_below_min(table[column])
            else:
                int_below_min = _calculate_integer_below_min(table)
                fill_values = {column: int_below_min for column in table.columns}
        self._sample_fill_values = fill_values
        return self

    def is_fitted(self) -> bool:
        """Returns True if the FixedValueImputer has been fitted."""
        return len(self._sample_fill_values) != 0

    def transform(self, sample: str, values: Iterable[float]) -> np.ndarray:
        """Impute all missing values in 'values'.

        Args:
            sample: Use the imputation value calculated for the specified 'sample'.
                Must correspond to one of the column names from the table that was used
                for the fitting.
            values: Iterable of numeric values that will be completed.

        Returns:
            A numpy array with the length of 'values' with imputed missing values.
        """
        confirm_is_fitted(self)

        data = np.array(values, dtype=float)
        mask = ~np.isfinite(data)
        data[mask] = self._sample_fill_values[sample]
        return data

    def transform_table(self, table: pd.DataFrame) -> pd.DataFrame:
        """Impute all missing values in 'table'.

        Args:
            table: A dataframe of numeric values that will be completed. Each column
                name must correspond to a column name from the table that was used for
                the fitting.

        Returns:
            'table' with imputed missing values.
        """
        _table = table.copy()
        for sample in table.columns:
            _table[sample] = self.transform(sample, _table[sample])
        return _table


class GaussianImputer:
    """Imputer for completing missing values by drawing from a gaussian distribution."""

    def __init__(self, mu: float, sigma: float, seed: Optional[int] = None):
        """Initializes the GaussianImputer.

        Args:
            mu: Mean of the gaussian distribution.
            sigma: Standard deviation of the gaussian distribution, must be positive.
            seed: Optional, allows specifying a number for initializing the random
                number generator. Using the same seed for the same input table will
                generate the same set of imputed values each time. Default is None,
                which results in different imputed values being generated each time.
        """
        self.mu = mu
        self.sigma = sigma
        self.seed = seed

    def fit(self, table: pd.DataFrame) -> GaussianImputer:
        """Fits the GaussianImputer, altough this is not necessary.

        Args:
            table: Input Dataframe for fitting.

        Returns:
            Returns the fitted GaussianImputer instance.
        """
        return self

    def is_fitted(self) -> bool:
        """Returns always True, as the GaussianImputer does not need to be fitted."""
        return True

    def transform_table(self, table: pd.DataFrame) -> pd.DataFrame:
        """Impute all missing values in 'table'.

        Args:
            table: A dataframe of numeric values that will be completed. Each column
                name must correspond to a column name from the table that was used for
                the fitting.

        Returns:
            'table' with imputed missing values.
        """
        confirm_is_fitted(self)
        np.random.seed(self.seed)

        _table = table.copy()
        for column in table.columns:
            column_data = np.array(_table[column], dtype=float)
            mask = ~np.isfinite(column_data)
            column_data[mask] = np.random.normal(
                loc=self.mu, scale=self.sigma, size=mask.sum()
            )
            _table[column] = column_data
        return _table


def confirm_is_fitted(imputer: any, msg: Optional[str] = None) -> None:
    """Perform is_fitted validation for imputer instances.

    Checks if the imputer is fitted by verifying the presence of fitted attributes
    and otherwise raises a NotFittedError with the given message.

    Args:
        msg : str, default=None
            The default error message is, "This %(name) instance is not fitted
            yet. Call 'fit' with appropriate arguments before using this
            normalizer."
    """
    if msg is None:
        msg = (
            "This %(name)s instance is not fitted yet. Call 'fit' with "
            "appropriate arguments before using this imputer."
        )

    if not hasattr(imputer, "is_fitted"):
        raise TypeError(f"{imputer} is not an imputer instance.")
    else:
        fitted = imputer.is_fitted()

    if not fitted:
        raise NotFittedError(msg % {"name": type(imputer).__name__})


def _calculate_integer_below_min(table) -> int:
    minimal_value = np.nanmin(table.to_numpy().flatten())
    below_minimal = np.floor(minimal_value)
    if minimal_value <= below_minimal:
        below_minimal = below_minimal - 1
    return int(below_minimal)
