from __future__ import annotations

import functools
from typing import Protocol

import numpy as np
import pandas as pd
import scipy

import msreport.helper
from msreport.errors import NotFittedError


class Transformer(Protocol):
    def fit(self, table: pd.DataFrame) -> Transformer:
        """Fits the Transformer and returns a fitted Transformer instance."""

    def is_fitted(self) -> bool:
        """Returns True if the Transformer has been fitted."""

    def transform(self, table: pd.DataFrame) -> pd.DataFrame:
        """Transform values in 'table'."""


class IsotopeImpurityCorrecter:
    """Corrects isotope impurity interference in isobaric reporter expression values."""

    def __init__(self):
        self._impurity_matrix = None

    def fit(self, impurity_matrix: np.array) -> IsotopeImpurityCorrecter:
        """Fits the isotope impurity correcter to a given impurity matrix.

        Args:
            impurity_matrix: A reporter isotope impurity matrix in a diagonal format,
                where columns describe the isotope impurity of a specific channel, and
                the values in each row indicate the percentage of signal from the
                reporter that is present in each channel. Both dimensions of the
                impurity matrix must have the same length.

        Returns:
            Returns the fitted class IsotopeImpurityCorrecter instance.
        """
        if impurity_matrix.shape[0] != impurity_matrix.shape[1]:
            raise ValueError("The impurity matrix must be square.")
        if np.isnan(impurity_matrix).any():
            raise ValueError("The impurity matrix contains NaN values.")
        self._impurity_matrix = impurity_matrix
        return self

    def is_fitted(self) -> bool:
        """Returns True if the IsotopeImpurityCorrecter has been fitted."""
        return self._impurity_matrix is not None

    def get_fits(self) -> np.array:
        """Returns a copy of the fitted impurity matrix.

        returns:
            A numpy array representing a diagonal impurity matrix.
        """
        if not self.is_fitted():
            raise NotFittedError("The IsotopeImpurityCorrecter has not been fitted.")
        return self._impurity_matrix.copy()

    def transform(self, table: pd.DataFrame) -> pd.DataFrame:
        """Applies isotope impurity correction to the values of the table.

        Args:
            table: The data to normalize. The columns of the table must correspond to
                the channels of the impurity matrix used for fitting.

        Returns:
            A copy of the table with isotope impurity corrected values.
        """
        if not self.is_fitted():
            raise NotFittedError("The IsotopeImpurityCorrecter has not been fitted.")
        if table.shape[1] != self.get_fits().shape[1]:
            raise ValueError(
                "The number of columns in the table does not match the number "
                "of channels in the impurity matrix."
            )

        corrected_values = correct_isobaric_reporter_impurities(
            intensity_table=table.to_numpy(),
            diagonal_impurity_matrix=self._impurity_matrix,
        )
        corrected_table = table.copy()
        corrected_table[:] = corrected_values
        return corrected_table


def correct_isobaric_reporter_impurities(
    intensity_table: np.array,
    diagonal_impurity_matrix: np.array,
) -> np.array:
    """Performs isotope impurity correction on isobaric reporter expression values.

    Args:
        intensity_table: A two-dimenstional array with columns corresponding to isobaric
            reporter channels and rows to measured units such as PSMs, peptides or
            proteins.
        diagonal_impurity_matrix: A reporter isotope impurity matrix in a diagonal
            format, where columns describe the isotope impurity of a specific channel,
            and the values in each row indicate the percentage of signal from the
            reporter that is present in each channel.
    """
    apply_impurity_correction = functools.partial(
        _correct_impurity_contamination,
        impurity_matrix=diagonal_impurity_matrix,
    )

    data_was_in_logpsace = msreport.helper.intensities_in_logspace(intensity_table)

    if data_was_in_logpsace:
        intensity_table = np.power(2, intensity_table)
    intensity_table[np.isnan(intensity_table)] = 0
    corrected_table = np.apply_along_axis(apply_impurity_correction, 1, intensity_table)
    corrected_table[corrected_table <= 0] = 0
    if data_was_in_logpsace:
        corrected_table = np.log2(corrected_table)

    return corrected_table


def _apply_impurity_contamination(
    intensities: np.array, impurity_matrix: np.array
) -> np.array:
    """Applies reporter isotope impurity interference to an intensity array.

    Args:
        intensities: An array containing non-contaminated isobaric reporter intensities.
        impurity_matrix: A reporter isotope impurity matrix in a diagonal format, where
            columns describe the isotope impurity of a specific channel, and the values
            in each row indicate the percentage of signal from the reporter that is
            present in each channel. Both dimensions of the impurity matrix must have
            the same length as the intensity array.

    Returns:
        An array containing contaminated intensities.
    """
    return np.sum(impurity_matrix * intensities, axis=1)


def _correct_impurity_contamination(
    intensities: np.array, impurity_matrix: np.array
) -> np.array:
    """Applies reporter isotope impurity interference correction to an intensity array.

    Args:
        intensities: An array containing isobaric reporter intensities affected by
            isotope impurity interference.
        impurity_matrix: A reporter isotope impurity matrix in a diagonal format, where
            columns describe the isotope impurity of a specific channel, and the values
            in each row indicate the percentage of signal from the reporter that is
            present in each channel. Both dimensions of the impurity matrix must have
            the same length as the intensity array.

    Returns:
        An array containing impurity corrected intensities.
    """
    corrected_intensities, _ = scipy.optimize.nnls(impurity_matrix, intensities)
    return corrected_intensities
