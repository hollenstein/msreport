import functools
import numpy as np
import scipy

import msreport.helper


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
    # TODO: not tested #
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
