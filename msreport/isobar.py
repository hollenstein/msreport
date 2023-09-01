import numpy as np
import scipy


def apply_isotope_impurity_contamination(
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


def correct_isotope_impurity_contamination(
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
