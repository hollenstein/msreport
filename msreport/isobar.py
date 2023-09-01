import numpy as np
import scipy


def apply_isotope_impurity_contamination(
    intensities: np.array, impurity_matrix: np.array
) -> np.array:
    return np.sum(impurity_matrix * intensities, axis=1)


def correct_isotope_impurity_contamination(
    intensities: np.array, impurity_matrix: np.array
) -> np.array:
    corrected_intensities, _ = scipy.optimize.nnls(impurity_matrix, intensities)
    return corrected_intensities
