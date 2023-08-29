import numpy as np
import pytest

import msreport.isobar


def test_correct_isotope_impurity_contamination():
    contaminated_intensities = np.array([190.0, 101.0, 98.0, 112.0, 82.0])
    impurity_matrix = np.array(
        [
            [0.90, 0.10, 0.00, 0.00, 0.00],
            [0.10, 0.80, 0.01, 0.00, 0.00],
            [0.00, 0.00, 0.95, 0.00, 0.03],
            [0.00, 0.10, 0.02, 1.00, 0.00],
            [0.00, 0.00, 0.02, 0.00, 0.80],
        ]
    )
    expected = np.array([200, 100, 100, 100, 100])
    corrected_intensities = msreport.isobar._correct_isotope_impurity_contamination(
        contaminated_intensities, impurity_matrix
    )
    np.testing.assert_allclose(corrected_intensities, expected)


def test_apply_isotope_impurity_contamination():
    true_intensities = np.array([200.0, 100.0, 100.0, 100.0, 100.0])
    impurity_matrix = np.array(
        [
            [0.90, 0.10, 0.00, 0.00, 0.00],
            [0.10, 0.80, 0.01, 0.00, 0.00],
            [0.00, 0.00, 0.95, 0.00, 0.03],
            [0.00, 0.10, 0.02, 1.00, 0.00],
            [0.00, 0.00, 0.02, 0.00, 0.80],
        ]
    )
    expected = np.array([190.0, 101.0, 98.0, 112.0, 82.0])
    contaminated_intensities = msreport.isobar._apply_isotope_impurity_contamination(
        true_intensities, impurity_matrix
    )
    np.testing.assert_allclose(contaminated_intensities, expected)


def test_applying_and_correcting_impurity_contamination():
    true_intensities = np.array([200.0, 100.0, 100.0, 100.0, 100.0])
    impurity_matrix = np.array(
        [
            [0.90, 0.10, 0.00, 0.00, 0.00],
            [0.10, 0.80, 0.01, 0.00, 0.00],
            [0.00, 0.00, 0.95, 0.00, 0.03],
            [0.00, 0.10, 0.02, 1.00, 0.00],
            [0.00, 0.00, 0.02, 0.00, 0.80],
        ]
    )
    contaminated_intensities = msreport.isobar._apply_isotope_impurity_contamination(
        true_intensities, impurity_matrix
    )
    corrected_intensities = msreport.isobar._correct_isotope_impurity_contamination(
        contaminated_intensities, impurity_matrix
    )
    np.testing.assert_allclose(true_intensities, corrected_intensities)
