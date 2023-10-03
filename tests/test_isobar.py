import numpy as np
import pandas as pd
import pytest

import msreport.isobar
from msreport.errors import NotFittedError


class TestCorrectionOfIsotopeImpurities:
    @pytest.fixture(autouse=True)
    def _init_matrix_and_intensities(self):
        self.pure_intensities = np.array([400, 200, 100, 100, 50])
        # The impurity matrix is a square matrix where each row and column represent a
        # distinct reporter channel. In this matrix each column describes the isotope
        # impurity of a specific channel. The values in each row indicate the percentage
        # of signal from the reporter that is present in each channel.
        self.impurity_matrix = np.array(
            [
                [0.90, 0.10, 0.00, 0.00, 0.00],
                [0.10, 0.80, 0.01, 0.00, 0.00],
                [0.00, 0.01, 0.95, 0.01, 0.03],
                [0.00, 0.09, 0.02, 0.98, 0.10],
                [0.00, 0.00, 0.02, 0.01, 0.80],
            ]
        )
        # Given the pure intensities and the impurity matrix, one can calculate the
        # contaminated intensities by multiplying the columns with the pure intensities
        # and then summing up each row.
        self.contaminated_intensities = (
            self.impurity_matrix * self.pure_intensities
        ).sum(axis=1)
        np.testing.assert_allclose(
            self.contaminated_intensities, np.array([380, 201, 99.5, 123, 43])
        )

    def test_apply_impurity_contamination(self):
        contaminated_intensities = msreport.isobar._apply_impurity_contamination(
            self.pure_intensities, self.impurity_matrix
        )
        np.testing.assert_allclose(
            contaminated_intensities, self.contaminated_intensities
        )

    def test_correct_impurity_contamination(self):
        corrected_intensities = msreport.isobar._correct_impurity_contamination(
            self.contaminated_intensities, self.impurity_matrix
        )
        np.testing.assert_allclose(corrected_intensities, self.pure_intensities)

    def test_applying_and_correcting_impurity_contamination(self):
        contaminated_intensities = msreport.isobar._apply_impurity_contamination(
            self.pure_intensities, self.impurity_matrix
        )
        corrected_intensities = msreport.isobar._correct_impurity_contamination(
            contaminated_intensities, self.impurity_matrix
        )
        np.testing.assert_allclose(self.pure_intensities, corrected_intensities)

    def test_correct_isobaric_reporter_impurities(self):
        intensity_table = self.contaminated_intensities.reshape(1, -1)
        expected_corrected_table = self.pure_intensities.reshape(1, -1)
        corrected_table = msreport.isobar.correct_isobaric_reporter_impurities(
            intensity_table, self.impurity_matrix
        )
        np.testing.assert_allclose(corrected_table, expected_corrected_table)


class TestFittingOfIsotopeImpurityCorrecter:
    @pytest.fixture(autouse=True)
    def _init_impurity_matrix(self):
        self.impurity_matrix = np.array(
            [
                [0.90, 0.10, 0.00, 0.00, 0.00],
                [0.10, 0.80, 0.01, 0.00, 0.00],
                [0.00, 0.01, 0.95, 0.01, 0.03],
                [0.00, 0.09, 0.02, 0.98, 0.10],
                [0.00, 0.00, 0.02, 0.01, 0.80],
            ]
        )

    def test_is_not_fitted_when_created(self):
        transformer = msreport.isobar.IsotopeImpurityCorrecter()
        assert not transformer.is_fitted()

    def test_is_fitted_after_fitting(self):
        transformer = msreport.isobar.IsotopeImpurityCorrecter().fit(self.impurity_matrix)  # fmt: skip
        assert transformer.is_fitted()

    def test_fitting_stores_correct_fits(self):
        transformer = msreport.isobar.IsotopeImpurityCorrecter().fit(self.impurity_matrix)  # fmt: skip
        np.testing.assert_equal(transformer.get_fits(), self.impurity_matrix)

    def test_fitting_raises_value_error_when_impurity_matrix_is_not_square(self):
        transformer = msreport.isobar.IsotopeImpurityCorrecter()
        with pytest.raises(ValueError):
            transformer.fit(self.impurity_matrix[:, :-1])

    def test_fitting_raises_value_error_when_impurity_matrix_contains_nan(self):
        transformer = msreport.isobar.IsotopeImpurityCorrecter()
        self.impurity_matrix[0, -1] = np.nan
        with pytest.raises(ValueError):
            transformer.fit(self.impurity_matrix)


class TestIsotopeImpurityCorrecterTransform:
    @pytest.fixture(autouse=True)
    def _init_impurity_matrix(self):
        self.impurity_matrix = np.array(
            [
                [0.90, 0.10, 0.00, 0.00, 0.00],
                [0.10, 0.80, 0.01, 0.00, 0.00],
                [0.00, 0.01, 0.95, 0.01, 0.03],
                [0.00, 0.09, 0.02, 0.98, 0.10],
                [0.00, 0.00, 0.02, 0.01, 0.80],
            ]
        )
        self.table = pd.DataFrame(np.array([[380, 201, 99.5, 123, 43]], dtype=float))
        self.corrected_table = pd.DataFrame(np.array([[400, 200, 100, 100, 50]], dtype=float))  # fmt: skip

    def test_transform_on_unfitted_normalizer_raises_NotFittedError(self):
        transformer = msreport.isobar.IsotopeImpurityCorrecter()
        with pytest.raises(NotFittedError):
            transformer.transform(self.table)

    def test_transform_of_table_with_wrong_column_number_raises_value_error(self):
        transformer = msreport.isobar.IsotopeImpurityCorrecter().fit(self.impurity_matrix)  # fmt: skip
        with pytest.raises(ValueError):
            transformer.transform(self.table[self.table.columns[:-1]])

    def test_transform_does_not_modify_input_table(self):
        transformer = msreport.isobar.IsotopeImpurityCorrecter().fit(self.impurity_matrix)  # fmt: skip
        transformer.transform(self.table)
        pd.testing.assert_frame_equal(self.table, self.table)

    def test_transform_applies_correct_correction(self):
        transformer = msreport.isobar.IsotopeImpurityCorrecter().fit(self.impurity_matrix)  # fmt: skip
        corrected_table = transformer.transform(self.table)
        pd.testing.assert_frame_equal(corrected_table, self.corrected_table)
