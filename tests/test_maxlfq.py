import itertools

import numpy as np
import pandas as pd
import pytest

import msreport.helper.maxlfq as MAXLFQ


@pytest.fixture
def example_array():
    dataframe = pd.DataFrame(
        {
            "Peptides": ["1", "2", "3", "4"],
            "Intensity A": [4, 8, 16, 32],
            "Intensity B": [4, 8, 16, 64],
            "Intensity C": [8, 16, np.nan, 64],
        }
    )
    intensity_columns = ["Intensity A", "Intensity B", "Intensity C"]
    example_array = dataframe[intensity_columns].to_numpy()
    return example_array


@pytest.fixture
def multi_row_ratio_matrix():
    # fmt: off
    multi_row_ratio_matrix = np.array([
        [
            [0., 0., -1.],
            [0., 0., -1.],
            [1., 1., 0.]
        ],
        [
            [0., 0., -1.],
            [0., 0., -1.],
            [1., 1., 0.]
        ],
        [
            [0., 0., np.nan],
            [0., 0., np.nan],
            [np.nan, np.nan, np.nan]
        ],
        [
            [0., -1., -1.],
            [1., 0., 0.],
            [1., 0., 0.]
        ]
    ])
    # fmt: on
    return multi_row_ratio_matrix


@pytest.fixture
def single_row_ratio_matrix():
    single_row_ratio_matrix = np.array(
        [
            [0.0, 0.0, -1.0],
            [0.0, 0.0, np.nan],
            [1.0, np.nan, 0.0],
        ],
    )
    return single_row_ratio_matrix


class TestCalculatePairWiseLogRatioMatrix:
    def test_corect_shape_with_multi_row_array(self, example_array):
        matrix = MAXLFQ.calculate_pairwise_log_ratio_matrix(example_array)

        # Matrix mus have three dimensions
        assert len(matrix.shape) == 3
        # Inner matrices must be square
        assert matrix.shape[1] == matrix.shape[2]
        # Number of inner matrices must correspond to array rows
        assert matrix.shape[0] == example_array.shape[0]
        # Size of inner matrices must correspond to array columns
        assert matrix.shape[1] == matrix.shape[2] == example_array.shape[1]

    def test_corect_shape_with_single_row_array(self, example_array):
        single_row_array = example_array[0].reshape(1, 3)
        matrix = MAXLFQ.calculate_pairwise_log_ratio_matrix(single_row_array)

        # Condensed matrix must have two dimensions
        assert len(matrix.shape) == 3
        # Inner matrices must be square
        assert matrix.shape[1] == matrix.shape[2]
        # Number of inner matrices must correspond to array rows
        assert matrix.shape[0] == single_row_array.shape[0]
        # Size of inner matrices must correspond to array columns
        assert matrix.shape[1] == matrix.shape[2] == single_row_array.shape[1]

    def test_correct_values_with_multi_row_array(self, example_array):
        expected_median_matrix = np.array(
            [
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
                [1.0, 1.0, 0.0],
            ]
        )
        matrix = MAXLFQ.calculate_pairwise_log_ratio_matrix(example_array)
        median_matrix = np.nanmedian(matrix, axis=0)
        np.testing.assert_array_equal(expected_median_matrix, median_matrix)

    @pytest.mark.parametrize(
        "input_array",
        [
            np.empty((1, 2), dtype=np.int32),
            np.empty((1, 2), dtype=np.int64),
            np.empty((1, 2), dtype=int),
        ],
    )
    def test_with_integer_input_array(self, input_array):
        try:
            MAXLFQ.calculate_pairwise_log_ratio_matrix(input_array, log_transformed=True)  # fmt: skip
        except ValueError:
            assert False, "Using an integer input array raised an exception"


class TestCalculatePairWiseMedianLogRatioMatrix:
    def test_corect_shape_with_multi_row_array(self, example_array):
        matrix = MAXLFQ.calculate_pairwise_median_log_ratio_matrix(example_array)
        # Condensed matrix must have two dimensions
        assert len(matrix.shape) == 2
        # Matrix must be square
        assert matrix.shape[0] == matrix.shape[1]
        # Size of matrix must correspond to array columns
        assert matrix.shape[0] == matrix.shape[1] == example_array.shape[1]

    def test_corect_shape_with_single_row_array(self, example_array):
        single_row_array = example_array[0].reshape(1, 3)
        matrix = MAXLFQ.calculate_pairwise_median_log_ratio_matrix(single_row_array)
        # Condensed matrix must have two dimensions
        assert len(matrix.shape) == 2
        # Matrix must be square
        assert matrix.shape[0] == matrix.shape[1]
        # Size of matrix must correspond to array columns
        assert matrix.shape[0] == matrix.shape[1] == single_row_array.shape[1]

    def test_correct_values_with_multi_row_array(self, example_array):
        expected_matrix = np.array(
            [
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
                [1.0, 1.0, 0.0],
            ]
        )
        matrix = MAXLFQ.calculate_pairwise_median_log_ratio_matrix(example_array)
        np.testing.assert_array_equal(expected_matrix, matrix)


class TestCalculatePairWiseModeLogRatioMatrix:
    def test_corect_shape_with_multi_row_array(self, example_array):
        matrix = MAXLFQ.calculate_pairwise_mode_log_ratio_matrix(example_array)
        # Condensed matrix must have two dimensions
        assert len(matrix.shape) == 2
        # Matrix must be square
        assert matrix.shape[0] == matrix.shape[1]
        # Size of matrix must correspond to array columns
        assert matrix.shape[0] == matrix.shape[1] == example_array.shape[1]

    def test_corect_shape_with_single_row_array(self, example_array):
        single_row_array = example_array[0].reshape(1, 3)
        matrix = MAXLFQ.calculate_pairwise_mode_log_ratio_matrix(single_row_array)
        # Condensed matrix must have two dimensions
        assert len(matrix.shape) == 2
        # Matrix must be square
        assert matrix.shape[0] == matrix.shape[1]
        # Size of matrix must correspond to array columns
        assert matrix.shape[0] == matrix.shape[1] == single_row_array.shape[1]

    def test_correct_values_with_multi_row_array(self, example_array):
        expected_matrix = np.array(
            [
                [0.0, -0.010936, -1.0],
                [0.010936, 0.0, -0.939233],
                [1.0, 0.939233, 0.0],
            ]
        )
        matrix = MAXLFQ.calculate_pairwise_mode_log_ratio_matrix(example_array)
        np.testing.assert_allclose(expected_matrix, matrix, rtol=1e-04, atol=1e-04)


class TestPrepareCoefficientMatrix:
    @pytest.mark.parametrize(
        "ratio_matrix",
        [
            np.empty((3, 3)),
            np.empty((4, 4)),
            np.empty((5, 5)),
        ],
    )
    def test_correct_coef_matrix_shape_from_single_row_ratio_matrix(self, ratio_matrix):
        coef_matrix, ratio_array, initial_rows = MAXLFQ.prepare_coefficient_matrix(ratio_matrix)  # fmt: skip
        ratio_matrix_shape = ratio_matrix.shape
        coefficient_combinations = len(
            list(itertools.combinations(range(ratio_matrix_shape[-1]), 2))
        )
        expected_coefficient_rows = coefficient_combinations

        assert coef_matrix.shape[0] == expected_coefficient_rows
        assert coef_matrix.shape[1] == ratio_matrix_shape[-1]
        assert ratio_array.shape == (expected_coefficient_rows,)
        assert initial_rows.shape == (expected_coefficient_rows,)

    @pytest.mark.parametrize(
        "ratio_matrix",
        [
            np.empty((1, 2, 2)),
            np.empty((1, 9, 9)),
            np.empty((2, 3, 3)),
            np.empty((3, 3, 3)),
            np.empty((8, 3, 3)),
            np.empty((2, 6, 6)),
            np.empty((3, 6, 6)),
            np.empty((4, 6, 6)),
        ],
    )
    def test_correct_coef_matrix_shape_from_multi_row_ratio_matrix(self, ratio_matrix):
        coef_matrix, ratio_array, initial_rows = MAXLFQ.prepare_coefficient_matrix(ratio_matrix)  # fmt: skip
        ratio_matrix_shape = ratio_matrix.shape
        coefficient_combinations = len(
            list(itertools.combinations(range(ratio_matrix_shape[-1]), 2))
        )
        num_inner_ratio_matrices = ratio_matrix_shape[0]
        expected_coefficient_rows = coefficient_combinations * num_inner_ratio_matrices

        assert coef_matrix.shape[0] == expected_coefficient_rows
        assert coef_matrix.shape[1] == ratio_matrix_shape[-1]
        assert ratio_array.shape == (expected_coefficient_rows,)
        assert initial_rows.shape == (expected_coefficient_rows,)

    def test_correct_coef_matrix_prepared_from_single_row_ratio_matrix(self, single_row_ratio_matrix):  # fmt: skip
        coef_matrix, ratio_array, initial_rows = MAXLFQ.prepare_coefficient_matrix(
            single_row_ratio_matrix
        )
        for coefs, ratio, row in zip(coef_matrix, ratio_array, initial_rows):
            coef_1_column = np.where(coefs == 1)
            coef_2_column = np.where(coefs == -1)
            expected_ratio = single_row_ratio_matrix[coef_1_column, coef_2_column]
            np.testing.assert_equal(ratio, expected_ratio)

    def test_correct_coef_matrix_prepared_from_multi_row_ratio_matrix(self, multi_row_ratio_matrix):  # fmt: skip
        coef_matrix, ratio_array, initial_rows = MAXLFQ.prepare_coefficient_matrix(
            multi_row_ratio_matrix
        )
        for coefs, ratio, row in zip(coef_matrix, ratio_array, initial_rows):
            ratio_matrix = multi_row_ratio_matrix[row]
            coef_1_column = np.where(coefs == 1)
            coef_2_column = np.where(coefs == -1)
            expected_ratio = ratio_matrix[coef_1_column, coef_2_column]
            np.testing.assert_equal(ratio, expected_ratio)


class TestLogProfilesByLastsq:
    def test_expected_profiles_returned_with_simple_input(self):
        coef_matrix = np.array(
            [
                [1, -1, 0],
                [1, 0, -1],
                [0, 1, -1],
            ]
        )
        ratio_matrix = np.array([-1.0, -2.0, -1.0])
        log_profiles = MAXLFQ.log_profiles_by_lstsq(coef_matrix, ratio_matrix)

        expected_profile = np.array([0, 1, 2])
        observed_profile = log_profiles - np.nanmin(log_profiles)
        np.testing.assert_allclose(observed_profile, expected_profile)

    def test_expected_profiles_returned_with_long_input_and_nan(self):
        coef_matrix = np.array(
            [
                [1, -1, 0],
                [1, 0, -1],
                [0, 1, -1],
                [1, -1, 0],
                [1, 0, -1],
                [0, 1, -1],
                [1, -1, 0],
                [1, 0, -1],
                [0, 1, -1],
                [1, -1, 0],
                [1, 0, -1],
                [0, 1, -1],
            ]
        )
        ratio_array = np.array([0, -1, -1, 0, -1, -1, 0, np.nan, np.nan, -1, -1, 0])
        log_profiles = MAXLFQ.log_profiles_by_lstsq(coef_matrix, ratio_array)

        expected_profile = np.array([0, 0.27272727, 0.96969697])
        observed_profile = log_profiles - np.nanmin(log_profiles)
        np.testing.assert_allclose(
            observed_profile, expected_profile, atol=1e-08, equal_nan=True
        )

    def test_empty_matrix_returns_all_nans(self):
        coef_matrix = np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )
        ratio_matrix = np.array([1, 2, 3])
        observed_profile = MAXLFQ.log_profiles_by_lstsq(coef_matrix, ratio_matrix)
        expected_profile = np.array([np.nan, np.nan, np.nan])
        np.testing.assert_allclose(observed_profile, expected_profile)

    def test_columns_not_present_in_coef_matrix_return_nan(self):
        coef_matrix = np.array(
            [
                [0, 1, -1, 0, 0],
                [0, 1, 0, -1, 0],
                [0, 0, 1, -1, 0],
            ]
        )
        ratio_matrix = np.array([-1.0, -2.0, -1.0])
        log_profiles = MAXLFQ.log_profiles_by_lstsq(coef_matrix, ratio_matrix)
        assert np.all(np.isnan(log_profiles) == [True, False, False, False, True])


class TestCalculationOfProfilesFromIntensities:
    # fmt: off
    def test_with_complete_ratio_matrix_calculation(self, example_array):
        ratio_matrix = MAXLFQ.calculate_pairwise_log_ratio_matrix(example_array, log_transformed=False)
        coef_matrix, ratio_array, initial_rows = MAXLFQ.prepare_coefficient_matrix(ratio_matrix)
        log_profiles = MAXLFQ.log_profiles_by_lstsq(coef_matrix, ratio_array)

        expected_profile = np.array([0, 0.27272727, 0.96969697])
        observed_profile = log_profiles - np.nanmin(log_profiles)
        np.testing.assert_allclose(observed_profile, expected_profile, atol=1e-08, equal_nan=True)

    def test_with_median_ratio_matrix_calculation(self, example_array):
        ratio_matrix = MAXLFQ.calculate_pairwise_median_log_ratio_matrix(example_array, log_transformed=False)
        coef_matrix, ratio_array, initial_rows = MAXLFQ.prepare_coefficient_matrix(ratio_matrix)
        log_profiles = MAXLFQ.log_profiles_by_lstsq(coef_matrix, ratio_array)

        expected_profile = np.array([0., 0., 1.])
        observed_profile = log_profiles - np.nanmin(log_profiles)
        np.testing.assert_allclose(observed_profile, expected_profile, atol=1e-08, equal_nan=True)

    def test_with_missing_values(self, example_array):
        example_array[:, 1] = np.nan
        ratio_matrix = MAXLFQ.calculate_pairwise_median_log_ratio_matrix(example_array, log_transformed=False)
        coef_matrix, ratio_array, initial_rows = MAXLFQ.prepare_coefficient_matrix(ratio_matrix)
        log_profiles = MAXLFQ.log_profiles_by_lstsq(coef_matrix, ratio_array)

        expected_profile = np.array([0., np.nan, 1.])
        observed_profile = log_profiles - np.nanmin(log_profiles)
        np.testing.assert_allclose(observed_profile, expected_profile, atol=1e-08, equal_nan=True)
    # fmt: on
