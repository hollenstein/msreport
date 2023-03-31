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


class TestCalculatePairWiseRatioMatrix:
    def test_corect_shape_with_multi_row_array(self, example_array):
        matrix = MAXLFQ.calculate_pairwise_ratio_matrix(example_array)

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
        matrix = MAXLFQ.calculate_pairwise_ratio_matrix(single_row_array)

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
        matrix = MAXLFQ.calculate_pairwise_ratio_matrix(example_array)
        median_matrix = np.nanmedian(matrix, axis=0)
        np.testing.assert_array_equal(expected_median_matrix, median_matrix)


class TestCalculatePairWiseMedianRatioMatrix:
    def test_corect_shape_with_multi_row_array(self, example_array):
        matrix = MAXLFQ.calculate_pairwise_median_ratio_matrix(example_array)
        # Condensed matrix must have two dimensions
        assert len(matrix.shape) == 2
        # Matrix must be square
        assert matrix.shape[0] == matrix.shape[1]
        # Size of matrix must correspond to array columns
        assert matrix.shape[0] == matrix.shape[1] == example_array.shape[1]

    def test_corect_shape_with_single_row_array(self, example_array):
        single_row_array = example_array[0].reshape(1, 3)
        matrix = MAXLFQ.calculate_pairwise_median_ratio_matrix(single_row_array)
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
        matrix = MAXLFQ.calculate_pairwise_median_ratio_matrix(example_array)
        np.testing.assert_array_equal(expected_matrix, matrix)
