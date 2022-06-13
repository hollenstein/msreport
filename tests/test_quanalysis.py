import numpy as np
import pandas as pd
import pytest
import quantable
import quanalysis


@pytest.fixture
def example_data():
    design = pd.DataFrame(
        [('Sample_A1', 'Experiment_A'), ('Sample_A2', 'Experiment_A'),
         ('Sample_B1', 'Experiment_B'), ('Sample_B2', 'Experiment_B')],
        columns=['Sample', 'Experiment'])
    data = pd.DataFrame({
        'Total peptides': [2, 1, 2],
        'Representative protein': ['A', 'B', 'C'],
        'Intensity Sample_A1': [10, np.nan, 10.3],
        'Intensity Sample_A2': [10, np.nan, 10.3],
        'Intensity Sample_B1': [15, np.nan, 0],  # <- 0 is considered missing
        'Intensity Sample_B2': [15, np.nan, 10.3],
    })
    missing_values = pd.DataFrame({
        'Missing total': [0, 4, 1],
        'Missing Experiment_A': [0, 2, 0],
        'Missing Experiment_B': [0, 2, 1],
    })

    example_data = {
        'data': data, 'design': design, 'missing_values': missing_values
    }
    return example_data


@pytest.fixture
def example_qtable(example_data):
    qtable = quantable.Qtable(
        example_data['data'], design=example_data['design']
    )
    qtable.set_expression_by_tag('Intensity')
    return qtable


def test_count_missing_values(example_data, example_qtable):
    missing_values = quanalysis.count_missing_values(example_qtable)
    expected = example_data['missing_values']
    assert missing_values.to_dict() == expected.to_dict()


def test_gaussian_imputation(example_data):
    table = pd.DataFrame({
        'A': [90000, 100000, 110000, np.nan],
        'B': [1, 1, 1, np.nan],
    })
    median_downshift = 1
    std_width = 1
    imputed = quanalysis.gaussian_imputation(
        table, median_downshift, std_width
    )

    number_missing_values = imputed.isna().sum().sum()
    assert number_missing_values == 0

    imputed_column_A = imputed['A'][3]
    imputed_column_B = imputed['B'][3]
    assert imputed_column_A >= imputed_column_B


def test_impute_missing_values(example_qtable):
    quanalysis.impute_missing_values(example_qtable)

    samples = example_qtable.get_samples()
    expr_table = example_qtable.get_expression_table(
        samples_as_columns=True, zerotonan=True
    )
    number_missing_values = expr_table[samples].isna().sum().sum()
    assert number_missing_values == 0


def test_add_experiment_means(example_qtable):
    quanalysis.add_experiment_means(example_qtable)
