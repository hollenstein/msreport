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
        columns=['Sample', 'Experiment']
    )
    data = pd.DataFrame({
        'Total peptides': [2, 1, 2],
        'Representative protein': ['A', 'B', 'C'],
        'Intensity Sample_A1': [10, np.nan, 10.3],
        'Intensity Sample_A2': [10, np.nan, 10.3],
        'Intensity Sample_B1': [15, np.nan, np.nan],
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
