import numpy as np
import pandas as pd
import pytest
import quantable


@pytest.fixture
def example_data():
    table = pd.DataFrame(
        data=[['A', 11, 2, 3, 6],
              ['B', 11, 0, 3, 6],
              ['C', 11, 2, 3, 6]],
        columns=['id', 'Tag', 'Tag SampleA_1',
                 'Tag SampleA_2', 'Tag SampleB_1'],
    )
    design = pd.DataFrame(
        data=[['Experiment_A', 'SampleA_1'],
              ['Experiment_A', 'SampleA_2'],
              ['Experiment_B', 'SampleB_1']],
        columns=['Experiment', 'Sample'],
    )
    example_data = {'table': table, 'design': design, 'expression_tag': 'Tag',
                    'expression_columns': ['Tag SampleA_1', 'Tag SampleA_2',
                                           'Tag SampleB_1']
                    }
    return example_data


def test_that_always_passes():
    assert True


def test_str_to_substr_mapping():
    strings = ['Tag SampleB_1', 'Tag SampleA_1', 'Tag SampleA_2']
    substrs = ['SampleA_1', 'SampleB_1', 'SampleA_2']
    true_mapping = {
        'Tag SampleA_1': 'SampleA_1',
        'Tag SampleA_2': 'SampleA_2',
        'Tag SampleB_1': 'SampleB_1',
    }
    assert quantable._str_to_substr_mapping(strings, substrs) == true_mapping


def test_map_samples_to_columns():
    samples = ['SamASamA_1', 'SamA_1', 'SamA_2', 'X']
    columns = ['Tag SamASamA_1', 'Tag SamA_1', 'Tag SamA_2', 'Tag']
    tag = 'Tag'
    true_mapping = {
        'Tag SamASamA_1': 'SamASamA_1',
        'Tag SamA_1': 'SamA_1',
        'Tag SamA_2': 'SamA_2',
    }
    mapping = quantable._map_samples_to_columns(columns, samples, tag)
    assert mapping == true_mapping


def test_qtable_setup():
    qtable = quantable.Qtable(pd.DataFrame())
    assert isinstance(qtable.data, pd.DataFrame)


def test_qtable_add_design(example_data):
    qtable = quantable.Qtable(pd.DataFrame())
    qtable.add_design(example_data['design'])
    assert qtable._design.equals(example_data['design'])

    with pytest.raises(ValueError):
        qtable.add_design(pd.DataFrame(columns=['Sample']))
    with pytest.raises(ValueError):
        qtable.add_design(pd.DataFrame(columns=['Experiment']))


def test_qtable_setup_with_design(example_data):
    qtable = quantable.Qtable(pd.DataFrame(), design=example_data['design'])
    assert qtable._design.equals(example_data['design'])


def test_qtable_get_design(example_data):
    qtable = quantable.Qtable(example_data['table'],
                              design=example_data['design'])
    assert qtable.get_design().equals(example_data['design'])


def test_qtable_get_samples(example_data):
    qtable = quantable.Qtable(example_data['table'],
                              design=example_data['design'])
    design = example_data['design']

    samples = design['Sample'].tolist()
    assert qtable.get_samples() == samples

    for exp in design['Experiment'].unique():
        samples = design[design['Experiment'] == exp]['Sample'].tolist()
        assert qtable.get_samples(exp) == samples


def test_qtable_get_experiments(example_data):
    qtable = quantable.Qtable(example_data['table'],
                              design=example_data['design'])
    experiment_set = set(example_data['design']['Experiment'])
    assert set(qtable.get_experiments()) == experiment_set


def test_qtable_set_expression(example_data):
    qtable = quantable.Qtable(example_data['table'])

    # Raise error when columns are not present in the data table
    with pytest.raises(ValueError):
        qtable._set_expression(['not_present_column'], 'not')
    # Raise error when the tag is not present in the expression columns
    with pytest.raises(ValueError):
        qtable._set_expression(example_data['expression_columns'], 'not')

    qtable._set_expression(example_data['expression_columns'],
                           example_data['expression_tag'])
    assert qtable._expression_columns == example_data['expression_columns']
    assert qtable._expression_tag == example_data['expression_tag']


def test_qtable_set_expression_by_tag(example_data):
    qtable = quantable.Qtable(example_data['table'])
    qtable.set_expression_by_tag(example_data['expression_tag'])
    assert qtable._expression_columns == example_data['expression_columns']


def test_qtable_set_expression_by_column(example_data):
    qtable = quantable.Qtable(example_data['table'])
    qtable.set_expression_by_column(example_data['expression_columns'],
                                    example_data['expression_tag'])
    assert qtable._expression_columns == example_data['expression_columns']


def test_qtable_get_expression_table(example_data):
    qtable = quantable.Qtable(example_data['table'])
    qtable.set_expression_by_tag(example_data['expression_tag'])
    expr_table = example_data['table'][example_data['expression_columns']]

    # Test for correct values in dataframe
    expr_table_by_qtable = qtable.get_expression_table()
    assert expr_table_by_qtable.equals(expr_table)


def test_qtable_get_expression_table_with_arguments(example_data):
    qtable = quantable.Qtable(example_data['table'])
    qtable.set_expression_by_tag(example_data['expression_tag'])

    # Test replacing expression columns with sample column names
    with pytest.raises(ValueError):
        qtable.get_expression_table(samples_as_columns=True)
    qtable.add_design(example_data['design'])
    expr_table_by_qtable = qtable.get_expression_table(samples_as_columns=True)
    expr_table_columns = expr_table_by_qtable.columns.tolist()
    replaced_column_names = example_data['design']['Sample'].tolist()
    assert expr_table_columns == replaced_column_names

    # Test that additional feature columns have been included
    expr_table_by_qtable = qtable.get_expression_table(features=['id'])
    assert 'id' in expr_table_by_qtable.columns
    assert example_data['table']['id'].equals(expr_table_by_qtable['id'])

    # Test for correct values in dataframe when zeros are not replace by nan
    expr_table_by_qtable = qtable.get_expression_table(zerotonan=False)
    num_zero = (expr_table_by_qtable == 0).sum().sum()
    num_nan = expr_table_by_qtable.isna().sum().sum()
    assert num_zero == 1
    assert num_nan == 0

    # Test for correct values in dataframe when zeros are replace by nan
    expr_table_by_qtable = qtable.get_expression_table(zerotonan=True)
    num_zero = (expr_table_by_qtable == 0).sum().sum()
    num_nan = expr_table_by_qtable.isna().sum().sum()
    assert num_zero == 0
    assert num_nan == 1

    # Test for correct values in dataframe when values are log2 transformed
    expr_table = example_data['table'][example_data['expression_columns']]
    expr_table = expr_table.replace({0: np.nan})
    expr_table = np.log2(expr_table)
    expr_table_by_qtable = qtable.get_expression_table(log2=True)
    assert expr_table_by_qtable.equals(expr_table)

    # Test all optional arguments together
    expr_table_by_qtable = qtable.get_expression_table(
        features=['id'], samples_as_columns=True, zerotonan=True, log2=True
    )
    assert isinstance(expr_table_by_qtable, pd.DataFrame)
