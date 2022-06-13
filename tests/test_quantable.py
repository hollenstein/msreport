import numpy as np
import pandas as pd
import pytest
import quantable


@pytest.fixture
def example_data():
    design = pd.DataFrame(
        [('Sample_A1', 'Experiment_A'), ('Sample_A2', 'Experiment_A'),
         ('Sample_B1', 'Experiment_B'), ('Sample_B2', 'Experiment_B')],
        columns=['Sample', 'Experiment']
    )

    data = pd.DataFrame({
        'id': [11, 11, 11],
        'Total peptides': [2, 1, 2],
        'Representative protein': ['A', 'B', 'C'],
        'Intensity Sample_A1': [10, np.nan, 6],
        'Intensity Sample_A2': [10, np.nan, 2],
        'Intensity Sample_B1': [15, np.nan, 0],  # <- 0 is considered missing
        'Intensity Sample_B2': [15, np.nan, 4],
    })

    example_data = {
        'data': data,
        'design': design,
        'expression_tag': 'Intensity',
        'expression_columns': [
            'Intensity Sample_A1', 'Intensity Sample_A2',
            'Intensity Sample_B1', 'Intensity Sample_B2'
        ],
        'expr_sample_mapping': {
            'Intensity Sample_A1': 'Sample_A1',
            'Intensity Sample_A2': 'Sample_A2',
            'Intensity Sample_B1': 'Sample_B1',
            'Intensity Sample_B2': 'Sample_B2'
        },
    }
    return example_data


@pytest.fixture
def example_qtable(example_data):
    qtable = quantable.Qtable(
        example_data['data'], design=example_data['design']
    )
    qtable.set_expression_by_tag('Intensity')
    return qtable


def test_str_to_substr_mapping():
    strings = ['Tag SampleB_1', 'Tag SampleA_1', 'Tag SampleA_2']
    substrs = ['SampleA_1', 'SampleB_1', 'SampleA_2']
    true_mapping = {
        'Tag SampleA_1': 'SampleA_1',
        'Tag SampleA_2': 'SampleA_2',
        'Tag SampleB_1': 'SampleB_1',
    }
    assert quantable._str_to_substr_mapping(strings, substrs) == true_mapping


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
    qtable = quantable.Qtable(
        example_data['data'], design=example_data['design']
    )
    assert qtable.get_design().equals(example_data['design'])


def test_qtable_get_samples(example_data, example_qtable):
    design = example_data['design']

    samples = design['Sample'].tolist()
    assert example_qtable.get_samples() == samples

    for exp in design['Experiment'].unique():
        samples = design[design['Experiment'] == exp]['Sample'].tolist()
        assert example_qtable.get_samples(exp) == samples


def test_qtable_get_experiments(example_data, example_qtable):
    experiment_set = set(example_data['design']['Experiment'])
    assert set(example_qtable.get_experiments()) == experiment_set


def test_qtable_set_expression_raises_value_error(example_data):
    qtable = quantable.Qtable(
        example_data['data'], design=example_data['design']
    )
    # Raise error when expression columns are not present in the data table
    with pytest.raises(ValueError):
        qtable._set_expression({'column_not_present': 'Sample_A1'})

    # Raise error when expr_column_mapping samples are not present in the design
    with pytest.raises(ValueError):
        qtable._set_expression({'Intensity Sample_A1': 'sample not present'})


def test_qtable_set_expression(example_data):
    qtable = quantable.Qtable(
        example_data['data'], design=example_data['design']
    )
    qtable._set_expression(example_data['expr_sample_mapping'])
    assert qtable._expression_columns == example_data['expression_columns']
    assert qtable._expression_sample_mapping == example_data['expr_sample_mapping']


def test_qtable_set_expression_with_zerotonan_false(example_data):
    qtable = quantable.Qtable(
        example_data['data'], design=example_data['design']
    )
    qtable._set_expression(
        example_data['expr_sample_mapping'], zerotonan=False
    )
    expr_table = qtable.data[example_data['expression_columns']]
    num_zero = (expr_table == 0).sum().sum()
    num_nan = expr_table.isna().sum().sum()
    assert num_zero == 1
    assert num_nan == 4


def test_qtable_set_expression_with_zerotonan(example_data):
    qtable = quantable.Qtable(
        example_data['data'], design=example_data['design']
    )
    qtable._set_expression(
        example_data['expr_sample_mapping'], zerotonan=True
    )

    expr_table = qtable.data[example_data['expression_columns']]
    num_zero = (expr_table == 0).sum().sum()
    num_nan = expr_table.isna().sum().sum()
    assert num_zero == 0
    assert num_nan == 5


def test_qtable_set_expression_with_log2(example_data):
    qtable = quantable.Qtable(
        example_data['data'], design=example_data['design']
    )
    qtable._set_expression(
        example_data['expr_sample_mapping'], log2=True
    )

    expected_table = example_data['data'][example_data['expression_columns']]
    expected_table = np.log2(expected_table.replace({0: np.nan}))
    expr_table = qtable.data[example_data['expression_columns']]
    assert expr_table.equals(expected_table)


def test_qtable_set_expression_by_tag(example_data):
    qtable = quantable.Qtable(example_data['data'], design=example_data['design'])
    qtable.set_expression_by_tag(example_data['expression_tag'])
    assert qtable._expression_columns == example_data['expression_columns']
    assert qtable._expression_sample_mapping == example_data['expr_sample_mapping']


def test_qtable_set_expression_by_tag_with_zerotonan(example_data):
    qtable = quantable.Qtable(example_data['data'], design=example_data['design'])
    qtable.set_expression_by_tag(
        example_data['expression_tag'], zerotonan=True
    )

    expr_table = qtable.data[example_data['expression_columns']]
    num_zero = (expr_table == 0).sum().sum()
    assert num_zero == 0


def test_qtable_set_expression_by_tag_with_log2(example_data):
    qtable = quantable.Qtable(example_data['data'], design=example_data['design'])
    qtable.set_expression_by_tag(example_data['expression_tag'], log2=True)

    expr_table = qtable.data[example_data['expression_columns']]
    expected_table = example_data['data'][example_data['expression_columns']]
    expected_table = np.log2(expected_table.replace({0: np.nan}))
    assert expr_table.equals(expected_table)


def test_qtable_set_expression_by_column(example_data):
    qtable = quantable.Qtable(example_data['data'], design=example_data['design'])
    qtable.set_expression_by_column(example_data['expr_sample_mapping'])
    assert qtable._expression_columns == example_data['expression_columns']
    assert qtable._expression_sample_mapping == example_data['expr_sample_mapping']


def test_qtable_set_expression_by_column_with_zerotonan(example_data):
    qtable = quantable.Qtable(example_data['data'], design=example_data['design'])
    qtable.set_expression_by_column(
        example_data['expr_sample_mapping'], zerotonan=True
    )

    expr_table = qtable.data[example_data['expression_columns']]
    num_zero = (expr_table == 0).sum().sum()
    assert num_zero == 0


def test_qtable_set_expression_by_column_with_log2(example_data):
    qtable = quantable.Qtable(example_data['data'], design=example_data['design'])
    qtable.set_expression_by_column(
        example_data['expr_sample_mapping'], log2=True
    )

    expr_table = qtable.data[example_data['expression_columns']]
    expected_table = example_data['data'][example_data['expression_columns']]
    expected_table = np.log2(expected_table.replace({0: np.nan}))
    assert expr_table.equals(expected_table)


def test_get_expression_column(example_data, example_qtable):
    expected_columns = [c for c in example_data['expr_sample_mapping']]
    samples = [example_data['expr_sample_mapping'][e] for e in expected_columns]
    columns = [example_qtable.get_expression_column(s) for s in samples]
    assert expected_columns == columns


def test_qtable_make_expression_table(example_data, example_qtable):
    expected_table = example_data['data'][example_data['expression_columns']]

    # Test for correct values in dataframe
    expr_table = example_qtable.make_expression_table()
    assert expr_table.equals(expected_table)


def test_qtable_make_expression_table_with_samples_as_columns(example_data, example_qtable):
    expr_table = example_qtable.make_expression_table(samples_as_columns=True)
    expr_table_columns = expr_table.columns.tolist()

    sample_names = example_data['design']['Sample'].tolist()
    assert expr_table_columns == sample_names


def test_qtable_make_expression_table_with_additional_features(example_data, example_qtable):
    expr_table = example_qtable.make_expression_table(features=['id'])
    assert 'id' in expr_table.columns
    assert example_data['data']['id'].equals(expr_table['id'])


def test_qtable_make_expression_table_with_all_arguments(example_data, example_qtable):
    expr_table_by_qtable = example_qtable.make_expression_table(
        features=['id'], samples_as_columns=True
    )
    assert isinstance(expr_table_by_qtable, pd.DataFrame)


def test_impute_missing_values(example_qtable):
    example_qtable.impute_missing_values()

    samples = example_qtable.get_samples()
    expr_table = example_qtable.make_expression_table(samples_as_columns=True)
    number_missing_values = expr_table[samples].isna().sum().sum()
    assert number_missing_values == 0


def test_calculate_experiment_means(example_data, example_qtable):
    example_qtable.calculate_experiment_means()

    experiments = example_qtable.get_experiments()
    assert all([e in example_qtable.data for e in experiments])
    assert all([e in example_qtable._expression_features for e in experiments])
    assert np.allclose(
        example_qtable.data['Experiment_A'],
        [10., np.nan, 4.],
        equal_nan=True
    )
