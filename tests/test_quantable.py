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
        'id': ['1', '2', '3'],
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
        'intensity_columns': [
            'Intensity Sample_A1',
            'Intensity Sample_A2',
            'Intensity Sample_B1',
            'Intensity Sample_B2'
        ],
        'expression_columns': [
            'Expression Sample_A1',
            'Expression Sample_A2',
            'Expression Sample_B1',
            'Expression Sample_B2'
        ],
        'intensity_cols_to_samples': {
            'Intensity Sample_A1': 'Sample_A1',
            'Intensity Sample_A2': 'Sample_A2',
            'Intensity Sample_B1': 'Sample_B1',
            'Intensity Sample_B2': 'Sample_B2'
        },
        'expr_cols_to_samples': {
            'Expression Sample_A1': 'Sample_A1',
            'Expression Sample_A2': 'Sample_A2',
            'Expression Sample_B1': 'Sample_B1',
            'Expression Sample_B2': 'Sample_B2'
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
    assert qtable.design.equals(example_data['design'])

    with pytest.raises(ValueError):
        qtable.add_design(pd.DataFrame(columns=['Sample']))
    with pytest.raises(ValueError):
        qtable.add_design(pd.DataFrame(columns=['Experiment']))


def test_qtable_setup_with_design(example_data):
    qtable = quantable.Qtable(pd.DataFrame(), design=example_data['design'])
    assert qtable.design.equals(example_data['design'])


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


class TestQtableResetExpression:
    @pytest.fixture(autouse=True)
    def _init_qtable(self, example_data):
        self.qtable = quantable.Qtable(
            example_data['data'], design=example_data['design']
        )

    def test_reset_of_parameters(self):
        self.qtable._expression_columns = ['test']
        self.qtable._expression_features = ['test']
        self.qtable._expression_sample_mapping = {'test': 'test'}

        self.qtable._reset_expression()
        assert self.qtable._expression_columns == []
        assert self.qtable._expression_features == []
        assert self.qtable._expression_sample_mapping == {}

    def test_reset_of_data_columns(self, example_data):
        self.qtable._set_expression(example_data['intensity_cols_to_samples'])

        self.qtable._reset_expression()
        data_columns = self.qtable.data.columns
        all_expression_columns_absent_in_data = not any(
            [c in data_columns for c in example_data['expression_columns']]
        )
        assert all_expression_columns_absent_in_data

    def test_reset_of_expression_features(self, example_data):
        # TODO: do this via add_expression_features function, once implemented
        new_feature = example_data['data']['id']
        new_feature.name = 'Feature'

        self.qtable.add_expression_features(new_feature)
        assert 'Feature' in self.qtable.data.columns

        self.qtable._reset_expression()
        assert 'Feature' not in self.qtable.data.columns


class TestQtableSetExpression:
    @pytest.fixture(autouse=True)
    def _init_qtable(self, example_data):
        self.qtable = quantable.Qtable(
            example_data['data'], design=example_data['design']
        )

    def test_set_expression(self, example_data):
        self.qtable._set_expression(example_data['intensity_cols_to_samples'])

        assert self.qtable._expression_columns == example_data['expression_columns']
        assert self.qtable._expression_sample_mapping == example_data['expr_cols_to_samples']

    def test_with_zerotonan_false(self, example_data):
        self.qtable._set_expression(
            example_data['intensity_cols_to_samples'], zerotonan=False
        )

        expr_table = self.qtable.data[example_data['expression_columns']]
        num_zero = (expr_table == 0).sum().sum()
        num_nan = expr_table.isna().sum().sum()
        assert num_zero == 1
        assert num_nan == 4

    def test_with_zerotonan(self, example_data):
        self.qtable._set_expression(
            example_data['intensity_cols_to_samples'], zerotonan=True
        )

        expr_table = self.qtable.data[example_data['expression_columns']]
        num_zero = (expr_table == 0).sum().sum()
        num_nan = expr_table.isna().sum().sum()
        assert num_zero == 0
        assert num_nan == 5

    def test_with_log2(self, example_data):
        self.qtable._set_expression(
            example_data['intensity_cols_to_samples'], log2=True
        )

        expr_table = self.qtable.data[example_data['expression_columns']]
        expected = example_data['data'][example_data['intensity_columns']]
        expected = np.log2(expected.replace({0: np.nan}))
        assert np.array_equal(expr_table.to_numpy(), expected.to_numpy(), equal_nan=True)

    def test_raises_value_errors(self, example_data):
        # Raise error when expression columns are not present in the data table
        with pytest.raises(ValueError):
            self.qtable._set_expression({'column_not_present': 'Sample_A1'})

        # Raise error when expr_column_mapping samples are not present in the design
        with pytest.raises(ValueError):
            self.qtable._set_expression({'Intensity Sample_A1': 'sample not present'})


class TestQtableSetExpressionByTag:
    @pytest.fixture(autouse=True)
    def _init_qtable(self, example_data):
        self.qtable = quantable.Qtable(
            example_data['data'], design=example_data['design']
        )

    def test_set_expression_by_tag(self, example_data):
        self.qtable.set_expression_by_tag(example_data['expression_tag'])

        assert self.qtable._expression_columns == example_data['expression_columns']
        assert self.qtable._expression_sample_mapping == example_data['expr_cols_to_samples']

    def test_with_zerotonan(self, example_data):
        self.qtable.set_expression_by_tag(example_data['expression_tag'], zerotonan=True)

        expr_table = self.qtable.data[example_data['expression_columns']]
        num_zero = (expr_table == 0).sum().sum()
        assert num_zero == 0

    def test_with_log2(self, example_data):
        self.qtable.set_expression_by_tag(example_data['expression_tag'], log2=True)

        expr_table = self.qtable.data[example_data['expression_columns']]
        expected = example_data['data'][example_data['intensity_columns']]
        expected = np.log2(expected.replace({0: np.nan}))
        assert np.array_equal(expr_table.to_numpy(), expected.to_numpy(), equal_nan=True)


class TestQtableSetExpressionByColumn:
    @pytest.fixture(autouse=True)
    def _init_qtable(self, example_data):
        self.qtable = quantable.Qtable(
            example_data['data'], design=example_data['design']
        )

    def test_set_expression_by_column(self, example_data):
        self.qtable.set_expression_by_column(example_data['intensity_cols_to_samples'])
        assert self.qtable._expression_columns == example_data['expression_columns']
        assert self.qtable._expression_sample_mapping == example_data['expr_cols_to_samples']

    def test_with_zerotonan(self, example_data):
        self.qtable.set_expression_by_column(
            example_data['intensity_cols_to_samples'], zerotonan=True
        )

        expr_table = self.qtable.data[example_data['expression_columns']]
        num_zero = (expr_table == 0).sum().sum()
        assert num_zero == 0

    def test_with_log2(self, example_data):
        self.qtable.set_expression_by_column(
            example_data['intensity_cols_to_samples'], log2=True
        )

        expr_table = self.qtable.data[example_data['expression_columns']]
        expected = example_data['data'][example_data['intensity_columns']]
        expected = np.log2(expected.replace({0: np.nan}))
        assert np.array_equal(expr_table.to_numpy(), expected.to_numpy(), equal_nan=True)


class TestQtableAddExpressionFeature:
    @pytest.fixture(autouse=True)
    def _init_qtable(self, example_data):
        self.qtable = quantable.Qtable(
            example_data['data'], design=example_data['design']
        )

    def test_with_series(self):
        new_data = self.qtable.data['id'].copy()
        new_data.name = 'Feature'
        self.qtable.add_expression_features(new_data)

        qtable_columns = self.qtable.data.columns.to_list()
        assert 'Feature' in qtable_columns
        assert 'Feature' in self.qtable._expression_features

    def test_with_dataframe(self):
        new_data = self.qtable.data[['id', 'id']].copy()
        new_data.columns = ['Feature 1', 'Feature 2']
        self.qtable.add_expression_features(new_data)

        qtable_columns = self.qtable.data.columns.to_list()
        for new_column in new_data.columns:
            assert new_column in qtable_columns
            assert new_column in self.qtable._expression_features

    def test_qtable_data_integrity(self):
        old_columns = self.qtable.data.columns.to_list()
        old_shape = self.qtable.data.shape

        new_data = self.qtable.data['id'].copy()
        new_data.name = 'Feature'
        self.qtable.add_expression_features(new_data)

        assert [column in self.qtable.data for column in old_columns]
        assert self.qtable.data.shape[0] == old_shape[0]
        assert self.qtable.data.shape[1] == old_shape[1] + 1


def test_qtable_get_expression_column(example_data, example_qtable):
    expected_columns = [c for c in example_data['expr_cols_to_samples']]
    samples = [example_data['expr_cols_to_samples'][e] for e in expected_columns]
    columns = [example_qtable.get_expression_column(s) for s in samples]
    assert expected_columns == columns


class TestQtableMakeExpressionMatrix:
    @pytest.fixture(autouse=True)
    def _init_qtable(self, example_qtable):
        self.qtable = example_qtable

    def test_default_args(self, example_data):
        expected = example_data['data'][example_data['intensity_columns']]

        # Test for correct values in dataframe
        expr_matrix = self.qtable.make_expression_matrix()
        assert np.array_equal(expr_matrix.to_numpy(), expected.to_numpy(), equal_nan=True)

    def test_with_samples_as_columns(self, example_data):
        expr_matrix = self.qtable.make_expression_matrix(samples_as_columns=True)
        expr_matrix_columns = expr_matrix.columns.tolist()

        sample_names = example_data['design']['Sample'].tolist()
        assert expr_matrix_columns == sample_names


class TestQtableMakeExpressionTable:
    @pytest.fixture(autouse=True)
    def _init_qtable(self, example_qtable):
        self.qtable = example_qtable

    def test_default_args(self, example_data):
        expected = example_data['data'][example_data['intensity_columns']]

        # Test for correct values in dataframe
        expr_table = self.qtable.make_expression_table()
        assert np.array_equal(expr_table.to_numpy(), expected.to_numpy(), equal_nan=True)

    def test_with_samples_as_columns(self, example_data):
        expr_table = self.qtable.make_expression_table(samples_as_columns=True)
        expr_table_columns = expr_table.columns.tolist()

        sample_names = example_data['design']['Sample'].tolist()
        assert expr_table_columns == sample_names

    def test_with_additional_features(self, example_data):
        expr_table = self.qtable.make_expression_table(features=['id'])
        assert 'id' in expr_table.columns
        assert example_data['data']['id'].equals(expr_table['id'])

    def test_with_all_arguments(self, example_data):
        expr_table_by_qtable = self.qtable.make_expression_table(
            features=['id'], samples_as_columns=True
        )
        assert isinstance(expr_table_by_qtable, pd.DataFrame)


def test_qtable_impute_missing_values(example_qtable):
    example_qtable.impute_missing_values()

    expr_matrix = example_qtable.make_expression_matrix()
    number_missing_values = expr_matrix.isna().sum().sum()
    assert number_missing_values == 0


def test_qtable_calculate_experiment_means(example_data, example_qtable):
    example_qtable.calculate_experiment_means()

    experiments = example_qtable.get_experiments()
    assert all([e in example_qtable.data for e in experiments])
    assert all([e in example_qtable._expression_features for e in experiments])
    assert np.allclose(
        example_qtable.data['Experiment_A'],
        [10., np.nan, 4.],
        equal_nan=True
    )
