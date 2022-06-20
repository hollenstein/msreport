import numpy as np
import pandas as pd
import warnings

import helper


class Qtable():
    def __init__(self, table: pd.DataFrame, design: pd.DataFrame = None):
        self._design: pd.DataFrame
        self._expression_columns: list[str]
        self._expression_tag: str
        self._expression_sample_mapping: dict[str, str]

        self.data: pd.DataFrame = table.copy()
        self._expression_features: list[str] = []
        if design is not None:
            self._design = design

    def get_design(self) -> pd.DataFrame:
        return self._design

    def get_samples(self, experiment: str = None) -> pd.DataFrame:
        design = self.get_design()
        if experiment is not None:
            samples = design[design['Experiment'] == experiment]['Sample']
        else:
            samples = design['Sample']
        return samples.tolist()

    def get_experiments(self) -> pd.DataFrame:
        return self.get_design()['Experiment'].unique().tolist()

    def get_expression_column(self, sample: str) -> str:
        """ Return expression column associated with a sample. """
        column_to_sample = self._expression_sample_mapping
        sample_to_column = {v: k for k, v in column_to_sample.items()}
        if sample in sample_to_column:
            expression_column = sample_to_column[sample]
        else:
            expression_column = ''
        return expression_column

    def make_expression_table(
            self, features: list[str] = None,
            samples_as_columns: bool = False,
        ) -> pd.DataFrame:
        """ Returns a new dataframe containing the expression data and
        additional expression related features.

        Attributes:
            features: A list of additional feature columns that will be added
                from the data table to the expression table.
            samples_as_columns: If true, replace expression column names with
                sample names. Requires that the experimental design is set.
        """
        target_columns = []
        target_columns.extend(self._expression_columns)
        target_columns.extend(self._expression_features)
        if features is not None:
            target_columns.extend(features)
        expr_table = self.data[target_columns].copy()

        if samples_as_columns:
            expr_table.rename(
                columns=self._expression_sample_mapping,
                inplace=True
            )

        return expr_table

    def add_design(self, design_matrix: pd.DataFrame) -> None:
        """ Add an experimental design matrix

        Attributes:
            design_matrix: A dataframe that must contain the columns 'Sample'
                and 'Experiment'. The 'Sample' entries should correspond to the
                Sample names present in the quantitative columns of the table.
        """
        matrix_columns = design_matrix.columns.tolist()
        required_columns = ['Experiment', 'Sample']
        if not all([c in matrix_columns for c in required_columns]):
            exception_message = ''.join([
                'The design matrix must at least contain the columns: ',
                ', '.join(f'"{c}"' for c in required_columns), '. '
                'It contains the columns: ',
                ', '.join(f'"{c}"' for c in matrix_columns), '.'
            ])
            raise ValueError(exception_message)
        self._design = design_matrix

    def set_expression_by_tag(self, tag: str,
                              zerotonan: bool = False,
                              log2: bool = False) -> None:
        """ Sets columns that contain the 'tag' as expression columns.

        Attributes:
            tag: Identifies columns that contain this substring
            zerotonan: If true, zeros in expression columns are replace by NaN
            log2: If true, expression column values are log2 transformed and 0
                are replaced by NaN.
        """
        columns = helper.find_columns(self.data, tag)
        column_mapping = {}
        for column in columns:
            if column.strip() != tag:
                sample = column.replace(tag, '').strip()
                column_mapping[column] = sample
        self._set_expression(column_mapping, zerotonan=zerotonan, log2=log2)

    def set_expression_by_column(self,
                                 columns_to_samples: dict[str, str],
                                 zerotonan: bool = False,
                                 log2: bool = False) -> None:
        """ Defines a list of expression columns.

        Attributes:
            columns_to_samples: Mapping of expression columns to sample names-
                They keys of the dictionary must correspond to columns of the
                data table, and are used to define expression columns.
            zerotonan: If true, zeros in expression columns are replace by NaN
            log2: If true, expression column values are log2 transformed and 0
                are replaced by NaN.
        """
        self._set_expression(
            columns_to_samples, zerotonan=zerotonan, log2=log2
        )

    def calculate_experiment_means(self) -> None:
        """ Calculate mean expression values for each experiment. """
        for experiment in self.get_experiments():
            samples = self.get_samples(experiment)
            columns = [self.get_expression_column(s) for s in samples]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                row_means = np.nanmean(self.data[columns], axis=1)
            self.data[experiment] = row_means
            if experiment not in self._expression_features:
                self._expression_features.append(experiment)

    def impute_missing_values(self) -> None:
        """ Impute missing expression values.

        Missing values are imputed independently for each column by drawing
        random values from a normal distribution. The parameters of the normal
        distribution are calculated from the observed values. Mu is the
        observed median, downshifted by 1.8 standard deviations. Sigma is the
        observed standard deviation multiplied by 0.3.
        """
        median_downshift = 1.8
        std_width = 0.3

        expr = self.make_expression_table()
        imputed = helper.gaussian_imputation(expr, median_downshift, std_width)
        self.data[expr.columns] = imputed[expr.columns]

    def _set_expression(self, expr_sample_mapping: dict[str, str],
                        zerotonan: bool = False, log2: bool = False) -> None:
        """ Define expresssion columns.

        Arguments:
            expr_sample_mapping: mapping of expression columns to sample names
            zerotonan: If true, zeros in expression columns are replace by NaN
            log2: If true, expression column values are log2 transformed and 0
                are replaced by NaN.
        """
        table_columns = self.data.columns.tolist()
        expression_columns = [e for e in expr_sample_mapping.keys()]
        sample_columns = [s for s in expr_sample_mapping.values()]

        if not all([e in table_columns for e in expression_columns]):
            exception_message = ('Not all expression columns from'
                                 ' "expr_sample_mapping" are present as'
                                 ' columns in the data table')
            raise ValueError(exception_message)
        if not all([s in self.get_samples() for s in sample_columns]):
            exception_message = ('Note all samples from "expr_sample_mapping"'
                                 ' are present in self.design')
            raise ValueError(exception_message)
        self._expression_columns = expression_columns
        self._expression_sample_mapping = expr_sample_mapping

        if zerotonan or log2:
            values = self.data[expression_columns].replace({0: np.nan})
            self.data[expression_columns] = values
        if log2:
            values = np.log2(self.data[expression_columns])
            self.data[expression_columns] = values


def _str_to_substr_mapping(strings, substrings) -> dict[str, str]:
    """ DEPRECATED: Mapping of strings to substrings.

    Strings point to a matching substring. If multiple substrings
    are found in a string, only one is reported. """
    mapping = dict()
    for sub in substrings:
        mapping.update({s: sub for s in strings if sub in s})
    return mapping


"""
Todo:
    - QTable.data_to_tsv()
    - QTable.design_to_tsv()
    - QTable.from_tsv(data_path, design_path)

def reshape() -> pd.DataFrame
    raise NotImplementedError
"""
