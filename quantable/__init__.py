import numpy as np
import pandas as pd
import helper


"""
Todo:
- Change name from get_expression_table to create_expression_table
- _map_samples_to_columns()
    replace the requirement for expression_tag. Not sure how to do
- When setting expression columns, they should be linked to the sample names
    in the design table, so that get_expression_table can simply look up and
    replace the expression column names with sample names.
"""


class Qtable():
    def __init__(self, table: pd.DataFrame, design: pd.DataFrame = None):
        self.data: pd.DataFrame
        self._design: pd.DataFrame
        self._expression_columns: list[str]
        self._expression_tag: str

        self.data = table.copy()
        if design is not None:
            self._design = design

    def get_design(self) -> pd.DataFrame:
        return self._design

    def get_samples(self, experiment=None) -> pd.DataFrame:
        design = self.get_design()
        if experiment is not None:
            samples = design[design['Experiment'] == experiment]['Sample']
        else:
            samples = design['Sample']
        return samples.tolist()

    def get_experiments(self) -> pd.DataFrame:
        return self.get_design()['Experiment'].unique().tolist()

    def get_expression_table(self, features: list[str] = None,
                             samples_as_columns: bool = False,
                             zerotonan: bool = False,
                             log2: bool = False) -> pd.DataFrame:
        """ Returns a new dataframe containing the expression data.

        Attributes:
            features: A list of additional feature columns that will be added
                from the data table to the expression table.
            samples_as_columns: If true, replace expression column names with
                sample names. Requires that the experimental design is set.
            zerotonan: If true, 0 in expression columns are replace by NaN.
            log2: If true, expression column values are log2 transformed and 0
                are replaced by NaN.
        """
        expr_col = self._expression_columns
        target_columns = list(expr_col)
        if features is not None:
            target_columns.extend(features)
        expr_table = self.data[target_columns].copy()

        if zerotonan or log2:
            expr_table[expr_col] = expr_table[expr_col].replace({0: np.nan})
        if log2:
            expr_table[expr_col] = np.log2(expr_table[expr_col])

        if samples_as_columns:
            try:
                samples = self.get_samples()
            except AttributeError:
                exception_message = ''.join([
                    'A design matrix must be set to allow replacing ',
                    'expression column names with sample names.'
                ])
                raise ValueError(exception_message)
            expr_tag = self._expression_tag
            mapping = _map_samples_to_columns(expr_col, samples, expr_tag)
            expr_table.rename(columns=mapping, inplace=True)

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

    def set_expression_by_tag(self, tag: str) -> None:
        """ Sets columns that contain the 'tag' as expression columns.

        Attributes:
            tag: Identifies columns that contain this substring
        """
        columns = helper.find_columns(self.data, tag)
        expression_columns = []
        for column in columns:
            if column.strip() != tag:
                expression_columns.append(column)
        self._set_expression(expression_columns, tag)

    def set_expression_by_column(self, expression_columns: list[str],
                                 expression_tag: str) -> None:
        """ Defines a list of expression columns.

        Attributes:
            expression_columns: Columns that must be present in the data table.
            expression_tag: String that must be present in every expression
                column (e.g. 'Intensity').
        """
        self._set_expression(expression_columns, expression_tag)

    def _set_expression(self, expression_columns: list[str],
                        expression_tag: str) -> None:
        table_columns = self.data.columns.tolist()
        if not all([c in table_columns for c in expression_columns]):
            exception_message = ''.join([
                'Not all values of expression_columns are present as columns',
                ' in the data table.'
            ])
            raise ValueError(exception_message)
        if not all([expression_tag in c for c in expression_columns]):
            exception_message = ''.join([
                'Expression tag not present in all expression columns.',
            ])
            raise ValueError(exception_message)
        self._expression_columns = expression_columns
        self._expression_tag = expression_tag

    # def transform_expression(self, how) -> None:
    #    raise NotImplementedError


def _map_samples_to_columns(columns: list[str],
                            samples: list[str], tag: str) -> dict[str, str]:
    """ Mapping of columns to samples that are present as substring. """
    mapping = {}
    for column in columns:
        col = column.replace(tag, '').strip()
        mapping.update({column: sample for sample in samples if sample == col})
    return mapping


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

# def reshape() -> pd.DataFrame
#     raise NotImplementedError
"""
