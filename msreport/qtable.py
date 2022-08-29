from typing import Optional
import warnings

import numpy as np
import pandas as pd

import msreport.helper as helper


class Qtable:
    def __init__(self, table: pd.DataFrame, design: Optional[pd.DataFrame] = None):
        self.design: pd.DataFrame
        self._expression_columns: list[str] = []
        self._expression_features: list[str] = []
        self._expression_sample_mapping: dict[str, str] = {}

        self.data: pd.DataFrame = table.copy()
        if design is not None:
            self.add_design(design)

    def to_tsv(self, path: str, index: bool = False):
        """Writes table to a tab-separated values (tsv) file."""
        self.data.to_csv(path, sep="\t", index=index)

    def to_clipboard(self, index: bool = False):
        """Writes table to the system clipboard, which can be pasted into Excel."""
        self.data.to_clipboard(sep="\t", index=index)

    def get_design(self) -> pd.DataFrame:
        return self.design

    def get_samples(self, experiment: str = None) -> list[str]:
        design = self.get_design()
        if experiment is not None:
            samples = design[design["Experiment"] == experiment]["Sample"]
        else:
            samples = design["Sample"]
        return samples.tolist()

    def get_experiment(self, sample: str) -> str:
        design = self.get_design()
        experiment = design[design["Sample"] == sample]["Experiment"].values[0]
        return experiment

    def get_experiments(self, samples: list[str] = None) -> list[str]:
        if samples is not None:
            experiments = []
            for sample in samples:
                experiments.append(self.get_experiment(sample))
        else:
            experiments = self.get_design()["Experiment"].unique().tolist()

        return experiments

    def get_expression_column(self, sample: str) -> str:
        """Return expression column associated with a sample."""
        column_to_sample = self._expression_sample_mapping
        sample_to_column = {v: k for k, v in column_to_sample.items()}
        if sample in sample_to_column:
            expression_column = sample_to_column[sample]
        else:
            expression_column = ""
        return expression_column

    def make_sample_table(
        self, tag: str, samples_as_columns: bool = False
    ) -> pd.DataFrame:
        """Returns a new dataframe with sample columns containing the 'tag'.

        Args:
            samples_as_columns: If true, replace expression column names with
                sample names. Requires that the experimental design is set.
        """
        samples = self.get_samples()
        columns = helper.find_sample_columns(self.data, tag, samples)
        table = self.data[columns].copy()
        if samples_as_columns:
            mapping = _str_to_substr_mapping(columns, samples)
            table.rename(columns=mapping, inplace=True)
        return table

    def make_expression_matrix(self, samples_as_columns: bool = False) -> pd.DataFrame:
        """Returns a new dataframe containing the expression columns.

        Args:
            samples_as_columns: If true, replace expression column names with
                sample names. Requires that the experimental design is set.
        """
        raise DeprecationWarning()
        matrix = self.data[self._expression_columns].copy()
        if samples_as_columns:
            matrix.rename(columns=self._expression_sample_mapping, inplace=True)
        return matrix

    def make_expression_table(
        self,
        samples_as_columns: bool = False,
        features: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Returns a new dataframe containing the expression columns.

        Args:
            features: A list of additional columns that will be added from qtable.data
                to the expression table.
            samples_as_columns: If true, replace expression column names with
                sample names. Requires that the experimental design is set.
        """
        columns = []
        columns.extend(self._expression_columns)
        if features is not None:
            columns.extend(features)

        table = self.data[columns].copy()
        if samples_as_columns:
            table.rename(columns=self._expression_sample_mapping, inplace=True)

        return table

    def add_design(self, design_table: pd.DataFrame) -> None:
        """Add an experimental design table

        Args:
            design_table: A dataframe that must contain the columns 'Sample'
                and 'Experiment'. The 'Sample' entries should correspond to the
                Sample names present in the quantitative columns of the table.
        """
        columns = design_table.columns.tolist()
        required_columns = ["Experiment", "Sample"]
        if not all([c in columns for c in required_columns]):
            exception_message = "".join(
                [
                    "The design table must at least contain the columns: ",
                    ", ".join(f'"{c}"' for c in required_columns),
                    ". " "It only contains the columns: ",
                    ", ".join(f'"{c}"' for c in columns),
                    ".",
                ]
            )
            raise ValueError(exception_message)
        self.design = design_table

    def set_expression_by_tag(
        self, tag: str, zerotonan: bool = False, log2: bool = False
    ) -> None:
        """Sets columns that contain the 'tag' as expression columns.

        Previous expression columns and expression features are deleted.

        Args:
            tag: Identifies columns that contain this substring
            zerotonan: If true, zeros in expression columns are replace by NaN
            log2: If true, expression column values are log2 transformed and 0
                are replaced by NaN. Evaluates wheter intensities are likely to
                be already in logspace, which prevents another log2
                transformation.
        """
        columns = helper.find_columns(self.data, tag, must_be_substring=True)
        column_mapping = {}
        for column in columns:
            sample = column.replace(tag, "").strip()
            column_mapping[column] = sample
        self._set_expression(column_mapping, zerotonan=zerotonan, log2=log2)

    def set_expression_by_column(
        self,
        columns_to_samples: dict[str, str],
        zerotonan: bool = False,
        log2: bool = False,
    ) -> None:
        """Defines a list of expression columns and their sample mapping.

        Previous expression columns and expression features are deleted.

        Args:
            columns_to_samples: Mapping of expression columns to sample names-
                They keys of the dictionary must correspond to columns of the
                data table, and are used to define expression columns.
            zerotonan: If true, zeros in expression columns are replace by NaN
            log2: If true, expression column values are log2 transformed and 0
                are replaced by NaN. Evaluates wheter intensities are likely to
                be already in logspace, which prevents another log2
                transformation.
        """
        self._set_expression(columns_to_samples, zerotonan=zerotonan, log2=log2)

    def add_expression_features(self, new_data: pd.DataFrame) -> None:
        """Add expression features as new columns to qtable.data

        Args:
            new_data: DataFrame or Series that will be added to qtable.data as
                new columns, and added to the list of expression features. The
                number and order of new_data rows must be equal to qtable.data.
        """
        # TODO: join all columns at once using pd.concat(axis=1) instead
        assert isinstance(new_data, (pd.DataFrame, pd.Series))
        assert self.data.shape[0] == new_data.shape[0]

        if isinstance(new_data, pd.Series):
            new_data = new_data.to_frame()

        for new_column in new_data.columns:
            # to_numpy() is used to remove the index
            self.data[new_column] = new_data[new_column].to_numpy()
            if new_column not in self._expression_features:
                self._expression_features.append(new_column)

    def calculate_experiment_means(self) -> None:
        """Calculate mean expression values for each experiment."""
        # TODO: move to quanalysis
        warnings.warn(
            "This method will be deprecated, "
            "use qanalysis.calculate_experiment_means() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        experiment_means = {}
        for experiment in self.get_experiments():
            samples = self.get_samples(experiment)
            columns = [self.get_expression_column(s) for s in samples]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                row_means = np.nanmean(self.data[columns], axis=1)
            experiment_means[experiment] = row_means
        self.add_expression_features(pd.DataFrame(experiment_means))

    def impute_missing_values(self) -> None:
        """Impute missing expression values.

        Missing values are imputed independently for each column by drawing
        random values from a normal distribution. The parameters of the normal
        distribution are calculated from the observed values. Mu is the
        observed median, downshifted by 1.8 standard deviations. Sigma is the
        observed standard deviation multiplied by 0.3.
        """
        warnings.warn(
            "This method will be deprecated", DeprecationWarning, stacklevel=2
        )

        median_downshift = 1.8
        std_width = 0.3

        expr = self.make_expression_table()
        imputed = helper.gaussian_imputation(expr, median_downshift, std_width)
        self.data[expr.columns] = imputed[expr.columns]

    def _set_expression(
        self,
        expr_sample_mapping: dict[str, str],
        zerotonan: bool = False,
        log2: bool = False,
    ) -> None:
        """Define expresssion columns and delete previous expression features.

        Args:
            expr_sample_mapping: mapping of expression columns to sample names
            zerotonan: If true, zeros in expression columns are replace by NaN
            log2: If true, expression column values are log2 transformed and 0
                are replaced by NaN. Uses helper.intensities_in_logspace() to
                evaluate wheter intensities are already in logspace, which
                prevents another log2 transformation.
        """
        data_columns = self.data.columns.tolist()
        expression_columns = list(expr_sample_mapping.keys())
        samples = list(expr_sample_mapping.values())

        if not all([e in data_columns for e in expression_columns]):
            exception_message = (
                "Not all expression columns from"
                f" {expression_columns} are present as"
                " columns in the data table!"
            )
            raise ValueError(exception_message)
        if not all([s in self.get_samples() for s in samples]):
            exception_message = (
                f"Not all samples from {samples}" " are present in the qtable design!"
            )
            raise ValueError(exception_message)

        self._reset_expression()
        new_column_names = [f"Expression {sample}" for sample in samples]
        new_sample_mapping = dict(zip(new_column_names, samples))

        self._expression_columns = new_column_names
        self._expression_sample_mapping = new_sample_mapping
        expression_data = self.data[expression_columns].copy()
        expression_data.columns = new_column_names

        if zerotonan or log2:
            expression_data = expression_data.replace({0: np.nan})
        if log2:
            if helper.intensities_in_logspace(expression_data):
                warnings.warn(
                    (
                        "Prevented log2 transformation of intensities that "
                        "appear to be already in log space."
                    ),
                    UserWarning,
                    stacklevel=2,
                )
            else:
                expression_data = np.log2(expression_data)
        self.data[new_column_names] = expression_data

    def _reset_expression(self) -> None:
        """Remove previously added expression and expression feature data."""
        no_expression_columns = []
        for col in self.data.columns:
            if col in self._expression_columns:
                continue
            elif col in self._expression_features:
                continue
            else:
                no_expression_columns.append(col)
        self.data = self.data[no_expression_columns]
        self._expression_columns = []
        self._expression_features = []
        self._expression_sample_mapping = {}

    def copy(self) -> "Self":
        # not tested #
        return self.__copy__()

    def __copy__(self) -> "Self":
        # not tested #
        new_instance = Qtable(self.data, self.design)
        # Copy all private attributes
        for attr in dir(self):
            if (
                not callable(getattr(self, attr))
                and attr.startswith("_")
                and not attr.startswith("__")
            ):
                attr_values = self.__getattribute__(attr).copy()
                new_instance.__setattr__(attr, attr_values)
        return new_instance


def _str_to_substr_mapping(strings, substrings) -> dict[str, str]:
    """DEPRECATED: Mapping of strings to substrings.

    Strings point to a matching substring. If multiple substrings
    are found in a string, only one is reported."""
    mapping = dict()
    for sub in substrings:
        mapping.update({s: sub for s in strings if sub in s})
    return mapping
