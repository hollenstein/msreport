from __future__ import annotations

import os
import warnings
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import yaml

import msreport.helper as helper


class Qtable:
    """Stores and provides access to quantitative proteomics data in a tabular form.

    Qtable contains proteomics data in a tabular form, which is stored as 'qtable.data',
    and an experimental design table, stored in 'qtable.design'. Columns from
    'qtable.data' can directly be accessed by indexing with [], column values can be set
    with [], and the 'in' operator can be used to check whether a column is present in
    'qtable.data', e.g. 'qtable[key]', 'qtable[key] = value', 'key in qtable'.

    Attributes:
        data: A pandas.DataFrame containing quantitative proteomics data.
        design: A pandas.DataFrame describing the experimental design.
    """

    def __init__(self, data: pd.DataFrame, design: Optional[pd.DataFrame] = None):
        """Initializes the Qtable.

        If data does not contain a "Valid" column, this column is added and all its row
        values are set to True.

        Args:
            data: A dataframe containing quantitative proteomics data in a wide format.
            design: A dataframe describing the experimental design that must at least
                contain the columns "Sample" and "Experiment". The "Sample" entries
                should correspond to the Sample names present in the quantitative
                columns of the data.
        """
        self.design: pd.DataFrame
        self.data: pd.DataFrame

        self.data = data.copy()
        if "Valid" not in self.data.columns:
            self.data["Valid"] = True
        if design is not None:
            self.add_design(design)

        self._expression_columns: list[str] = []
        self._expression_features: list[str] = []
        self._expression_sample_mapping: dict[str, str] = {}

    def __getitem__(self, key: Any) -> pd.DataFrame:
        """Evaluation of self.data[key]"""
        return self.data[key]

    def __setitem__(self, key: Any, value: Any):
        """Item assignment of self.data[key]"""
        self.data[key] = value

    def __contains__(self, key: Any) -> bool:
        """True if key is in the info axis of self.data"""
        return key in self.data

    def add_design(self, design: pd.DataFrame) -> None:
        """Adds an experimental design table.

        Args:
            design: A dataframe describing the experimental design that must at least
                contain the columns "Sample" and "Experiment". The "Sample" entries
                should correspond to the Sample names present in the quantitative
                columns of the table.
        """
        columns = design.columns.tolist()
        required_columns = ["Experiment", "Sample", "Replicate"]
        if not all(c in columns for c in required_columns):
            exception_message = "".join(
                [
                    "The design table must at least contain the columns: ",
                    ", ".join(f'"{c}"' for c in required_columns),
                    ". It only contains the columns: ",
                    ", ".join(f'"{c}"' for c in columns),
                    ".",
                ]
            )
            raise ValueError(exception_message)
        self.design = design.copy()

    def get_data(self, exclude_invalid: bool = False) -> pd.DataFrame:
        """Returns a copy of the data table.

        Args:
            exclude_invalid: Optional, if true the returned dataframe is filtered by
                the "Valid" column. Default false.

        Returns:
            A copy of the qtable.data dataframe.
        """
        data = self.data.copy()
        if exclude_invalid:
            data = _exclude_invalid(data)
        return data

    def get_design(self) -> pd.DataFrame:
        """Returns a copy of the design table."""
        return self.design.copy()

    def get_samples(self, experiment: Optional[str] = None) -> list[str]:
        """Returns a list of samples present in the design table.

        Args:
            experiment: If specified, only samples from this experiment are returned.

        Returns:
            A list of sample names.
        """
        design = self.get_design()
        if experiment is not None:
            samples = design[design["Experiment"] == experiment]["Sample"]
        else:
            samples = design["Sample"]
        return samples.tolist()

    def get_experiment(self, sample: str) -> str:
        """Looks up the experiment of the specified sample from the design table.

        Args:
            sample: A sample name.

        Returns:
            An experiment name.
        """
        design = self.get_design()
        experiment = design[design["Sample"] == sample]["Experiment"].values[0]
        return experiment

    def get_experiments(self, samples: Optional[list[str]] = None) -> list[str]:
        """Returns a list of experiments present in the design table.

        Args:
            samples: If specified, only experiments from these samples are returned.

        Returns:
            A list of experiments names.
        """
        if samples is not None:
            experiments = []
            for sample in samples:
                experiments.append(self.get_experiment(sample))
        else:
            experiments = self.get_design()["Experiment"].unique().tolist()

        return experiments

    def get_expression_column(self, sample: str) -> str:
        """Returns the expression column associated with a sample.

        Args:
            sample: A sample name.

        Returns:
            The name of the expression column associated with the sample.
        """
        column_to_sample = self._expression_sample_mapping
        sample_to_column = {v: k for k, v in column_to_sample.items()}
        if sample in sample_to_column:
            expression_column = sample_to_column[sample]
        else:
            expression_column = ""
        return expression_column

    def make_sample_table(
        self,
        tag: str,
        samples_as_columns: bool = False,
        exclude_invalid: bool = False,
    ) -> pd.DataFrame:
        """Returns a new dataframe with sample columns containing the 'tag'.

        Args:
            tag: Substring that must be present in selected columns.
            samples_as_columns: If true, replaces expression column names with
                sample names. Requires that the experimental design is set.
            exclude_invalid: Optional, if true the returned dataframe is filtered by
                the "Valid" column. Default false.

        Returns:
            A new dataframe generated from self.data with sample columns that also
                contained the specified 'tag'.

        Returns:
            A copied dataframe that contains only the specified columns from the
            quantitative proteomics data.
        """
        samples = self.get_samples()
        columns = helper.find_sample_columns(self.data, tag, samples)
        table = self.get_data(exclude_invalid=exclude_invalid)[columns]
        if samples_as_columns:
            sample_to_columns = _match_samples_to_tag_columns(samples, columns, tag)
            columns_to_samples = {v: k for k, v in sample_to_columns.items()}
            table.rename(columns=columns_to_samples, inplace=True)
        return table

    def make_expression_table(
        self,
        samples_as_columns: bool = False,
        features: Optional[list[str]] = None,
        exclude_invalid: bool = False,
    ) -> pd.DataFrame:
        """Returns a new dataframe containing the expression columns.

        Args:
            samples_as_columns: If true, replaces expression column names with
                sample names. Requires that the experimental design is set.
            features: A list of additional columns that will be added from qtable.data
                to the newly generated datarame.
            exclude_invalid: Optional, if true the returned dataframe is filtered by
                the "Valid" column. Default false.

        Returns:
            A copy of tbhe qtable.data dataframe that only contains expression columns
            and additionally specified columns.
        """
        columns = []
        columns.extend(self._expression_columns)
        if features is not None:
            columns.extend(features)

        table = self.get_data(exclude_invalid=exclude_invalid)[columns]
        if samples_as_columns:
            table.rename(columns=self._expression_sample_mapping, inplace=True)

        return table

    def set_expression_by_tag(
        self, tag: str, zerotonan: bool = False, log2: bool = False
    ) -> None:
        """Selects and sets expression columns from those that contain the 'tag'.


        A copy of all identified expression columns is generated and columns are renamed
        to "Expression sample_name". Only columns containing a sample name that is
        present in qtable.design are selected as expression columns. For all samples
        present inqtable.design an expression column must be present in qtable.data.
        When this method is called, previously generated expression columns and
        expression features are deleted.

        Args:
            tag: Columns that contain 'tag' as a substring are selected as potential
                expression columns.
            zerotonan: If true, zeros in expression columns are replace by NaN.
            log2: If true, expression column values are log2 transformed and zeros are
                replaced by NaN. Evaluates whether intensities are likely to be already
                in log-space, which prevents another log2 transformation.
        """
        columns = helper.find_columns(self.data, tag, must_be_substring=True)
        samples_from_design = self.get_samples()
        column_mapping = {}
        for column in columns:
            sample = column.replace(tag, "").strip()
            if sample in samples_from_design:
                column_mapping[column] = sample
        self._set_expression(column_mapping, zerotonan=zerotonan, log2=log2)

    def set_expression_by_column(
        self,
        columns_to_samples: dict[str, str],
        zerotonan: bool = False,
        log2: bool = False,
    ) -> None:
        """Sets as expression columns by using the keys from 'columns_to_samples'.

        Generates a copy of all specified expression columns and renames them to
        "Expression sample_name", according to the 'columns_to_samples' mapping. When
        this method is called, previously generated expression columns and expression
        features are deleted.

        Args:
            columns_to_samples: Mapping of expression columns to sample names. The keys
                of the dictionary must correspond to columns of the proteomics data and
                are used to identify expression columns. The value of each expression
                column specifies the sample name and must correspond to an entry of the
                experimental design table.
            zerotonan: If true, zeros in expression columns are replace by NaN
            log2: If true, expression column values are log2 transformed and zeros are
                replaced by NaN. Evaluates whether intensities are likely to be already
                in log-space, which prevents another log2 transformation.
        """
        self._set_expression(columns_to_samples, zerotonan=zerotonan, log2=log2)

    def add_expression_features(self, expression_features: pd.DataFrame) -> None:
        """Adds expression features as new columns to the proteomics data.

        Args:
            expression_features: dataframe or Series that will be added to qtable.data
                as new columns, column names are added to the list of expression
                features. The number and order of rows in 'expression_features' must
                correspond to qtable.data.
        """
        assert isinstance(expression_features, (pd.DataFrame, pd.Series))
        assert self.data.shape[0] == expression_features.shape[0]

        if isinstance(expression_features, pd.Series):
            expression_features = expression_features.to_frame()

        old_columns = self.data.columns.difference(expression_features.columns)
        old_columns = self.data.columns[self.data.columns.isin(old_columns)]
        self.data = self.data[old_columns]

        # Adopt index to assure row by row joining, assumes identical order of entries
        expression_features.index = self.data.index
        self.data = self.data.join(expression_features, how="left")

        self._expression_features.extend(
            expression_features.columns.difference(self._expression_features)
        )

    def save(self, directory: str, basename: str):
        """Save qtable to disk, creating a data, design, and config file.

        Saving the qtable will generate three files, each starting with the specified
        basename, followed by an individual extension. The generated files are:
        "basename.data.tsv", "basename.design.tsv" and "basename.config.yaml"

        Args:
            directory: The path of the directory where to save the generated files.
            basename: Basename of files that will be generated.
        """
        filepaths = _get_qtable_export_filepaths(directory, basename)

        config_data = {
            "Expression columns": self._expression_columns,
            "Expression features": self._expression_features,
            "Expression sample mapping": self._expression_sample_mapping,
            "Data dtypes": self.data.dtypes.astype(str).to_dict(),
        }
        with open(filepaths["config"], "w") as openfile:
            yaml.safe_dump(config_data, openfile)
        self.data.to_csv(filepaths["data"], sep="\t", index=True)
        self.design.to_csv(filepaths["design"], sep="\t", index=True)

    @classmethod
    def load(cls, directory: str, basename: str) -> Qtable:
        """Load a qtable from disk by reading a data, design, and config file.

        Loading a qtable will first import the three files generated during saving, then
        create and configure a new qtable instance. Each of the filename starts with the
        specified basename, followed by an individual extension. The loaded files are:
        "basename.data.tsv", "basename.design.tsv" and "basename.config.yaml"

        Args:
            directory: The path of the directory where saved qtable files are located.
            basename: Basename of saved files.

        Returns:
            An instance of Qtable loaded from the specified files.
        """
        filepaths = _get_qtable_export_filepaths(directory, basename)
        with open(filepaths["config"]) as openfile:
            config_data = yaml.safe_load(openfile)

        dtypes = config_data["Data dtypes"]
        data = _read_csv_str_safe(
            filepaths["data"], dtypes, **{"sep": "\t", "index_col": 0}
        )
        design = pd.read_csv(
            filepaths["design"], sep="\t", index_col=0, keep_default_na=True
        )

        qtable = Qtable(data, design)
        qtable._expression_columns = config_data["Expression columns"]
        qtable._expression_features = config_data["Expression features"]
        qtable._expression_sample_mapping = config_data["Expression sample mapping"]
        return qtable

    def to_tsv(self, path: str, index: bool = False):
        """Writes the data table to a .tsv (tab-separated values) file."""
        warnings.warn(
            "This function is deprecated, use Qtable.save() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.data.to_csv(path, sep="\t", index=index)

    def to_clipboard(self, index: bool = False):
        """Writes the data table to the system clipboard."""
        self.data.to_clipboard(sep="\t", index=index)

    def copy(self) -> Qtable:
        """Returns a copy of this Qtable instance."""
        # not tested #
        return self.__copy__()

    def _set_expression(
        self,
        columns_to_samples: dict[str, str],
        zerotonan: bool = False,
        log2: bool = False,
    ) -> None:
        """Defines expresssion columns and deletes previous expression features.

        Generates a copy of all specified expression columns and renames them to
        "Expression sample_name", according to the 'columns_to_samples' mapping.

        Args:
            columns_to_samples: Mapping of expression columns to sample names. The keys
                of the dictionary must correspond to columns of self.data, the values
                specify the sample name and must correspond to entries in
                self.design["Sample"].
            zerotonan: If true, zeros in expression columns are replace by NaN
            log2: If true, expression column values are log2 transformed and zeros are
                replaced by NaN. Evaluates whether intensities are likely to be already
                in log-space, which prevents another log2 transformation.
        """
        data_columns = self.data.columns.tolist()
        expression_columns = list(columns_to_samples.keys())
        samples = list(columns_to_samples.values())

        if not expression_columns:
            raise KeyError("No expression columns matched in qtable")
        if not all(e in data_columns for e in expression_columns):
            exception_message = (
                f"Not all specified columns {expression_columns} are present in the"
                " qtable"
            )
            raise KeyError(exception_message)
        if not all(s in self.get_samples() for s in samples):
            exception_message = (
                f"Not all specified samples {samples} are present in the qtable.design"
            )
            raise ValueError(exception_message)
        if not all(s in samples for s in self.get_samples()):
            exception_message = (
                "Not all samples from qtable.design are also present in the specified"
                "samples."
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
        """Removes previously added expression and expression feature columns."""
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

    def __copy__(self) -> Qtable:
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


def _exclude_invalid(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a filterd dataframe only containing valid entries.

    Returns:
        A copy of the dataframe that is filtered according to the boolean values in the
        column "Valid".
    """
    if "Valid" not in df:
        raise KeyError("'Valid' column not present in qtable")
    return df[df["Valid"]].copy()


def _match_samples_to_tag_columns(
    samples: Iterable[str],
    columns: Iterable[str],
    tag: str,
) -> dict:
    """Mapping of samples to columns which contain the sample and the tag.

    Args:
        samples: A list of sample names.
        columns: A list of column names.
        tag: A string that must be present in the column names.

    Returns:
        A dictionary that maps sample names to column names that contain the sample
        name and the tag.
    """
    WHITESPACE_CHARS = " ."

    mapping = {}
    for sample in samples:
        for col in columns:
            if col.replace(tag, "").replace(sample, "").strip(WHITESPACE_CHARS) == "":
                mapping[sample] = col
                break
    return mapping


def _get_qtable_export_filepaths(directory: str, name: str):
    """Returns a dictionary of standard filepaths for loading and saving a qtable."""
    filenames = {
        "data": f"{name}.data.tsv",
        "design": f"{name}.design.tsv",
        "config": f"{name}.config.yaml",
    }
    filepaths = {k: os.path.join(directory, fn) for k, fn in filenames.items()}
    return filepaths


def _read_csv_str_safe(filepath: str, dtypes: dict[str, str], **kwargs):
    """Uses pands.read_csv to read a csv file and preserves empty strings."""
    converters = {}
    dtypes_used = {}
    for column, dtype in dtypes.items():
        if dtype in ["object", "o"]:
            converters[column] = lambda x: "" if x == "" else x
        else:
            dtypes_used[column] = dtype
    return pd.read_csv(filepath, dtype=dtypes_used, converters=converters, **kwargs)
