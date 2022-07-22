import numpy as np
import pandas as pd
import re


def guess_design(table: pd.DataFrame, tag: str) -> pd.DataFrame:
    """ Extract sample names and experiments from intensity columns.

    First a subset of columns containing a column tag are identified. Then
    sample names are extracted by removing the column tag from each column
    name. And finally, experiment names are extracted from sample names
    by splitting sample names at the last underscore.

    This requires that the sample naming follows a specific convention. It must
    start with the experiment name, followed by an underscore and a unique
    identifier of the sample, for example the replicate number. The experiment
    name can also contain underscores, as it is split only by the last
    underscore. Example of valid samples names are "ExperimentA_r1" or
    "Experiment_A_r1".

    Args:
        table: Table that contains intensity columns with sample names.
        tag: Columns containing the tag are selected for sample extraction.

    Returns:
        A new DataFrame containing the columns "Sample" and "Experiment"
    """
    sample_entries = []
    for column in find_columns(table, tag, must_be_substring=True):
        sample = column.replace(tag, '').strip()
        experiment = '_'.join(sample.split('_')[:-1])
        sample_entries.append([sample, experiment])
    design = pd.DataFrame(sample_entries, columns=['Sample', 'Experiment'])
    return design


def rename_mq_reporter_channels(
        table: pd.DataFrame, channel_names: list[str]) -> None:
    """ Renames reporter channel numbers with sample names.

    MaxQuant writes reporter channel names either in the format
    'Reporter intensity 1' or 'Reporter intensity 1 Experiment Name',
    depending on if an experiment name was specified.

    NOTE: This might not work for the peptides.txt table, as there are columns
    present with the experiment name and without it.
    """
    pattern = re.compile('Reporter intensity [0-9]+')
    reporter_columns = list(filter(pattern.match, table.columns.tolist()))
    assert len(reporter_columns) == len(channel_names)

    column_mapping = {}
    base_name = 'Reporter intensity '
    for column, channel_name in zip(reporter_columns, channel_names):
        for tag in ['', 'count ', 'corrected ']:
            old_column = column.replace(f'{base_name}', f'{base_name}{tag}')
            new_column = f'{base_name}{tag}{channel_name}'
            column_mapping[old_column] = new_column
    table.rename(columns=column_mapping, inplace=True)


def find_columns(df: pd.DataFrame, substring: str,
                 must_be_substring: bool = False) -> list[str]:
    """ Returns a list column names containing the substring.

    Args:
        df: Columns of this pandas.DataFrame are queried.
        substring: String that must be part of column names.
        must_be_substring: If true than column names are not reported if they
            are exactly equal to the substring.

    Returns:
        A list of column names that contain the substring.
    """
    matches = [substring in col for col in df.columns]
    matched_columns = np.array(df.columns)[matches].tolist()
    if must_be_substring:
        matched_columns = [col for col in matched_columns if col != substring]
    return matched_columns
