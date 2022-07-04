import numpy as np
import pandas as pd
import re


def guess_design(table, tag):
    """ Extract sample names and experiments from intensity columns. """
    # Not tested #
    sample_entries = []
    for column in find_columns(table, tag):
        sample = column.replace(tag, '').strip()
        experiment = '_'.join(sample.split('_')[:-1])
        sample_entries.append([sample, experiment])
    design = pd.DataFrame(sample_entries, columns=['Sample', 'Experiment'])
    return design


def find_columns(df: pd.DataFrame, substring: str) -> list[str]:
    """ Returns a list column names containing the substring """
    matched_columns = [substring in col for col in df.columns]
    matched_column_names = np.array(df.columns)[matched_columns].tolist()
    return matched_column_names


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
