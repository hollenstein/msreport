import numpy as np
import pandas as pd
import pytest
import helper


def test_find_columns():
    df = pd.DataFrame(columns=['Test', 'Test A', 'Test B', 'Something else'])
    columns = helper.find_columns(df, 'Test')
    assert len(columns) == 3
    assert columns == ['Test', 'Test A', 'Test B']


def test_gaussian_imputation():
    table = pd.DataFrame({
        'A': [90000, 100000, 110000, np.nan],
        'B': [1, 1, 1, np.nan],
    })
    median_downshift = 1
    std_width = 1
    imputed = helper.gaussian_imputation(
        table, median_downshift, std_width
    )

    number_missing_values = imputed.isna().sum().sum()
    assert number_missing_values == 0

    imputed_column_A = imputed['A'][3]
    imputed_column_B = imputed['B'][3]
    assert imputed_column_A >= imputed_column_B


def test_rename_mq_reporter_channels_only_intensity():
    table = pd.DataFrame(columns=[
        'Reporter intensity 1',
        'Reporter intensity 2',
    ])
    channel_names = ['Channel 1', 'Channel 2']
    expected_columns = [f'Reporter intensity {i}' for i in channel_names]
    helper.rename_mq_reporter_channels(table, channel_names)
    assert table.columns.tolist() == expected_columns


def test_rename_mq_reporter_channels_with_other_columns():
    table = pd.DataFrame(columns=[
        'Reporter intensity 1',
        'Reporter intensity 2',
        'Reporter intensity',
        'Reporter count',
        'Something else'
    ])
    channel_names = ['Channel 1', 'Channel 2']

    expected_columns = [
        'Reporter intensity Channel 1',
        'Reporter intensity Channel 2',
        'Reporter intensity',
        'Reporter count',
        'Something else'
    ]
    helper.rename_mq_reporter_channels(table, channel_names)
    assert table.columns.tolist() == expected_columns


def test_rename_mq_reporter_channels_with_count_and_corrected():
    table = pd.DataFrame(columns=[
        'Reporter intensity 1',
        'Reporter intensity 2',
        'Reporter intensity count 1',
        'Reporter intensity count 2',
        'Reporter intensity corrected 1',
        'Reporter intensity corrected 2',
    ])
    channel_names = ['Channel 1', 'Channel 2']

    expected_columns = [
        'Reporter intensity Channel 1',
        'Reporter intensity Channel 2',
        'Reporter intensity count Channel 1',
        'Reporter intensity count Channel 2',
        'Reporter intensity corrected Channel 1',
        'Reporter intensity corrected Channel 2',
    ]
    helper.rename_mq_reporter_channels(table, channel_names)
    assert table.columns.tolist() == expected_columns
