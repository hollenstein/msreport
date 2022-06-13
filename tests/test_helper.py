import pandas as pd
import pytest
import helper


def test_find_columns():
    df = pd.DataFrame(columns=['Test', 'Test A', 'Test B', 'Something else'])
    columns = helper.find_columns(df, 'Test')
    assert len(columns) == 3
    assert columns == ['Test', 'Test A', 'Test B']
