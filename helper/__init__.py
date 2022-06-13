import numpy as np
import pandas as pd


def find_columns(df: pd.DataFrame, substring: str) -> list[str]:
    """ Returns a list column names containing the substring """
    matched_columns = [substring in col for col in df.columns]
    matched_column_names = np.array(df.columns)[matched_columns].tolist()
    return matched_column_names
