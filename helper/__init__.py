import numpy as np
import pandas as pd


def find_columns(df: pd.DataFrame, substring: str) -> list[str]:
    """ Returns a list column names containing the substring """
    matched_columns = [substring in col for col in df.columns]
    matched_column_names = np.array(df.columns)[matched_columns].tolist()
    return matched_column_names


def gaussian_imputation(table: pd.DataFrame, median_downshift: float,
                        std_width: float) -> pd.DataFrame:
    """ Imput missing values by drawing values from a normal distribution.

    Imputation is performed column wise, and the parameters for the normal
    distribution are calculated independently for each column.

    Attributes:
        table: table containing missing values that will be replaced
        median_downshift: number of standard deviations the median of the
            measured values is downshifted for the normal distribution
        std_width: width of the normal distribution relative to the
            standard deviation of the measured values

    Returns:
        A new pandas.DataFrame containing imputed values
    """
    imputed_table = table.copy()
    for column in imputed_table:
        median = np.nanmedian(imputed_table[column])
        std = np.nanstd(imputed_table[column])

        mu = median - (std * median_downshift)
        sigma = std * std_width
        missing_values = imputed_table[column].isnull()
        num_missing_values = missing_values.sum()
        imputed_values = np.random.normal(mu, sigma, num_missing_values)

        imputed_table.loc[missing_values, column] = imputed_values
    return imputed_table
