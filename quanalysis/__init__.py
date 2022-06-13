""" The quanalysis module contains methods for analysing quantification
results.


Scope
-----

- define valid protein quantification rows, based on number of peptides and
    quantified values in experimental groups.
- Calculate missing values
- Perform normalization
- Calculate average intensities per experiment
- Impute missing values
- DE with limma


Required test data
------------------

* for filtering
    - 'Total peptides'
    - Missing values per group / Num values per group
    - entries with "contam_" tag
* for calculating the number of missing values
    - rows with and without missing values
* for imputation
    - rows with and without missing values
* for Limma
    - ???


Interface
---------


quanalysis.add_missing_value_count(qtable)
--> Adds missing value counts as columns to "data"; and corresponding entries
    to expression_features.
! Requires expression columns to be set


quanalysis.validate_protein_quantification(
    qtable,
    min_peptides=None,
    min_group_quantification=None,
    keep_contaminants=False
)
--> Adds a column "Valid quantification" containing true or false to
    qtable.data; and an entry "Valid quantification" to expression_features
! Requires expression columns to be set
! Requires a "Total peptides" column
! Expects that contaminants are marked with "contam_"
? Do missing value calculation on the fly


quanalysis.analyse_differential_expression(
    qtable, groups, method='limma_trend'
)
--> Perform differential expression analysis for expression columns in qtable.
    "groups" defines pairs of experiments that should be compared. Extract
    expression columns according to experiments from the design. Run limma for
    each pair of experiments, and add p-value, adjusted p-value and log FC to
    qtable.data, and the column names to expression_features.
    Naming: 'P-value: exp1 / exp2', 'Adjusted p-value: exp1 / exp2',
    'Fold change [log2]: exp1 / exp2'
! Requires expression columns to be set
! If a "Valid quantification" column is present, expression values are filtered
    according to this column before differential expression analysis.
! Removes rows which contain NaN before differential expression analysis.


Use cases
---------
mqreader = reader.MQReader(search_dir, contaminant_tag='contam_')
table = mqreader.import_proteins(special_proteins=['ups'])
qtable = quantable.Qtable(table, design=design)
qtable.set_expression_by_tag('LFQ intensity')

validate_protein_quantification(
    qtable, min_peptides=2, min_group_quantification=2
)
impute_missing_values(qtable)
analyse_differential_expression(qtable)
"""

import numpy as np
import pandas as pd
import quantable


def add_experiment_means(qtable: quantable.Qtable):
    raise NotImplementedError


def impute_missing_values(qtable: quantable.Qtable) -> None:
    """ Impute missing expression values.

    Missing values are imputed independently for each column by drawing
    random values from a normal distribution. The parameters of the normal
    distribution are calculated from the observed values. Mu is the observed
    median, downshifted by 1.8 standard deviations. Sigma is the observed
    standard deviation multiplied by 0.3

    """
    median_downshift = 1.8
    std_width = 0.3

    expr = qtable.get_expression_table(zerotonan=True)
    imputed = gaussian_imputation(expr, median_downshift, std_width)
    qtable.data[expr.columns] = imputed[expr.columns]


def count_missing_values(qtable: quantable.Qtable) -> pd.DataFrame:
    """ Returns a quantification of missing values in expression columns.

    --> Returns a dataframe with missing value counts in expression columns per
        row, for all sample columns and per experiment. 'Missing total'
    ! Requires expression columns to be set
    """
    missingness = pd.DataFrame()
    expr_table = qtable.get_expression_table(
        samples_as_columns=True, zerotonan=True
    )
    num_missing = expr_table.isna().sum(axis=1)
    missingness['Missing total'] = num_missing
    for experiment in qtable.get_experiments():
        exp_samples = qtable.get_samples(experiment)
        num_missing = expr_table[exp_samples].isna().sum(axis=1)
        column_name = ' '.join(['Missing', experiment])
        missingness[column_name] = num_missing
    return missingness


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
