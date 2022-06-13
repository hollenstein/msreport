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
qtable.set_expression_by_tag('LFQ intensity', log2=True)

validate_protein_quantification(
    qtable, min_peptides=2, min_group_quantification=2
)
qtable.impute_missing_values()
analyse_differential_expression(qtable)
"""

import pandas as pd
import quantable
import helper


def count_missing_values(qtable: quantable.Qtable) -> pd.DataFrame:
    """ Returns a quantification of missing values in expression columns.

    --> Returns a dataframe with missing value counts in expression columns per
        row, for all sample columns and per experiment. 'Missing total'
    ! Requires expression columns to be set
    """
    missingness = pd.DataFrame()
    expr_table = qtable.make_expression_table(samples_as_columns=True)
    num_missing = expr_table.isna().sum(axis=1)
    missingness['Missing total'] = num_missing
    for experiment in qtable.get_experiments():
        exp_samples = qtable.get_samples(experiment)
        num_missing = expr_table[exp_samples].isna().sum(axis=1)
        column_name = ' '.join(['Missing', experiment])
        missingness[column_name] = num_missing
    return missingness
