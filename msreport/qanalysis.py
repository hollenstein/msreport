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
qtable = Qtable(table, design=design)
qtable.set_expression_by_tag('LFQ intensity', log2=True)

validate_protein_quantification(
    qtable, min_peptides=2, min_group_quantification=2
)
qtable.impute_missing_values()
analyse_differential_expression(qtable)
"""
import itertools
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

import msreport.helper as helper
from msreport.qtable import Qtable
import msreport.rinterface


def count_missing_values(qtable: Qtable) -> pd.DataFrame:
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


def validate_proteins(qtable: Qtable, min_peptides: int = 0,
                      max_missing: int = None) -> None:
    """ Validate protein entries and add a 'Valid' column to the qtable.

    Attributes:
        min_peptides: minimum number of unique peptides
        max_missing: requires at least one experiment with this maximum number
            of missing values.
    """
    valid_entries = (qtable.data['Total peptides'] >= min_peptides)
    qtable.data['Valid'] = valid_entries

    # NOT TESTED from here #
    if max_missing is not None:
        missing_values = count_missing_values(qtable)
        cols = [' '.join(['Missing', e]) for e in qtable.get_experiments()]
        min_two_quant_events = np.any(missing_values[cols] <= max_missing, axis=1)
        qtable.data['Valid'] = min_two_quant_events & qtable.data['Valid']


def median_normalize_samples(qtable: Qtable) -> None:
    """ Normalize samples with median profiles. """
    # NOT TESTED #
    samples = qtable.get_samples()
    num_samples = len(samples)
    expr_table = qtable.make_expression_table(samples_as_columns=True)

    # calculate ratio matrix
    sample_combinations = list(itertools.combinations(range(num_samples), 2))
    matrix = np.full((num_samples, num_samples), np.nan)
    for i, j in sample_combinations:
        ratios = expr_table[samples[i]] - expr_table[samples[j]]
        ratios = ratios[np.isfinite(ratios)]
        median = np.median(ratios)
        matrix[i, j] = median

    # Correct intensities
    profile = helper.solve_ratio_matrix(matrix)
    for i, sample in enumerate(samples):
        col = qtable.get_expression_column(sample)
        qtable.data[col] -= profile[i]


def mode_normalize_samples(qtable: Qtable) -> None:
    """ Normalize samples with median profiles. """
    # NOT TESTED #
    # Is a duplication of median_normalize_samples -> create common function #
    samples = qtable.get_samples()
    num_samples = len(samples)
    expr_table = qtable.make_expression_table(samples_as_columns=True)

    # calculate ratio matrix
    sample_combinations = list(itertools.combinations(range(num_samples), 2))
    matrix = np.full((num_samples, num_samples), np.nan)
    for i, j in sample_combinations:
        ratios = expr_table[samples[i]] - expr_table[samples[j]]
        ratios = ratios[np.isfinite(ratios)]
        mode = helper.mode(ratios)
        matrix[i, j] = mode

    # Correct intensities
    profile = helper.solve_ratio_matrix(matrix)
    for i, sample in enumerate(samples):
        col = qtable.get_expression_column(sample)
        qtable.data[col] -= profile[i]


def lowess_normalize_samples(qtable: Qtable) -> None:
    """ Normalize samples to pseudo reference with lowess. """
    # NOT TESTED #
    samples = qtable.get_samples()
    expr_table = qtable.make_expression_table(samples_as_columns=True)

    ref_mask = (expr_table[samples].isna().sum(axis=1) == 0)
    ref_intensities = expr_table.loc[ref_mask, samples].mean(axis=1)
    lowess_delta = (ref_intensities.max() - ref_intensities.min()) * 0.05

    sample_fits = {}
    for sample in samples:
        sample_intensities = expr_table.loc[ref_mask, sample]
        ratios = sample_intensities - ref_intensities
        sample_fits[sample] = lowess(
            ratios, sample_intensities, delta=lowess_delta, it=5
        )

    for sample, fit in sample_fits.items():
        fit_int, fit_ratio = [np.array(i) for i in zip(*fit)]

        # Correct intensities
        sample_mask = np.isfinite(expr_table[sample])
        raw_intensities = expr_table.loc[sample_mask, sample]
        normalized_intensities = []
        for intensity in raw_intensities:
            norm_value = fit_ratio[np.argmin(np.abs(fit_int - intensity))]
            normalized_intensity = intensity - norm_value
            normalized_intensities.append(normalized_intensity)
        expr_table.loc[sample_mask, sample] = normalized_intensities

    for sample in samples:
        expr_column = qtable.get_expression_column(sample)
        qtable.data[expr_column] = expr_table[sample]


def impute_missing_values(qtable: Qtable) -> None:
    """ Impute missing expression values.

    Imputes missing values (nan) from expression columns and thus requires that
    expression columns are defined.

    Missing values are imputed independently for each column by drawing
    random values from a normal distribution. The parameters of the normal
    distribution are calculated from the observed values. Mu is the
    observed median, downshifted by 1.8 standard deviations. Sigma is the
    observed standard deviation multiplied by 0.3.
    """
    median_downshift = 1.8
    std_width = 0.3

    expr = qtable.make_expression_matrix()
    imputed = helper.gaussian_imputation(expr, median_downshift, std_width)
    qtable.data[expr.columns] = imputed[expr.columns]


def calculate_two_group_limma(qtable: Qtable, groups: list[str],
                              filter_valid: bool = True,
                              limma_trend: bool = True) -> pd.DataFrame:
    """ Use limma to calculate two sample differential expression from qtable.

    Requires that expression columns are set. All rows with missing values are
    ignored, use imputation of missing values to prevent this. The qtable.data
    column 'Representative protein' is used as the index.

    Attributes:
        qtable
        groups: two experiments to compare
        filter_valid: if true, use column 'Valid' to filter rows
        limma_trend: if true, an intensity-dependent trend is fitted to the
            prior variances

    Returns:
        A dataframe containing 'logFC', 'P-value', and 'Adjusted p-value'
        The logFC is calculated as the mean intensity of group2 - group1
    """
    # NOT TESTED #
    expression_table = qtable.make_expression_table(
        samples_as_columns=True, features=['Representative protein']
    )

    # TODO: filtering should be able via "make_expression_table"
    if filter_valid:
        valid = qtable.data['Valid']
    else:
        valid = np.full(expression_table.shape[0], True)

    samples_to_experiment = {}
    for experiment in groups:
        mapping = {s: experiment for s in qtable.get_samples(experiment)}
        samples_to_experiment.update(mapping)

    table_columns = ['Representative protein']
    table_columns.extend(samples_to_experiment.keys())
    table = expression_table[table_columns]
    table = table.set_index('Representative protein')
    not_nan = (table.isna().sum(axis=1) == 0)

    mask = np.all([valid, not_nan], axis=0)
    column_groups = list(samples_to_experiment.values())
    group1 = groups[0]
    group2 = groups[1]

    limma_result = msreport.rinterface.two_group_limma(
        table[mask], column_groups, group1, group2, limma_trend
    )

    # For adding expression features to the qtable it is necessary that the
    # the limma_results have the same number of rows.
    limma_table = pd.DataFrame(index=table.index, columns=limma_result.columns)
    limma_table[mask] = limma_result
    limma_table.fillna(np.nan, inplace=True)

    group_name = f'{group2} vs {group1}'
    mapping = {col: f'{col}: {group_name}' for col in limma_table.columns}
    limma_table.rename(columns=mapping, inplace=True)
    qtable.add_expression_features(limma_table)

    return limma_result
