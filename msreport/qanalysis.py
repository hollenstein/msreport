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

msreport.quanalysis.analyze_missingness(qtable)
--> Adds missing value counts as columns to "data"; and corresponding entries
    to expression_features.
! Requires expression columns to be set


msreport.quanalysis.validate_proteins(
    qtable,
    min_peptides=None,
    min_group_quantification=None,
    keep_contaminants=False
)
--> Adds a column "Valid quantification" containing true or false to
    qtable.data; and an entry "Valid quantification" to expression_features
! Requires expression columns to be set
! Requires missingness to be analyzed
! Requires a "Total peptides" column
! Requires a "Potential contaminant" column


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

"""
import itertools
from typing import Optional
import warnings

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

import msreport.helper as helper
import msreport.normalize
from msreport.qtable import Qtable
import msreport.rinterface


def analyze_missingness(qtable: Qtable) -> None:
    """Adds a quantification of missing values in expression columns."""
    # TODO: not tested #
    missing_events = pd.DataFrame()
    quant_events = pd.DataFrame()
    table = qtable.make_expression_table(samples_as_columns=True)
    num_missing = np.isnan(table).sum(axis=1)
    num_events = np.isfinite(table).sum(axis=1)
    quant_events["Events total"] = num_events
    missing_events["Missing total"] = num_missing
    for experiment in qtable.get_experiments():
        exp_samples = qtable.get_samples(experiment)
        num_events = np.isfinite(table[exp_samples]).sum(axis=1)
        quant_events[f"Events {experiment}"] = num_events
        num_missing = np.isnan(table[exp_samples]).sum(axis=1)
        missing_events[f"Missing {experiment}"] = num_missing
    for sample in qtable.get_samples():
        sample_missing = np.isnan(table[sample])
        missing_events[f"Missing {sample}"] = sample_missing
    qtable.add_expression_features(missing_events)
    qtable.add_expression_features(quant_events)


def validate_proteins(
    qtable: Qtable,
    min_peptides: int = 0,
    remove_contaminants: bool = True,
    min_events: Optional[int] = None,
    max_missing: Optional[int] = None,
) -> None:
    """Validates protein entries by adding a 'Valid' column to the qtable.

    Args:
        min_peptides: Minimum number of unique peptides.
        remove_contaminants: If true, the 'Potential contaminant' column is
            used to remove invalid entries.
        min_events: Requires at least one experiment with this minimum number
            of quantified samples.
        max_missing: Requires at least one experiment with this maximum number
            of missing values.
    """
    valid_entries = qtable.data["Total peptides"] >= min_peptides
    # TODO: not tested from here #
    if remove_contaminants and "Potential contaminant" in qtable.data:
        valid_entries = np.all(
            [valid_entries, np.invert(qtable.data["Potential contaminant"])], axis=0
        )

    if max_missing is not None:
        if "Missing total" not in qtable.data:
            raise Exception("Missing values need to be analyzed before.")
        cols = [" ".join(["Missing", e]) for e in qtable.get_experiments()]
        max_missing_valid = np.any(qtable.data[cols] <= max_missing, axis=1)
        valid_entries = max_missing_valid & valid_entries

    if min_events is not None:
        cols = [" ".join(["Events", e]) for e in qtable.get_experiments()]
        min_events_valid = np.any(qtable.data[cols] >= min_events, axis=1)
        valid_entries = min_events_valid & valid_entries

    qtable.data["Valid"] = valid_entries


def normalize_expression(
    qtable: Qtable,
    method: Optional[str] = None,
    normalizer: Optional[msreport.normalize.BaseSampleNormalizer] = None,
) -> msreport.normalize.BaseSampleNormalizer:
    """Normalizes expression values and returns a Normalizer instance.

    Args:
        qtable: Qtable instance that contains expression values for
            normalization.
        method: Normalization method "median", "mode" or "lowess".
        normalizer: Optional, if specified an already fitted normalizer is used
            for normalization of expression values and the "method" argument is
            ignored.
    """
    # TODO not tested #
    default_normalizers = {
        "median": msreport.normalize.MedianNormalizer(),
        "mode": msreport.normalize.ModeNormalizer(),
        "lowess": msreport.normalize.LowessNormalizer(),
    }

    if method is None and normalizer is None:
        raise ValueError(f"Either 'method' or 'normalizer' must be specified.")
    elif normalizer is not None:
        if not normalizer.is_fitted():
            raise Exception("'normalizer' must be fitted, call normalizer.fit()")
    else:
        if method not in default_normalizers:
            raise ValueError(
                f"'method' = '{method}'' not allowed, "
                f"must be one of {*default_normalizers,}."
            )

    expression_table = qtable.make_expression_table(samples_as_columns=True)

    if normalizer is None:
        if "Valid" in qtable.data:
            fitting_mask = qtable.data["Valid"].to_numpy()
        else:
            fitting_mask = np.ones(expression_table.shape[0], dtype=bool)

        normalizer = default_normalizers[method]
        normalizer.fit(expression_table[fitting_mask])

    samples = expression_table.columns.tolist()
    columns = [qtable.get_expression_column(s) for s in samples]
    qtable.data[columns] = normalizer.transform_table(expression_table)
    return normalizer


def impute_missing_values(qtable: Qtable) -> None:
    """Impute missing expression values.

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

    table = qtable.make_expression_table()
    imputed = helper.gaussian_imputation(table, median_downshift, std_width)
    qtable.data[table.columns] = imputed[table.columns]


def calculate_experiment_means(qtable: Qtable) -> None:
    """Calculate mean expression values for each experiment."""
    # TODO not tested #
    experiment_means = {}
    for experiment in qtable.get_experiments():
        samples = qtable.get_samples(experiment)
        columns = [qtable.get_expression_column(s) for s in samples]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            row_means = np.nanmean(qtable.data[columns], axis=1)
        experiment_means[f"Expression {experiment}"] = row_means
    qtable.add_expression_features(pd.DataFrame(experiment_means))


def two_group_comparison(
    qtable: Qtable, groups: list[str], filter_valid: bool = False
) -> None:
    """Calculates comparison values for two experiments.

    Expects log2 transformed values
    """
    # TODO: not tested #
    table = qtable.make_expression_table(samples_as_columns=True)

    # TODO: filtering should be able via "make_expression_table"
    if filter_valid:
        invalid = np.invert(qtable.data["Valid"].to_numpy())
    else:
        invalid = np.full(table.shape[0], False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        group_expressions = []
        for experiment in groups:
            samples = qtable.get_samples(experiment)
            group_expressions.append(np.nanmean(table[samples], axis=1))
        ratios = group_expressions[0] - group_expressions[1]
        average_expressions = np.nanmean(group_expressions, axis=0)

    comparison_table = pd.DataFrame(
        {
            f"Average expression {groups[0]} vs {groups[1]}": average_expressions,
            f"logFC {groups[0]} vs {groups[1]}": ratios,
        }
    )
    comparison_table[invalid] = np.nan
    qtable.add_expression_features(comparison_table)


def calculate_two_group_limma(
    qtable: Qtable,
    groups: list[str],
    filter_valid: bool = True,
    limma_trend: bool = True,
) -> pd.DataFrame:
    """Use limma to calculate two sample differential expression from qtable.

    Requires that expression columns are set. All rows with missing values are
    ignored, use imputation of missing values to prevent this. The qtable.data
    column 'Representative protein' is used as the index.

    Args:
        qtable: Qtable instance that contains expresion values for differential
            expression analysis.
        groups: two experiments to compare
        filter_valid: if true, use column 'Valid' to filter rows
        limma_trend: if true, an intensity-dependent trend is fitted to the
            prior variances

    Returns:
        A dataframe containing 'logFC', 'P-value', and 'Adjusted p-value'
        The logFC is calculated as the mean intensity of group2 - group1
    """
    # TODO: not tested #
    expression_table = qtable.make_expression_table(
        samples_as_columns=True, features=["Representative protein"]
    )

    # TODO: filtering should be able via "make_expression_table"
    if filter_valid:
        valid = qtable.data["Valid"]
    else:
        valid = np.full(expression_table.shape[0], True)

    samples_to_experiment = {}
    for experiment in groups:
        mapping = {s: experiment for s in qtable.get_samples(experiment)}
        samples_to_experiment.update(mapping)

    table_columns = ["Representative protein"]
    table_columns.extend(samples_to_experiment.keys())
    table = expression_table[table_columns]
    table = table.set_index("Representative protein")
    not_nan = table.isna().sum(axis=1) == 0

    mask = np.all([valid, not_nan], axis=0)
    column_groups = list(samples_to_experiment.values())
    group1 = groups[0]
    group2 = groups[1]

    # Note that the order of groups for calling limma is reversed
    limma_result = msreport.rinterface.two_group_limma(
        table[mask], column_groups, group2, group1, limma_trend
    )

    # For adding expression features to the qtable it is necessary that the
    # the limma_results have the same number of rows.
    limma_table = pd.DataFrame(index=table.index, columns=limma_result.columns)
    limma_table[mask] = limma_result
    limma_table.fillna(np.nan, inplace=True)

    group_name = f"{group1} vs {group2}"
    mapping = {col: f"{col} {group_name}" for col in limma_table.columns}
    limma_table.rename(columns=mapping, inplace=True)
    qtable.add_expression_features(limma_table)

    return limma_result


def count_missing_values(qtable: Qtable) -> pd.DataFrame:
    """Returns a quantification of missing values in expression columns.

    --> Returns a dataframe with missing value counts in expression columns per
        row, for all sample columns and per experiment. 'Missing total'
    ! Requires expression columns to be set
    """
    warnings.warn(
        "This method will be deprecated, use analyze_missingness() instead",
        DeprecationWarning,
        stacklevel=2,
    )

    missingness = pd.DataFrame()
    expr_table = qtable.make_expression_table(samples_as_columns=True)
    num_missing = expr_table.isna().sum(axis=1)
    missingness["Missing total"] = num_missing
    for experiment in qtable.get_experiments():
        exp_samples = qtable.get_samples(experiment)
        num_missing = expr_table[exp_samples].isna().sum(axis=1)
        column_name = " ".join(["Missing", experiment])
        missingness[column_name] = num_missing
    return missingness


def median_normalize_samples(qtable: Qtable) -> None:
    """Normalize samples with median profiles."""
    warnings.warn(
        "This method will be deprecated, use normalize_expression() instead",
        DeprecationWarning,
        stacklevel=2,
    )

    samples = qtable.get_samples()
    num_samples = len(samples)
    expr_table = qtable.make_expression_table(samples_as_columns=True)
    if "Valid" in qtable.data:
        expr_table = expr_table[qtable.data["Valid"]]

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
    """Normalize samples with median profiles."""
    warnings.warn(
        "This method will be deprecated, use normalize_expression() instead",
        DeprecationWarning,
        stacklevel=2,
    )

    samples = qtable.get_samples()
    num_samples = len(samples)
    expr_table = qtable.make_expression_table(samples_as_columns=True)
    if "Valid" in qtable.data:
        expr_table = expr_table[qtable.data["Valid"]]

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

    # Write normalized intensities to qtable
    for i, sample in enumerate(samples):
        sample_column = qtable.get_expression_column(sample)
        corrected_values = qtable.data[sample_column] - profile[i]
        qtable.data[sample_column] = corrected_values


def lowess_normalize_samples(qtable: Qtable) -> None:
    """Normalize samples to pseudo reference with lowess."""
    warnings.warn(
        "This method will be deprecated, use normalize_expression() instead",
        DeprecationWarning,
        stacklevel=2,
    )

    samples = qtable.get_samples()
    expr_table = qtable.make_expression_table(samples_as_columns=True)

    ref_mask = expr_table[samples].isna().sum(axis=1) == 0
    if "Valid" in qtable.data:
        ref_mask = np.all([ref_mask, qtable.data["Valid"]], axis=0)
    ref_intensities = expr_table.loc[ref_mask, samples].mean(axis=1)

    delta_span_percentage = 0.05
    lowess_delta = (
        ref_intensities.max() - ref_intensities.min()
    ) * delta_span_percentage
    sample_fits = {}
    for sample in samples:
        sample_intensities = expr_table.loc[ref_mask, sample]
        ratios = sample_intensities - ref_intensities
        sample_fits[sample] = lowess(
            ratios, sample_intensities, delta=lowess_delta, it=5
        )

    for sample, fit in sample_fits.items():
        fit_int, fit_ratio = [np.array(i) for i in zip(*fit)]

        # Get raw intensities
        sample_mask = np.isfinite(expr_table[sample])
        raw_intensities = expr_table.loc[sample_mask, sample]
        normalized_intensities = raw_intensities - np.interp(
            raw_intensities, fit_int, fit_ratio
        )

        # Store normalized intensities
        expr_table.loc[sample_mask, sample] = normalized_intensities

    # Write normalized intensities to qtable
    for sample in samples:
        sample_column = qtable.get_expression_column(sample)
        corrected_values = expr_table[sample]
        qtable.data[sample_column] = corrected_values
