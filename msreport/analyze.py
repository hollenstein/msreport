""" The quanalysis module contains methods for analysing quantification
results.


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
    """Quantifies missing values of expression columns.

    Adds additional columns to the qtable; for the number of missing values per sample
    "Missing sample_name", per experiment "Missing experiment_name" and in total
    "Missing total"; and for the number of quantification events per experiment
    "Events experiment_name" and in total "Events total".

    Requires expression columns to be set. Missing values in expression columns must be
    present as NaN, and not as zero or an empty string.

    Args:
        qtable: A Qtable instance.
    """
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
    """Validates protein entries (rows).

    Adds an additional column "Valid" to the qtable, containing boolean values.

    Requires expression columns to be set. Depending on the arguments requires the
    columns "Total peptides", "Potential contaminant", and the experiment columns
    "Missing experiment_name" and "Events experiment_name".

    Args:
        qtable: A Qtable instance.
        min_peptides: Minimum number of unique peptides, default 0.
        remove_contaminants: If true, the "Potential contaminant" column is used to
            remove invalid entries, default True. If no "Potential contaminant" column
            is present 'remove_contaminants' is ignored.
        min_events: If specified, at least one experiment must have the minimum number
            of quantified events for the protein entry to be valid.
        max_missing: If specified, at least one experiment must have no more than the
            maximum number of missing values.
    """
    valid_entries = np.ones(qtable.data.shape[0], dtype=bool)

    if min_peptides > 0:
        if "Total peptides" not in qtable.data:
            raise Exception("'Total peptides' column not present in qtable")
        valid_entries = np.all(
            [valid_entries, qtable.data["Total peptides"] >= min_peptides], axis=0
        )

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
    """Normalizes expression values and returns a fitted Normalizer instance.

    If a column "Valid" is present, only valid entries are used for fitting the
    normalizer. The expression values of all rows, including non valid ones, are
    corrected.

    Args:
        qtable: A Qtable instance, which expression values will be normalized.
        method: Normalization method "median", "mode" or "lowess".
        normalizer: Optional, if specified an already fitted normalizer can be used for
            normalization of expression values and the 'method' argument is ignored.

    Returns:
        The fitted normalizer instance.
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
            raise Exception("'normalizer' must be fitted by calling normalizer.fit()")
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
    """Imputes missing expression values.

    Imputes missing values (nan) present in the expression columns and thus requires
    that expression columns are set.

    Missing values are imputed independently for each column by drawing random values
    from a normal distribution. The parameters of the normal distribution are
    calculated from the observed values. Mu is the observed median, downshifted by 1.8
    standard deviations. Sigma is the observed standard deviation multiplied by 0.3.

    Args:
        qtable: A Qtable instance, which missing expression values will be imputed.
    """
    median_downshift = 1.8
    std_width = 0.3

    table = qtable.make_expression_table()
    imputed = helper.gaussian_imputation(table, median_downshift, std_width)
    qtable.data[table.columns] = imputed[table.columns]


def calculate_experiment_means(qtable: Qtable) -> None:
    """Calculates mean expression values for each experiment.

    Adds a new column "Expression experiment_name" for each experiment, containing the
    mean expression values of the corresponding samples.

    Args:
        qtable: A Qtable instance, which mean experiment expression values will be
            calculated.
    """
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
    qtable: Qtable, experiment_pair: list[str, str], exclude_invalid: bool = False
) -> None:
    """Calculates comparison values for two experiments.

    Adds new columns "Average expression Experiment_1 vs Experiment_2" and
    "logFC Experiment_1 vs Experiment_2" to the qtable. Expects that expression values
    are log2 transformed.

    Args:
        qtable: A Qtable instance, containing expression values.
        experiment_pair: The two experiments that will be compared, experiments must be
            present in qtable.design
        exclude_invalid: If true, the column "Valid" is used to determine for which rows
            comparison values are calculated.
    """
    # TODO: not tested #
    table = qtable.make_expression_table(samples_as_columns=True)
    comparison_tag = f"{experiment_pair[0]} vs {experiment_pair[1]}"

    if exclude_invalid:
        invalid = np.invert(qtable.data["Valid"].to_numpy())
    else:
        invalid = np.zeros(table.shape[0], dtype=bool)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        group_expressions = []
        for experiment in experiment_pair:
            samples = qtable.get_samples(experiment)
            group_expressions.append(np.nanmean(table[samples], axis=1))
        ratios = group_expressions[0] - group_expressions[1]
        average_expressions = np.nanmean(group_expressions, axis=0)

    comparison_table = pd.DataFrame(
        {
            f"Average expression {comparison_tag}": average_expressions,
            f"logFC {comparison_tag}": ratios,
        }
    )
    comparison_table[invalid] = np.nan
    qtable.add_expression_features(comparison_table)


def calculate_two_group_limma(
    qtable: Qtable,
    experiment_pair: list[str, str],
    exclude_invalid: bool = True,
    limma_trend: bool = True,
) -> pd.DataFrame:
    """Uses limma to perform a differential expression analysis of two experiments.

    Adds new columns "P-value Experiment_1 vs Experiment_2",
    "Adjusted p-value Experiment_1 vs Experiment_2",
    "Average expression Experiment_1 vs Experiment_2", and
    "logFC Experiment_1 vs Experiment_2" to the qtable.

    Requires that expression columns are set, and expression values are log2
    transformed. All rows with missing values are ignored, impute missing values to
    allow differential expression analysis of all rows. The qtable.data
    column "Representative protein" is used as the index.

    Args:
        qtable: Qtable instance that contains expresion values for differential
            expression analysis.
        experiment_pair: The names of the two experiments that will be compared,
            experiments must be present in qtable.design
        exclude_invalid: If true, the column "Valid" is used to determine which rows are
            used for the differential expression analysis; default True.
        limma_trend: If true, an intensity-dependent trend is fitted to the prior
            variances; default True.

    Returns:
        A dataframe containing "logFC", "P-value", "Adjusted p-value" and
        "Average expression". The logFC is calculated as the mean intensity of
        experiment 2 - experiment 1.
    """
    # TODO: not tested #
    expression_table = qtable.make_expression_table(
        samples_as_columns=True, features=["Representative protein"]
    )

    if exclude_invalid:
        valid = qtable.data["Valid"]
    else:
        valid = np.full(expression_table.shape[0], True)

    samples_to_experiment = {}
    for experiment in experiment_pair:
        mapping = {s: experiment for s in qtable.get_samples(experiment)}
        samples_to_experiment.update(mapping)

    table_columns = ["Representative protein"]
    table_columns.extend(samples_to_experiment.keys())
    table = expression_table[table_columns]
    table = table.set_index("Representative protein")
    not_nan = table.isna().sum(axis=1) == 0

    mask = np.all([valid, not_nan], axis=0)
    column_groups = list(samples_to_experiment.values())

    # Note that the order of experiments for calling limma is reversed
    limma_result = msreport.rinterface.two_group_limma(
        table[mask], column_groups, experiment_pair[1], experiment_pair[0], limma_trend
    )

    # For adding expression features to the qtable it is necessary that the
    # the limma_results have the same number of rows.
    limma_table = pd.DataFrame(index=table.index, columns=limma_result.columns)
    limma_table[mask] = limma_result
    limma_table.fillna(np.nan, inplace=True)

    comparison_tag = f"{experiment_pair[0]} vs {experiment_pair[1]}"
    mapping = {col: f"{col} {comparison_tag}" for col in limma_table.columns}
    limma_table.rename(columns=mapping, inplace=True)
    qtable.add_expression_features(limma_table)

    return limma_result


def count_missing_values(qtable: Qtable) -> pd.DataFrame:
    """Returns a quantification of missing values in expression columns.

    This function will be deprecated, use analyze_missingness() instead.
    """
    warnings.warn(
        "This function will be deprecated, use analyze_missingness() instead",
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
    """Normalize samples with median profiles.

    This function will be deprecated, use normalize_expression() instead.
    """
    warnings.warn(
        "This function will be deprecated, use normalize_expression() instead",
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
    """Normalize samples with median profiles.

    This function will be deprecated, use normalize_expression() instead.
    """
    warnings.warn(
        "This function will be deprecated, use normalize_expression() instead",
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
    """Normalize samples to pseudo reference with lowess.

    This function will be deprecated, use normalize_expression() instead.
    """
    warnings.warn(
        "This function will be deprecated, use normalize_expression() instead",
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
