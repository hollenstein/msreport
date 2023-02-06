""" Python interface to custome R scripts. """
import os

import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.conversion import localconverter

from .rinstaller import install_limma_if_missing


def multi_group_limma(
    table: pd.DataFrame,
    design: pd.DataFrame,
    comparison_groups: list[str],
    batch: bool,
    trend: bool,
) -> dict[str, pd.DataFrame]:
    """Use limma to calculate differential expression analysis of multiple groups.

    Args:
        table: Contains quantitative data for differential expression analysis. Column
            names must correspond to entries from `design["Sample"]`.
        design: Dataframe describing the experimental design of the 'table', where each
            row must correspond to a column in 'table'. The 'Design' must contain the
            columns "Sample" and "Experiment". If batch correction should be applied,
            batches must be described in the "Batch" column. Names must be valid R
            names, for reference see the R function make.names.
        comparison_groups: A list containing experiment pairs for which the results of
            the differential expression analysis should be reported. Each experiment
            pair must be written as one string with a dash between the experiment names,
            for example "Experimen1-Experiment2".
        batch: If true batch effects are considered for the differential expression
            analysis. Batches must be specified in the design in a "Batch" column.
        trend: If true an intensity-dependent trend is fitted to the prior variance
            during calculation of the moderated t-statistics, refer to limma.eBayes for
            details.

    Returns:
        A dictionary with keys being the comparison groups and values being a
        dataframe that contains the respective results of the differential expression
        analysis. Dataframes contain the following columns: "Average expression",
        "Ratio [log2]", "P-value", and "Adjusted p-value". Note that the
        "Average expression" calcualted by limma corresponds to the row mean of all
        samples, and not the average of the two experiments that were compared.
    """
    install_limma_if_missing()
    rscript_path = _find_rscript_paths()["limma.R"]
    robjects.r["source"](rscript_path)
    R_multi_group_limma = robjects.globalenv[".multi_group_limma"]

    column_mapping = {
        "AveExpr": "Average expression",
        "logFC": "Ratio [log2]",
        "P.Value": "P-value",
        "adj.P.Val": "Adjusted p-value",
    }
    columns_to_keep = column_mapping.keys()

    # `R_multi_group_limma` expects that the sample order in table and design are equal
    table = table[design["Sample"]]

    group_results = {}
    with localconverter(
        robjects.default_converter + numpy2ri.converter + pandas2ri.converter
    ):
        limma_results = R_multi_group_limma(
            table, design, comparison_groups, batch, trend
        )
        for comparison_group, limma_table in limma_results.items():
            group_results[comparison_group] = limma_table[columns_to_keep].rename(
                columns=column_mapping
            )
    return group_results


def two_group_limma(
    table: pd.DataFrame, groups: list[str], group1: str, group2: str, trend: bool
) -> pd.DataFrame:
    """Use limma to calculate differential expression analysis of two groups.

    Args:
        table: Contains quantitative data for differential expression analysis.
        groups: A list that contains a group name for each column. List entries must
            be equal to 'group1' or 'group2'.
        group1: Experimental group 1
        group2: Experimental group 2, used as the coefficient
        trend: If true an intensity-dependent trend is fitted to the prior variance
            during calculation of the moderated t-statistics, refer to limma.eBayes for
            details.

    Returns:
        A dataframe containing "Average expression", "Ratio [log2]", "P-value", and
        "Adjusted p-value".
    """
    install_limma_if_missing()
    rscript_path = _find_rscript_paths()["limma.R"]
    robjects.r["source"](rscript_path)
    R_two_group_limma = robjects.globalenv[".two_group_limma"]

    with localconverter(robjects.default_converter + pandas2ri.converter):
        limma_result = R_two_group_limma(table, groups, group1, group2, trend)

    column_mapping = {
        "AveExpr": "Average expression",
        "logFC": "Ratio [log2]",
        "P.Value": "P-value",
        "adj.P.Val": "Adjusted p-value",
    }
    columns_to_keep = column_mapping.keys()
    return limma_result[columns_to_keep].rename(columns=column_mapping)


def _find_rscript_paths() -> dict[str, str]:
    """Returns a mapping for filepaths from the msreport.rinterface.rscripts folder.

    Returns:
        A dictionary with filenames as keys and filepaths as values.
    """
    script_paths = {}
    _module_path = os.path.dirname(os.path.realpath(__file__))
    _scripts_path = os.path.join(_module_path, "rscripts")
    for filename in os.listdir(_scripts_path):
        filepath = os.path.join(_scripts_path, filename)
        script_paths[filename] = filepath
    return script_paths
