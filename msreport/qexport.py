"""
Columns that are not yet present in the amica ouptut at the moment:
Index(['Protein Probability', 'Top Peptide Probability', 'Total peptides',
       'LFQ intensity H900Y000_1', 'LFQ intensity H900Y000_2',
       'LFQ intensity H900Y000_3', 'LFQ intensity H900Y030_1',
       'LFQ intensity H900Y030_2', 'LFQ intensity H900Y030_3',
       'LFQ intensity H900Y100_1', 'LFQ intensity H900Y100_2',
       'LFQ intensity H900Y100_3',
       'Leading proteins', 'Protein entry name',
       'Fasta header', 'Protein length', 'iBAQ peptides',
       'iBAQ intensity H900Y000_1', 'iBAQ intensity H900Y000_2',
       'iBAQ intensity H900Y000_3', 'iBAQ intensity H900Y030_1',
       'iBAQ intensity H900Y030_2', 'iBAQ intensity H900Y030_3',
       'iBAQ intensity H900Y100_1', 'iBAQ intensity H900Y100_2',
       'iBAQ intensity H900Y100_3', 'Sequence coverage',
       ], dtype='object')
"""
import os

import numpy as np
import pandas as pd

import msreport.helper as helper
from msreport.qtable import Qtable


def to_amica(
    qtable: Qtable,
    directory,
    table_name: str = "amica_table.tsv",
    design_name: str = "amica_design.tsv",
) -> None:
    amica_table = _amica_table_from(qtable.data)
    amica_table_path = os.path.join(directory, table_name)
    amica_table.to_csv(amica_table_path, sep="\t", index=False)

    amica_design = _amica_design_from(qtable.design)
    amica_design_path = os.path.join(directory, design_name)
    amica_design.to_csv(amica_design_path, sep="\t", index=False)


def _amica_table_from(table: pd.DataFrame) -> pd.DataFrame:
    """Returns a dataframe in the amica format.

    Args:
        table: A dataframe containing experimental data. Requires that columns are named
        according to the MsReport defaults.
    """
    amica_column_mapping = {
        "Representative protein": "Majority.protein.IDs",
        "Gene name": "Gene.names",
        "Valid": "quantified",
        "Potential contaminant": "Potential.contaminant",
    }
    amica_column_tags = {
        "LFQ intensity ": "LFQIntensity_",
        "Expression ": "ImputedIntensity_",
        "Spectral count ": "razorUniqueCount_",
        "iBAQ intensity ": "iBAQ_",
        "Average expression ": "AveExpr_",
        "logFC ": "logFC_",
        "P-value ": "P.Value_",
        "Adjusted p-value ": "adj.P.Val_",
    }
    intensity_column_tags = [
        "Intensity",
        "LFQ intensity",
        "Expression",
        "iBAQ intensity",
    ]
    amica_comparison_tag = (" vs ", "__vs__")

    amica_table = table.copy()
    # Log transform columns if necessary
    for tag in intensity_column_tags:
        for column in helper.find_columns(amica_table, tag):
            if not helper.intensities_in_logspace(amica_table[column]):
                amica_table[column] = amica_table[column].replace({0: np.nan})
                amica_table[column] = np.log2(amica_table[column])

    for old_column in helper.find_columns(amica_table, amica_comparison_tag[0]):
        new_column = old_column.replace(*amica_comparison_tag)
        amica_table.rename(columns={old_column: new_column}, inplace=True)

    for column in ["Valid", "Potential contaminant"]:
        amica_table[column] = ["+" if i else "" for i in amica_table[column]]

    for old_tag, new_tag in amica_column_tags.items():
        for old_column in helper.find_columns(amica_table, old_tag):
            new_column = old_column.replace(old_tag, new_tag)
            amica_column_mapping[old_column] = new_column
            amica_table.rename(columns={old_column: new_column}, inplace=True)
    amica_table.rename(columns=amica_column_mapping, inplace=True)
    amica_columns = list(amica_column_mapping.values())
    return amica_table[amica_columns]


def _amica_design_from(design: pd.DataFrame) -> pd.DataFrame:
    """Returns an experimental design table in the amica format.

    Args:
        design: A dataframe that must contain the columns 'Sample' and 'Experiment'.
    """
    amica_design_columns = {"Sample": "samples", "Experiment": "groups"}
    amica_design = design.copy().rename(columns=amica_design_columns)
    return amica_design
