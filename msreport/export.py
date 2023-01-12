"""
Columns that are not yet present in the amica output at the moment:
Index([
    'Protein Probability',
    'Top Peptide Probability',
    'Total peptides',
    'Leading proteins',
    'Protein entry name',
    'Fasta header',
    'Protein length',
    'iBAQ peptides',
    'Sequence coverage',
], dtype='object')
"""
import os

import numpy as np
import pandas as pd

import msreport.helper as helper
from msreport.qtable import Qtable


def contaminants_to_clipboard(qtable: Qtable) -> None:
    """Creates a contaminant table and writes it to the system clipboard.

    The contaminant table contains "iBAQ rank", "riBAQ", "iBAQ intensity", "Intensity",
    and "Expression" columns for each sample. Imputed values in the "Expression" columns
    are set to NaN.

    The qtable must at least contain "iBAQ intensity" and "Missing" sample columns, and
    a "Potential contaminant" column, expression columns must be set. For calculation
    of iBAQ intensities refer to msreport.reader.add_ibaq_intensities(). "Missing"
    sample columns can be added with msreport.analyze.analyze_missingness().

    Args:
        qtable: A Qtable instance. Requires that column names follow the MsReport
            conventions.
    """
    columns = [
        "Representative protein",
        "Protein entry name",
        "Gene name",
        "Fasta header",
        "Protein length",
        "Total peptides",
        "iBAQ peptides",
        "iBAQ intensity total",
    ]
    column_tags = ["iBAQ rank", "riBAQ", "iBAQ intensity", "Intensity", "Expression"]

    samples = qtable.get_samples()
    data = qtable.get_data()

    data["iBAQ intensity total"] = np.nansum(
        data[[f"iBAQ intensity {s}" for s in samples]], axis=1
    ) / len(samples)
    for sample in samples:
        data.loc[data[f"Missing {sample}"], f"Expression {sample}"] = np.nan

        ibaq_values = data[f"iBAQ intensity {sample}"]
        order = np.argsort(ibaq_values)[::-1]
        rank = np.empty_like(ibaq_values, dtype=int)
        rank[order] = np.arange(1, len(ibaq_values) + 1)
        data[f"iBAQ rank {sample}"] = rank
        data[f"riBAQ {sample}"] = ibaq_values / ibaq_values.sum() * 100

    for column_tag in column_tags:
        columns.extend(helper.find_sample_columns(data, column_tag, samples))
    columns = np.array(columns)[[c in data.columns for c in columns]]

    contaminants = qtable["Potential contaminant"]
    data = data.loc[contaminants, columns]

    data.sort_values("iBAQ intensity total", ascending=False, inplace=True)
    data.to_clipboard(index=False)


def to_amica(
    qtable: Qtable,
    directory,
    table_name: str = "amica_table.tsv",
    design_name: str = "amica_design.tsv",
) -> None:
    """Writes amica input and design tables from a qtable.

    As amica requires the same number of columns for each intensity group (LFQIntensity,
    ImputedIntensity, iBAQ), experiment intensity columns are removed and only sample
    intensity columns are written to the output.

    Args:
        qtable: A Qtable instance.
        directory: Output path of the generated files.
        table_name: Optional, filename of the amica table file. Default is
            "amica_table.tsv".
        design_name: Optional, filename of the amica design file. Default is
            "amica_design.tsv".
    """
    amica_table = _amica_table_from(qtable)
    amica_table_path = os.path.join(directory, table_name)
    amica_table.to_csv(amica_table_path, sep="\t", index=False)

    amica_design = _amica_design_from(qtable)
    amica_design_path = os.path.join(directory, design_name)
    amica_design.to_csv(amica_design_path, sep="\t", index=False)


def _amica_table_from(qtable: Qtable) -> pd.DataFrame:
    """Returns a dataframe in the amica format.

    Args:
        table: A dataframe containing experimental data. Requires that column names
            follow the MsReport conventions.
    """
    filter_columns = ["Valid", "Potential contaminant"]
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
        "Ratio [log2] ": "logFC_",
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

    amica_table = qtable.get_data()

    # Drop intensity columns that are not sample columns (e.g. experiment columns)
    for tag in intensity_column_tags[2:3]:
        columns = helper.find_columns(amica_table, tag)
        sample_columns = helper.find_sample_columns(
            amica_table, tag, qtable.get_samples()
        )
        non_sample_columns = set(columns).difference(set(sample_columns))
        amica_table.drop(non_sample_columns, inplace=True, axis=1)

    # Log transform columns if necessary
    for tag in intensity_column_tags:
        for column in helper.find_columns(amica_table, tag):
            if not helper.intensities_in_logspace(amica_table[column]):
                amica_table[column] = amica_table[column].replace({0: np.nan})
                amica_table[column] = np.log2(amica_table[column])

    for old_column in helper.find_columns(amica_table, amica_comparison_tag[0]):
        new_column = old_column.replace(*amica_comparison_tag)
        amica_table.rename(columns={old_column: new_column}, inplace=True)

    for column in filter_columns:
        if column in amica_table.columns:
            amica_table[column] = ["+" if i else "" for i in amica_table[column]]

    for old_tag, new_tag in amica_column_tags.items():
        for old_column in helper.find_columns(amica_table, old_tag):
            new_column = old_column.replace(old_tag, new_tag)
            amica_column_mapping[old_column] = new_column
    amica_table.rename(columns=amica_column_mapping, inplace=True)

    amica_columns = [
        col for col in amica_column_mapping.values() if col in amica_table.columns
    ]
    return amica_table[amica_columns]


def _amica_design_from(qtable: Qtable) -> pd.DataFrame:
    """Returns an experimental design table in the amica format.

    Args:
        design: A dataframe that must contain the columns "Sample" and "Experiment".
    """
    design = qtable.get_design()
    amica_design_columns = {"Sample": "samples", "Experiment": "groups"}
    amica_design = design.rename(columns=amica_design_columns)
    return amica_design
