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


# NOTE all intensity columns need to be log2 transformed
import numpy as np
for column_tag in ['iBAQ intensity', 'LFQ intensity']:
    for column in helper.find_columns(qtable.data, column_tag):
        qtable.data[column] = qtable.data[column].replace({0: np.nan})
        qtable.data[column] = np.log2(qtable.data[column])

write_amica_input(qtable, 'C:/Users/david.hollenstein/Desktop')
"""
import os

import pandas as pd
import helper
import quantable


def write_amica_input(qtable: quantable.Qtable, directory,
                      table_name: str = 'amica_table.tsv',
                      design_name: str = 'amica_design.tsv') -> None:
    amica_table = _amica_table_from(qtable)
    amica_table_path = os.path.join(directory, table_name)
    amica_table.to_csv(amica_table_path, sep='\t', index=False)

    amica_design = _amica_design_from(qtable)
    amica_design_path = os.path.join(directory, design_name)
    amica_design.to_csv(amica_design_path, sep='\t', index=False)


def _amica_table_from(qtable: quantable.Qtable) -> pd.DataFrame:
    amica_column_mapping = {
        'Representative protein': 'Majority.protein.IDs',
        'Gene name': 'Gene.names',
        'Valid': 'quantified',
        'Potential contaminant': 'Potential.contaminant',
    }
    amica_column_tags = {
        'LFQ intensity ': 'LFQIntensity_',
        'Expression ': 'ImputedIntensity_',
        'Spectral count ': 'razorUniqueCount_',
        'iBAQ intensity ': 'iBAQ_',
        'Average expression: ': 'AveExpr_',
        'logFC: ': 'logFC_',
        'P-value: ': 'P.Value_',
        'Adjusted p-value: ': 'adj.P.Val_',
    }
    amica_comparison_tag = (' vs ', '__vs__')

    table = qtable.data.copy()
    for old_column in helper.find_columns(table, amica_comparison_tag[0]):
        new_column = old_column.replace(*amica_comparison_tag)
        table.rename(columns={old_column: new_column}, inplace=True)

    for column in ['Valid', 'Potential contaminant']:
        table[column] = ['+' if i else '' for i in table[column]]

    for old_tag, new_tag in amica_column_tags.items():
        for old_column in helper.find_columns(table, old_tag):
            new_column = old_column.replace(old_tag, new_tag)
            amica_column_mapping[old_column] = new_column
            table.rename(columns={old_column: new_column}, inplace=True)
    table.rename(columns=amica_column_mapping, inplace=True)
    amica_columns = list(amica_column_mapping.values())
    return table[amica_columns]


def _amica_design_from(qtable: quantable.Qtable) -> pd.DataFrame:
    amica_design_columns = {'Sample': 'samples', 'Experiment': 'groups'}
    amica_design = qtable.design.copy().rename(columns=amica_design_columns)
    return amica_design
