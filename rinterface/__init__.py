""" Python interface to custome R scripts. """

import os
import pandas as pd
import rpy2.robjects as RO
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter


def two_group_limma(table: pd.DataFrame, column_groups: list[str],
                    group1: str, group2: str, trend: bool) -> pd.DataFrame:
    """ Use limma to calculate differential expression of two groups.

    Returns:
        A dataframe containing 'logFC', 'P-value', and 'Adjusted p-value'
    """
    script_path = _get_rscript_paths()['limma']
    RO.r['source'](script_path)
    R_two_group_limma = RO.globalenv['.two_group_limma']

    with localconverter(RO.default_converter + pandas2ri.converter):
        limma_result = R_two_group_limma(
            table, column_groups, group1, group2, trend
        )

    keep_columns = ['logFC', 'P.Value', 'adj.P.Val']
    column_mapping = {
        'P.Value': 'P-value', 'adj.P.Val': 'Adjusted p-value',
    }
    return limma_result[keep_columns].rename(columns=column_mapping)


def _get_rscript_paths():
    """ Returns a mapping of files from the 'r_scripts' folder. """
    script_paths = {}
    _module_path = os.path.dirname(os.path.realpath(__file__))
    _scripts_path = os.path.join(_module_path, 'rscripts')
    for filename in os.listdir(_scripts_path):
        filepath = os.path.join(_scripts_path, filename)
        genericname, extension = os.path.splitext(filename)
        script_paths[genericname] = filepath
    return script_paths
