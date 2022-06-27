""" Python interface to custome R scripts. """
import os

import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

from .rinstaller import _install_missing_r_packages
from .rinstaller import _install_missing_bioconductor_packages


_install_missing_r_packages(['BiocManager'])
_install_missing_bioconductor_packages(['limma'])


def two_group_limma(table: pd.DataFrame, column_groups: list[str],
                    group1: str, group2: str, trend: bool) -> pd.DataFrame:
    """ Use limma to calculate differential expression of two groups.

    Attributes:
        column_groups: A list that contains a group name for each column.
            Group names must correspond either to 'group1' or 'group2'.
        group1: Experimental group 1, corresponds to the coefficient
        group2: Experimental group 2

    Returns:
        A dataframe containing 'logFC', 'P-value', and 'Adjusted p-value'
    """
    rscript_path = _find_rscript_paths()['limma.R']
    robjects.r['source'](rscript_path)
    R_two_group_limma = robjects.globalenv['.two_group_limma']

    with localconverter(robjects.default_converter + pandas2ri.converter):
        limma_result = R_two_group_limma(
            table, column_groups, group1, group2, trend
        )

    keep_columns = ['logFC', 'P.Value', 'adj.P.Val']
    column_mapping = {
        'P.Value': 'P-value', 'adj.P.Val': 'Adjusted p-value',
    }
    return limma_result[keep_columns].rename(columns=column_mapping)


def _find_rscript_paths():
    """ Returns a mapping of files from the 'r_scripts' folder. """
    script_paths = {}
    _module_path = os.path.dirname(os.path.realpath(__file__))
    _scripts_path = os.path.join(_module_path, 'rscripts')
    for filename in os.listdir(_scripts_path):
        filepath = os.path.join(_scripts_path, filename)
        script_paths[filename] = filepath
    return script_paths
