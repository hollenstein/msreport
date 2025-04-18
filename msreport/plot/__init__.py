"""This module provides various plotting functions for visualizing data within a Qtable.

The functions in this module generate a wide range of plots, including heatmaps, PCA
plots, volcano plots, and histograms, to analyze and compare expression values,
missingness, contaminants, and other features in proteomics datasets. The plots are
designed to work with the Qtable class as input, which provides structured access to
proteomics data and experimental design information.
"""

from .plots import (
    ColorWheelDict,
    contaminants,
    experiment_ratios,
    expression_clustermap,
    expression_comparison,
    missing_values_horizontal,
    missing_values_vertical,
    pvalue_histogram,
    replicate_ratios,
    sample_correlation,
    sample_intensities,
    sample_pca,
    set_dpi,
    volcano_ma,
)

__all__ = [
    "ColorWheelDict",
    "set_dpi",
    "missing_values_vertical",
    "missing_values_horizontal",
    "contaminants",
    "sample_intensities",
    "replicate_ratios",
    "experiment_ratios",
    "sample_pca",
    "volcano_ma",
    "expression_comparison",
    "expression_clustermap",
    "pvalue_histogram",
    "sample_correlation",
]
