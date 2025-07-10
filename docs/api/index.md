# API Overview

## Importing Software Results and FASTA Files
The `reader` and `fasta` modules allow integrating external data into `msreport` workflows. The `fasta` module enables import of protein sequence databases from FASTA files into protein databases. The `reader` module handles the parsing and standardization of quantitative proteomics output from various software packages, and annotation with FASTA-related metadata, such as information from FASTA headers.

## High-Level Data Interface

The `qtable`, `analyze`, `plot`, and `export` modules form the **high-level interface** for working with quantitative proteomics data in `msreport`. The `qtable` module defines the central `Qtable` data structure itself. Building upon this, the `analyze` module offers post-processing and statistical analysis capabilities. For tasks like normalization and imputation, it provides functions that can utilize a variety of **transformer classes** available from other `msreport` modules. The `plot` module enables data visualization, and the `export` module provides data conversion to various external formats. By directly operating on `Qtable` instances, these high-level modules provide a straightforward and effective way for analyzing quantitative proteomics data.

## Data Transformation Classes

The `impute`, `normalize`, and `isobar` modules provide a collection of **transformer classes** for processing quantitative proteomics data like imputation or normalization. These transformers follow a consistent "fit-and-transform" pattern, they can be fitted to a training table to learn specific parameters, and then be applied to other tables to perform the transformation.

## Auxiliary Modules
These modules provide supplementary functionalities that support various stages of proteomics data processing. The `aggregate` module offers comprehensive tools for summarizing and reshaping tabular data, including specialized algorithms like MaxLFQ. The `peptidoform` module defines the core Peptide class for robust representation and manipulation of peptide sequences and modifications. Finally, the `helper` module  provides various utility and helper functions.

## Overview of available modules

{{ api_module_overview_table() }}