"""A comprehensive set of tools for aggregating and reshaping tabular proteomics data.

The `aggregation` module contains submodules that offer functionalities to transform
data from lower levels of abstraction (e.g. ions, peptides) to higher levels (e.g.
peptides, proteins, PTMs) through various summarization and condensation techniques.
It also includes methods for reshaping tables from "long" to "wide" format, a common
prerequisite for aggregation. The MaxLFQ algorithm is integrated for specific
quantitative summarizations, enabling users to build customized, higher-level data
tables.
"""
