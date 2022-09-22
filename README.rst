MsReport
========


Introduction
------------
MsReport is a python library that aims to allow simple and standardized post processing
of quantitative proteomics data from bottom up, mass spectrometry experiments.
Currently working with label free protein quantification reports from MaxQuant and
FragPipe is supported. Other data analysis pipelines can be added by writing a software
specific reader module.

MsReport is primarily developed as a tool for the Mass Spectrometry Facility at the Max
Perutz Labs (University of Vienna) to allow the generation of Quantitative Protein and
PTM reports, and to facilitate project specific data analysis tasks.


Release
-------
Development is currently in early alpha and the interface is not yet stable.


Scope
-----
The "Reader" module contains software specific reader classes that provide access to the
outputs of the respective software. The reader has to be aware of the file structure
and naming conventions of the respective software and allows importing protein and ion
tables and standardizing column names and data formats.

The "Qtable" class from allows storing and accessing quantitative data from a specific
abstraction level, such as ions or proteins, and an experimental design table, which
describes samples and experiments. The quantitative data is in a wide format, meaning
that the quantification from each sample is stored in a separate column. The Qtable
allows convenient handling and access to the quantitative data by utilizing the 
information from the experimental design, and represents the data structure used by
analysis and plotting methods.

The "Analyze" module provides a high level interface for post-processing of data present
in the Qtable class, such as filtering valid values, normalization between samples,
imputation of missing values, and statistical testing with the R package LIMMA.

The "Plot" module allows simple generation of plots for quality control and data
analysis directly from a Qtable instance. 

Using methods from the "Export" module allows converting and exporting data from a
Qtable into the Amica input format, and generating contaminant tables for the
inspection of potential contaminants.

Additional scripts
- Generate a formatted excel protein report (Uses the XlsxReport library)
