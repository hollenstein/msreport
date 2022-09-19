MsReport
========


Introduction
------------
MsReport is a Python library that aims to allow simple and standardized post
processing of proteomics data from bottom up, mass spectrometry experiments.
Currently MsReport supports label free protein quantification reports from
MaxQuant, FragPipe, and Spectronaut. Other data analysis pipelines can be added
by writing a software specific reader module.

MsReport is primarily developed as a tool for the Mass Spectrometry Facility at
the Max Perutz Labs (University of Vienna) to allow the generation of
Quantitative Protein and PTM reports, and to facilitate project specific data
anylsis tasks.


Release
-------
Development is currently in early alpha and the interface is not yet stable.


Scope
-----
The "Reader" module contains software specific reader classes that provide
access to the outputs of the respective software. The reader is aware of the
file structure and naming conventions of the respective software and allows
access to protein and ion tables with standardized column names and data
format.

"Qtable" contains quantitative data from one specific abstraction level, such as
ions or proteins. The data are in a wide format, meaning that the quantitative
data from each sample is stored in a separate column. The QTable is aware of
the experimental design, which are samples and experiments, and should provide
convenient access to this data.

"Analyze" provides a high level interface for post-processing of data present in
a qtable class, such as filtering, normalization, imputation, and statistical
testing with the R package LIMMA.

"Plot" allows generation of plots directly from the QTable for quality control
and data analysis.

"Export"
- Generate amica input files
- Generate a contaminant table

Additional scripts
- excel report: protein (Uses the ExcelReport library)
- planned excel reports: ptm / combined (Uses the ExcelReport library)
