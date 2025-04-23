[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)


# MsReport


## Introduction

MsReport is a python library that allows simple and standardized post processing of
quantitative proteomics data from bottom up, mass spectrometry experiments. Currently
working with label free protein quantification reports from MaxQuant and FragPipe is
fully supported. Other data analysis pipelines can be added by writing a software
specific reader function.

MsReport is primarily developed as a tool for the Mass Spectrometry Facility at the Max
Perutz Labs (University of Vienna), to allow the generation of Quantitative Protein and
PTM reports, and to facilitate project specific data analysis tasks.


## Release

Development is currently in early alpha and the interface is not yet stable.


## Scope

The `reader` module contains software specific reader classes that provide access to the
outputs of the respective software. Reader instances allow importing protein and ion
tables, and provide the ability to standardize column names and data formats during the
import. To do so, reader classes must know the file structure and naming conventions of
the respective software.

The `qtable` class allows storing and accessing quantitative data from a particular
level of abstraction, such as proteins or ions, and an experimental design table that
describes to which experiment a sample belongs to. The quantitative data are in the wide
format, i.e. the quantification data of each sample is stored in a separate column. The
`Qtable` allows convenient handling and access to quantitative data through information
from the experimental design, and represents the data structure used by the `analyze`,
`plot`, and `export` modules.

The `analyze` module provides a high-level interface for post-processing of quantitative
data, such as filtering valid values, normalization between samples, imputation of
missing values, and statistical testing with the R package LIMMA.

The `plot` module allows generation of quality control and data analysis plots.

Using methods from the `export` module allows conversion and export of quantitative data
into the Amica input format, and generating contaminant tables for the inspection of
potential contaminants.

Additional scripts

- The `excel_report` module enables the creation of a formatted excel protein report
  by using the XlsxReport library.
- The `benchmark` module contains functions to generate benchmark plots from multiple
  `Qtable` instances, and can be used for method or software comparison.


## Install

If you do not already have a Python installation, we recommend installing the
[Anaconda distribution](https://www.continuum.io/downloads) of Continuum Analytics,
which already contains a large number of popular Python packages for Data Science.
Alternatively, you can also get Python from the
[Python homepage](https://www.python.org/downloads/windows). MsReport requires Python
version 3.10 or higher.

You can use pip to install MsReport from the distribution file with the following
command:

```
pip install msreport-X.Y.Z-py3-none-any.whl
```

To uninstall the MsReport library type:

```
pip uninstall msreport
```


### Installation when using Anaconda
If you are using Anaconda, you will need to install the MsReport package into a conda
environment. Open the Anaconda navigator, activate the conda environment you want to
use, run the "CMD.exe" application to open a terminal, and then use the pip install
command as described above.


### Additional requirements

MsReport provides an interface to the R package LIMMA for differential expression
analysis, which requires a local installation of R (R version 4.0 or higher) and the
system environment variable "R_HOME" to be set to the R home directory. Note that it
might be necessary to restart the computer after adding the "R_HOME" variable. The R
home directory can also be found from within R by using the command below, and might
look similar to "C:\Program Files\R\R-4.2.1" on windows.

```
normalizePath(R.home("home"))
```