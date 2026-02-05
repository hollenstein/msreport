# MsReport documentation

MsReport is a Python library for post-processing quantitative proteomics data from
bottom-up mass spectrometry.

## Project documentation status

Currently, only the [[api/index.md|API]] documentation and [installation instructions](#installation) are available. The rest of the documentation is a work in progress and may take some time to complete. Please refer to the API reference for detailed information about MsReport's modules, classes and functions.

## Installation

If you do not already have a Python installation, we recommend installing the [Anaconda distribution](https://www.anaconda.com/download) or [Miniconda](https://docs.anaconda.com/free/miniconda/index.html) distribution from Continuum Analytics, which already contains a large number of popular Python packages for Data Science. Alternatively, you can also get Python from the [Python homepage](https://www.python.org/downloads/windows). Note that MsReport requires Python version 3.11 or higher.

The following command will install MsReport and its dependencies by using a wheel file.

```shell
pip install msreport
```

To uninstall the MsReport library use:

```shell
pip uninstall msreport
```

### Installation when using Anaconda

To install the MsReport library using Anaconda, you need to either activate a custom conda environment or install it into the default base environment. Open the Anaconda Navigator, activate the desired conda environment or use the base environment, and then open a command line by running the "CMD.exe" application. Finally, use the `pip install` command as before.

### Optional Dependencies

#### R Integration

MsReport provides an interface to the R package LIMMA for differential expression analysis. To use this functionality, you need:

- A local installation of **R (version 4.0 or higher)**.
- The system environment variable R_HOME set to the R home directory.
- To install msreport with the optional dependencies for R integration.

```shell
pip install msreport[R]
```

#### Setting the R_HOME environment variable

On Windows, you may need to restart your computer after modifying the system environment variables for the changes to take effect. To find the R home directory, you can run the following command in R:

```R
normalizePath(R.home("home"))
```

For example, the R home directory might look like this on Windows: `C:\Program Files\R\R-4.2.1`
