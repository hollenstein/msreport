Changelog
=========


0.0.3
-----

- Reader module
  - Added function for converting peptide sites from the "Modifications" column
    to protein sites, and adding them to the "Protein modifications" column.

- Helper
  - guess_design() now also extracts the replicate from sample names and returns
    a dataframe with the columns "Sample", "Experiment", "Replicate". If no
    experiment was extracted the sample name is used as experiment.

- Analysis
  - Added function that allows calculating multi group differential expression
    analysis with limma and taking batch effects into account.

- Plot
  - Specifying a different "pvalue_tag" in volcano_ma() now allows plotting of
    the "Adjusted p-value" instead of the "P-value".

- Qtable:
  - Added get_data() function to qtable, which returns a copy of qtable.data and
    allows exclusion of invalid values.

- Fixes
  - Parsing of incomplete or non-standard FASTA headers should be possible now.
  - Fixed inconsistent y-axis labelling of plots generated with
    plot.expression_comparison().
  - Fixed issue with exported amica tables being incompatible with amica.


0.0.2
-----

- Plot module
  - Plot for Comparing expression values between replicates of each experiments.
  - volcano_ma() now requires specifying the experiment pair that will be compared.

- Reader module
  - Added support for importing peptide tables to MQReader and FPReader
  - Added support for importing ion tables to MQReader and FPReader. Importing ion
    tables adds a "Modifications" column and changes entries in the "Modified sequence"
    column to comply with the MsRepport conventions. 
  - Changed the behaviour of 'drop_decoy' and 'drop_idbysite' when importing files
    using the MQReader, which now also removes the "Reverse" and "Only identified by
    site" columns from the table.

- MsReport scripts
  - Added a benchmark plotting script that allows comparing ground truth data sets
    analyzed with different software, settings or methods.
  - Added a excel protein report script that uses the XlsxReport library to write a
    formatted excel protein report from a Qtable.


0.0.1
-----

- Initial unstable release of MsReport

- Reader module
  - Fully implemented import of protein tables from MaxQuant and FragPipe.
  - Partially implemented import of peptide and ion tables from FragPipe.
  - Update peptide table with protein start and end positions from a FASTA file.
  - Update protein table with protein annotations from a FASTA file.
  - Update protein table with sequence coverage, using information from a peptide table.
  - Update protein table with iBAQ intensities. 

- Qtable class
  - Allows retrieving information from the experimental design, like sample names.
    associated with an experiment or the experiment associated with a sample name.
  - Allows setting a group of quantification columns to be used as expression columns,
    which are then automatically used for subsequent analysis.
  - Generate a new dataframe, only containing the expression columns.
  - Export data to a tab separated file or to the clipboard.

- Analyze module
  - Analyze missing values and quantified replicates per experiment and in total.
  - Validate proteins according to the number of identified peptides and quantified
    replicates.
  - Normalize expression values between samples.
  - Impute missing expression values by sampling from a gaussian distribution.
  - Calculate mean experiment expression values.
  - Calculate log fold change and average expression values of two experiments.
  - Analyze differential expression between experiments by using LIMMA.

- Plot module
  - Display relative abundance of all contaminants per sample.
  - Analyze completeness of quantification per experiment.
  - Compare similarity of raw intensities between samples.
  - Compare similarity of expression values between all experiments.
  - Analyze sample similarity with a PCA plot.
  - Compare expression values between two experiments.
  - Compare two experiments with a volcano and MA plot.

- Export module
  - Generate a contaminant table.
  - Export data from a Qtable to the Amica input format. 

- Normalize module
  - Provides several normalizer classes that can be fitted with expression data from
    multiple samples and then be used to apply the fitting to quantitative data.
  - Fixed value normalizer with median or mode.
  - Value dependent normalizer with LOWESS.

- Rinterface module
  - Provides an interface to call R code from python and automatically install required
    packages from CRAN and Bioconductor.
  - Two experiment differential expression analysis by using LIMMA.