Changelog
=========


0.0.1
-----

- Initial unstable release of MsReport
- Reader module
  - Fully implemented import of protein tables from MaxQuant and FragPipe.
  - Partially implemented import of peptide and ion tables from FragPipe.
  - Update peptide table with protein start and end positions from a fasta file.
  - Update protein table with protein annotations from a fasta file.
  - Update protein table with sequence coverage, using information from a peptide table.
  - Update protein table with iBAQ intensities. 
- Qtable class
  - Allows retrieving information from the experimental design, like sample names.
    associated with an experiment or the experiment associated with a sample name.
  - Allows setting a group of quantification columns to be used as expression columns,
    which are then automatically used for subsequent analysis.
  - Generate a new DataFrame, only containing the expression columns.
  - Export data to a tab separated file or to the clipboard.
- Analyze module
  - Analyze missing values and quantified replicates per experiment and in total.
  - Validate proteins according to the number of identified peptides and quantified
    replicates.
  - Normalize expression values between samples.
  - Impute missing expression values by sampling from a gaussian distribution.
  - Calculate mean experiment expression values.
  - Calculate log fold change and average expression values of two experiments.
  - Analyse differential expression between experiments by using LIMMA.
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