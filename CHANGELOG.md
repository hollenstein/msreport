# Changelog

## 0.0.28 - Improved visual consistency of plots and Qtable QOL features 

### Added
- Added a new plot function, `plot.sample_correlation`, to display a heatmap of pairwise sample correlations based on expression values.
- Added matplotlib rcParams style sheets to the `plot` module to ensure consistent styling across all plots.
  - The default active style sheet is "msreport-notebook", which is automatically applied to all plotting functions in the `plot` module using a context manager.
  - The new submodule `plot.style` manages the active style sheet and provides functions to set an active style sheet and provide overrides for individual rcParams.
- Added customization options to `plot.expression_clustermap`:
  - `remove_imputation`: Allows retaining imputed values instead of setting them to 0.
  - `mean_center`: Enables row-wise mean-centering of the data before creating the clustermap.
  - `cluster_samples`: Provides the option to disable sample clustering and use the sample order from the qtable design.
- Introduced `id_column` to the `Qtable` class to replace the hardcoded "Representative protein" column. This allows specifying a column in `qtable.data` that contains unique identifiers to enable compatibility with other data types, such as peptide tables.
  - The default value for `id_column` is set to "Representative protein" for backward compatibility.
  - Updated modules `analyze` and `plot` to access `qtable.id_column` instead of relying on the hardcoded "Representative protein" column.
- Added the context manager method `Qtable.temp_design` to temporarily use an alternative design table. This allows using a different design table for plotting or analysis without directly modifying the `Qtable` instance.

### Changed
- Cosmetic improvements to all plots.
  - Instead of globally applying seaborn styles in the plot functions, the "msreport-notebook" style sheet is now applied using a context manager, ensuring a consistent style and that the style is reset after plotting.
  - Added descriptive suptitles for better context.
  - Adjusted font sizes for consistency across plots.
  - Adjusted spine visibility and thickness for consistency.
  - Improved bar and scatter plot aesthetics by refining colors and outlines.
- Instead of raising an error when not enough samples or experiments are present in the design, the functions `plot.replicate_ratios`, `plot.experiment_ratios`, and `plot.sample_pca` now display an empty plot with a warning message.

### Fixed
- Improved robustness of all plots for long experiment or samples names that could lead to overlapping labels or shrinked subplot sizes.
- Ensure consistent plot layout and subplot sizes and spacing independent of the number of samples, experiments and subplots.
- Corrected the number of principal components in the "explained variance" plot of `plot.sample_pca` to be one less than the number of samples.

### Deprecated
- `plot.missing_values_vertical` is now deprecated and will be removed in a future release.
- The 'special_proteins' argument in `plot.volcano_ma` and `plot.expression_comparison` is now deprecated and will be removed in a future release. The argument is replaced by the 'special_entries' argument.

### Dependencies
- Added `pytest` as an optional development dependency.
- Added `mypy` as an optional development dependency.
- Excluded `rpy2` version 3.5.13 due to compatibility issues.

### Internal
- Applied code formatting, linting and import sorting with `ruff`.
- Fixed (most) wrong or missing type hints using `mypy`.
- Restructured `plot` module into submodules for better organization.

----------------------------------------------------------------------------------------

## 0.0.27 - PercentageScaler, minor features and fixes

### Added
- Added a new normalizer, `normalize.PercentageScaler`, which scales the data to transform the values to percentages of the sum of the values in each column. This is useful for calculating relative iBAQ values.
- Added `analyze.apply_transformer`, a universal function for applying fitted transformers (such as normalizers or imputers) to a qtable.
- Added the option to validate proteins based on total spectral counts by adding a `min_spectral_count` argument to the `analyze.validate_proteins` function.

### Fixed
- Fixed an issue in the `plot.volcano_ma` function that occured when plotting special proteins with missing values.
- Fixed a rare issue with the `mode` calculation function, which caused it to not return a local maximum instead of the global maximum.
- Improved error messages for `analyze.calculate_multi_group_limma`

----------------------------------------------------------------------------------------

## 0.0.26 - Minor fixes

### Changed
- `msreport.helper.table.find_sample_columns` now returns the list of sample columns in the same order as in the sample argument.

### Fixed

- Fixed issue of wrong column selection with `msreport.helper.table.find_sample_columns` when samples contain a substring of another sample name.

----------------------------------------------------------------------------------------

## 0.0.25 - Minor fixes

### Fixed

- Added missing x-label to the pvalue histogram.
- Fixed wrong y-label in volcano-ma plot.
- Fixed issue of wrong column renaming when calling `Qtable.make_sample_table` with `samples_as_columns=True` when the design contained sample names that are substrings of other sample names.
- Fixed faulty modification site probability extraction for peptides with multiple modifications from FragPipe results, which was caused by a change in the site localization format in FragPipe version 22.0

----------------------------------------------------------------------------------------

## 0.0.24 - Removed msreport_scripts and dependency updates

### Changed

- Replaced the FASTA parsing functionality from the `helper.temp` module with the `profasta` library.
- Added a `fasta` module which now contains the `import_protein_database` function, which previously was located in the `helper.temp` module.

### Removed

- Removed the `msreport_scripts` package, which is now an independent package.
- Removed the deprecated `helper.temp.importProteinDatabase` function.

### Dependencies

- Added minimal required versions for all package dependencies
- Removed the `xslxreport` as a dependency.
- Added `profasta` as a dependency.
- Set required version of `adjustText` < 1.0.0 to avoid breaking changes.

### Fixed

- Fixed non-breaking issues with seaborn version >= 0.13.0
- Fixed future warnings caused by using pandas >= 2.0.0 and >= 2.2.0

----------------------------------------------------------------------------------------

## 0.0.23 - iBAQ transformer and zscore scaler

### Added

- Added `analyze.create_ibaq_transformer` function that creates a `CategoricalNormalizer` object for converting protein intensity values to iBAQ values. The created normalizer can be passed to `analyze.normalize_expression_by_category` together with a `Qtable` to apply the iBAQ transformation on the expression values.
- Added a new normalizer, `normalize.ZscoreScaler`, to apply z-score transformation to a table. As other normalizer, it can be passed to `analyze.normalize_expression` to apply the normalization to the expression values in a `Qtable`.

----------------------------------------------------------------------------------------

## 0.0.22 - Isotope impurity correction

### Added

- Added a new module, `msreport.isobar`, which contains logic specific for datasets utilizing isobaric labeling for quantification.
- Added an `IsotopeImpurityCorrecter` transformer to the `isobar` module that can be used together with `analyze.normalize_expression` to perform reporter isotope impurity correction.
- Added an `import_ion_evidence` method to the `reader.SpectronautReader` for reading ion evidence files in long format. Note that modified sequence and modification localization probabilities are currently not processed.

----------------------------------------------------------------------------------------

## 0.0.21 - Generation of quantified protein sites table

### Added

- Added the `aggregate_ions_to_site_quant_table` function to the `msreport_scripts.aggregation` module. This function allows the generation of a quantiative protein sites table from a qtable containing quantitative ion or peptide data.
- Added a `CategoricalNormalizer` to the `normalize` module, which forms the basis for site to protein normalization.
- Added `create_site_to_protein_normalizer` and `normalize_expression_by_category` to the `analyze` module, which can be used to create a fitted site to protein normalizer from a protein containing qtable, and apply the normalization to a qtable containing ion, peptide or site data.

### Changed

- Added a `cluster_method` argument to `plot.expression_clustermap` that allows selecting the preferred linkage method to use for calculating clusters.
- Changed the default linkage method used in `plot.expression_clustermap` for calculating clusters from "median" to "average", which in some cases greatly improves the quality of the formed clusters.

### Fixed

- Fixed an issue in `msreport_scripts.excel_report.proteins.write_protein_report` that prevented config files outside of the xlsxreport appdata directory to be found when passing the `config` argument.
- Add missing "[-log10]" to the y-label of the `volcano_ma` plot
- Resolved an issue in the `import_design` method of `reader.SpectronautReader`. This problem occurred when the "Run Label" column of the Spectronaut ConditionSetup file contained only numeric values. Now `import_design` converts all columns to string.
- Resolved an issue in the `import_protein` method of `reader.FragPipeReader` that was caused by an empty "Indistinguishable Proteins" column in the "combined_protein.tsv" file.

----------------------------------------------------------------------------------------

## 0.0.20 - Minor fixes and tweaks

### Added

- Added an argument to `msreport_scripts.aggregation.aggregate_ions_to_site_id_table` that allows applying a site localization probability filter on the PSM level.

### Changed

- More robust renaming of columns when using `helper.rename_sample_columns`.

### Fixed

- Resolved an issue in the `import_design` method of `reader.SpectronautReader`. This problem occurred when the "Condition" column of the Spectronaut ConditionSetup file contained only numeric values.

----------------------------------------------------------------------------------------

## 0.0.19 - Spectral counts per sample for protein sites table

### Added

- The table created with `msreport_scripts.aggregation.aggregate_ions_to_site_id_table` now contains additional spectral counts columns for each sample.

### Changed

- During the import of tables with the `reader.MaxQuantReader` the "Experiment" column is now renamed to "Sample" to comply with the MsReport conventions.
- Renamed "Spectral count" to "Total spectral count" in the aggregated site id table.

----------------------------------------------------------------------------------------

## 0.0.18 - Identified protein sites table from MaxQuant

### Added

- Added two functions to the `helper` module that allow filtering of a dataframe based on partial matches of a string to column values, keeping or removing matched rows: `keep_rows_by_partial_match` and `remove_rows_by_partial_match`.
- Added a method to the `peptidoform.Peptide` class that allows calculation of the best isoform probability.

### Changed

- Added a "score_column" argument to the functions for generating the site id table in the `msreport_scripts.aggregation` module. Instead of using the columns "Probability" and "Expectation" from the FragPipe output, it is now required to specify the name of the column that is used to extract the best score for a given modified protein site. This change enables the creation of site id tables also from the results of other software, such as MaxQuant.
- Updated the `import_ion_evidence` method of the `reader.MaxQuantReader` to add a column containing standardized localization probabilities strings.
- Updated the `import_ions` method of the `reader.FragPipeReader` to add a column containing standardized localization probabilities strings.

### Fixed

- The "filename" argument was not used when calling the `import_ion_evidence` method of the `reader.MaxQuantReader`.

----------------------------------------------------------------------------------------

## 0.0.17 - Generation of identified protein sites table

### Added

- New module `msreport.peptidoform` for standardized processing of modified peptide information, including modification localization probabilities.
  - The `Peptide` class represents a modified peptide identified by mass spectrometry with convenience method for extracting modifications and localization probabilities and for generating a modified sequence string.
  - Contains functions to make and read a site localization probabilities string that represents all modifications and their localization probabilities of a peptide. The localization string is used to encode this information in a standardized string during the import of mass spectrometry analysis results from different software.
  - Contains functions to parse and write a modified peptide sequence.
- Added the `add_protein_site_annotation` function to the `reader` module, which adds two protein site annotation columns to a table containing protein sites. The column "Modified residue" contains single amino acid characters that correspond to the respective protein site; "Sequence window" contains protein sequence windows of eleven amino acids centered on the respective protein site.
- The new module `msreport_scripts.aggregation` will provide functions for the user friendly creation of aggregated tables, e.g. protein or a protein site tables, from tables of a lower abstraction level, such as ion or peptide tables.
  - Added `aggregate_ions_to_site_id_table` function that requires an ion evidence table and a target modification as input and creates an identified protein sites table by aggregation of the ion entries.

### Changed

- Updated the `import_ion_evidence` method of the `reader.FragPipeReader` to add a column containing standardized localization probabilities strings. 

----------------------------------------------------------------------------------------

## 0.0.16 - Minimal peptide import from Spectronaut

### Added
- Added a method for importing peptide tables to the `SpectronautReader`. Note that currently no processing of peptide entries is performed and the imported peptide table is mainly used for protein sequence coverage calculation.

----------------------------------------------------------------------------------------

## 0.0.15 - Improved html coverage map

### Changed

- Changed the output of the HTML page generated with `msreport.export.write_html_coverage_map`
  - The default displayed name now contains the protein name and protein ID.
  - The coverage map contains the percent of covered protein sequence.

### Fixed

- In some situations protein entries were not properly parsed when importing ion tables with `FragPipeReader.import_ions`
- Calling `msreport.export.write_html_coverage_map` caused an error when the covered regions started after the first row of the protein sequence was already written.

----------------------------------------------------------------------------------------

## 0.0.14 - Adding color

### Changed

- Changed the scatter point colors to cyan with red highlights in `msreport.plot.volcano_ma` and `msreport.plot.expression_comparison`.
- The `msreport.plot.expression_comparison` plots now always display strip plots for the left and right subplots, instead of mixing strip plots and swarm plots.
- Changed the interface of `msreport.export.write_html_coverage_map` to use a peptide table and a protein database to automatically extract the protein sequence from the FASTA file and covered regions from the identified peptides.
  - Note that this function is still experimental and the interface might be changed.
- Changed the appearance of the html coverage map:
  - Added amino acid position indicators to the beginning of each row.
  - Added underscores to highlighted position to improve visibility.
  - Changed the default font colors to increase the contrast.
- Added more detailed explanations to the docstrings of the plotting functions.

### Fixed

- Fixed an error when using `msreport.plot.replicate_ratios()` with only one experiment present in the design table.

----------------------------------------------------------------------------------------

## 0.0.13 - The export update

### Added

- New function `msreport.export.to_perseus_matrix` exports data from a qtable into a tsv file that can be imported into perseus and already contains column annotation types.
- `Qtable` got a `Qtable.save` and `Qtable.load` method, which allows exporting and importing a configured qtable instance. Saving a qtable generates three separate files, for the tabular data, the design table, and configuration information.
- Added a function to generate an html file containing a nicely formated protein coverage map. Using `msreport.export.write_html_coverage_map` allows specifying covered regions that will be indicated with a specific color, as well as individual protein positions that will be highlighted with a different color (which can be used for example to highlight modified sites).

### Fixed

- Fixed an issue of the exported amica table containing unequal numbers of samples for different intensity columns. The issue emerged when only a subset of the samples present in a protein table were used in the design, and thus as expression columns. In this case the amica table contained the full set of sample intensity columns, but only the subset of sample expression columns that were present in the design. Now, only sample intensity columns are exported for samples that are present in the design.

### Deprecated

- The `Qtable.to_tsv` method will be removed in the future, use `Qtable.save` insted.

----------------------------------------------------------------------------------------

## 0.0.12 - Methods for data aggregation and MaxLFQ

### Added

- New module `msreport.aggregate`, containing the submodules:
  - `pivot` for converting a table in long format to wide format.
  - `condense` provides various low-level functions to condense multiple rows into a single one, for example by summing up values, counting unique values, are summing up data by median ratio regression similar to the MaxLFQ algorithm.
  - `summarize` provides high-level functions to aggregate a table on unique entries and condense the row information in various ways. 
- New module `msreport.helper.maxlfq` that contains functions to perform the various steps required for aggregating a table with the MaxLFQ algorithm.
- Added `reader.FragPipeReader` method `import_ion_evidence` to read and concatenate all ion.tsv files from a FragPipe result folder.

### Fixed

- Mode calculation did now work with np.nan or when all values are identical.
- Replaced wrong "sklearn" library requirement with the correct one: "scikit-learn"

### Changed

- (!) Renamed `reader.MaxQuantReader` method `import_ions` to `import_ion_evidence`

### Deprecated

- Removed deprecated `msreport.helper.calc.solve_ratio_matrix`, which functionality is now contained in the `msreport.helper.maxlfq` module

----------------------------------------------------------------------------------------

## 0.0.11 - New imputation and normalization interface

### Added

- Added new module `msreport.imputer`, which includes a number of classes for various imputation strategies. To apply a particular imputation strategy, an `Imputer` object can be passed to `msreport.analyze.impute_missing_values()` together with a qtable.
- Added `msreport.imputer.FixedValueImputer`. The `FixedValueImputer` can be used with on of two different imputation strategies: either to replace missing values with a user defined fixed value (strategy="constant") or to replace missing values with a value that is smaller than the lowest observed value (strategy="below").
- Added `msreport.imputer.GaussianImputer`, which allows replacing missing values by drawing values from a gaussian distribution with specified mu and sigma.
- Added `msreport.imputer.PerseusImputer`, which is an implementation of the Perseus-style imputation. Missing values are replaced by drawing values from a gaussian distribution, which parameters are calculated using the standard deviation and median of the observed values.
- Added `msreport.helper.apply_intensity_cutoff()` function for replacing intensity values below a specified threshold with NaN.
- Made relevant msreport submodules available when importing `msreport`, so they don't need to be imported separately: `reader`, `normalize`, `impute`, `analyze`, `plot`, and `export`
- The `msreport.reader.SpectronautReader` can now also be accessed via `msreport.SpectronautReader`, similar as for the other reader classes.

### Changed

- (!) Changed the interface of `msreport.analyze.impute_missing_values()`. Instead of performing a Perseus-style imputation, `impute_missing_values()` now takes an `Imputer` object as input, which specifies the imputation strategy that is used to replace missing values. The new interface allows to add new imputation strategies by creating additional `Imputer` classes, without requiring additional changes to the `impute_missing_values()` interface.
- (!) Changed the interface of `msreport.analyze.normalize_expression()`. Instead of specifying the normalization method with a keyword `impute_missing_values()` now takes an `Normalizer` object as input, which specifies the normalization strategy that is used. The changed interface of `normalize_expression()` is identical to the new interface of `impute_missing_values()`.

### Fixed

- Added missing "Intensity" columns to amica tables exported from qtable.
- Fixed negative iBAQ intensities, which were caused by calculation of iBAQ intensities with negative iBAQ peptides. Negative or zero iBAQ peptides now results in iBAQ intensities being reported as NaN.
  - This also fixes the issue of all iBAQ intensities being zero when calling `add_ibaq_intensities` with `normlize=True` in the presence of negative iBAQ peptide entries.
- The scitkit learn requirement was wrongly specified as sklearn, and was now replaced with Replaced wrong "sklearn" requirement with "scikit-learn>=1.0.0"

----------------------------------------------------------------------------------------

## 0.0.10 - Fix annotated scatter plot issues

### Changed

- Added minimal version for seaborn library (>= 0.12.0)

### Fixed

- Error that might occur when using `plot.expression_comparison()` and for any suplot no special protein is specified for annotation.
- Add missing y-label in the expression comparison plot
- Annotation of wrong data points in scatter plots when using seaborn version < 0.12.0 

----------------------------------------------------------------------------------------

## 0.0.9 - SpectronautReader for LFQ protein reports

### Added
- A prototype reader for Spectronaut has been added. The `reader.SpectronautReader` class currently supports import of DIA LFQ protein reports and Spectronaut "ConditionSetup" files.
- Added `helper.rename_sample_columns()` for cautious renaming sample names that appear as substring of dataframe columns.
- Added `helper.import_protein_database()` to replace `helper.importProteinDatabase()`.
- Added `sort_by` argument to `msreport_scripts.proteins.write_protein_report()` which can be used to specify by which column the Excel table is sorted.

### Changed
- The "Replicate" column has been made mandatory in the `Qtable` experimental design.
- The "Raw files" column has been changed to "Filename" in the "Info" tab of the Excel report generated by `msreport_scripts.proteins.write_protein_report()`.

### Fixed

- The default `tag` parameter in `plot.sample_pca()` should have been "Expression" and not "Intensity".
- Unequal bar size in several plots when a different number of samples or experiments were plotted.
- X-axis boundaries in `plot.expression_comparison()` were displaced by adding special protein annotations.
- Error when using `plot.expression_comparison()` to plot data with no missing values. 

### Deprecated

- `helper.importProteinDatabase()` will be removed in the future.

----------------------------------------------------------------------------------------

## 0.0.8 - Analyze functions exclude invalid by default

### Added

- Added `exclude_invalid` parameter to `analyze.impute_missing_values()`.
- Added `analyze.calculate_multi_group_comparison()` for calculation of ratios and average expression for multiple pair wise experiment comparisons.

### Changed

- Default behavior of `analyze.impute_missing_values()` is now to impute missing values only for valid rows.
- Default behavior of `analyze.two_group_comparison()` is now to calculate values only for valid rows.

----------------------------------------------------------------------------------------

## 0.0.7 - More and better plots

### Added

- Added new plot `plot.pvalue_histogram()`.
- Added option to annotate and highlight special proteins in  `plot.expression_comparison()` by using the `special_proteins` argument.
- Added option to annotate and highlight special proteins in `plot.volcano_ma()` by using the `special_proteins` argument.
- Added option to specify which experiments are used in `plot.experiment_ratios()` by using the `experiments` argument.

### Changed

- Decreased the transparency of the density distributions in `plot.experiment_ratios()` and `plot.replicate_ratios()`.
- The first subplot from `plot.experiment_ratios()` now displays the number of data points used for generating the distribution as "n = xxxx".
- Overlapping replicate annotations in `plot.sample_pca()` are now automatically adjusted.
- Added the python package "adjustText" to the msreport requirements.
- Added missing docstring descriptions.
- Removed return value from `analyze.calculate_two_group_limma()`
- Switched from `setup.py` to `pyproject.toml` for specifying build instructions and package meta data.

### Fixes

- Fixed `plot.replicate_ratios()` displaying no or too few gridlines.
- Fixed `plot.contaminants()` y-axis label not being adjusted to the specified tag.
- Fixed wrong p-value calculation of multi group LIMMA when the sample order in `qtable.data` and `qtable.design` was different.
- Corrected wrong type hints for arguments and return values.
- Corrected docstring typos.

----------------------------------------------------------------------------------------

## 0.0.6 - Minor API changes, small extensions and bug fixes

### Added

- Qtable.data is now always initiliazed with a "Valid" column with all rows being True.
- `rinterface.package_version()` to return the version of an installed R package
- Add `plot.expression_clustermap()` for plotting sample expession values as a hierarchically-clustered heatmap.
- Added two additional arguments for `reader.add_protein_annotation()`
  - `molecular_weight` and `protein_name`

### Changed

- (!) Renamed MQReader to MaxQuantReader
- (!) Renamed FPReader to FragPipeReader
- The `rename_columns` argument in MaxQuantReader now renames additional columns:
  - "Intensity" to "Intensity combined"
  - "iBAQ" to "iBAQ intensity combined"
  - "Protein length" to "Molecular weight [kDA]"
- The `rename_columns` argument in FragPipeReader now renames additional columns:
  - "Description" to "Protein name"
  - "Gene" to "Gene name"
  - "Protein Length" to "Protein length"
  - "Entry Name" to "Protein entry name"
- Renamed the group comparison column tag "logFC" to "Ratio [log2]". This affects the output of the following functions:
  - `analyze.two_group_comparison()`
  - `analyze.calculate_multi_group_limma()`
  - `analyze.calculate_two_group_limma()`
- `reader.add_protein_annotation()` now returns -1 for missing "Protein length" and "iBAQ peptides" entries.
- "Total" and "Combined", and their lower case variants, are ignored as samples names when using `guess_design()`.
- When using `qtable.set_expression_by_tag()`, only sample names present in the design are considered. In addition.
- When setting expression columns for a qtable instance, the exact samples present in the design table must be present in the expression columns.
- Renamed outdated XlsxReport config file for LFQ protein reports from "qtable_proteins.yaml" to "msreport_lfq_protein.yaml"
- Removed maspy dependency

### Fixes

- Calling `reader.add_sequence_coverage()` with protein length as float instead of integer values does no longer raise an error.
- The replicate labels in `plot.replicate_ratios()` and `plot.sample_pca()` were previously extracted from sample names. Now they are read from the qtable.design.

----------------------------------------------------------------------------------------

## 0.0.5 - Overhaul sorting of leading proteins

- Reader module
  - Leading proteins are no longer sorted during the protein import. Protein sorting is now done by a dedicated function, `reader.sort_leading_proteins`.
  - Added `reader.sort_leading_proteins`, which allows sorting of leading proteins with four options: Alphanumeric sorting by protein ID, penalization of contaminants, promotion of special proteins, and sorting of protein entries by the database from which the entry originates.
  - Added `reader.add_protein_annotation`, which allows adding protein annotations from a protein database. Optional arguments allow to specify the added annotation columns.
  - Added `reader.add_leading_proteins_annotation`, which allows adding protein annotations for the "Leading proteins" column, i.e. multiple annotation per field, one for each protein ID. Uses the same syntax as `reader.add_protein_annotation` to specify the added annotation columns.

- Introduced breaking changes
  - Replaced `reader.add_protein_annotations` with `reader.add_protein_annotation`
  - `reader.add_peptide_positions`, now requires a protein database instead of a fasta path.

----------------------------------------------------------------------------------------

## 0.0.4 - Errors and Optional arguments

- Analyze module
  - Added several additional arguments to `impute_missing_values()`, which allow specifying a seed for the random number generator, performing imputation column wise or in total, specifying the median downshift and standard deviation width for calculation of the normal distribution parameters.

- Errors and warnings
  - Added the `errors` module, containing msreport specific errors and warning
  - Added warning to `reader.add_protein_annotations()` and  `reader.add_peptide_positions()` when proteins are absent from fasta files.
  - Added specific errors to the `analyze` and `normalize` modules.

- Normalize module
  - Normalizer classes got a `get_fits()` function to retrieve sample fits.

- Qtable module
  - Added indexing to the Qtable, which allows directly accessing and setting columns for qtable.data with [] by calling qtable[column_name].

- Reader module
  - Added arguments to `add_protein_annotations()` for specifying whether "Protein length" and "iBAQ peptides" should be added or not.

----------------------------------------------------------------------------------------

## 0.0.3 - Multi group limma and small extensions

- Reader module
  - Added function for converting peptide sites from the "Modifications" column to protein sites, and adding them to the "Protein modifications" column.

- Helper module
  - guess_design() now also extracts the replicate from sample names and returns a dataframe with the columns "Sample", "Experiment", "Replicate". If no experiment was extracted the sample name is used as experiment.

- Analyze module
  - Added function that allows calculating multi group differential expression analysis with limma and taking batch effects into account.

- Plot module
  - Specifying a different "pvalue_tag" in volcano_ma() now allows plotting of the "Adjusted p-value" instead of the "P-value".

- Qtable module
  - Added get_data() function to qtable, which returns a copy of qtable.data and allows exclusion of invalid values.

- Fixes
  - Parsing of incomplete or non-standard FASTA headers should be possible now.
  - Fixed inconsistent y-axis labelling of plots generated with plot.expression_comparison().
  - Fixed issue with exported amica tables being incompatible with amica.

----------------------------------------------------------------------------------------

## 0.0.2 - Extended plotting and import functionality

- Plot module
  - Plot for Comparing expression values between replicates of each experiments.
  - volcano_ma() now requires specifying the experiment pair that will be compared.

- Reader module
  - Added support for importing peptide tables to MQReader and FPReader
  - Added support for importing ion tables to MQReader and FPReader. Importing ion tables adds a "Modifications" column and changes entries in the "Modified sequence" column to comply with the MsReport conventions. 
  - Changed the behaviour of 'drop_decoy' and 'drop_idbysite' when importing files using the MQReader, which now also removes the "Reverse" and "Only identified by site" columns from the table.

- MsReport scripts
  - Added a benchmark plotting script that allows comparing ground truth data sets analyzed with different software, settings or methods.
  - Added a excel protein report script that uses the XlsxReport library to write a formatted excel protein report from a Qtable.

----------------------------------------------------------------------------------------

## 0.0.1 - initial release

- Initial unstable release of MsReport

- Reader module
  - Fully implemented import of protein tables from MaxQuant and FragPipe.
  - Partially implemented import of peptide and ion tables from FragPipe.
  - Update peptide table with protein start and end positions from a FASTA file.
  - Update protein table with protein annotations from a FASTA file.
  - Update protein table with sequence coverage, using information from a peptide table.
  - Update protein table with iBAQ intensities. 

- Qtable class
  - Allows retrieving information from the experimental design, like sample names associated with an experiment or the experiment associated with a sample name.
  - Allows setting a group of quantification columns to be used as expression columns which are then automatically used for subsequent analysis.
  - Generate a new dataframe, only containing the expression columns.
  - Export data to a tab separated file or to the clipboard.

- Analyze module
  - Analyze missing values and quantified replicates per experiment and in total.
  - Validate proteins according to the number of identified peptides and quantified replicates.
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
  - Provides several normalizer classes that can be fitted with expression data from multiple samples and then be used to apply the fitting to quantitative data.
  - Fixed value normalizer with median or mode.
  - Value dependent normalizer with LOWESS.

- Rinterface module
  - Provides an interface to call R code from python and automatically install required packages from CRAN and Bioconductor.
  - Two experiment differential expression analysis by using LIMMA.