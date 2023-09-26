from typing import Iterable, Optional

import numpy as np
import pandas as pd
import msreport
from msreport.aggregate.summarize import (
    count_unique,
    join_unique,
    sum_columns_maxlfq,
    sum_columns,
    aggregate_unique_groups,
)
import msreport.aggregate.condense as CONDENSE
from msreport.helper.table import keep_rows_by_partial_match


def aggregate_ions_to_site_quant_table(
    qtable: msreport.Qtable,
    target_modification: str,
    include_modifications: Optional[list[str]],
) -> pd.DataFrame:
    """Aggregates an ion qtable to an modified sites quantification table.

    Requires the following columns in the input table:
    - "Representative protein"
    - "Modification localization string" for different samples
    - "Modified sequence"
    - "Start position"
    - "Peptide sequence"
    - "Modified sequence"

    Args:
        qtable: The ion qtable to aggregate.
        target_modification: The identifier of the modification that should be used for
            aggregation to protein sites, must occur in the "Modified sequence" column.
        include_modifications: Optional, allows specifying additional modification
            identifiers that will be displayed together with the 'target_modification'
            in "Modified sequences" entries of the generated table.

    Returns:
        A quantitative protein sites table containing single or multiple sites.
    """
    modifications_column = "Modified sequence"
    intensity_tag = "Expression"
    table = qtable.get_data()
    samples = qtable.get_samples()
    expression_columns = qtable._expression_columns

    # Undo log transformation
    if msreport.helper.intensities_in_logspace(table[expression_columns]):
        table[expression_columns] = np.power(2, table[expression_columns])

    table = keep_rows_by_partial_match(
        table, modifications_column, [target_modification]
    )
    protein_sites_table = create_modification_centered_table(
        table, target_modification, include_modifications
    )
    aggregated_table = aggregate_protein_sites_table(
        protein_sites_table, "Site ID", intensity_tag, samples
    )
    return aggregated_table


def create_modification_centered_table(
    table: pd.DataFrame,
    target_modification: str,
    include_modifications: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Adds site columns to an ion or peptide table to create a protein sites table.

    Requires the following columns in the input table:
    - "Representative protein"
    - "Modification localization string" for different samples
    - "Modified sequence"
    - "Start position"

    Args:
        table: The input table that will be used for generating the protein sites table.
        target_modification: The identifier of the modification which will be used to
            populate the added columns.
        include_modifications: Optional, allows specifying additional modification
            identifiers that will be included in "Modified sequence" entries together
            with the 'target_modification'.

    Returns:
        A copy of the input table with additional columns containing information about
        the specified target modification.
        - "Site ID": A unique row id with the format "ProteinID_ProteinSites.
        - "Protein sites": Modified amino acid positions of a protein, with multiple
          sites being separated with ";".
        - "First protein site": First modified position from the "Protein sites" column.
        - "Last protein site": Last modified position from the "Protein sites" column.
        - "Isoform probability": Combined site probability of all target modifications
          of the modified peptide isoform.
        - "Modification count": Number of target modification occurences.
        - "Modified sequence": Peptide sequence containing modification tags for the
          specified target modification and those specified with the include
          modifications parameter.
    """
    displayed_modifications = set([target_modification])
    if include_modifications is not None:
        displayed_modifications.update(include_modifications)

    string_localization_column_tag = "Modification localization string"
    localization_string_columns = msreport.helper.find_columns(
        table, string_localization_column_tag
    )

    new_columns = {
        "Site ID": [],
        "Protein sites": [],
        "First protein site": [],
        "Last protein site": [],
        "Isoform probability": [],
        "Modification count": [],
        "Modified sequence": [],
    }
    for _, entry in table.iterrows():
        peptide = msreport.peptidoform.Peptide(
            entry["Modified sequence"],
            protein_position=entry["Start position"],
        )

        modified_sequence = peptide.make_modified_sequence(
            include=displayed_modifications
        )
        mod_count = peptide.count_modification(target_modification)
        protein_sites = peptide.list_modified_protein_sites(target_modification)
        protein_sites_entry = ";".join([str(i) for i in protein_sites])
        protein_id = entry["Representative protein"]
        protein_sites_id = "_".join([str(i) for i in protein_sites])
        unique_id = f"{protein_id}_{protein_sites_id}"

        isoform_probabilities = []
        for column in localization_string_columns:
            if localization_string := entry[column]:
                mod_probabilities = msreport.peptidoform.read_localization_string(
                    localization_string
                )
                peptide.localization_probabilities = mod_probabilities
                probability = peptide.isoform_probability(target_modification)
                isoform_probabilities.append(probability)
        best_isoform_probability = max(isoform_probabilities)

        new_columns["Site ID"].append(unique_id)
        new_columns["Protein sites"].append(protein_sites_entry)
        new_columns["First protein site"].append(min(protein_sites))
        new_columns["Last protein site"].append(max(protein_sites))
        new_columns["Isoform probability"].append(best_isoform_probability)
        new_columns["Modification count"].append(mod_count)
        new_columns["Modified sequence"].append(modified_sequence)

    sites_table = table.copy()
    for column_name, values in new_columns.items():
        sites_table[column_name] = values
    return sites_table


def aggregate_protein_sites_table(
    protein_sites_table: pd.DataFrame,
    group_by: str,
    intensity_tag: str,
    samples: Iterable[str],
) -> pd.DataFrame:
    """Aggregates a protein sites table on unique entries from the 'group_by' column.

    Note that for the aggregation it is essential that values in the intensity and
    expression columns are not log transformed.

    Requires the following columns in the input table, the column specified with
    the 'group_by' argument, and columns containing a combination of the 'intensity_tag'
    and the sample names from the 'samples' argument:
    - "Representative protein"
    - "Protein sites"
    - "Isoform probability"
    - "Modification count"
    - "Peptide sequence"
    - "Modified sequence"

    Args:
        protein_sites_table: The input table that will be used for aggregation of
            protein sites to generate a quantitative protein sites table.
        group_by: The name of the column used to determine unique groups for
            aggregation. Typically corresponds to a string containing the protein
            identifier and protein sites identifier.
        input_tag: Substring of column names, which is used together with the sample
            names to determine the columns whose values will be used for MaxLFQ
            calculation of protein sites.
        samples: List of sample names that appear in columns of the table as substrings.

    Returns:
        An aggregated protein sites table containing exactly one entry for each
        combination of modified protein sites. Contains the following columns, the
        column specified by 'group_by', and columns containing a combination of the
        'intensity_tag' and the sample names from the 'samples' argument:
        - "Representative protein": Protein identifier.
        - "Protein sites": Modified amino acid positions of a protein, with multiple
          sites being separated with ";".
        - "Best isoform probability": Best isoform probability.
        - "Modification count": Number of target modification occurences.
        - "Sequences": One or multiple unique plain sequences, separated by ";".
        - "Modified sequences": One or multiple unique modified sequences, separated
          by ";".
        - "# Sequences": Number of unique plain sequences.
        - "# Modified sequences": Number of unique modified sequences.
    """
    table = protein_sites_table.sort_values(by=group_by)

    protein_ids = join_unique(table, group_by, "Representative protein", is_sorted=True)
    protein_ids.columns = ["Representative protein"]

    protein_sites = join_unique(table, group_by, "Protein sites", is_sorted=True)
    protein_sites.columns = ["Protein sites"]

    aggregation, groups = aggregate_unique_groups(
        table, group_by, ["Isoform probability"], CONDENSE.maximum, is_sorted=True
    )
    best_probabilities = pd.DataFrame(
        columns=["Best isoform probability"], data=aggregation, index=groups
    )

    modification_count = join_unique(
        table, group_by, "Modification count", is_sorted=True
    )
    modification_count.columns = ["Modification count"]

    sequences = join_unique(table, group_by, "Peptide sequence", is_sorted=True)
    sequences.columns = ["Sequences"]

    modified_sequences = join_unique(
        table, group_by, "Modified sequence", is_sorted=True
    )
    modified_sequences.columns = ["Modified sequences"]

    n_sequences = count_unique(table, group_by, "Peptide sequence", is_sorted=True)
    n_sequences.columns = ["# Sequences"]

    n_modified_sequences = count_unique(
        table, group_by, "Modified sequence", is_sorted=True
    )
    n_modified_sequences.columns = ["# Modified sequences"]

    summed_intensity_columns = sum_columns(
        table, group_by, samples, intensity_tag, is_sorted=True
    )
    expression_columns = sum_columns_maxlfq(
        table, group_by, samples, intensity_tag, is_sorted=True
    )
    missing_lfq = (~expression_columns.isna()).sum(axis=1) == 0
    expression_columns[missing_lfq] = summed_intensity_columns[missing_lfq].to_numpy()

    aggregated_sub_tables = [
        protein_ids,
        protein_sites,
        best_probabilities,
        modification_count,
        sequences,
        modified_sequences,
        n_sequences,
        n_modified_sequences,
        expression_columns,
    ]
    aggregated_table = pd.DataFrame(index=table[group_by].unique())
    for sub_table in aggregated_sub_tables:
        aggregated_table = aggregated_table.join(sub_table, how="outer")
    aggregated_table.index.rename(group_by, inplace=True)
    aggregated_table.reset_index(inplace=True)
    return aggregated_table
