""" This module allows the aggregation of ions to protein sites. """
from typing import Optional

import pandas as pd

import msreport.aggregate.condense as CONDENSE
from msreport.aggregate.summarize import (
    aggregate_unique_groups,
    count_unique,
    join_unique,
)
import msreport.peptidoform
from msreport.helper import keep_rows_by_partial_match


def aggregate_ions_to_site_id_table(
    ion_table: pd.DataFrame,
    target_modification: str,
    score_column: str,
    include_modifications: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Aggregates an ion table to an modified sites identification table.

    Args:
        ion_table: The input table that will be used for aggregating.
        target_modification: The identifier of the modification that should be used for
            expanding modified protein sites, must occur in the "Modified sequence"
            column.
        score_column: Column containing peptide identification quality scores, where
            higher numbers indicate better scores.
        include_modifications: Optional, allows specifying additional modification
            identifiers that will be displayed together with the 'target_modification'
            in "Modified sequence" entries of the generated table.

    Returns:
        An aggregated protein sites table containing exactly one entry per protein site.
        Contains the following columns plus the column specified by 'group_by':
        - "Site ID":
        - "Representative protein": Protein identifier.
        - "Protein site": Modified amino acid position of the protein.
        - "Sequences": One or multiple unique plain sequences from the "Sequence" column
          of the 'expanded_site_table', separated by ";".
        - "Modified sequences" One or multiple unique modified sequences from the
          "Modified sequence" column of the 'expanded_site_table', separated by ";".
        - "# Sequences": Number of unique plain sequences.
        - "# Modified sequences": Number of unique modified sequences.
        - "Modification multiplicity": Indicates the count of modifications found in the
          peptides containing the respective site. Can contain one or mulitple
          entries, separated by ";".
        - "Best localization probability": Best observed site localization probability.
        - "Spectral count": Sum of spectral counts per protein site.
        - The column name specified with the 'score_column' argument, contains the best
          observed peptide score.
    """
    table = ion_table.copy()
    modifications_column = "Modified sequence"

    table = table[(table["Spectral count"] > 0)]
    table = keep_rows_by_partial_match(
        table, modifications_column, [target_modification]
    )

    expanded_protein_site_table = create_expanded_protein_site_table(
        table, target_modification, score_column, include_modifications
    )
    aggregated_table = aggregate_expanded_site_table(
        expanded_protein_site_table, "Site ID", score_column
    )
    return aggregated_table


def aggregate_expanded_site_table(
    expanded_site_table: pd.DataFrame,
    group_by: str,
    score_column: str,
) -> pd.DataFrame:
    """Aggregates an expanded site table by unique entries from the 'group_by' column.

    This aggregation creates a table containing identification information about
    individual modified protein sites. Each protein site, as defined by unique entries
    in the 'group_by' column, corresponds to one row in the aggregated table.

    Requires the following columns in the input table, besides the column specified
    with the 'group_by' and 'score_column' arguments:
    - "Protein site"
    - "Representative protein"
    - "Sequence"
    - "Modified sequence"
    - "Modification count"
    - "Site probability"
    - "Spectral count" --> This column should be optional for DIA compatibility.

    Args:
        expanded_site_table: The input table that will be used for aggregation of
            protein sites to generate a site identification table.
        group_by: The name of the column used to determine unique groups for
            aggregation. Typically corresponds to a string containing the protein
            identifier and protein site identifier.
        score_column: Column containing peptide identification quality scores, where
            higher numbers indicate better scores.

    Returns:
        An aggregated protein sites table containing exactly one entry per protein site.
        Contains the following columns plus the column specified by 'group_by':
        - "Site ID": A unique row id with the format "ProteinID_ProteinSite.
        - "Representative protein": Protein identifier.
        - "Protein site": Modified amino acid position of the protein.
        - "Sequences": One or multiple unique plain sequences, separated by ";".
        - "Modified sequences" One or multiple unique modified sequences, separated
          by ";".
        - "# Sequences": Number of unique plain sequences.
        - "# Modified sequences": Number of unique modified sequences.
        - "Modification multiplicity": Indicates the count of modifications found in the
          peptides containing the respective site. Can contain one or mulitple
          entries, separated by ";".
        - "Best localization probability": Best observed site localization probability.
        - "Spectral count": Sum of spectral counts per protein site.
        - The column name specified with the 'score_column' argument, contains the best
          observed peptide score.
    """
    table = expanded_site_table.sort_values(by=group_by)

    protein_sites = join_unique(table, group_by, "Protein site", is_sorted=True)
    protein_sites.columns = ["Protein site"]

    protein_ids = join_unique(table, group_by, "Representative protein", is_sorted=True)
    protein_ids.columns = ["Representative protein"]

    sequences = join_unique(table, group_by, "Sequence", is_sorted=True)
    sequences.columns = ["Sequences"]

    modified_sequences = join_unique(
        table, group_by, "Modified sequence", is_sorted=True
    )
    modified_sequences.columns = ["Modified sequences"]

    n_sequences = count_unique(table, group_by, "Sequence", is_sorted=True)
    n_sequences.columns = ["# Sequences"]

    n_modified_sequences = count_unique(
        table, group_by, "Modified sequence", is_sorted=True
    )
    n_modified_sequences.columns = ["# Modified sequences"]

    multiplicity = join_unique(table, group_by, "Modification count", is_sorted=True)
    multiplicity.columns = ["Modification multiplicity"]

    aggregation, groups = aggregate_unique_groups(
        table, group_by, ["Site probability"], CONDENSE.maximum, is_sorted=True
    )
    best_probabilities = pd.DataFrame(
        columns=["Best localization probability"], data=aggregation, index=groups
    )

    aggregation, groups = aggregate_unique_groups(
        table, group_by, ["Spectral count"], CONDENSE.sum, is_sorted=True
    )
    spectral_counts = pd.DataFrame(
        columns=["Spectral count"], data=aggregation, index=groups
    )

    aggregation, groups = aggregate_unique_groups(
        table, group_by, [score_column], CONDENSE.maximum, is_sorted=True
    )
    best_peptide_score = pd.DataFrame(
        columns=[score_column], data=aggregation, index=groups
    )

    aggregated_sub_tables = [
        protein_ids,
        protein_sites,
        n_sequences,
        n_modified_sequences,
        sequences,
        modified_sequences,
        multiplicity,
        best_probabilities,
        spectral_counts,
        best_peptide_score,
    ]
    aggregated_table = pd.DataFrame(index=table[group_by].unique())
    for sub_table in aggregated_sub_tables:
        aggregated_table = aggregated_table.join(sub_table, how="outer")
    aggregated_table.index.rename(group_by, inplace=True)
    aggregated_table.reset_index(inplace=True)

    aggregated_table["Protein site"] = aggregated_table["Protein site"].astype(int)
    aggregated_table.sort_values(
        ["Representative protein", "Protein site"], inplace=True
    )
    return aggregated_table


def create_expanded_protein_site_table(
    table: pd.DataFrame,
    target_modification: str,
    score_column: str,
    include_modifications: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Expand entries into individual protein sites of the target modifications.

    First, entries from the "Modified sequence" and "Start position" are used to
    parse peptide modifications and map them to modified protein sites. Then, for each
    modified protein site of a peptide, a new entry row in the expanded protein sites
    table is created. This means that a peptide with two occurences of the target
    modification will generate two rows, one for each modified site. Hence, each row
    corresponds to exactly one modified protein site, but multiple rows can occur for
    the same protein site.

    Requires the following columns in the input table, plus the column specified with
    the 'score_column' argument:
    - "Representative protein"
    - "Modification localization string"
    - "Modified sequence"
    - "Start position"
    - "Spectral count" --> This column should be optional for DIA compatibility.

    Args:
        table: The input table that will be used for expanding individual protein sites.
        target_modification: The identifier of the modification that should be used for
            expanding modified protein sites.
        score_column: Column containing peptide identification quality scores, where
            higher numbers indicate better scores.
        include_modifications: Optional, allows specifying additional modification
            identifiers that will be included in "Modified sequence" entries together
            with the 'target_modification'.

    Returns:
        A new table containing expanded protein sites with the following columns:
        - "Site ID": A unique row id with the format "ProteinID_ProteinSite.
        - "Representative protein": Protein identifier.
        - "Protein site": Modified amino acid position of the protein.
        - "Sequence": Plain sequence of the identified peptide.
        - "Modified sequence": Peptide sequence containing modification identifiers,
          enclosed by square brackets, to the right of the modified amino acid.
        - "Modification count": Number of target modification occurences on the peptide.
        - "Site probability": Localization probability of the protein site.
        - "Spectral count": Number of spectral counts for the original peptide entry.
        - The column specified with the 'score_column' attribute.
    """
    if include_modifications is None:
        include_modifications = []
    if target_modification not in include_modifications:
        include_modifications.append(target_modification)

    site_collection = {
        "Site ID": [],
        "Representative protein": [],
        "Protein site": [],
        "Sequence": [],
        "Modified sequence": [],
        "Site probability": [],
        "Modification count": [],
        "Spectral count": [],
        score_column: [],
        "Sample": [],
    }

    for _, entry in table.iterrows():
        localization_probabilities = msreport.peptidoform.read_localization_string(
            entry["Modification localization string"]
        )
        modified_peptide = msreport.peptidoform.Peptide(
            entry["Modified sequence"],
            localization_probabilities,
            protein_position=entry["Start position"],
        )
        sequence = modified_peptide.plain_sequence
        modified_sequence = modified_peptide.make_modified_sequence(
            include=include_modifications
        )
        multiplicity = modified_peptide.count_modification(target_modification)
        peptide_score = entry[score_column]
        spectral_count = entry["Spectral count"]
        protein_id = entry["Representative protein"]
        sample = entry["Sample"]

        for protein_site in modified_peptide.list_modified_protein_sites(
            target_modification
        ):
            site_probability = modified_peptide.get_protein_site_probability(
                protein_site
            )
            site_id = f"{protein_id}_{protein_site}"
            site_collection["Site ID"].append(site_id)
            site_collection["Protein site"].append(protein_site)
            site_collection["Sequence"].append(sequence)
            site_collection["Modified sequence"].append(modified_sequence)
            site_collection["Modification count"].append(multiplicity)
            site_collection["Site probability"].append(site_probability)
            site_collection[score_column].append(peptide_score)
            site_collection["Spectral count"].append(spectral_count)
            site_collection["Representative protein"].append(protein_id)
            site_collection["Sample"].append(sample)
    expanded_protein_site_table = pd.DataFrame(site_collection)
    return expanded_protein_site_table
