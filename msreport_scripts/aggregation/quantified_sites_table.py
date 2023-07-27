from typing import Optional

import pandas as pd
import msreport


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
