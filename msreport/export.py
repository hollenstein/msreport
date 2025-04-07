"""
Columns that are not yet present in the amica output at the moment:
Index([
    'Protein Probability',
    'Top Peptide Probability',
    'Total peptides',
    'Leading proteins',
    'Protein entry name',
    'Fasta header',
    'Protein length',
    'iBAQ peptides',
    'Sequence coverage',
], dtype='object')
"""

import os
import warnings
from collections import defaultdict as ddict
from typing import Iterable, Optional, Protocol, Sequence

import numpy as np
import pandas as pd

import msreport.helper as helper
import msreport.reader
from msreport.qtable import Qtable


class Protein(Protocol):
    """Abstract protein entry"""

    header: str
    sequence: str
    header_fields: dict[str, str]


class ProteinDatabase(Protocol):
    """Abstract protein database"""

    def __getitem__(self, protein_id: str) -> Protein: ...

    def __contains__(self, protein_id: str) -> bool: ...


def contaminants_to_clipboard(qtable: Qtable) -> None:
    """Creates a contaminant table and writes it to the system clipboard.

    The contaminant table contains "iBAQ rank", "riBAQ", "iBAQ intensity", "Intensity",
    and "Expression" columns for each sample. Imputed values in the "Expression" columns
    are set to NaN.

    The qtable must at least contain "iBAQ intensity" and "Missing" sample columns, and
    a "Potential contaminant" column, expression columns must be set. For calculation
    of iBAQ intensities refer to msreport.reader.add_ibaq_intensities(). "Missing"
    sample columns can be added with msreport.analyze.analyze_missingness().

    Args:
        qtable: A Qtable instance. Requires that column names follow the MsReport
            conventions.
    """
    columns = [
        "Representative protein",
        "Protein entry name",
        "Gene name",
        "Fasta header",
        "Protein length",
        "Total peptides",
        "iBAQ peptides",
        "iBAQ intensity total",
    ]
    column_tags = ["iBAQ rank", "riBAQ", "iBAQ intensity", "Intensity", "Expression"]

    samples = qtable.get_samples()
    data = qtable.get_data()

    data["iBAQ intensity total"] = np.nansum(
        data[[f"iBAQ intensity {s}" for s in samples]], axis=1
    ) / len(samples)
    for sample in samples:
        data.loc[data[f"Missing {sample}"], f"Expression {sample}"] = np.nan

        ibaq_values = data[f"iBAQ intensity {sample}"]
        order = np.argsort(ibaq_values)[::-1]
        rank = np.empty_like(ibaq_values, dtype=int)
        rank[order] = np.arange(1, len(ibaq_values) + 1)
        data[f"iBAQ rank {sample}"] = rank
        data[f"riBAQ {sample}"] = ibaq_values / ibaq_values.sum() * 100

    for column_tag in column_tags:
        columns.extend(helper.find_sample_columns(data, column_tag, samples))
    columns = [c for c in columns if c in data.columns]

    contaminants = qtable["Potential contaminant"]
    data = data.loc[contaminants, columns]

    data.sort_values("iBAQ intensity total", ascending=False, inplace=True)
    data.to_clipboard(index=False)


def to_perseus_matrix(
    qtable: Qtable,
    directory,
    table_name: str = "perseus_matrix.tsv",
) -> None:
    """Exports a qtable to a perseus matrix file in tsv format.

    The Perseus matrix file has a second header row that contains single-letter entries
    for column annotations. The first entry starts with the string "#!{Type}" followed
    by an annotation letter, such as "#!{Type}E".

    The annotation single letter code is:
        E = Expression
        N = numerical
        C = Categorical
        T = Text

    Args:
        qtable: A Qtable instance.
        directory: Output path of the generated files.
        table_name: Optional, filename of the perseus matrix file. Default is
            "perseus_matrix.tsv".
    """
    table = qtable.data
    default_category = "T"
    annotation_row_prefix = "#!{Type}"
    categorical_tags = ["Events", "Missing"]

    categorical_columns = ["Potential contaminant", "Valid"]
    for tag in categorical_tags:
        categorical_columns.extend([c for c in table.columns if tag in c])

    expression_columns = [qtable.get_expression_column(s) for s in qtable.get_samples()]

    numeric_columns = table.select_dtypes(include="number").columns.tolist()
    numeric_columns = set(numeric_columns).difference(expression_columns)
    numeric_columns = set(numeric_columns).difference(categorical_columns)

    column_categories: ddict[str, str] = ddict(lambda: default_category)
    column_categories.update(dict.fromkeys(numeric_columns, "N"))
    column_categories.update(dict.fromkeys(categorical_columns, "C"))
    column_categories.update(dict.fromkeys(expression_columns, "E"))

    column_annotation = [column_categories[column] for column in table.columns]
    column_annotation[0] = f"{annotation_row_prefix}{column_annotation[0]}"
    annotation_frame = pd.DataFrame(columns=table.columns, data=[column_annotation])

    perseus_matrix = pd.concat([annotation_frame, table])
    perseus_matrix_path = os.path.join(directory, table_name)
    perseus_matrix.to_csv(perseus_matrix_path, sep="\t", index=False)


def to_amica(
    qtable: Qtable,
    directory,
    table_name: str = "amica_table.tsv",
    design_name: str = "amica_design.tsv",
) -> None:
    """Exports a qtable to an amica protein table and design files.

    Note that amica expects the same number of columns for each group of intensity
    columns (Intensity, LFQIntensity, ImputedIntensity, iBAQ), therefore only sample
    columns are included from samples that are present in the qtable design.

    Args:
        qtable: A Qtable instance.
        directory: Output path of the generated files.
        table_name: Optional, filename of the amica table file. Default is
            "amica_table.tsv".
        design_name: Optional, filename of the amica design file. Default is
            "amica_design.tsv".
    """
    amica_table = _amica_table_from(qtable)
    amica_table_path = os.path.join(directory, table_name)
    amica_table.to_csv(amica_table_path, sep="\t", index=False)

    amica_design = _amica_design_from(qtable)
    amica_design_path = os.path.join(directory, design_name)
    amica_design.to_csv(amica_design_path, sep="\t", index=False)


def write_html_coverage_map(
    filepath: str,
    protein_id: str,
    peptide_table: pd.DataFrame,
    protein_db: ProteinDatabase,
    displayed_name: Optional[str] = None,
    coverage_color: str = "#E73C40",
    highlight_positions: Optional[Iterable[int]] = None,
    highlight_color: str = "#1E90FF",
    column_length: int = 10,
    row_length: int = 50,
):
    """Generates an html file containing a protein coverage map.

    Args:
        filepath: The filepath where the generated html file will be saved.
        protein_id: ID of the protein that will be displayed on the html page. Must
            correspond to an entry in the specified `protein_db`.
        peptide_table: Dataframe which contains peptide information required for
            calculation of the protein sequence coverage.
        protein_db: A protein database containing entries from one or multiple FASTA
            files.
        displayed_name: Allows specifying a custom displayed name. By default, the
            protein name and protein id are shown.
        coverage_color: Hex color code for highlighting amino acids that correspond to
            covered regions from the coverage mask, for example "#FF0000" for red.
        highlight_positions: Optional, allows specifying a list of amino acid positions
            that are highlighted in a different color. Note that positions specified
            here will overwrite the coloring from the coverage mask. Positions are
            one-indexed, which means that the first amino acid positions is 1.
        highlight_color: Hex color code for highlighting amino acids specified with the
            'highlight_positions' variable.
        column_length: Number of amino acids after which a space is inserted.
        row_length: Number of amino acids after which a new line is inserted.
    """
    warnings.warn(
        (
            "`write_html_coverage_map` is still experimental, and the interface might "
            "change in a future release."
        ),
        FutureWarning,
        stacklevel=2,
    )
    # Get protein information from the protein database
    protein_entry = protein_db[protein_id]
    sequence = protein_entry.sequence
    protein_length = len(sequence)

    if displayed_name is None:
        protein_name = msreport.reader._get_annotation_protein_name(
            protein_entry, default_value=protein_id
        )
        if protein_name == protein_id:
            displayed_name = protein_id
        else:
            displayed_name = f"{protein_name} ({protein_id})"

    # Generate coverage boundaries from a peptide table
    id_column = "Representative protein"
    peptide_group = peptide_table[peptide_table[id_column] == protein_id]
    peptide_positions = list(
        zip(peptide_group["Start position"], peptide_group["End position"])
    )
    coverage_mask = helper.make_coverage_mask(protein_length, peptide_positions)
    boundaries = _find_covered_region_boundaries(coverage_mask)

    # Define highlight positions
    highlight_positions = highlight_positions if highlight_positions is not None else ()
    highlights = {pos - 1: highlight_color for pos in highlight_positions}
    html_title = f"Coverage map: {displayed_name}"

    # Generate and save the html page
    sequence_coverage = helper.calculate_sequence_coverage(
        protein_length, peptide_positions, ndigits=1
    )
    html_sequence_map = _generate_html_sequence_map(
        sequence,
        boundaries,
        coverage_color,
        highlights=highlights,
        column_length=column_length,
        row_length=row_length,
    )
    html_text = _generate_html_coverage_map_page(
        html_sequence_map, sequence_coverage, title=html_title
    )
    with open(filepath, "w") as openfile:
        openfile.write(html_text)


def _amica_table_from(qtable: Qtable) -> pd.DataFrame:
    """Returns a dataframe in the amica format.

    Args:
        table: A dataframe containing experimental data. Requires that column names
            follow the MsReport conventions.

    Returns:
        A dataframe which columns are in the amica data table format. Note that only
        intensity columns are included from samples that are present in the qtable
        design.
    """
    filter_columns = ["Valid", "Potential contaminant"]
    amica_column_mapping = {
        "Representative protein": "Majority.protein.IDs",
        "Gene name": "Gene.names",
        "Valid": "quantified",
        "Potential contaminant": "Potential.contaminant",
    }
    amica_column_tag_mapping = {
        "Intensity ": "Intensity_",
        "LFQ intensity ": "LFQIntensity_",
        "Expression ": "ImputedIntensity_",
        "iBAQ intensity ": "iBAQ_",
        "Spectral count ": "razorUniqueCount_",
        "Average expression ": "AveExpr_",
        "Ratio [log2] ": "logFC_",
        "P-value ": "P.Value_",
        "Adjusted p-value ": "adj.P.Val_",
    }
    intensity_column_tags = [
        "Intensity",
        "LFQ intensity",
        "Expression",
        "iBAQ intensity",
    ]
    sample_columns_tags = ["Spectral count"] + intensity_column_tags
    amica_comparison_tag = (" vs ", "__vs__")

    amica_table = qtable.get_data()

    # Drop intensity columns from samples that are not present in the design
    for tag in sample_columns_tags:
        columns = helper.find_columns(amica_table, tag)
        sample_columns = helper.find_sample_columns(
            amica_table, tag, qtable.get_samples()
        )
        non_sample_columns = list(set(columns).difference(set(sample_columns)))
        amica_table.drop(columns=non_sample_columns, inplace=True, axis=1)

    # Log transform columns if necessary
    for tag in intensity_column_tags:
        for column in helper.find_columns(amica_table, tag):
            if not helper.intensities_in_logspace(amica_table[column]):
                amica_table[column] = amica_table[column].replace({0: np.nan})
                amica_table[column] = np.log2(amica_table[column])

    for old_column in helper.find_columns(amica_table, amica_comparison_tag[0]):
        new_column = old_column.replace(*amica_comparison_tag)
        amica_table.rename(columns={old_column: new_column}, inplace=True)

    for column in filter_columns:
        if column in amica_table.columns:
            amica_table[column] = ["+" if i else "" for i in amica_table[column]]

    for old_tag, new_tag in amica_column_tag_mapping.items():
        for old_column in helper.find_columns(amica_table, old_tag):
            new_column = old_column.replace(old_tag, new_tag)
            amica_column_mapping[old_column] = new_column
    amica_table.rename(columns=amica_column_mapping, inplace=True)

    amica_columns = [
        col for col in amica_column_mapping.values() if col in amica_table.columns
    ]
    return amica_table[amica_columns]


def _amica_design_from(qtable: Qtable) -> pd.DataFrame:
    """Returns an experimental design table in the amica format.

    Args:
        design: A dataframe that must contain the columns "Sample" and "Experiment".

    Returns:
        A dataframe which columns are in the amica design table format.
    """
    design = qtable.get_design()
    amica_design_columns = {"Sample": "samples", "Experiment": "groups"}
    amica_design = design.rename(columns=amica_design_columns)
    return amica_design


def _generate_html_coverage_map_page(
    html_sequence_map: str, coverage: float, title: str = "Protein coverage map"
) -> str:
    """Generates the code for an html pag displaying a protein coverage map.

    Args:
        html_sequence_map: A string containing html code that represents a protein
            coverage map.
        coverage: Sequence coverage in percent.
        title: Title of coverage page, is displayed in the browser tab as well as a
            title on the page itself.

    Returns:
        A string containing the html code of the sequence coverage html page.

    """
    # fmt: off
    html_lines = (
        '<!-- index.html -->',
        '',
        '<!DOCTYPE html>',
        '<html lang="en">',
        '    <head>',
        '        <meta charset="utf-8">',
        f'        <title>{title}</title>',
        '        <style>',
        '           h1 {font-family: "Arial", sans-serif;}'
        '           body {',
        '               font-family: "Lucida Console", "Courier new", monospace;',
        '               font-size: 100%;'
        '           }',
        '        </style>',
        '    </head>',
        '    <body>',
        f'        <h1>{title}</h1>',
        f'        <p>Sequence coverage: {coverage}%</p>',
        f'        <p><PRE>{html_sequence_map}</PRE></p>',
        '    </body>',
        '</html>',
    )
    # fmt: on
    html_string = "\n".join(html_lines)
    return html_string


def _generate_html_sequence_map(
    sequence: str,
    covered_regions: Iterable[Iterable[int]],
    coverage_color: str,
    highlights: Optional[dict[int, str]] = None,
    column_length: int = 10,
    row_length: int = 50,
) -> str:
    """Generates the html code for a sequence coverage map with colored highlighting.

    Args:
        sequence: Amino acid sequence of a protein
        covered_regions: A list of tuples, where each tuple specifies the start and end
            positions of the continuously covered regions in the protein sequence. Note
            that the positions are zero-indexed.
        coverage_color: Hex color code for highlighting amino acids from the covered
            regions.
        highlights: Optional, allows specifying amino acid positions that should be
            highlighted with a specific color. Must be a dictionary with keys being
            zero indexed protein positions and values hex color codes.
        column_length: Number of amino acids after which a space is inserted.
        row_length: Number of amino acids after which a new line is inserted.

    Returns:
        A string containing the html code of the sequence coverage map.
    """
    if covered_regions:
        coverage_start_idx, coverage_stop_idx = list(zip(*covered_regions))
    else:
        coverage_start_idx, coverage_stop_idx = (), ()
    highlights = highlights if highlights is not None else {}
    sequence_length = len(sequence)

    def write_row_index(pos: int, strings: list):
        ndigits = len(str(sequence_length))
        row_index = str(pos + 1).rjust(ndigits)
        html_entry = '<FONT COLOR="#000000">' + row_index + "   " + "</FONT>"
        strings.append(html_entry)

    def open_coverage_region(strings: list):
        strings.append(f'<FONT COLOR="{coverage_color}">')

    def close_coverage_region(strings: list):
        strings.append("</FONT>")

    def is_end_of_row(pos: int):
        return (pos != 0) and (pos % row_length == 0)

    def is_end_of_column(pos: int):
        return (pos != 0) and (pos % column_length == 0) and not is_end_of_row(pos)

    in_covered_region: bool = False
    strings = []
    strings.append('<FONT COLOR="#606060">')  # Set default text color to grey
    write_row_index(0, strings)
    for pos, character in enumerate(sequence):
        if pos in coverage_start_idx:
            in_covered_region = True
            open_coverage_region(strings)

        if is_end_of_row(pos):
            if in_covered_region:
                close_coverage_region(strings)
            strings.append("<br>")
            write_row_index(pos, strings)
            if in_covered_region:
                open_coverage_region(strings)
        elif is_end_of_column(pos):
            strings.append(" ")

        if pos in highlights:
            color = highlights[pos]
            strings.append(f'<FONT COLOR="{color}"><u>{character}</u></FONT>')
        else:
            strings.append(character)

        if pos in coverage_stop_idx:
            in_covered_region = False
            close_coverage_region(strings)
    strings.append("</FONT>")

    html_sequence_block = "".join(strings)
    return html_sequence_block


def _find_covered_region_boundaries(
    coverage_mask: Sequence[bool],
) -> list[tuple[int, int]]:
    """Returns a list of boundaries from continuously covered regions in a protein.

    Args:
        coverage_mask: An iterable of boolean values that represents the coverage map of
            a protein sequence. A True value at a specific position indicates that the
            corresponding amino acid was covered by the identified peptides.

    Returns:
        A list of tuples, where each tuple specifies the start and end positions of the
        continuously covered regions in the coverage mask. Note that the positions are
        zero-indexed.

    Examples:
        >>> coverage_mask = [True, True, False, False, True]
        >>> _find_covered_region_boundaries(coverage_mask)
        ... [(0, 1), (4, 4)]
    """
    start = []
    stop = []

    start_index = 0

    previous_was_covered = coverage_mask[0]
    if previous_was_covered:
        start.append(start_index)
    for i, is_covered in enumerate(coverage_mask[1:], start=start_index + 1):
        if is_covered and not previous_was_covered:
            start.append(i)
        if not is_covered and previous_was_covered:
            stop.append(i - 1)
        previous_was_covered = is_covered
    if previous_was_covered:
        stop.append(i)
    return list(zip(start, stop))
