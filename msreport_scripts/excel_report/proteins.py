from typing import Optional, Callable

import pandas as pd
import xlsxreport


def write_protein_report(
    protein_table: pd.DataFrame,
    experimental_design: pd.DataFrame,
    outpath: str,
    config: str = "msreport_lfq_protein.yaml",
    special_proteins: Optional[list] = None,
) -> None:
    """Writes an excel protein report from an MsReport protein table.

    Args:
        table: A dataframe that contains a protein table, will be written to the excel
            sheet "Proteins" and formated by using the specified config file.
        experimental_design: A dataframe that contains an experimental design table,
            will be written to the excel sheet "Info".
        outpath: Output path of the generated files.
        config: Path of an XlsxReport config file. If the config file is present in the
            default XlsxReport app directory, it is enough to specify the filename.
        special_proteins: Optional, allows specifying a list of entries in the
            "Representative protein" column from the 'protein_table', which will be
            sorted to the top.
    """
    design = experimental_design.copy()
    table = protein_table.copy()
    special_proteins = special_proteins if special_proteins is not None else []
    table.sort_values(
        ["Representative protein", "Spectral count Combined"],
        key=_create_protein_sorter(special_proteins),
        ascending=[False, False],
        na_position="last",
        inplace=True,
    )

    config_path = xlsxreport.get_config_file(config)
    with xlsxreport.Reportbook(outpath) as reportbook:
        _write_design(reportbook, design)
        _write_proteins(reportbook, table, config_path)


def _write_design(reportbook: xlsxreport.Reportbook, design: pd.DataFrame) -> None:
    """Adds an "Info" excel sheet containing data from the 'design' table.

    Args:
        reportbook: An XlsxReport Reportbook class instance for writing the Excel
            Workbook file.
        design: Dataframe that will be written to the "Info" excel sheet.
    """
    design_columns = ["Raw files", "Sample", "Experiment"]
    for column in design_columns:
        if column not in design.columns:
            design[column] = ""
    design = design[design_columns]

    info_sheet = reportbook.add_infosheet()

    header_row = 2
    data_row_start = header_row + 1
    data_row_end = data_row_start + len(design)
    column_width = 125

    header_format = reportbook.add_format({"bold": True, "top": 2, "bottom": 2})
    for col_idx, column in enumerate(design.columns):
        info_sheet.write(header_row, col_idx, column, header_format)
        info_sheet.write_column(data_row_start, col_idx, design[column])
        info_sheet.set_column_pixels(col_idx, col_idx, column_width)

    top_border_format = reportbook.add_format({"top": 2})
    for col_idx, column in enumerate(design.columns):
        info_sheet.write(data_row_end, col_idx, "", top_border_format)


def _write_proteins(
    reportbook: xlsxreport.Reportbook, table: pd.DataFrame, config_path: str
) -> None:
    """Adds a "Proteins" excel sheet containing formatted data from the 'table'.

    Args:
        reportbook: An XlsxReport Reportbook class instance for writing the Excel
            Workbook file.
        table: Dataframe that will be written to the "Proteins" excel sheet.
        config_path: Path of an XlsxReport config file.
    """
    protein_sheet = reportbook.add_datasheet("Proteins")
    protein_sheet.apply_configuration(config_path)
    protein_sheet.add_data(table)
    protein_sheet.write_data()


def _create_protein_sorter(
    special_proteins: list[str],
) -> Callable[pd.Series, pd.Series]:
    """Returns a sorting function that puts entries from 'special_proteins' in front."""

    def sorter(series: pd.Series) -> pd.Series:
        if series.name == "Representative protein":
            return series.isin(special_proteins)
        else:
            return series

    return sorter
