""" TODOs
- Add different conditional formatting for "Combined" column.
    - Maybe only consider columns that also contain a sample entry
    - Specify a tag for the combined column (such as '', 'combined',
      'total'). This should remove e.g. "iBAQ peptides" from the
      "iBAQ" tag.
    - Move the combined column to the front.
- Ignore columns that don't exist without causing an exception
    - Could this be a switch argument?
- Parse arguments from the config file
- Remove columns that have already been parsed to avoid duplication
- Add column comments
- Add option to specify sample order
- Add option to sort the table before writing data
"""

import numpy as np
import pandas as pd
import xlsxwriter
import yaml

import helper


class ReportSheet():
    default_format = {'align': 'left', 'num_format': '0'}

    def __init__(self, workbook: xlsxwriter.Workbook):
        self.args = {
            'border_weight': 2,
            'log2_tag': '[log2]',
            'nan_symbol': 'n.a.',
            'supheader_height': 30,
            'header_height': 105,
        }
        self.border_weight = 2
        self.log2_tag = '[log2]'  # args['nan_symbol']
        self.nan_symbol = 'n.a.'  # args['nan_symbol']
        self.supheader_height = 30  # args['upheader_height']
        self.header_height = 105  # args['header_height']

        self.workbook = workbook
        self.worksheet = workbook.add_worksheet('Proteins')
        self._config = None
        self._format_templates = {}
        self._workbook_formats = {}
        self._conditional_formats = {}

    def apply_configuration(self, config_file: str) -> None:
        """ Reads a config file and prepares formats. """
        self._config = parse_config_file(config_file)
        self.add_formats(self._config['formats'])
        self.extend_header_format(self._config['groups'])
        self.extend_supheader_format(self._config['groups'])
        self.extend_border_formats()
        self.add_formats_to_workbook()
        self.add_conditionals(self._config['conditionals'])

    def get_format(self, format_name: str) -> xlsxwriter.format.Format:
        """ Returns an excel format. """
        return self._workbook_formats[format_name]

    def get_conditional(self, format_name: str) -> dict[str, object]:
        """ Returns an excel conditional format. """
        return self._conditional_formats[format_name]

    def add_formats(self, formats: dict[str, dict[str, object]]) -> None:
        """ Add formats. """
        for format_name in formats:
            format_properties = formats[format_name].copy()
            self._format_templates[format_name] = format_properties

    def extend_border_formats(self) -> None:
        """ Add format variants with borders to the format templates.

        For each format adds a variant with a left or a right border, the
        format name is extended by 'format_left' or 'format_right'.
        """
        for name in list(self._format_templates):
            for border in ['left', 'right']:
                format_properties = self._format_templates[name].copy()
                format_name = f'{name}_{border}'
                format_properties[border] = self.border_weight
                self._format_templates[format_name] = format_properties

    def extend_header_format(self, groups: dict[str, object]) -> None:
        """ Adds individual header formats per group.

        This allows to individualize header formats, such as defining a
        different background color. The default 'header' format is extended
        and modified by all entries from the groups 'header_format' entry.
        """
        self._extend_formats('header', groups)

    def extend_supheader_format(self, groups: dict[str, object]) -> None:
        """ Adds individual supheader formats per group.

        This allows to individualize supheader formats, such as defining a
        different background color. The default 'supheader' format is extended
        and modified by all entries from the groups 'supheader_format' entry.
        """
        self._extend_formats('supheader', groups)

    def _extend_formats(
            self, key: str, groups: dict[str, object]) -> None:
        """ Adds individual format types per group.

        This allows to individualize header or supheader formats, such as
        defining a different background color or vertical rotation.
        The default format is extended and modified by all entries from the
        groups 'KEY_format' entry.
        """
        for group_name, group_info in groups.items():
            base_format = self._format_templates[key]
            group_format = base_format.copy()
            if f'{key}_format' in group_info:
                group_format.update(group_info[f'{key}_format'])
            group_format_name = f'{key}_{group_name}'
            self._format_templates[group_format_name] = group_format

    def add_conditionals(self, formats: dict[str, dict[str, object]]) -> None:
        """ Add conditional formats to the conditional templates. """
        for format_name, format_properties in formats.items():
            self._conditional_formats[format_name] = format_properties

    def add_formats_to_workbook(self):
        """ Add the template formats to the workbook. """
        for name, properties in self._format_templates.items():
            self._workbook_formats[name] = self.workbook.add_format(
                properties
            )

    def write_data(self, table: pd.DataFrame) -> None:
        """ Writes data from the table to the excel sheet. """
        if self._config is None:
            raise Exception('Configuration was not applied.')

        supheader_row = 0
        header_row = 1
        data_row = 2
        data_row_end = table.shape[0] + data_row - 1

        column_position = 0
        first_column_position = column_position

        # Parse column groups that were defined in the config.yaml
        # TODO: maye extract the groups from self._config['groups'] before
        for group_name, group_info in self._config['groups'].items():
            # Get keyword arguments
            write_supheader = 'supheader' in group_info
            remove_tag = 'remove_tag' in group_info and group_info['remove_tag']
            log2_transform = 'log2' in group_info and group_info['log2']
            conditional_column_formats = 'column_conditional' in group_info
            conditional_group_format = 'conditional' in group_info

            # Define group columns
            if 'tag' in group_info:
                group_columns = helper.find_columns(table, group_info['tag'])
            else:
                group_columns = group_info['columns']

            # Define positions of first and last column
            group_length = len(group_columns)
            start_col = column_position
            end_col = column_position + group_length - 1

            # Define format per column, consider default and individual formats
            header_format = f'header_{group_name}'
            header_formats = {c: header_format for c in group_columns}
            col_formats = {c: group_info['format'] for c in group_columns}
            if 'column_format' in group_info:
                for column, format_name in group_info['column_format'].items():
                    col_formats[column] = format_name

            # Change format to the border version if applicable
            if 'border' in group_info and group_info['border']:
                first_column = group_columns[0]
                col_formats[first_column] = f'{col_formats[first_column]}_left'
                header_formats[first_column] = f'{header_format}_left'
                last_column = group_columns[-1]
                col_formats[last_column] = f'{col_formats[last_column]}_right'
                header_formats[last_column] = f'{header_format}_right'

            # Define column widths
            if 'width' in group_info:
                self.worksheet.set_column_pixels(
                    start_col, end_col, group_info['width']
                )

            # Write supheaders
            if write_supheader:
                text = group_info['supheader']
                if log2_transform:
                    text = f'{text} {self.log2_tag}'.strip()
                format_name = f'supheader_{group_name}'
                supheader_format = self.get_format(format_name)

                self.worksheet.merge_range(
                    supheader_row, start_col, supheader_row, end_col,
                    text, supheader_format
                )

            # Write headers
            for current_position, column in enumerate(group_columns):
                col_position = start_col + current_position
                text = column
                if remove_tag:
                    text = text.replace(group_info['tag'], '').strip()
                if log2_transform and not remove_tag:
                    text = f'{text} {self.log2_tag}'.strip()
                header_format = self.get_format(header_formats[column])
                self.worksheet.write(
                    header_row, col_position, text, header_format
                )

            # Write data
            for current_position, column in enumerate(group_columns):
                col_position = start_col + current_position
                col_format = self.get_format(col_formats[column])
                data = table[column]
                if log2_transform:
                    data = data.replace(0, np.nan)
                    if not helper.intensities_in_logspace(data):
                        data = np.log2(data)
                    data = data.replace(np.nan, self.nan_symbol)
                self.worksheet.write_column(
                    data_row, col_position, data, col_format
                )

            # Add conditional formats for individual columns
            if conditional_column_formats:
                for current_position, column in enumerate(group_columns):
                    if column not in group_info['column_conditional']:
                        continue
                    col_position = start_col + current_position
                    conditional_name = group_info['column_conditional'][column]
                    conditional_format = self.get_conditional(conditional_name)
                    self.worksheet.conditional_format(
                        data_row, col_position, data_row_end, col_position,
                        conditional_format
                    )
            # Add conditional formats per group
            if conditional_group_format:
                conditional_format = self.get_conditional(group_info['conditional'])
                self.worksheet.conditional_format(
                    data_row, start_col, data_row_end, end_col,
                    conditional_format
                )

            # Advance column position by number of written columns
            column_position = end_col + 1
        last_column_position = column_position - 1
        # Set header height
        self.worksheet.set_row_pixels(supheader_row, self.supheader_height)
        self.worksheet.set_row_pixels(header_row, self.header_height)
        self.worksheet.freeze_panes(data_row, 1)  # Freeze only first column
        self.worksheet.autofilter(
            data_row - 1, first_column_position,
            data_row_end, last_column_position
        )


class Report():
    def __init__(self):
        self.workbook = None
        self.report_sheets = {}


def parse_config_file(file: str) -> dict[str, dict]:
    """ Parses excel report config file and returns entries as dictionaries.

    Returns:
        Dictionary containing the keys 'formats', 'conditionals', 'groups',
            'args', each pointing to another dictionary.
    """
    with open(file) as open_file:
        yaml_file = yaml.safe_load(open_file)
    config = {
        'args': _extract_config_entry(yaml_file, 'args'),
        'groups': _extract_config_entry(yaml_file, 'groups'),
        'formats': _extract_config_entry(yaml_file, 'formats'),
        'conditionals': _extract_config_entry(
            yaml_file, 'conditional_formats'
        ),
    }
    return config


def _extract_config_entry(
        config: dict[str, dict], name: str) -> dict[str, object]:
    return config.pop(name) if name in config else dict()
