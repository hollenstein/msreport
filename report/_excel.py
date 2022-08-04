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
- Add option to append all remaining columns (and hide them)
- Add option to specify sample order
- Add column comments
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
        self.column_width = 64  # args['column_width']
        self.border_weight = 2
        self.log2_tag = '[log2]'  # args['nan_symbol']
        self.nan_symbol = 'n.a.'  # args['nan_symbol']
        self.supheader_height = 30  # args['upheader_height']
        self.header_height = 105  # args['header_height']
        self.sample_extraction_tag = 'Spectral count'  # args['sample_extraction_tag']

        self.workbook = workbook
        self.worksheet = workbook.add_worksheet('Proteins')
        self._config = None
        self._table = None
        self._samples = None
        self._sample_groups = None
        self._format_templates = {}
        self._workbook_formats = {}
        self._conditional_formats = {}

    def apply_configuration(self, config_file: str) -> None:
        """ Reads a config file and prepares formats. """
        self._config = parse_config_file(config_file)
        self._add_formats(self._config['formats'])
        self._extend_header_format(self._config['groups'])
        self._extend_supheader_format(self._config['groups'])
        self._extend_border_formats()
        self._add_formats_to_workbook()
        self._add_conditionals(self._config['conditionals'])

    def add_data(self, table: pd.DataFrame) -> None:
        """ Adds table that will be used for filing the worksheet with data.

        Also extracts samples and generates possible sample comparison groups.
        """
        self._table = table.copy()
        if self._samples is None and self.sample_extraction_tag is not None:
            self._samples = extract_samples_with_column_tag(
                self._table, self.sample_extraction_tag
            )
        if self._sample_groups is None:
            # TODO: generate combinations of samples, separated by the
            #   comparison_symbol.
            pass

    def write_data(self) -> None:
        """ Writes data to the excel sheet and applies formats. """
        if self._config is None:
            raise Exception('Configuration has not applied. Call '
                            '"ReportSheet.apply_configuration()" to do so.')
        if self._table is None:
            raise Exception('No data for writing has been added. '
                            'Call "ReportSheet.add_data()" to add data.')

        coordinates = {
            'supheader_row': 0,
            'header_row': 1,
            'data_row_start': 2,
            'data_row_end': 2,
            'first_column': 0,
        }
        coordinates['data_row_end'] += self._table.shape[0] - 1
        coordinates['start_column'] = coordinates['first_column']

        # Prepare data
        data_groups = self._prepare_data_groups()

        ##############
        # Write data #
        ##############

        # for data_group in data_groups:
        #    coordinates['last_column'] = self._write_data_group(
        #        data_group, coordinates
        #    )
        #    coordinates['start_column'] = coordinates['last_column'] + 1
        def _write_data_group(self, data_group: dict,
                              coordinates: dict[str, int]) -> int:
            """ ...

            Args:
                data_group:
                coordinates: Dicionary containing information about row and
                    column positions. Keys are 'supheader_row', 'header_row',
                    'data_row_start', 'data_row_end', 'start_column'.

            Returns:
                Last column position that was filled with data
            """
            pass
        supheader_row = 0
        header_row = 1
        data_row_start = 2
        first_column = 0
        data_row_end = self._table.shape[0] + data_row_start - 1

        last_column = -1
        for data_group in data_groups:
            start_column = last_column + 1
            # Start column is input

            group_length = len(data_group['data'])
            end_column = start_column + group_length - 1

            # Write column data
            curr_column = start_column
            for values, format_name, conditional_name in data_group['data']:
                excel_format = self.get_format(format_name)
                self.worksheet.write_column(
                    data_row_start, curr_column, values, excel_format
                )
                if conditional_name:
                    excel_conditional = self.get_conditional(conditional_name)
                    self.worksheet.conditional_format(
                        data_row_start, curr_column,
                        data_row_end, curr_column, excel_conditional
                    )
                curr_column += 1

            # Write header data
            curr_column = start_column
            for text, format_name in data_group['header']:
                excel_format = self.get_format(format_name)
                self.worksheet.write(
                    header_row, curr_column, text, excel_format
                )
                curr_column += 1

            # Write supheader
            supheader_text, format_name = data_group['supheader']
            if supheader_text:
                supheader_format = self.get_format(format_name)
                self.worksheet.merge_range(
                    supheader_row, start_column, supheader_row, end_column,
                    supheader_text, supheader_format
                )

            # Set column width
            self.worksheet.set_column_pixels(
                start_column, end_column, data_group['col_width']
            )

            # Apply conditional formats to the group
            for conditional in data_group['conditional_formats']:
                if conditional is not None:
                    conditional_format = self.get_conditional(conditional)
                    self.worksheet.conditional_format(
                        data_row_start, start_column,
                        data_row_end, end_column, conditional_format
                    )

            # Return last column
            last_column = end_column

        # Set header height, freeze panes, and autofilter
        self.worksheet.set_row_pixels(supheader_row, self.supheader_height)
        self.worksheet.set_row_pixels(header_row, self.header_height)
        self.worksheet.freeze_panes(data_row_start, 1)  # Freeze only first column
        self.worksheet.autofilter(
            data_row_start - 1, first_column, data_row_end, last_column
        )

    def _prepare_data_groups(self):
        data_groups = []
        for name, config in self._config['groups'].items():
            if _eval_arg('comparison_group', config):
                pass
                # comp_group_data = self._prepare_comparison_group(name, config)
                # data_groups.extend(comp_group_data)
            elif _eval_arg('tag', config):
                group_data = self._prepare_sample_group(name, config)
                data_groups.append(group_data)
            else:
                group_data = self._prepare_feature_group(name, config)
                data_groups.append(group_data)
        return data_groups

    def _prepare_feature_group(self, name, config):
        """ Prepare data required to write a feature group. """
        columns = config['columns']
        group_data = {
            'data': self._prepare_column_data(config, columns),
            'header': self._prepare_column_headers(config, columns, name),
            'supheader': self._prepare_supheader(config, name),
            'col_width': config.get('width', self.column_width),
            'conditional_formats': [None]
        }
        return group_data

    def _prepare_sample_group(self, name, config):
        """ Prepare data required to write a sample group. """
        columns = helper.find_columns(self._table, config['tag'])
        group_data = {
            'data': self._prepare_column_data(config, columns),
            'header': self._prepare_column_headers(config, columns, name),
            'supheader': self._prepare_supheader(config, name),
            'col_width': config.get('width', self.column_width),
            'conditional_formats': [config['conditional']]
        }
        return group_data

    def _prepare_comparison_group(self, name, config):
        # use 'tag' and 'column_conditional' to find groups
        # comparison_group_data = []
        # for columns, comparison_name in some_function():
        #     supheader = comparison_name
        #     if _eval_arg('replace_comparison_tag', config):
        #         supheader = supheader.replace(
        #             config['tag'], config['replace_comparison_tag']
        #         )
        #     sub_config = config.copy()
        #     sub_config['columns'] = columns
        #     sub_config['supheader'] = comparison_name
        #     sub_config['tag'] = comparison_name
        #     sub_config['column_conditional'] = {}
        #     for tag, conditional in config['column_conditional'].items:
        #         match = None
        #         for column in columns:
        #             if column.find(tag) != -1:
        #                 match = column
        #         if match is not None:
        #             sub_config['column_conditional'][match] = conditional
        #     group_data = _prepare_sample_group(name, sub_config)
        #     comparison_group_data.append(group_data)
        # return comparison_group_data
        #
        # Change format to the border version if applicable
        #   - Special case for "sample group comparison group", add border to every
        #       sample pair group.
        # Write supheaders
        #   - Only if applicable. Special case for "sample group comparison group",
        #       add border to every sample pair group.
        group_data = {}
        return group_data


    def _prepare_column_data(self, config: dict, columns: list[str]) -> dict:
        data_info = []
        for column in columns:
            data = self._table[column]
            if _eval_arg('log2', config):
                data = data.replace(0, np.nan)
                if not helper.intensities_in_logspace(data):
                    data = np.log2(data)
                data = data.replace(np.nan, self.nan_symbol)

            format_name = config['format']
            conditional = None
            if 'column_format' in config and column in config['column_format']:
                format_name = config['column_format'][column]
            if 'column_conditional' in config:
                conditional = config['column_conditional'].get(column, None)
            data_info.append([data, format_name, conditional])
        if _eval_arg('border', config):
            data_info[0][1] = f'{data_info[0][1]}_left'
            data_info[-1][1] = f'{data_info[-1][1]}_right'
        return data_info

    def _prepare_column_headers(self, config: dict, columns: list[str], name: str) -> dict:
        header_info = []
        for text in columns:
            if _eval_arg('remove_tag', config):
                text = text.replace(config['tag'], '').strip()
            elif _eval_arg('log2', config):
                text = f'{text} {self.log2_tag}'.strip()
            format_name = f'header_{name}'
            header_info.append([text, format_name])
        if _eval_arg('border', config):
            header_info[0][1] = f'{header_info[0][1]}_left'
            header_info[-1][1] = f'{header_info[-1][1]}_right'
        return header_info

    def _prepare_supheader(self, config: dict, name: str) -> dict:
        text = config.get('supheader', None)
        if text and _eval_arg('log2', config):
            text = f'{text} {self.log2_tag}'.strip()
        format_name = f'supheader_{name}'
        supheader_info = [text, format_name]
        return supheader_info

        """
        # -> perform log2 transformation on data if required
        # -> change column names if required
        # -> define a list of data columns {data, format}
        # -> define a list of headers {text, format}
        # -> define a subheader {text, format} or None
        # -> define column width: from group or default
        # --> return group objects (can be multiple group objects in case of comparison groups)
        #
        # iterate over groups and write data
        #   - always pass on the last column position

        # Parse column groups that were defined in the config.yaml
        # TODO: maye extract the groups from self._config['groups'] before
        for group_name, group_info in self._config['groups'].items():
            # Get keyword arguments
            write_supheader = _eval_arg('supheader', group_info)
            remove_tag = _eval_arg('remove_tag', group_info)
            log2_transform = _eval_arg('log2', group_info)
            conditional_column_formats = _eval_arg('column_conditional', group_info)
            conditional_group_format = _eval_arg('conditional', group_info)

            # Define group columns
            if 'tag' in group_info:
                group_columns = helper.find_columns(self._table, group_info['tag'])
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

            # Apply column widths
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
                data = self._table[column]
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
        """

    def get_format(self, format_name: str) -> xlsxwriter.format.Format:
        """ Returns an excel format. """
        return self._workbook_formats[format_name]

    def get_conditional(self, format_name: str) -> dict[str, object]:
        """ Returns an excel conditional format. """
        return self._conditional_formats[format_name]

    def _add_formats(self, formats: dict[str, dict[str, object]]) -> None:
        """ Add formats. """
        for format_name in formats:
            format_properties = formats[format_name].copy()
            self._format_templates[format_name] = format_properties

    def _extend_header_format(self, groups: dict[str, object]) -> None:
        """ Adds individual header formats per group.

        This allows to individualize header formats, such as defining a
        different background color. The default 'header' format is extended
        and modified by all entries from the groups 'header_format' entry.
        """
        self._extend_formats('header', groups)

    def _extend_supheader_format(self, groups: dict[str, object]) -> None:
        """ Adds individual supheader formats per group.

        This allows to individualize supheader formats, such as defining a
        different background color. The default 'supheader' format is extended
        and modified by all entries from the groups 'supheader_format' entry.
        """
        self._extend_formats('supheader', groups)

    def _extend_border_formats(self) -> None:
        """ Add format variants with borders to the format templates.

        For each format adds a variant with a left or a right border, the
        format name is extended by 'format_left' or 'format_right'.
        """
        for name in list(self._format_templates):
            for border in ['left', 'right', 'left_right']:
                format_name = f'{name}_{border}'
                format_properties = self._format_templates[name].copy()
                for direction in border.split('_'):
                    format_properties[direction] = self.border_weight
                self._format_templates[format_name] = format_properties

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

    def _add_formats_to_workbook(self):
        """ Add the template formats to the workbook. """
        for name, properties in self._format_templates.items():
            self._workbook_formats[name] = self.workbook.add_format(
                properties
            )

    def _add_conditionals(self, formats: dict[str, dict[str, object]]) -> None:
        """ Add conditional formats to the conditional templates. """
        for format_name, format_properties in formats.items():
            self._conditional_formats[format_name] = format_properties


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


def _eval_arg(arg: str, args: dict) -> bool:
    """ Evaluates wheter arg is present in args and is not False. """
    return arg in args and args[arg] is not False


def extract_samples_with_column_tag(table: pd.DataFrame, tag: str):
    """ Extract sample names from columns containing the specified tag """
    # TODO: move to helper and add test #
    columns = helper.find_columns(table, tag, must_be_substring=True)
    samples = [c.replace(tag, '').strip() for c in columns]
    return samples




def write_data(self) -> None:
    """ Writes data to the excel sheet and applies formats. """
    if self._config is None:
        raise Exception('Configuration has not applied. Call '
                        '"ReportSheet.apply_configuration()" to do so.')
    if self._table is None:
        raise Exception('No data for writing has been added. '
                        'Call "ReportSheet.add_data()" to add data.')

    supheader_row = 0
    header_row = 1
    data_row = 2
    data_row_end = self._table.shape[0] + data_row - 1

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
            group_columns = helper.find_columns(self._table, group_info['tag'])
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

        # Apply column widths
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
            data = self._table[column]
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