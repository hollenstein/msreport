import pandas as pd
import pytest

import msreport.reader


class TestResultReader:
    @pytest.fixture(autouse=True)
    def _init_reader(self):
        self.reader = msreport.reader.ResultReader()
        self.reader._add_data_directory("./tests/testdata/common")
        self.reader.filenames = {"table": "table.txt"}
        self.reader.protected_columns = []
        self.reader.column_mapping = {}
        self.reader.column_tag_mapping = {}
        self.reader.sample_column_tags = []

        self.table = self.reader._read_file("table")
        self.table_nrows = 5
        self.table_ncolumns = 8

    def test_read_file_with_filename_lookup(self):
        table = self.reader._read_file("table")
        assert isinstance(table, pd.DataFrame)
        assert table.shape == (self.table_nrows, self.table_ncolumns)

    def test_read_file_with_filename(self):
        table = self.reader._read_file("table.txt")
        assert isinstance(table, pd.DataFrame)
        assert table.shape == (self.table_nrows, self.table_ncolumns)

    def test_rename_columns_with_mapping(self):
        self.reader.column_mapping = {"Column 1": "Renamed column"}

        assert "Column 1" in self.table.columns
        assert "Renamed column" not in self.table.columns
        self.table = self.reader._rename_columns(self.table, prefix_tag=False)
        assert "Column 1" not in self.table.columns
        assert "Renamed column" in self.table.columns

    def test_rename_columns_with_column_tag_mapping(self):
        self.reader.column_tag_mapping = {"Another_tag": "B_tag"}

        assert "B_tag Column 1" not in self.table.columns
        self.table = self.reader._rename_columns(self.table, prefix_tag=False)
        assert all(["Another_tag" not in c for c in self.table.columns])
        assert "B_tag Column 1" in self.table.columns

    @pytest.mark.parametrize(
        "prefix, expected_columns",
        [
            (True, ["A_tag Column 1", "A_tag Column 2", "A_tag Column 3"]),
            (False, ["Column 1 A_tag", "Column 2 A_tag", "Column 3 A_tag"]),
        ],
    )
    def test_rename_columns_with_sample_column_tags(self, prefix, expected_columns):
        self.reader.sample_column_tags = ["A_tag"]

        self.table = self.reader._rename_columns(self.table, prefix_tag=prefix)
        assert all([c in self.table.columns for c in expected_columns])

    def test_rename_columns_with_protected_columns(self):
        self.reader.protected_columns = ["Column 3 A_tag"]
        self.reader.column_tag_mapping = {"A_tag": "B_tag"}

        self.table = self.reader._rename_columns(self.table, prefix_tag=False)
        table_columns = self.table.columns.tolist()
        assert "Column 3 A_tag" in table_columns
        assert "B_tag Column 1" in table_columns
        assert "B_tag Column 2" in table_columns

    @pytest.mark.parametrize(
        "drop_columns, num_dropped_columns",
        [
            (["Column 1"], 1),
            (["Column 1", "Column 2", "Column 3"], 3),
            ([], 0),
            (["Column that does not exist"], 0),
        ],
    )
    def test_drop_columns(self, drop_columns, num_dropped_columns):
        self.table = self.reader._drop_columns(self.table, drop_columns)
        assert self.table.shape[1] == self.table_ncolumns - num_dropped_columns

    def test_drop_columns_by_tag(self):
        self.table = self.reader._drop_columns_by_tag(self.table, "A_tag")
        assert all(["A_tag" not in c for c in self.table.columns])
        assert self.table.shape[1] < self.table_ncolumns


class TestResultReaderSpectronautStyleTags:
    # Test file with dots instead of spaces in column names.
    @pytest.fixture(autouse=True)
    def _init_reader(self):
        self.reader = msreport.reader.ResultReader()
        self.reader._add_data_directory("./tests/testdata/spectronaut")
        self.reader.filenames = {"table": "table_spectronaut_style_tag.txt"}
        self.reader.protected_columns = []
        self.reader.column_mapping = {}
        self.reader.column_tag_mapping = {
            ".A_tag": " A_tag",
            "Another_tag.": "Another_tag ",
        }
        self.reader.sample_column_tags = [".A_tag", "Another_tag."]
        self.table = self.reader._read_file("table")

    def test_rename_columns_with_column_tag_mapping(self):
        self.table = self.reader._rename_columns(self.table, prefix_tag=True)
        rearranged_columns = [
            "A_tag Column 1",
            "A_tag Column 2",
            "A_tag Column 3",
            "Another_tag Column 1",
            "Another_tag Column 2",
        ]
        for column in rearranged_columns:
            assert column in self.table.columns.to_list()

        self.table = self.reader._rename_columns(self.table, prefix_tag=False)
        rearranged_columns = [
            "Column 1 A_tag",
            "Column 2 A_tag",
            "Column 3 A_tag",
            "Column 1 Another_tag",
            "Column 2 Another_tag",
        ]
        for column in rearranged_columns:
            assert column in self.table.columns.to_list()
