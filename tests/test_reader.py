import os

import numpy as np
import pandas as pd
import pytest

import msreport.reader
from msreport.helper.temp import ProteinDatabase, Protein


@pytest.fixture
def example_fpreader():
    return msreport.reader.FPReader("./tests/testdata/fragpipe_")


@pytest.fixture
def example_mqreader():
    return msreport.reader.MQReader("./tests/testdata/maxquant_")


def test_that_always_passes():
    assert True


def test_extract_sample_names():
    reader = msreport.reader.ResultReader()
    reader._add_data_directory("./tests/testdata/common")
    table = reader._read_file("table.txt")
    sample_names = msreport.reader.extract_sample_names(table, "A_tag")
    assert set(sample_names) == set(["Column 1", "Column 2", "Column 3"])


@pytest.mark.parametrize(
    "tag, prefixed, columns, rearranged_columns",
    [
        (
            "Tag1",
            True,
            ["Tag1 Text1", "Text2 Tag1", "Tag1"],
            ["Tag1 Text1", "Tag1 Text2", "Tag1"],
        ),
        (
            "Tag1",
            False,
            ["Tag1 Text1", "Tag2 Text2", "Tag1"],
            ["Text1 Tag1", "Tag2 Text2", "Tag1"],
        ),
    ],
)
def test_rearrange_column_tag(tag, prefixed, columns, rearranged_columns):
    df = msreport.reader._rearrange_column_tag(
        pd.DataFrame(columns=columns), tag, prefixed
    )
    assert df.columns.tolist() == rearranged_columns


def test_find_remaining_substrings():
    strings_list = [
        ["Test Sub1", "Test Sub2", "Test Sub3", "Test", "Test Sub3"],
        ["Sub1 Test", "Sub2 Test", "Sub3 Test", "Test"],
    ]
    split_with = "Test"
    for strings in strings_list:
        substrings = msreport.reader._find_remaining_substrings(strings, split_with)
        assert len(substrings) == 3
        assert substrings == ["Sub1", "Sub2", "Sub3"]


class TestSortLeadingProteins:
    @pytest.fixture(autouse=True)
    def _init_input_table(self):
        self.table_columns = [
            "Leading proteins",
            "Leading potential contaminants",
            "Representative protein",
            "Potential contaminant",
            "Leading proteins database origin",
        ]
        self.input_table = pd.DataFrame(
            data=[
                ["C;A_cont;B", "False;True;False", "C", False, "sp;tr;sp"],
                ["A;C;B", "False;False;False", "A", False, "tr;xx;sp"],
            ],
            columns=self.table_columns,
        )
        self.expected_table_columns = [
            "Leading proteins",
            "Leading potential contaminants",
            "Representative protein",
            "Potential contaminant",
        ]

    def test_alhpanumeric_sorting(self):
        expected = pd.DataFrame(
            data=[
                ["A_cont;B;C", "True;False;False", "A_cont", True],
                ["A;B;C", "False;False;False", "A", False],
            ],
            columns=self.expected_table_columns,
        )

        sorted_table = msreport.reader.sort_leading_proteins(
            self.input_table, alphanumeric=True, penalize_contaminants=False
        )
        for column in expected.columns:
            assert sorted_table[column].tolist() == expected[column].tolist()

    def test_penalize_contaminant_sorting(self):
        expected = pd.DataFrame(
            data=[
                ["C;B;A_cont", "False;False;True", "C", False],
                ["A;C;B", "False;False;False", "A", False],
            ],
            columns=self.expected_table_columns,
        )

        sorted_table = msreport.reader.sort_leading_proteins(
            self.input_table, alphanumeric=False, penalize_contaminants=True
        )
        for column in expected.columns:
            assert sorted_table[column].tolist() == expected[column].tolist()

    def test_special_protein_sorting(self):
        special_proteins = ["B"]
        expected = pd.DataFrame(
            data=[
                ["B;C;A_cont", "False;False;True", "B", False],
                ["B;A;C", "False;False;False", "B", False],
            ],
            columns=self.expected_table_columns,
        )

        sorted_table = msreport.reader.sort_leading_proteins(
            self.input_table,
            alphanumeric=False,
            penalize_contaminants=True,
            special_proteins=special_proteins,
        )
        for column in expected.columns:
            assert sorted_table[column].tolist() == expected[column].tolist()

    def test_database_order_sorting(self):
        database_order = ["sp", "tr"]
        expected = pd.DataFrame(
            data=[
                ["C;B;A_cont", "False;False;True", "C", False],
                ["B;A;C", "False;False;False", "B", False],
            ],
            columns=self.expected_table_columns,
        )

        sorted_table = msreport.reader.sort_leading_proteins(
            self.input_table,
            alphanumeric=False,
            penalize_contaminants=False,
            database_order=database_order,
        )
        for column in expected.columns:
            assert sorted_table[column].tolist() == expected[column].tolist()

    def test_multiple_sorting_options(self):
        special_proteins = ["C"]
        expected = pd.DataFrame(
            data=[
                ["C;B;A_cont", "False;False;True", "C", False],
                ["C;A;B", "False;False;False", "C", False],
            ],
            columns=self.expected_table_columns,
        )

        sorted_table = msreport.reader.sort_leading_proteins(
            self.input_table,
            alphanumeric=False,
            penalize_contaminants=True,
            special_proteins=special_proteins,
        )
        for column in expected.columns:
            assert sorted_table[column].tolist() == expected[column].tolist()


def test_process_protein_entries():
    contaminant_tag = "CON__"
    leading_protein_entries = [
        ["A"],
        ["B", "C"],
        ["CON__x|D|x", "E"],
        ["x|F|x", "x|CON__G|x"],
    ]
    expected = {
        "Representative protein": ["A", "B", "D", "F"],
        "Protein reported by software": ["A", "B", "D", "F"],
        "Potential contaminant": [False, False, True, False],
        "Leading proteins": ["A", "B;C", "D;E", "F;CON__G"],
        "Leading potential contaminants": [
            "False",
            "False;False",
            "True;False",
            "False;True",
        ],
    }

    table = msreport.reader._process_protein_entries(
        leading_protein_entries, contaminant_tag
    )
    for column, expected_values in expected.items():
        assert table[column].tolist() == expected_values


def test_add_protein_modifications():
    table = pd.DataFrame(
        {
            "Modifications": ["0:ac", "4:ph", "2:ph;7:ox"],
            "Start position": [1, 10, 50],
        }
    )
    expected_protein_sites = ["0:ac", "13:ph", "51:ph;56:ox"]
    msreport.reader.add_protein_modifications(table)
    assert table["Protein modifications"].to_list() == expected_protein_sites


class TestAddIbaqIntensities:
    @pytest.fixture(autouse=True)
    def _init_qtable(self):
        self.table = pd.DataFrame(
            {
                "ibaq_petides": [2, 4],
                "intensity": [100.0, 200.0],
            }
        )

    def test_ibaq_intensity_added(self):
        msreport.reader.add_ibaq_intensities(
            self.table,
            ibaq_peptide_column="ibaq_petides",
            intensity_tag="intensity",
            ibaq_tag="ibaq",
        )
        assert "ibaq" in self.table.columns

    @pytest.mark.parametrize(
        "expected_ibaq, normalize_intensity",
        [
            ([50, 50], False),
            ([150, 150], True),
        ],
    )
    def test_correct_ibaq_intensities(self, expected_ibaq, normalize_intensity):
        msreport.reader.add_ibaq_intensities(
            self.table,
            normalize=normalize_intensity,
            ibaq_peptide_column="ibaq_petides",
            intensity_tag="intensity",
            ibaq_tag="ibaq",
        )
        assert np.all(np.equal(self.table["ibaq"], expected_ibaq))


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


class TestMQReader:
    @pytest.fixture(autouse=True)
    def _init_reader(self):
        self.reader = msreport.reader.MQReader(
            "./tests/testdata/maxquant",
            contaminant_tag="CON__",
        )

    def test_testdata_setup(self):
        assert os.path.isdir(self.reader.data_directory)

    def test_drop_decoy(self):
        table = self.reader._read_file("proteins")
        table = self.reader._drop_decoy(table)
        protein_entries = table["Majority protein IDs"].str.split(";", expand=True)[0]
        is_decoy = protein_entries.str.contains("REV__")
        assert not is_decoy.any()

    def test_drop_idbysite(self):
        table = self.reader._read_file("proteins")
        ids_by_site = table["Majority protein IDs"][
            table["Only identified by site"] == "+"
        ]

        table = self.reader._drop_idbysite(table)
        assert not ids_by_site.isin(table["Majority protein IDs"]).any()

    def test_collect_leading_protein_entries(self):
        table = pd.DataFrame(
            {
                "Majority protein IDs": ["B;A;C", "D", "E;F", "G;H;I"],
                "Peptide counts (all)": ["5;5;3", "3", "6;3", "6;6;6"],
            }
        )
        expected = [["B", "A"], ["D"], ["E"], ["G", "H", "I"]]
        leading_proteins = self.reader._collect_leading_protein_entries(table)
        assert leading_proteins == expected

    def test_integration_import_proteins(self):
        table = self.reader.import_proteins(
            rename_columns=True,
            prefix_column_tags=False,
            drop_decoy=True,
            drop_idbysite=True,
            drop_protein_info=True,
        )
        assert table["Potential contaminant"].dtype == bool
        assert "Total peptides" in table
        assert "SampleA_1 Intensity" in table
        assert not table.columns.isin(["Reverse", "Only identified by site"]).any()
        assert not table["Representative protein"].str.contains("REV__").any()
        assert "Sequence length" not in table.columns
        assert not table.columns.str.contains("iBAQ").any()
        assert not table.columns.str.contains("site positions").any()

    def test_integration_import_peptides(self):
        table = self.reader.import_peptides(
            rename_columns=True,
            prefix_column_tags=False,
            drop_decoy=True,
        )
        assert "Protein reported by software" in table
        assert "Representative protein" in table
        assert "Peptide sequence" in table
        assert "SampleA_1 Intensity" in table
        assert not table["Leading razor protein"].str.contains("REV__").any()

    def test_integration_import_ions(self):
        table = self.reader.import_ions(
            rename_columns=True,
            rewrite_modifications=True,
            drop_decoy=True,
        )
        assert "Protein reported by software" in table
        assert "Representative protein" in table
        assert "Peptide sequence" in table
        assert "Modified sequence" in table
        assert "Modifications" in table
        assert table["Peptide sequence"][4] == "AAGPISER"
        assert table["Modified sequence"][4] == "[Acetyl (Protein N-term)]AAGPISER"
        assert table["Modifications"][4] == "0:Acetyl (Protein N-term)"
        assert not table["Leading razor protein"].str.contains("REV__").any()


class TestFPReader:
    @pytest.fixture(autouse=True)
    def _init_reader(self):
        self.reader = msreport.reader.FPReader(
            "./tests/testdata/fragpipe",
            contaminant_tag="contam_",
        )

    def test_testdata_setup(self):
        assert os.path.isdir(self.reader.data_directory)

    def test_collect_leading_protein_entries(self):
        table = pd.DataFrame(
            {
                "Protein": ["x|B|b", "x|D|d", "x|E|e", "x|G|g"],
                "Indistinguishable Proteins": ["x|A|a", "", "", "x|H|h, x|I|i"],
            }
        )
        expected = [
            ["x|B|b", "x|A|a"],
            ["x|D|d"],
            ["x|E|e"],
            ["x|G|g", "x|H|h", "x|I|i"],
        ]
        leading_proteins = self.reader._collect_leading_protein_entries(table)
        assert leading_proteins == expected

    def test_add_protein_entries(self):
        table = pd.DataFrame(
            {
                "Protein": ["x|B|b", "x|D|d", "x|E|e", "x|G|g"],
                "Indistinguishable Proteins": ["x|A|a", "", "", "x|H|h, x|I|i"],
            }
        )
        leading_proteins = ["B;A", "D", "E", "G;H;I"]
        representative_protein = ["B", "D", "E", "G"]
        protein_reported_by_software = representative_protein

        table = self.reader._add_protein_entries(table)
        assert table["Leading proteins"].tolist() == leading_proteins
        assert table["Representative protein"].tolist() == representative_protein
        assert (
            table["Protein reported by software"].tolist()
            == protein_reported_by_software
        )

    def test_integration_import_proteins(self):
        table = self.reader.import_proteins(
            rename_columns=True,
            prefix_column_tags=True,
            drop_protein_info=True,
        )
        assert "Representative protein" in table
        assert table["Potential contaminant"].dtype == bool
        assert "Total peptides" in table
        assert "Intensity SampleA_1" in table
        assert "Protein Length" not in table.columns

    def test_integration_import_peptides(self):
        table = self.reader.import_peptides(
            rename_columns=True,
            prefix_column_tags=True,
        )
        assert "Protein reported by software" in table
        assert "Representative protein" in table
        assert "Peptide sequence" in table
        assert "Start position" in table
        assert "Intensity SampleA_1" in table

    def test_integration_import_ions(self):
        table = self.reader.import_ions(
            rename_columns=True,
            rewrite_modifications=True,
            prefix_column_tags=True,
        )
        assert "Protein reported by software" in table
        assert "Representative protein" in table
        assert "Start position" in table
        assert "Peptide sequence" in table
        assert "Modified sequence" in table
        assert "Modifications" in table
        assert table["Peptide sequence"][0] == "AACDLVR"
        assert table["Modified sequence"][0] == "AAC[57.0214]DLVR"
        assert table["Modifications"][0] == "3:57.0214"
        assert "Intensity SampleA_1" in table


class TestGetAnnotationFunctions:
    @pytest.fixture(autouse=True)
    def _init_protein_db(self):
        self.protein_id = "P60709"
        self.sequence = "MDDDIAALVVDNGSGMCKAGFAGDDAPRAVFPSIVGRPRHQGVMVGMGQK"
        self.tryptic_ibaq_peptides = 4
        self.sequence_length = len(self.sequence)
        self.fasta_header = "sp|P60709|ACTB_HUMAN Actin, cytoplasmic 1 OS=Homo sapiens OX=9606 GN=ACTB PE=1 SV=1"
        self.gene_name = "ACTB"
        self.entry_name = "ACTB_HUMAN"
        self.db_origin = "sp"
        self.default_value = "Default"

        self.protein_db = msreport.helper.ProteinDatabase()
        self.protein_db.proteins[self.protein_id] = Protein(
            sequence=self.sequence,
            header=self.fasta_header,
            info={
                "db": self.db_origin,
                "entry": self.entry_name,
                "gene_id": self.gene_name,
                "id": self.protein_id,
                "name": "Actin, cytoplasmic 1",
                "taxon": "HUMAN",
                "OS": "Homo sapiens",
                "OX": "9606",
                "GN": "ACTB",
                "PE": 1,
                "SV": 1,
            },
        )

    # fmt: off
    def test_get_annotation_sequence_length(self):
        assert self.sequence_length == msreport.reader._get_annotation_sequence_length(
            self.protein_id, self.protein_db, default_value=None
        )
        assert self.default_value == msreport.reader._get_annotation_sequence_length(
            "Absent database entry", self.protein_db, default_value=self.default_value
        )

    def test_get_annotation_fasta_header(self):
        assert self.fasta_header == msreport.reader._get_annotation_fasta_header(
            self.protein_id, self.protein_db, default_value=None
        )
        assert self.default_value == msreport.reader._get_annotation_fasta_header(
            "Absent database entry", self.protein_db, default_value=self.default_value
        )

    def test_get_annotation_gene_name(self):
        assert self.gene_name == msreport.reader._get_annotation_gene_name(
            self.protein_id, self.protein_db, default_value=None
        )
        assert self.default_value == msreport.reader._get_annotation_gene_name(
            "Absent database entry", self.protein_db, default_value=self.default_value
        )

    def test_get_annotation_protein_entry_name(self):
        assert self.entry_name == msreport.reader._get_annotation_protein_entry_name(
            self.protein_id, self.protein_db, default_value=None
        )
        assert self.default_value == msreport.reader._get_annotation_protein_entry_name(
            "Absent database entry", self.protein_db, default_value=self.default_value
        )

    def test_get_annotation_db_origin(self):
        assert self.db_origin == msreport.reader._get_annotation_db_origin(
            self.protein_id, self.protein_db, default_value=None
        )
        assert self.default_value == msreport.reader._get_annotation_db_origin(
            "Absent database entry", self.protein_db, default_value=self.default_value
        )

    def test_get_annotation_ibaq_peptides(self):
        assert self.tryptic_ibaq_peptides == msreport.reader._get_annotation_ibaq_peptides(
            self.protein_id, self.protein_db, default_value=None
        )
        assert self.default_value == msreport.reader._get_annotation_ibaq_peptides(
            "Absent database entry", self.protein_db, default_value=self.default_value
        )
    # fmt: on
