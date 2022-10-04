import os

import numpy as np
import pandas as pd
import pytest

import msreport.reader


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


@pytest.mark.parametrize(
    "input, expected_fastas, expected_proteins, expected_names",
    [
        (["x|A|a"], ["x|A|a"], ["A"], ["a"]),
        (["x|A|a something else"], ["x|A|a something else"], ["A"], ["a"]),
        (["x|A|a", "x|B|b"], ["x|A|a", "x|B|b"], ["A", "B"], ["a", "b"]),
        (["x|B|b", "x|A|a"], ["x|A|a", "x|B|b"], ["A", "B"], ["a", "b"]),
    ],
)
def test_sort_fasta_entries_(input, expected_fastas, expected_proteins, expected_names):
    fastas, proteins, names = msreport.reader._sort_fasta_entries(input)
    assert fastas == expected_fastas
    assert proteins == expected_proteins
    assert names == expected_names


def test_sort_fasta_entries_with_sorting_by_tag():
    fasta_headers = ["x|B|b", "x|Apost|a_post", "x|preA|pre_a"]
    sorting_tags = {"pre": -1, "post": 1}
    fastas, proteins, names = msreport.reader._sort_fasta_entries(
        fasta_headers, sorting_tags
    )
    assert fastas == ["x|preA|pre_a", "x|B|b", "x|Apost|a_post"]
    assert proteins == ["preA", "B", "Apost"]
    assert names == ["pre_a", "b", "a_post"]


# fmt: off
@pytest.mark.parametrize(
    "proteins, contaminants, special_proteins, expected_proteins, expected_contaminants",
    [
        (["C", "B", "A"], None, None, ["A", "B", "C"], [None, None, None]),
        (["C", "B", "A"], None, "C", ["C", "A", "B"], [None, None, None]),
        (["C", "B", "A"], [False, False, True], None, ["B", "C", "A"], [False, False, True]),
        (["C", "B", "A"], [False, False, True], ["C"], ["C", "B", "A"], [False, False, True]),
        (["C", "B", "A"], [False, True, False], ["B"], ["B", "A", "C"], [True, False, False]),
    ],
)
def test_sort_order_proteins(
    proteins, contaminants, special_proteins, expected_proteins, expected_contaminants
):
    proteins = ["C", "B", "A"]
    sorted_proteins, sorted_contaminants = msreport.reader._sort_proteins_and_contaminants(
        proteins, special_proteins=special_proteins, contaminants=contaminants
    )
    # sorted_proteins = [proteins[i] for i in sort_order]
    assert sorted_contaminants == expected_contaminants
    assert sorted_proteins == expected_proteins
# fmt: on


class TestSortLeadingProteins:
    def test_without_args(self):
        df = pd.DataFrame(
            {
                "Leading proteins": ["B;A", "D", "E;F", "G;I;H"],
            }
        )
        leading_proteins = ["A;B", "D", "E;F", "G;H;I"]
        representative_protein = ["A", "D", "E", "G"]

        df = msreport.reader._sort_leading_proteins(df)
        assert df["Leading proteins"].tolist() == leading_proteins
        assert df["Representative protein"].tolist() == representative_protein

    def test_with_contamination_tag(self):
        df = pd.DataFrame(
            {
                "Leading proteins": ["B;A", "D", "E;F", "Gtag;I;H"],
            }
        )
        leading_proteins = ["A;B", "D", "E;F", "H;I;Gtag"]
        representative_protein = ["A", "D", "E", "H"]

        df = msreport.reader._sort_leading_proteins(df, contaminant_tag="tag")
        assert df["Leading proteins"].tolist() == leading_proteins
        assert df["Representative protein"].tolist() == representative_protein

    def test_with_special_proteins(self):
        df = pd.DataFrame(
            {
                "Leading proteins": ["B;A", "D", "E;F", "G;I;H"],
            }
        )
        leading_proteins = ["A;B", "D", "F;E", "H;I;G"]
        representative_protein = ["A", "D", "F", "H"]

        df = msreport.reader._sort_leading_proteins(
            df, special_proteins=["F", "I", "H"]
        )
        assert df["Leading proteins"].tolist() == leading_proteins
        assert df["Representative protein"].tolist() == representative_protein


class TestProcessProteinEntries:
    @pytest.fixture(autouse=True)
    def _init_dataframe(self):
        self.leading_protein_entries = [
            ["B", "A"],
            ["D"],
            ["E"],
            ["G", "H", "I"],
            ["CON__x|J|x", "J"],
            ["CON__x|K|x"],
        ]

    def test_without_sorting(self):
        contaminant_tag = "CON__"
        sort_proteins = False
        table = msreport.reader._process_protein_entries(
            self.leading_protein_entries,
            contaminant_tag,
            sort_proteins,
        )

        leading_proteins = ["B;A", "D", "E", "G;H;I", "J;J", "K"]
        representative_protein = ["B", "D", "E", "G", "J", "K"]
        protein_reported_by_software = representative_protein
        is_contaminant = [False, False, False, False, True, True]

        assert table["Leading proteins"].tolist() == leading_proteins
        assert table["Representative protein"].tolist() == representative_protein
        assert (
            table["Protein reported by software"].tolist()
            == protein_reported_by_software
        )
        assert table["Potential contaminant"].tolist() == is_contaminant

    def test_with_sorting(self):
        contaminant_tag = "CON__"
        sort_proteins = True
        special_proteins = ["H"]
        table = msreport.reader._process_protein_entries(
            self.leading_protein_entries,
            contaminant_tag,
            sort_proteins,
            special_proteins,
        )
        leading_proteins = ["A;B", "D", "E", "H;G;I", "J;J", "K"]
        representative_protein = ["A", "D", "E", "H", "J", "K"]
        protein_reported_by_software = ["B", "D", "E", "G", "J", "K"]
        is_contaminant = [False, False, False, False, False, True]

        assert table["Leading proteins"].tolist() == leading_proteins
        assert table["Representative protein"].tolist() == representative_protein
        assert (
            table["Protein reported by software"].tolist()
            == protein_reported_by_software
        )
        assert table["Potential contaminant"].tolist() == is_contaminant


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
            sort_proteins=True,
            drop_protein_info=True,
            special_proteins=["Q13838"],
        )
        assert table["Potential contaminant"].dtype == bool
        assert "Total peptides" in table
        assert "SampleA_1 Intensity" in table
        assert not table.columns.isin(["Reverse", "Only identified by site"]).any()
        assert not table["Representative protein"].str.contains("REV__").any()
        assert "Sequence length" not in table.columns
        assert not table.columns.str.contains("iBAQ").any()
        assert not table.columns.str.contains("site positions").any()
        assert not table["Representative protein"].str.contains("CON__P12763").any()
        assert table["Representative protein"].str.contains("Q13838").any()

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

    def test_add_protein_entries_without_sorting(self):
        table = pd.DataFrame(
            {
                "Protein": ["x|B|b", "x|D|d", "x|E|e", "x|G|g"],
                "Indistinguishable Proteins": ["x|A|a", "", "", "x|H|h, x|I|i"],
            }
        )
        leading_proteins = ["B;A", "D", "E", "G;H;I"]
        representative_protein = ["B", "D", "E", "G"]
        protein_reported_by_software = representative_protein

        table = self.reader._add_protein_entries(table, False)
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
            sort_proteins=True,
            drop_protein_info=True,
            special_proteins=["Q13838"],
        )
        assert "Representative protein" in table
        assert table["Potential contaminant"].dtype == bool
        assert "Total peptides" in table
        assert "Intensity SampleA_1" in table
        assert "Protein Length" not in table.columns
        assert table["Representative protein"].str.contains("Q13838").any()

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
