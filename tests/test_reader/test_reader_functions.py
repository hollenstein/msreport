import numpy as np
import pandas as pd
import pytest

import msreport.reader


@pytest.fixture
def example_protein_db():
    class Protein:
        def __init__(self):
            self.header = "sp|P60709|ACTB_HUMAN Actin, cytoplasmic 1 OS=Homo sapiens OX=9606 GN=ACTB PE=1 SV=1"
            self.sequence = "MDDDIAALVVDNGSGMCKAGFAGDDAPRAVFPSIVGRPRHQGVMVGMGQK"
            self.header_fields = {
                "gene_name": "ACTB",
                "protein_name": "Actin, cytoplasmic 1",
                "entry_name": "ACTB_HUMAN",
                "db": "sp",
            }

    class ProteinDatabase:
        def __init__(self):
            self.proteins = {"P60709": Protein()}

        def __getitem__(self, protein_id: str):
            return self.proteins[protein_id]

        def __contains__(self, protein_id: str):
            return True if protein_id in self.proteins else False

    return ProteinDatabase()


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
            "Leading proteins database origin",
        ]

    def observed_protein_order_after_sorting(self, sorted_table) -> list[list[int]]:
        """Returns the observed leading protein sorting order in sorted_table."""
        observed_orders = []
        for entry_before, entry_after in zip(
            self.input_table["Leading proteins"], sorted_table["Leading proteins"]
        ):
            proteins_before = entry_before.split(";")
            proteins_after = entry_after.split(";")
            observed_orders.append(
                [proteins_before.index(protein) for protein in proteins_after]
            )
        return observed_orders

    def test_alhpanumeric_sorting(self):
        sorted_table = msreport.reader.sort_leading_proteins(
            self.input_table, alphanumeric=True, penalize_contaminants=False
        )
        expected_protein_order = [[1, 2, 0], [0, 2, 1]]

        observed_protein_order = self.observed_protein_order_after_sorting(sorted_table)
        assert observed_protein_order == expected_protein_order

    def test_penalize_contaminant_sorting(self):
        sorted_table = msreport.reader.sort_leading_proteins(
            self.input_table, alphanumeric=False, penalize_contaminants=True
        )
        expected_protein_order = [[0, 2, 1], [0, 1, 2]]

        observed_protein_order = self.observed_protein_order_after_sorting(sorted_table)
        assert observed_protein_order == expected_protein_order

    def test_special_protein_sorting(self):
        special_proteins = ["B"]
        sorted_table = msreport.reader.sort_leading_proteins(
            self.input_table,
            alphanumeric=False,
            penalize_contaminants=True,
            special_proteins=special_proteins,
        )
        expected_protein_order = [[2, 0, 1], [2, 0, 1]]

        observed_protein_order = self.observed_protein_order_after_sorting(sorted_table)
        assert observed_protein_order == expected_protein_order

    def test_database_order_sorting(self):
        database_order = ["sp", "tr"]
        sorted_table = msreport.reader.sort_leading_proteins(
            self.input_table,
            alphanumeric=False,
            penalize_contaminants=False,
            database_order=database_order,
        )
        expected_protein_order = [[0, 2, 1], [2, 0, 1]]

        observed_protein_order = self.observed_protein_order_after_sorting(sorted_table)
        assert observed_protein_order == expected_protein_order

    def test_all_leading_columns_are_sorted(self):
        expected = pd.DataFrame(
            data=[
                ["A_cont;B;C", "True;False;False", "A_cont", True, "tr;sp;sp"],
                ["A;B;C", "False;False;False", "A", False, "tr;sp;xx"],
            ],
            columns=[
                "Leading proteins",
                "Leading potential contaminants",
                "Representative protein",
                "Potential contaminant",
                "Leading proteins database origin",
            ],
        )

        sorted_table = msreport.reader.sort_leading_proteins(
            self.input_table,
            alphanumeric=True,
            penalize_contaminants=False,
        )
        for column in expected.columns:
            assert sorted_table[column].tolist() == expected[column].tolist()

    def test_absent_columns_are_ignored(self):
        input_table = pd.DataFrame(
            data=[["C;B;A_cont", "C"], ["A;C;B", "A"]],
            columns=["Leading proteins", "Representative protein"],
        )
        expected = pd.DataFrame(
            data=[["A_cont;B;C", "A_cont"], ["A;B;C", "A"]],
            columns=["Leading proteins", "Representative protein"],
        )
        sorted_table = msreport.reader.sort_leading_proteins(
            input_table, alphanumeric=True, penalize_contaminants=False
        )
        for column in expected:
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


class TestAddSequenceCoverage:
    def test_calculate_correct_coverage(self):
        peptide_table = pd.DataFrame(
            {"Start position": [1, 1], "End position": [10, 10], "ID": ["A", "B"]}
        )
        protein_table = pd.DataFrame({"ID": ["A", "B"], "Protein length": [10, 20]})
        expected_coverage = np.array([100, 50])

        msreport.reader.add_sequence_coverage(
            protein_table, peptide_table, id_column="ID"
        )
        observed_coverage = protein_table["Sequence coverage"].to_numpy()
        assert np.array_equal(observed_coverage, expected_coverage, equal_nan=True)

    def test_absent_peptide_entries_returns_nan_coverage(self):
        peptide_table = pd.DataFrame(
            {"Start position": [1], "End position": [10], "ID": ["A"]}
        )
        protein_table = pd.DataFrame({"ID": ["A", "B"], "Protein length": [10, 20]})
        expected_coverage = np.array([100, np.nan])

        msreport.reader.add_sequence_coverage(
            protein_table, peptide_table, id_column="ID"
        )
        observed_coverage = protein_table["Sequence coverage"].to_numpy()
        assert np.array_equal(observed_coverage, expected_coverage, equal_nan=True)

    def test_negative_protein_length_returns_nan_coverage(self):
        peptide_table = pd.DataFrame(
            {"Start position": [1], "End position": [10], "ID": ["A"]}
        )
        protein_table = pd.DataFrame({"ID": ["A"], "Protein length": [-1]})
        expected_coverage = np.array([np.nan])

        msreport.reader.add_sequence_coverage(
            protein_table, peptide_table, id_column="ID"
        )
        observed_coverage = protein_table["Sequence coverage"].to_numpy()
        assert np.array_equal(observed_coverage, expected_coverage, equal_nan=True)


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
    def test_correct_calculation_of_ibaq_intensities(
        self, expected_ibaq, normalize_intensity
    ):
        msreport.reader.add_ibaq_intensities(
            self.table,
            normalize=normalize_intensity,
            ibaq_peptide_column="ibaq_petides",
            intensity_tag="intensity",
            ibaq_tag="ibaq",
        )
        np.testing.assert_array_equal(self.table["ibaq"], np.array(expected_ibaq))

    @pytest.mark.parametrize(
        "expected_ibaq, normalize_intensity",
        [
            ([50, 50, np.nan, np.nan], False),
            ([150, 150, np.nan, np.nan], True),
        ],
    )
    def test_calculation_with_invalid_ibaq_peptides(
        self, expected_ibaq, normalize_intensity
    ):
        table = pd.DataFrame(
            {
                "ibaq_petides": [2, 4, 0, -1],
                "intensity": [100.0, 200.0, 50.0, 50.0],
            }
        )
        msreport.reader.add_ibaq_intensities(
            table,
            normalize=normalize_intensity,
            ibaq_peptide_column="ibaq_petides",
            intensity_tag="intensity",
            ibaq_tag="ibaq",
        )
        np.testing.assert_array_equal(table["ibaq"], np.array(expected_ibaq))


def test_create_protein_annotations_from_db():
    default_value = ""
    protein_db = {
        "entry 1": "value 1",
        "entry 2": "value 2",
    }
    protein_ids = ["entry 1", "entry 2", "missing entry"]
    expected_annotation = ["value 1", "value 2", default_value]

    annotation = msreport.reader._create_protein_annotations_from_db(
        protein_ids, protein_db, lambda return_value, _: return_value, default_value
    )
    assert annotation == expected_annotation


def test_create_multi_protein_annotations_from_db():
    protein_db = {
        "entry 1": "value 1",
        "entry 2": "value 2",
        "entry 3": "value 3",
        "entry 4": "value 4",
    }
    protein_entries = ["entry 1;entry 2", "entry 3", "entry 4;missing entry"]
    expected_annotation = ["value 1;value 2", "value 3", "value 4;"]

    annotation = msreport.reader._create_multi_protein_annotations_from_db(
        protein_entries, protein_db, lambda return_value, _: return_value
    )
    assert annotation == expected_annotation


class TestGetAnnotationFunctions:
    @pytest.fixture(autouse=True)
    def _init_protein_db(self, example_protein_db):
        self.protein_entry = example_protein_db["P60709"]
        self.tryptic_ibaq_peptides = 4
        self.molecular_weight_kda = 5.14
        self.sequence_length = len(self.protein_entry.sequence)
        self.fasta_header = self.protein_entry.header
        self.gene_name = self.protein_entry.header_fields["gene_name"]
        self.protein_name = self.protein_entry.header_fields["protein_name"]
        self.entry_name = self.protein_entry.header_fields["entry_name"]
        self.db_origin = self.protein_entry.header_fields["db"]
        self.default_value = "Default"

    # fmt: off
    def test_get_annotation_sequence_length(self):
        assert self.sequence_length == msreport.reader._get_annotation_sequence_length(
            self.protein_entry, default_value=None
        )

    def test_get_annotation_molecular_weight(self):
        assert self.molecular_weight_kda == msreport.reader._get_annotation_molecular_weight(
            self.protein_entry, default_value=None
        )

    def test_get_annotation_fasta_header(self):
        assert self.fasta_header == msreport.reader._get_annotation_fasta_header(
            self.protein_entry, default_value=None
        )

    def test_get_annotation_gene_name(self):
        assert self.gene_name == msreport.reader._get_annotation_gene_name(
            self.protein_entry, default_value=None
        )

    def test_get_annotation_protein_name(self):
        assert self.protein_name == msreport.reader._get_annotation_protein_name(
            self.protein_entry, default_value=None
        )

    def test_get_annotation_protein_entry_name(self):
        assert self.entry_name == msreport.reader._get_annotation_protein_entry_name(
            self.protein_entry, default_value=None
        )

    def test_get_annotation_db_origin(self):
        assert self.db_origin == msreport.reader._get_annotation_db_origin(
            self.protein_entry, default_value=None
        )

    def test_get_annotation_ibaq_peptides(self):
        assert self.tryptic_ibaq_peptides == msreport.reader._get_annotation_ibaq_peptides(
            self.protein_entry, default_value=None
        )
    # fmt: on


class TestAddProteinAnnotation:
    def test_add_protein_annotation(self, example_protein_db):
        expected_columns = [
            "Gene name",
            "Protein name",
            "Protein entry name",
            "Protein length",
            "Molecular weight [kDa]",
            "Fasta header",
            "Database origin",
            "iBAQ peptides",
        ]
        protein_table = pd.DataFrame({"Representative protein": ["P60709"]})
        msreport.reader.add_protein_annotation(
            protein_table,
            example_protein_db,
            id_column="Representative protein",
            gene_name=True,
            protein_name=True,
            protein_entry=True,
            protein_length=True,
            molecular_weight=True,
            fasta_header=True,
            ibaq_peptides=True,
            database_origin=True,
        )
        protein_table_columns = protein_table.columns.tolist()
        for column in expected_columns:
            assert column in protein_table_columns

    def test_proteins_not_in_db_warning(self, example_protein_db):
        with pytest.warns(msreport.errors.ProteinsNotInFastaWarning):
            protein_table = pd.DataFrame({"Representative protein": ["Missing"]})
            msreport.reader.add_protein_annotation(
                protein_table, example_protein_db, fasta_header=True
            )


class TestAddLeadingProteinsAnnotation:
    def test_add_leading_proteins_annotation(self, example_protein_db):
        expected_columns = [
            "Leading proteins database origin",
            "Leading proteins gene name",
            "Leading proteins entry name",
            "Leading proteins length",
            "Leading proteins fasta header",
            "Leading proteins iBAQ peptides",
        ]
        protein_table = pd.DataFrame(
            {"Representative protein": ["P60709"], "Leading proteins": ["P60709"]}
        )
        msreport.reader.add_leading_proteins_annotation(
            protein_table,
            example_protein_db,
            id_column="Representative protein",
            gene_name=True,
            protein_entry=True,
            protein_length=True,
            fasta_header=True,
            ibaq_peptides=True,
            database_origin=True,
        )
        for column in expected_columns:
            assert column in protein_table.columns

    def test_proteins_not_in_db_warning(self, example_protein_db):
        with pytest.warns(msreport.errors.ProteinsNotInFastaWarning):
            protein_table = pd.DataFrame(
                {
                    "Representative protein": ["P60709"],
                    "Leading proteins": ["P60709;Missing"],
                }
            )
            msreport.reader.add_leading_proteins_annotation(
                protein_table, example_protein_db, fasta_header=True
            )
