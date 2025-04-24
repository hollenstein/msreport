import os

import numpy as np
import pandas as pd
import pytest

import msreport.reader


@pytest.fixture(autouse=True)
def test_reader():
    reader = msreport.reader.FragPipeReader(
        "./tests/testdata/fragpipe",
        contaminant_tag="contam_",
    )
    return reader


class TestFragPipeReader:
    def test_testdata_setup(self, test_reader):
        assert os.path.isdir(test_reader.data_directory)

    def test_collect_leading_protein_entries(self, test_reader):
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
        leading_proteins = test_reader._collect_leading_protein_entries(table)
        assert leading_proteins == expected

    def test_add_protein_entries(self, test_reader):
        table = pd.DataFrame(
            {
                "Protein": ["x|B|b", "x|D|d", "x|E|e", "x|G|g"],
                "Indistinguishable Proteins": ["x|A|a", "", "", "x|H|h, x|I|i"],
            }
        )
        leading_proteins = ["B;A", "D", "E", "G;H;I"]
        representative_protein = ["B", "D", "E", "G"]
        protein_reported_by_software = representative_protein

        table = test_reader._add_protein_entries(table)
        assert table["Leading proteins"].tolist() == leading_proteins
        assert table["Representative protein"].tolist() == representative_protein
        assert (
            table["Protein reported by software"].tolist()
            == protein_reported_by_software
        )

    def test_integration_import_proteins(self, test_reader):
        table = test_reader.import_proteins(
            rename_columns=True,
            prefix_column_tags=True,
            drop_protein_info=True,
        )
        assert "Representative protein" in table
        assert table["Potential contaminant"].dtype == bool
        assert "Total peptides" in table
        assert "Intensity SampleA_1" in table
        assert "Protein Length" not in table.columns

    def test_integration_import_peptides(self, test_reader):
        table = test_reader.import_peptides(
            rename_columns=True,
            prefix_column_tags=True,
        )
        assert "Protein reported by software" in table
        assert "Representative protein" in table
        assert "Peptide sequence" in table
        assert "Start position" in table
        assert "Intensity SampleA_1" in table

    def test_integration_import_ions(self, test_reader):
        table = test_reader.import_ions(
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


class TestImportIonEvidence:
    def test_correct_columns_after_renaming(self, test_reader):
        table = test_reader.import_ion_evidence(
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
        assert "Intensity" in table

    def test_tables_from_different_samples_are_different(self, test_reader):
        table = test_reader.import_ion_evidence()
        table_1 = table[table["Sample"] == "SampleA_1"]
        table_2 = table[table["Sample"] == "SampleB_1"]
        assert len(table_1) != len(table_2)

    def test_sample_column_filled_with_ion_table_folder(self, test_reader):
        table = test_reader.import_ion_evidence()
        np.testing.assert_array_equal(
            table["Sample"].unique(), ["SampleA_1", "SampleB_1"]
        )

    def test_concatenated_table_is_reindexed(self, test_reader):
        table = test_reader.import_ion_evidence()
        assert table.index.nunique() == len(table)


class TestImportPsmEvidence:
    def test_correct_columns_after_renaming(self, test_reader):
        table = test_reader.import_psm_evidence(
            rename_columns=True,
            rewrite_modifications=True,
        )
        assert "Protein reported by software" in table
        assert "Representative protein" in table
        assert "Mapped proteins" in table
        assert "Probability" in table
        assert "Start position" in table
        assert "End position" in table
        assert "Peptide sequence" in table
        assert "Modified sequence" in table
        assert "Modifications" in table
        assert "Missed cleavage" in table
        assert "Intensity" in table

    def test_tables_from_different_samples_are_different(self, test_reader):
        table = test_reader.import_psm_evidence()
        table_1 = table[table["Sample"] == "SampleA_1"]
        table_2 = table[table["Sample"] == "SampleB_1"]
        assert len(table_1) != len(table_2)

    def test_sample_column_filled_with_ion_table_folder(self, test_reader):
        table = test_reader.import_psm_evidence()
        np.testing.assert_array_equal(
            table["Sample"].unique(), ["SampleA_1", "SampleB_1"]
        )

    def test_concatenated_table_is_reindexed(self, test_reader):
        table = test_reader.import_psm_evidence()
        assert table.index.nunique() == len(table)


class TestExtractFragpipeLocalizationProbabilities:
    def test_extract_single_modification_with_merged_amino_acid_entries(self):
        # Test case for FragPipe before version 22.0
        localization = msreport.reader.extract_fragpipe_localization_probabilities(
            "STY:79.9663@FIMS(0.334)PT(0.666)LK;"
        )
        expected = {"79.9663": {4: 0.334, 6: 0.666}}
        assert localization == expected

    def test_extract_single_modifications_with_split_amino_aicd_entries(self):
        # Test case for FragPipe version 22.0
        localization = msreport.reader.extract_fragpipe_localization_probabilities(
            "S:79.9663@FIMS(0.334)PTLK;T:79.9663@FIMSPT(0.666)LK;"
        )
        expected = {"79.9663": {4: 0.334, 6: 0.666}}
        assert localization == expected

    def test_extract_multiple_modifications(self):
        localization = msreport.reader.extract_fragpipe_localization_probabilities(
            "M:15.9949@FIM(1.000)SPTLK;S:79.9663@FIMS(0.334)PTLK"
        )
        expected = {"15.9949": {3: 1.0}, "79.9663": {4: 0.334}}
        assert localization == expected

    def test_empty_localization_string_returns_empty_dict(self):
        localization = msreport.reader.extract_fragpipe_localization_probabilities("")
        expected = {}
        assert localization == expected


@pytest.mark.parametrize(
    "modifications_entry, sequence, expected_modifications",
    [
        ("8C(5)", "AAAAAAAC", [(8, "5")]),
        ("3C(A),4C(5)", "AAAAAAAC", [(3, "A"), (4, "5")]),
        ("N-term(5),3A(M)", "AAAAAAAC", [(0, "5"), (3, "M")]),
        ("C-term(M)", "AAC", [(3, "M")]),
        ("C-term(M)", "AACAAC", [(6, "M")]),
        ("", "AAAAAAAC", []),
    ],
)
def test_extract_fragpipe_assigned_modifications(
    modifications_entry, sequence, expected_modifications
):
    extracted_modifications = msreport.reader._extract_fragpipe_assigned_modifications(
        modifications_entry, sequence
    )  # fmt: skip

    assert extracted_modifications == expected_modifications


def test_generate_modification_entries_from_assigned_modifications():
    sequences = ["STVHEILCK", "ATHGQTCAR"]
    assigned_modifications = ["N-term(42.0106),8C(57.0215)", "7C(57.0215)"]  # fmt:skip
    expected_result = {
        "Modified sequence": ["[42.0106]STVHEILC[57.0215]K", "ATHGQTC[57.0215]AR"],
        "Modifications": ["0:42.0106;8:57.0215", "7:57.0215"],
    }
    observed_result = msreport.reader._generate_modification_entries_from_assigned_modifications(
        sequences, assigned_modifications
    )  # fmt:skip
    assert observed_result == expected_result
