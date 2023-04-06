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

    def test_sample_column_filled_with_ion_table_folder(self, test_reader):
        table = test_reader.import_ion_evidence()
        np.testing.assert_array_equal(
            table["Sample"].unique(), ["SampleA_1", "SampleB_1"]
        )

    def test_concatenated_table_is_reindexed(self, test_reader):
        table = test_reader.import_ion_evidence()
        assert table.index.nunique() == len(table)
