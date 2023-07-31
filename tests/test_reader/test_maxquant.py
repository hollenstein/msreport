import os

import pandas as pd
import pytest

import msreport.reader


class TestMaxQuantReader:
    @pytest.fixture(autouse=True)
    def _init_reader(self):
        self.reader = msreport.reader.MaxQuantReader(
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

    def test_integration_import_ion_evidence(self):
        table = self.reader.import_ion_evidence(
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


class TestExtractMaxquantLocalizationProbabilities:
    def test_extract_normal_entry(self):
        localization = msreport.reader.extract_maxquant_localization_probabilities(
            "IRT(0.989)AMNS(0.011)IER"
        )
        expected = {3: 0.989, 7: 0.011}
        assert localization == expected

    def test_empty_localization_string_returns_empty_dict(self):
        localization = msreport.reader.extract_maxquant_localization_probabilities("")
        expected = {}
        assert localization == expected
