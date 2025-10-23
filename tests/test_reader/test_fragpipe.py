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

    def test_init_with_sil_and_isobar_raises_value_error(self):
        with pytest.raises(ValueError):
            msreport.reader.FragPipeReader("", sil=True, isobar=True)

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

    def test_collect_leading_protein_entries_in_sil_mode(self):
        reader = msreport.reader.FragPipeReader("", sil=True)
        table = pd.DataFrame({"Protein": ["x|B|b", "x|D|d", "x|E|e", "x|G|g"]})
        expected = [["x|B|b"], ["x|D|d"], ["x|E|e"], ["x|G|g"]]
        leading_proteins = reader._collect_leading_protein_entries(table)
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

    def test_collect_mapped_proteins(self):
        reader = msreport.reader.FragPipeReader("")
        table = pd.DataFrame(
            {
                "Representative protein": ["B", "D", "E", "G"],
                "Mapped Proteins": ["A", "", "", "H;I"],
            }
        )
        expected_mapped_proteins = ["B;A", "D", "E", "G;H;I"]
        mapped_proteins = reader._collect_mapped_proteins(table)
        assert mapped_proteins == expected_mapped_proteins


class TestImportManifest:
    def test_manifest_file_processed_correctly(self, test_reader):
        table = test_reader.import_manifest()
        expected_manifest_table = pd.DataFrame(
            {
                "Sample": ["SampleA_1", "SampleB_1"],
                "Experiment": ["SampleA", "SampleB"],
                "Replicate": ["1", "1"],
                "Rawfile": [
                    "20220926_E2_RSLC2_FAIMS_PepSep_Q4L_iso_0c4_cgf_4c4_01_CV70.raw",
                    "20220928_E2_RSLC2_FAIMS_PepSep_Q4L_iso_0c4_cgf_4c4_01_CV70.raw",
                ],
            }
        )
        pd.testing.assert_frame_equal(table, expected_manifest_table)

    def test_import_manifest_with_literal_nan_values(self, tmp_path):
        manifest_content = "\t".join(["C:\\rawfile.raw", "nan", "1", "DDA"])
        expected_manifest_table = pd.DataFrame(
            {
                "Sample": ["nan_1"],
                "Experiment": ["nan"],
                "Replicate": ["1"],
                "Rawfile": ["rawfile.raw"],
            }
        )
        manifest = self._write_temp_manifest_and_import_with_fragpipereader(tmp_path, manifest_content)  # fmt: skip
        pd.testing.assert_frame_equal(manifest, expected_manifest_table)

    def test_import_manifest_with_no_experiment_values(self, tmp_path):
        manifest_content = "\t".join(["C:\\rawfile.raw", "", "1", "DDA"])
        expected_manifest_table = pd.DataFrame(
            {
                "Sample": ["exp_1"],
                "Experiment": ["exp"],
                "Replicate": ["1"],
                "Rawfile": ["rawfile.raw"],
            }
        )
        manifest = self._write_temp_manifest_and_import_with_fragpipereader(tmp_path, manifest_content)  # fmt: skip
        pd.testing.assert_frame_equal(manifest, expected_manifest_table)

    def test_import_manifest_with_no_replicate_values(self, tmp_path):
        manifest_content = "\t".join(["C:\\rawfile.raw", "Sample", "", "DDA"])
        expected_manifest_table = pd.DataFrame(
            {
                "Sample": ["Sample"],
                "Experiment": ["Sample"],
                "Replicate": [""],
                "Rawfile": ["rawfile.raw"],
            }
        )
        manifest = self._write_temp_manifest_and_import_with_fragpipereader(tmp_path, manifest_content)  # fmt: skip
        pd.testing.assert_frame_equal(manifest, expected_manifest_table)

    def _write_temp_manifest_and_import_with_fragpipereader(self, path, manifest_content):  # fmt: skip
        with open(path / "fragpipe-files.fp-manifest", "w") as tmp:
            tmp.write(manifest_content)
        reader = msreport.reader.FragPipeReader(path)
        return reader.import_manifest()


class TestImportProteins:
    def test_correct_columns_after_renaming(self, test_reader):
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

    def test_non_existing_file_raises_error(self):
        with pytest.raises(FileNotFoundError):
            msreport.reader.FragPipeReader("non_existing_file").import_proteins()


class TestImportPeptides:
    def test_correct_columns_after_renaming(self, test_reader):
        table = test_reader.import_peptides(
            rename_columns=True,
            prefix_column_tags=True,
        )
        assert "Protein reported by software" in table
        assert "Representative protein" in table
        assert "Mapped proteins" in table
        assert "Peptide sequence" in table
        assert "Start position" in table
        assert "Intensity SampleA_1" in table

    def test_column_values_processed_after_import(self, test_reader):
        table = test_reader.import_psm_evidence(
            rename_columns=True,
            rewrite_modifications=True,
        )
        assert not (table["Mapped proteins"] == "").any()


class TestImportIons:
    def test_correct_columns_after_renaming(self, test_reader):
        table = test_reader.import_ions(
            rename_columns=True,
            rewrite_modifications=True,
            prefix_column_tags=True,
        )
        assert "Protein reported by software" in table
        assert "Representative protein" in table
        assert "Mapped proteins" in table
        assert "Start position" in table
        assert "Peptide sequence" in table
        assert "Modified sequence" in table
        assert "Modifications" in table
        assert "Intensity SampleA_1" in table
        assert "Ion ID" in table
        assert table["Peptide sequence"][1] == "CLAALASLR"
        assert table["Modified sequence"][1] == "C[57.0214]LAALASLR"
        assert table["Modifications"][1] == "1:57.0214"
        assert table["Ion ID"][1] == "C[57.0214]LAALASLR_c2"

    def test_column_values_processed_after_import(self, test_reader):
        table = test_reader.import_psm_evidence(
            rename_columns=True,
            rewrite_modifications=True,
        )
        assert not (table["Mapped proteins"] == "").any()


class TestImportIonEvidence:
    def test_correct_columns_after_renaming(self, test_reader):
        table = test_reader.import_ion_evidence(
            rename_columns=True,
            rewrite_modifications=True,
            prefix_column_tags=True,
        )
        assert "Protein reported by software" in table
        assert "Representative protein" in table
        assert "Mapped proteins" in table
        assert "Start position" in table
        assert "Peptide sequence" in table
        assert "Modified sequence" in table
        assert "Modifications" in table
        assert "Intensity" in table
        assert "Ion ID" in table

    def test_column_values_processed_after_import(self, test_reader):
        table = test_reader.import_psm_evidence(
            rename_columns=True,
            rewrite_modifications=True,
        )
        assert not (table["Mapped proteins"] == "").any()

    def test_integration_import_ion_evidence(self, test_reader):
        table = test_reader.import_ion_evidence(
            rename_columns=True,
            rewrite_modifications=True,
            prefix_column_tags=True,
        )
        assert sorted(table["Sample"].unique()) == ["SampleA_1", "SampleB_1"]

        table_sample_b = table[table["Sample"] == "SampleB_1"].reset_index()
        assert table_sample_b["Ion ID"][1] == "C[57.0214]YEM[15.9949]ASHLR_c3"
        assert table_sample_b["Modifications"][1] == "1:57.0214;4:15.9949"

    def test_tables_from_different_samples_are_different(self, test_reader):
        table = test_reader.import_ion_evidence()
        table_1 = table[table["Sample"] == "SampleA_1"]
        table_2 = table[table["Sample"] == "SampleB_1"]
        assert not table_1.equals(table_2)

    def test_sample_column_filled_with_parent_folder(self, test_reader):
        table = test_reader.import_ion_evidence()
        assert set(table["Sample"].unique()) == {"SampleA_1", "SampleB_1"}

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

    def test_column_values_processed_after_import(self, test_reader):
        table = test_reader.import_psm_evidence(
            rename_columns=True,
            rewrite_modifications=True,
        )
        assert not (table["Mapped proteins"] == "").any()

    def test_tables_from_different_samples_are_different(self, test_reader):
        table = test_reader.import_psm_evidence()
        table_1 = table[table["Sample"] == "SampleA_1"]
        table_2 = table[table["Sample"] == "SampleB_1"]
        assert not table_1.equals(table_2)

    def test_sample_column_filled_with_parent_folder(self, test_reader):
        table = test_reader.import_psm_evidence()
        assert set(table["Sample"].unique()) == {"SampleA_1", "SampleB_1"}

    def test_concatenated_table_is_reindexed(self, test_reader):
        table = test_reader.import_psm_evidence()
        assert table.index.nunique() == len(table)


def test_add_modification_localization_string_to_psm_evidence(test_reader):
    psm_table = pd.DataFrame(
        {
            "M:15.9949": ["SPESHM(1.0000)R", "M(1.0000)QAGPGSDR", ""],
            "M:15.9949 Best Localization": [1.0, 1.0, ""],
            "STY:79.9663": ["S(0.9255)PES(0.0745)HMR", "", ""],
            "STY:79.9663 Best Localization": [0.925, "", ""],
            "C:0.0000": [np.nan, np.nan, np.nan],
            "C:0.0000 Best Localization": ["", "", ""],
        }
    )
    expected_localization_strings = [
        "15.9949@6:1.000;79.9663@1:0.925,4:0.074",
        "15.9949@1:1.000",
        "",
    ]
    updated_table = test_reader._add_modification_localization_string_to_psm_evidence(
        psm_table
    )
    assert updated_table["Modification localization string"].tolist() == expected_localization_strings  # fmt: skip


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
