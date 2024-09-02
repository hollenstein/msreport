import numpy as np
import pandas as pd
import pytest

import msreport.helper


class TestFindColumns:
    def test_must_be_substring_false(self):
        df = pd.DataFrame(columns=["Test", "Test A", "Test B", "Something else"])
        columns = msreport.helper.find_columns(df, "Test")
        assert len(columns) == 3
        assert columns == ["Test", "Test A", "Test B"]

    def test_must_be_substring_True(self):
        df = pd.DataFrame(columns=["Test", "Test A", "Test B", "Something else"])
        columns = msreport.helper.find_columns(df, "Test", must_be_substring=True)
        assert len(columns) == 2
        assert columns == ["Test A", "Test B"]


class TestFindSampleColumns:
    def test_find_sample_columns(self):
        df = pd.DataFrame(
            columns=[
                "Tag",
                "Tag Not_a_sample",
                "Tag Sample_A",
                "Tag Sample_B",
                "Something else",
            ]
        )
        samples = ["Sample_A", "Sample_B"]
        tag = "Tag"
        columns = msreport.helper.find_sample_columns(df, tag, samples)
        assert columns == ["Tag Sample_A", "Tag Sample_B"]

    def test_columns_are_returned_in_order_of_samples(self):
        df = pd.DataFrame(columns=["Tag Sample_B", "Tag Sample_A"])
        samples = ["Sample_A", "Sample_B"]
        tag = "Tag"
        columns = msreport.helper.find_sample_columns(df, tag, samples)
        assert columns == ["Tag Sample_A", "Tag Sample_B"]

    def test_correct_mapping_with_samples_that_are_substrings(self):
        df = pd.DataFrame(columns=["Tag SampleB_1", "Tag B_1"])
        samples = ["B_1"]
        tag = "Tag"
        columns = msreport.helper.find_sample_columns(df, tag, samples)
        assert columns == ["Tag B_1"]


class TestKeepRowsByPartialMatch:
    def test_entries_partially_matched_to_one_value_are_kept(self):
        table = pd.DataFrame(
            {
                "Col1": [1, 2, 3],
                "Col2": ["A", "B", "CA"],
            }
        )
        matching_values = ["A"]
        matching_column = "Col2"
        filtered = msreport.helper.keep_rows_by_partial_match(
            table, matching_column, matching_values
        )
        assert filtered["Col2"].tolist() == ["A", "CA"]

    def test_entries_partially_matched_to_multiple_values_are_kept(self):
        table = pd.DataFrame(
            {
                "Col1": [1, 2, 3],
                "Col2": ["A", "B", "CA"],
            }
        )
        matching_values = ["B", "C"]
        matching_column = "Col2"
        filtered = msreport.helper.keep_rows_by_partial_match(
            table, matching_column, matching_values
        )
        assert filtered["Col2"].tolist() == ["B", "CA"]


class TestRemoveRowsByPartialMatch:
    def test_entries_partially_matched_to_one_value_are_removed(self):
        table = pd.DataFrame(
            {
                "Col1": [1, 2, 3],
                "Col2": ["A", "B", "CA"],
            }
        )
        matching_values = ["A"]
        matching_column = "Col2"
        filtered = msreport.helper.remove_rows_by_partial_match(
            table, matching_column, matching_values
        )
        assert filtered["Col2"].tolist() == ["B"]

    def test_entries_partially_matched_to_multiple_values_are_removed(self):
        table = pd.DataFrame(
            {
                "Col1": [1, 2, 3],
                "Col2": ["A", "B", "CA"],
            }
        )
        matching_values = ["B", "C"]
        matching_column = "Col2"
        filtered = msreport.helper.remove_rows_by_partial_match(
            table, matching_column, matching_values
        )
        assert filtered["Col2"].tolist() == ["A"]


class TestJoinTables:
    def test_join_two_tables_with_single_index(self):
        tables = [
            pd.DataFrame({"Column 1": [1, 2, 3]}, index=["A", "B", "C"]),
            pd.DataFrame({"Column 2": [6, 5, 4]}, index=["C", "B", "A"]),
        ]
        expected = pd.DataFrame(
            {"Column 1": [1, 2, 3], "Column 2": [4, 5, 6]}, index=["A", "B", "C"]
        )
        joined = msreport.helper.join_tables(tables, reset_index=False)
        assert joined.equals(expected)

    def test_join_three_tables_with_single_index(self):
        tables = [
            pd.DataFrame({"Column 1": [1, 2]}, index=["A", "B"]),
            pd.DataFrame({"Column 2": [4, 3]}, index=["B", "A"]),
            pd.DataFrame({"Column 3": [5, 6]}, index=["A", "B"]),
        ]
        expected = pd.DataFrame(
            {"Column 1": [1, 2], "Column 2": [3, 4], "Column 3": [5, 6]},
            index=["A", "B"],
        )
        joined = msreport.helper.join_tables(tables, reset_index=False)
        assert joined.equals(expected)

    def test_join_two_tables_with_multi_index(self):
        tables = [
            pd.DataFrame(
                {
                    "Column 1": [1, 2, 3],
                    "Index 1": ["A", "A", "B"],
                    "Index 2": ["1", "2", "-"],
                }
            ),
            pd.DataFrame(
                {
                    "Column 2": [4, 5, 6],
                    "Index 1": ["A", "B", "C"],
                    "Index 2": ["2", "-", "-"],
                }
            ),
        ]
        for table in tables:
            table.set_index(["Index 1", "Index 2"], inplace=True)
        expected = pd.DataFrame(
            {
                "Column 1": [1, 2, 3, np.nan],
                "Column 2": [np.nan, 4, 5, 6],
                "Index 1": ["A", "A", "B", "C"],
                "Index 2": ["1", "2", "-", "-"],
            }
        )
        expected.set_index(["Index 1", "Index 2"], inplace=True)
        joined = msreport.helper.join_tables(tables, reset_index=False)
        assert joined.equals(expected)

    def test_join_two_tables_partly_overlapping_single_index_multiple_columns(self):
        tables = [
            pd.DataFrame({"Column 1": [1, 2], "Column 2": [3, 4]}, index=["A", "B"]),
            pd.DataFrame({"Column 3": [6, 5]}, index=["C", "B"]),
        ]
        expected = pd.DataFrame(
            {
                "Column 1": [1, 2, np.nan],
                "Column 2": [3, 4, np.nan],
                "Column 3": [np.nan, 5, 6],
            },
            index=["A", "B", "C"],
        )
        joined = msreport.helper.join_tables(tables, reset_index=False)
        assert joined.equals(expected)

    def test_join_two_tables_with_multi_index_and_reset_index(self):
        tables = [
            pd.DataFrame(
                {
                    "Column 1": [1, 2, 3],
                    "Index 1": ["A", "A", "B"],
                    "Index 2": ["1", "2", "-"],
                }
            ),
            pd.DataFrame(
                {
                    "Column 2": [4, 5, 6],
                    "Index 1": ["A", "B", "C"],
                    "Index 2": ["2", "-", "-"],
                }
            ),
        ]
        for table in tables:
            table.set_index(["Index 1", "Index 2"], inplace=True)
        expected = pd.DataFrame(
            {
                "Index 1": ["A", "A", "B", "C"],
                "Index 2": ["1", "2", "-", "-"],
                "Column 1": [1, 2, 3, np.nan],
                "Column 2": [np.nan, 4, 5, 6],
            }
        )
        joined = msreport.helper.join_tables(tables, reset_index=True)
        assert joined.equals(expected)

    def test_join_two_tables_and_reset_index_with_no_index_name(self):
        tables = [
            pd.DataFrame({"Column 1": [1, 2, 3]}, index=["A", "B", "C"]),
            pd.DataFrame({"Column 2": [6, 5, 4]}, index=["C", "B", "A"]),
        ]
        expected = pd.DataFrame(
            {"index": ["A", "B", "C"], "Column 1": [1, 2, 3], "Column 2": [4, 5, 6]}
        )
        joined = msreport.helper.join_tables(tables, reset_index=True)
        assert joined.equals(expected)


def test_rename_mq_reporter_channels_only_intensity():
    table = pd.DataFrame(
        columns=[
            "Reporter intensity 1",
            "Reporter intensity 2",
        ]
    )
    channel_names = ["Channel 1", "Channel 2"]
    expected_columns = [
        "Reporter intensity Channel 1",
        "Reporter intensity Channel 2",
    ]
    msreport.helper.rename_mq_reporter_channels(table, channel_names)
    assert table.columns.tolist() == expected_columns


def test_rename_mq_reporter_channels_with_other_columns():
    table = pd.DataFrame(
        columns=[
            "Reporter intensity 1",
            "Reporter intensity 2",
            "Reporter intensity",
            "Reporter count",
            "Something else",
        ]
    )
    channel_names = ["Channel 1", "Channel 2"]

    expected_columns = [
        "Reporter intensity Channel 1",
        "Reporter intensity Channel 2",
        "Reporter intensity",
        "Reporter count",
        "Something else",
    ]
    msreport.helper.rename_mq_reporter_channels(table, channel_names)
    assert table.columns.tolist() == expected_columns


def test_rename_mq_reporter_channels_with_count_and_corrected():
    table = pd.DataFrame(
        columns=[
            "Reporter intensity 1",
            "Reporter intensity 2",
            "Reporter intensity count 1",
            "Reporter intensity count 2",
            "Reporter intensity corrected 1",
            "Reporter intensity corrected 2",
        ]
    )
    channel_names = ["Channel 1", "Channel 2"]

    expected_columns = [
        "Reporter intensity Channel 1",
        "Reporter intensity Channel 2",
        "Reporter intensity count Channel 1",
        "Reporter intensity count Channel 2",
        "Reporter intensity corrected Channel 1",
        "Reporter intensity corrected Channel 2",
    ]
    msreport.helper.rename_mq_reporter_channels(table, channel_names)
    assert table.columns.tolist() == expected_columns


class TestRenameSampleColumns:
    @pytest.fixture(autouse=True)
    def _init_table(self):
        self.table = pd.DataFrame(
            columns=[
                "Column 1",
                "Intensity Sample 1",
                "Intensity Sample 11",
                "Intensity Sample A",
                "Intensity Sample B",
            ]
        )

    def test_simple_renaming(self):
        mapping = {"Sample A": "Sample 2", "Sample B": "Sample 3"}
        renamed_table = msreport.helper.rename_sample_columns(self.table, mapping)
        expected_columns = [f"Intensity {name}" for name in mapping.values()]
        removed_columns = [f"Intensity {name}" for name in mapping.keys()]

        assert all([col in renamed_table for col in expected_columns])
        assert all([col not in renamed_table for col in removed_columns])

    def test_rename_with_original_names_being_substrings_of_orignal_names(self):
        mapping = {"Sample 1": "Sample C", "Sample 11": "Sample B"}
        renamed_table = msreport.helper.rename_sample_columns(self.table, mapping)
        expected_columns = [f"Intensity {name}" for name in mapping.values()]
        removed_columns = [f"Intensity {name}" for name in mapping.keys()]

        assert all([col in renamed_table for col in expected_columns])
        assert all([col not in renamed_table for col in removed_columns])

    def test_rename_with_target_names_being_substrings_of_original_names(self):
        table = pd.DataFrame(columns=["Intensity Sample 1", "Intensity Sample 11"])
        mapping = {"Sample 1": "Sample 11", "Sample 11": "Sample 1"}
        renamed_table = msreport.helper.rename_sample_columns(table, mapping)
        expected_columns = [f"Intensity {name}" for name in mapping.values()]

        assert renamed_table.columns.tolist() == expected_columns
        assert renamed_table.columns.tolist() != table.columns.tolist()


def test_rename_sample_columns():
    mapping = {
        "tes": "another name",
        "test": "reference",
        "test_1": "ctrl_1",
        "test_2": "ctrl_2",
        "treatment_test_1": "treatment_1",
        "treatment_test_2": "treatment_2",
    }
    tag = "Intensity"
    expected_renamed_columns = [f"{tag} {value}" for value in mapping.values()]

    table = pd.DataFrame(columns=[f"{tag} {k}" for k in mapping.keys()])
    renamed_table = msreport.helper.rename_sample_columns(table, mapping)
    observed_renamed_columns = renamed_table.columns.tolist()

    assert observed_renamed_columns == expected_renamed_columns


class TestApplyIntensityCutoff:
    @pytest.fixture(autouse=True)
    def _init_table(self):
        self.table = pd.DataFrame(
            {
                "No tag": [2, 1, 2],
                "Tag Sample_A1": [9.9, 10.9, 11.9],
                "Tag Sample_A2": [12, 12, 12],
            }
        )

    @pytest.mark.parametrize("threshold", [9, 10, 11, 12])
    def test_correct_application_of_cutoff(self, threshold):
        expected_nan = (self.table["Tag Sample_A1"] < threshold).sum()
        msreport.helper.apply_intensity_cutoff(self.table, "Tag", threshold=threshold)
        observed_nan = self.table["Tag Sample_A1"].isna().sum()
        assert expected_nan == observed_nan

    def test_columns_without_tag_are_not_modified(self):
        no_tag_data = self.table["No tag"].tolist()
        msreport.helper.apply_intensity_cutoff(self.table, "Tag", threshold=99)
        np.testing.assert_array_equal(no_tag_data, self.table["No tag"])


def test_extract_modifications():
    modified_sequence = "(Acetyl (Protein N-term))ADSRDPASDQM(Oxidation (M))QHWK"
    expected_modifications = [(0, "Acetyl (Protein N-term)"), (11, "Oxidation (M)")]
    modifications = msreport.helper.extract_modifications(modified_sequence, "(", ")")
    assert modifications == expected_modifications


@pytest.mark.parametrize(
    "sequence, modifications, expected_mofified_sequence",
    [
        ("ADSRDPASDQMQHWK", [], "ADSRDPASDQMQHWK"),
        (
            "ADSRDPASDQMQHWK",
            [(0, "Acetyl (Protein N-term)"), (11, "Oxidation (M)")],
            "[Acetyl (Protein N-term)]ADSRDPASDQM[Oxidation (M)]QHWK",
        ),
        (
            "ADSRDPASDQMQHWK",
            [(11, "Oxidation (M)"), (0, "Acetyl (Protein N-term)")],
            "[Acetyl (Protein N-term)]ADSRDPASDQM[Oxidation (M)]QHWK",
        ),
        (
            "ADSRDPASDQMQHWK",
            [(0, "Oxidation (M)"), (0, "Acetyl (Protein N-term)")],
            "[Acetyl (Protein N-term)][Oxidation (M)]ADSRDPASDQMQHWK",
        ),
    ],
)
def test_modify_peptide(sequence, modifications, expected_mofified_sequence):
    modified_sequence = msreport.helper.modify_peptide(sequence, modifications)
    assert modified_sequence == expected_mofified_sequence


class TestGuessDesign:
    def test_well_formated_sample_names(self):
        table = pd.DataFrame(
            columns=[
                "Intensity ExperimentA_R1",
                "Intensity ExperimentB_R1",
                "Intensity ExperimentB_R2",
                "Intensity",
                "Other columns",
            ]
        )
        tag = "Intensity"
        expected_design = pd.DataFrame(
            {
                "Sample": ["ExperimentA_R1", "ExperimentB_R1", "ExperimentB_R2"],
                "Experiment": ["ExperimentA", "ExperimentB", "ExperimentB"],
                "Replicate": ["R1", "R1", "R2"],
            }
        )

        design = msreport.helper.guess_design(table, tag)
        assert expected_design.equals(design)

    def test_single_experiment(self):
        table = pd.DataFrame(
            columns=[
                "Intensity ExperimentA",
                "Intensity",
                "Other columns",
            ]
        )
        tag = "Intensity"
        expected_design = pd.DataFrame(
            {
                "Sample": ["ExperimentA"],
                "Experiment": ["ExperimentA"],
                "Replicate": ["-1"],
            }
        )

        design = msreport.helper.guess_design(table, tag)
        assert expected_design.equals(design)

    def test_ignore_total_and_combined_as_sample_names(self):
        table = pd.DataFrame(
            columns=[
                "Intensity ValidSampleName",
                "Intensity total",
                "Intensity Total",
                "Intensity combined",
                "Intensity Combined",
                "Intensity COMBINED",
            ]
        )
        tag = "Intensity"

        design = msreport.helper.guess_design(table, tag)
        assert design["Sample"].to_list() == ["ValidSampleName"]


@pytest.mark.parametrize(
    "data, data_in_logspace",
    [
        ([32, 45], True),
        ([np.nan, 64], True),
        ([100, 1], False),
        ([100, np.nan], False),
        ([32, 64.1], False),
        (np.array([32, 45]), True),
        (np.array([[32, 45], [32, 45]]), True),
        (np.array([[32, 45], [32, 64.1]]), False),
        (pd.DataFrame([[32, 45, 64], [32, 45, 64]]), True),
        (pd.DataFrame([[32, 45, 64.1]]), False),
    ],
)
def test_intensities_in_logspace(data, data_in_logspace):
    assert msreport.helper.intensities_in_logspace(data) == data_in_logspace


class TestMode:
    @pytest.fixture(autouse=True)
    def _init_random_values(self):
        np.random.seed(0)
        self.values = np.random.normal(size=100)

    def test_mode_is_calculated_properly(self):
        mode = msreport.helper.mode(self.values)
        np.testing.assert_allclose(mode, 0.0879, rtol=1e-02, atol=1e-02, equal_nan=True)  # fmt: skip

    def test_mode_calculation_with_some_nan_returns_a_number(self):
        self.values[[i for i in range(1, 100, 10)]] = np.nan
        mode = msreport.helper.mode(self.values)
        assert ~np.isnan(mode)

    def test_mode_calculation_with_all_nan_returns_nan(self):
        self.values[:] = np.nan
        mode = msreport.helper.mode(self.values)
        assert np.isnan(mode)

    def test_mode_calculation_with_all_identical_values(self):
        self.values[:] = 1
        mode = msreport.helper.mode(self.values)
        assert mode == 1


def test_calculate_tryptic_ibaq_peptides():
    peptides = [
        "MGSCCSCLK",
        "DSSDEASVSPIADNER",
        "EAVTLLLGYLEDK",
        "DQLDFYSGGPLK",
        "ALTTLVYSDNLNLQR",
        "SAALAFAEITEK",
        "YVR",
        "QVSR",
        "EVLEPILILLQSQDPQIQVAACAALGNLAVNNENK",
        "EVLEPILILLQSQDPQIQVAACAALGNLAK",
        "LEAPQE",
    ]
    min_len = 7
    max_len = 30
    protein_sequence = "".join(peptides)
    expected_ibaq_peptides = sum(
        [len(p) >= min_len and len(p) <= max_len for p in peptides]
    )
    ibaq_peptides = msreport.helper.calculate_tryptic_ibaq_peptides(protein_sequence)
    assert ibaq_peptides == expected_ibaq_peptides


def test_calculate_monoisotopic_mass():
    protein_sequence = "MDDDIAALVVDNGSGMCKAGFAGDDAPRAVFPSIVGRPRHQGVMVGMGQK"
    monoisotopic_mass = 5142.47
    calculated_mass = msreport.helper.calculate_monoisotopic_mass(protein_sequence)
    assert round(calculated_mass, 2) == monoisotopic_mass


@pytest.mark.parametrize(
    "length, expected_coverage, peptide_positions",
    [
        (10, 9, [(1, 5), (3, 6), (8, 10)]),
        (20, 9, [(1, 5), (3, 6), (8, 10)]),
        (10, 5, [(1, 5), (1, 5), (1, 5)]),
        (10, 0, []),
    ],
)
def test_make_coverage_mask(length, expected_coverage, peptide_positions):
    coverage_mask = msreport.helper.make_coverage_mask(length, peptide_positions)
    assert coverage_mask.sum() == expected_coverage


@pytest.mark.parametrize(
    "length, expected_coverage, ndigits, peptide_positions",
    [
        (15, round(7 / 15 * 100, 0), 0, [(1, 7)]),
        (15, round(7 / 15 * 100, 1), 1, [(1, 7)]),
        (15, round(7 / 15 * 100, 10), 10, [(1, 7)]),
    ],
)
def test_calculate_sequence_coverage(
    length, expected_coverage, ndigits, peptide_positions
):
    calculated_coverage = msreport.helper.calculate_sequence_coverage(
        length, peptide_positions, ndigits=ndigits
    )
    assert calculated_coverage == expected_coverage
