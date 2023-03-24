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


def test_find_sample_columns():
    df = pd.DataFrame(
        columns=[
            "Test",
            "Test Not_a_sample",
            "Test Sample_A",
            "Test Sample_B",
            "Something else",
        ]
    )
    samples = ["Sample_A", "Sample_B"]
    tag = "Test"
    columns = msreport.helper.find_sample_columns(df, tag, samples)
    assert columns == ["Test Sample_A", "Test Sample_B"]


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
    expected_renamed_columns = [f"{tag} {mapping[k]}" for k in mapping.keys()]

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


def test_mode():
    values = np.random.normal(size=100)
    mode = msreport.helper.mode(values)
    assert isinstance(mode, float)
    assert mode < 1 and mode > -1


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
