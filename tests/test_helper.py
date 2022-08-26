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


def test_guess_design():
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
        }
    )

    design = msreport.helper.guess_design(table, tag)
    assert expected_design.equals(design)


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


def test_gaussian_imputation():
    table = pd.DataFrame(
        {
            "A": [90000, 100000, 110000, np.nan],
            "B": [1, 1, 1, np.nan],
        }
    )
    median_downshift = 1
    std_width = 1
    imputed = msreport.helper.gaussian_imputation(table, median_downshift, std_width)

    number_missing_values = imputed.isna().sum().sum()
    assert number_missing_values == 0

    imputed_column_A = imputed["A"][3]
    imputed_column_B = imputed["B"][3]
    assert imputed_column_A >= imputed_column_B


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
