import pytest
import msreport.peptidoform


class TestParseModifiedSequence:
    @pytest.mark.parametrize(
        "modified_sequence, expected_plain_sequence",
        [
            (
                "(Acetyl (Protein N-term))ADSRDPASDQM(Oxidation (M))QHWK",
                "ADSRDPASDQMQHWK",
            ),
            ("ADSRDPASDQMQHWK", "ADSRDPASDQMQHWK"),
            ("ADSRDPASDQMQHWK(Mod)", "ADSRDPASDQMQHWK"),
        ],
    )
    def test_correct_sequence_extracted(self, modified_sequence, expected_plain_sequence):  # fmt: skip
        plain_sequence, _ = msreport.peptidoform.parse_modified_sequence(
            modified_sequence, "(", ")"
        )
        assert plain_sequence == expected_plain_sequence


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
    modified_sequence = msreport.peptidoform.modify_peptide(sequence, modifications)
    assert modified_sequence == expected_mofified_sequence


def test_make_localization_string():
    modification_localization_probabilities = {
        "15.9949": {11: 1.000},
        "79.9663": {3: 0.080, 4: 0.219, 5: 0.840, 13: 0.860},
    }
    expected_string = "15.9949@11:1.000;79.9663@3:0.080,4:0.219,5:0.840,13:0.860"

    localization_string = msreport.peptidoform.make_localization_string(
        modification_localization_probabilities
    )
    assert localization_string == expected_string


def test_read_localization_string():
    localization_string = "15.9949@11:1.000;79.9663@3:0.080,4:0.219,5:0.840,13:0.860"
    expected_localization = {
        "15.9949": {11: 1.000},
        "79.9663": {3: 0.080, 4: 0.219, 5: 0.840, 13: 0.860},
    }

    localization = msreport.peptidoform.read_localization_string(localization_string)
    assert localization == expected_localization


def test_make_and_read_localization_string():
    initial_localization = {
        "15.9949": {11: 1.000},
        "79.9663": {3: 0.080, 4: 0.219, 5: 0.840, 13: 0.860},
    }
    generated_localization = msreport.peptidoform.read_localization_string(
        msreport.peptidoform.make_localization_string(initial_localization)
    )
    assert initial_localization == generated_localization
