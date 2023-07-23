import pytest
import msreport.peptidoform


class TestPeptide:
    def test_make_modified_sequence_without_arguments(self):
        modified_peptide = msreport.peptidoform.Peptide("PE[mod1]PT[mod2]IDESR")
        modified_sequence = modified_peptide.make_modified_sequence()
        assert modified_sequence == "PE[mod1]PT[mod2]IDESR"

    def test_make_modified_sequence_with_including_specific_modifications(self):
        modified_peptide = msreport.peptidoform.Peptide("PE[mod1]PT[mod2]ID[mod3]ESR")
        modified_sequence = modified_peptide.make_modified_sequence(include=["mod2", "mod3"])  # fmt: skip
        assert modified_sequence == "PEPT[mod2]ID[mod3]ESR"

    @pytest.mark.parametrize(
        "modification_tag, expected_count",
        [("mod1", 1), ("mod2", 2), ("mod3", 0)],
    )
    def test_count_modification(self, modification_tag, expected_count):
        modified_peptide = msreport.peptidoform.Peptide("PE[mod1]PT[mod2]ID[mod2]ESR")
        modification_count = modified_peptide.count_modification(modification_tag)
        assert modification_count == expected_count

    @pytest.mark.parametrize(
        "modification_tag, expected_sites",
        [("mod1", [2]), ("mod2", [4, 6]), ("mod3", [])],
    )
    def test_list_modified_peptide_sites(self, modification_tag, expected_sites):
        modified_peptide = msreport.peptidoform.Peptide("PE[mod1]PT[mod2]ID[mod2]ESR")
        modified_sites = modified_peptide.list_modified_peptide_sites(modification_tag)  # fmt: skip
        assert modified_sites == expected_sites

    @pytest.mark.parametrize(
        "modification_tag, expected_sites",
        [("mod1", [11]), ("mod2", [13, 15]), ("mod3", [])],
    )
    def test_list_modified_protein_sites(self, modification_tag, expected_sites):
        modified_peptide = msreport.peptidoform.Peptide("PE[mod1]PT[mod2]ID[mod2]ESR", protein_position=10)  # fmt: skip
        modified_sites = modified_peptide.list_modified_protein_sites(modification_tag)
        assert modified_sites == expected_sites

    def test_list_modified_protein_sites_when_protein_start_is_not_specified(self):  # fmt: skip
        modified_peptide = msreport.peptidoform.Peptide("PE[mod1]PT[mod2]ID[mod2]ESR")
        modified_sites = modified_peptide.list_modified_protein_sites("mod1")
        assert modified_sites == [2]

    @pytest.mark.parametrize(
        "site, expected_probability", [(4, 0.8), (6, 0.2), (2, None), (1, None)]
    )
    def test_get_peptide_site_probability(self, site, expected_probability):
        modified_peptide = msreport.peptidoform.Peptide(
            "PE[mod1]PT[mod2]ID[mod2]ESR",
            localization_probabilities={"mod2": {4: 0.800, 6: 0.200}},
        )
        probability = modified_peptide.get_peptide_site_probability(site)
        assert probability == expected_probability

    def test_get_peptide_site_probability_when_no_localization_probabilites_are_specified(self):  # fmt: skip
        modified_peptide = msreport.peptidoform.Peptide("PE[mod1]PT[mod2]ID[mod2]ESR")
        probability = modified_peptide.get_peptide_site_probability(2)
        assert probability == None

    @pytest.mark.parametrize(
        "site, expected_probability", [(13, 0.8), (15, 0.2), (2, None)]
    )
    def test_get_protein_site_probability(self, site, expected_probability):
        modified_peptide = msreport.peptidoform.Peptide(
            "PE[mod1]PT[mod2]ID[mod2]ESR",
            localization_probabilities={"mod2": {4: 0.800, 6: 0.200}},
            protein_position=10,
        )
        probability = modified_peptide.get_protein_site_probability(site)
        assert probability == expected_probability

    def test_get_protein_site_probability_when_protein_start_is_not_specified(self):
        modified_peptide = msreport.peptidoform.Peptide(
            "PE[mod1]PT[mod2]ID[mod2]ESR",
            localization_probabilities={"mod2": {4: 0.800, 6: 0.200}},
        )
        probability = modified_peptide.get_protein_site_probability(4)
        assert probability == 0.800


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


class TestMakeLocalizationString:
    def test_with_single_modifications(self):
        modification_localization_probabilities = {"15.9949": {11: 1.000}}
        expected_string = "15.9949@11:1.000"

        localization_string = msreport.peptidoform.make_localization_string(
            modification_localization_probabilities
        )
        assert localization_string == expected_string

    def test_with_multiple_modifications(self):
        modification_localization_probabilities = {
            "15.9949": {11: 1.000},
            "79.9663": {3: 0.080, 4: 0.219, 5: 0.840, 13: 0.860},
        }
        expected_string = "15.9949@11:1.000;79.9663@3:0.080,4:0.219,5:0.840,13:0.860"

        localization_string = msreport.peptidoform.make_localization_string(
            modification_localization_probabilities
        )
        assert localization_string == expected_string

    def test_with_no_probabilities_in_input_dictionary(self):
        modification_localization_probabilities = {}
        expected_string = ""

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
