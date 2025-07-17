import msreport.reader


class TestExtractSpectronautLocalizationProbabilities:
    def test_extract_single_modification_with_merged_amino_acid_entries(self):
        localization = msreport.reader.extract_spectronaut_localization_probabilities(
            "_FIMS[Phospho (STY): 33.0%]PT[Phospho (STY): 67.0%]LK_"
        )
        expected = {"Phospho (STY)": {4: 0.33, 6: 0.67}}
        assert localization == expected

    def test_extract_multiple_modifications(self):
        localization = msreport.reader.extract_spectronaut_localization_probabilities(
            "_FIM[Oxidation (M): 100.0%]S[Phospho (STY): 100.0%]PTLK_"
        )
        expected = {"Oxidation (M)": {3: 1.0}, "Phospho (STY)": {4: 1.0}}
        assert localization == expected

    def test_empty_localization_string_returns_empty_dict(self):
        localization = msreport.reader.extract_spectronaut_localization_probabilities(
            ""
        )
        expected = {}
        assert localization == expected
