import pytest
import msreport.peptidoform


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
