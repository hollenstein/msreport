import numpy as np
import pandas as pd
import pytest

import msreport.export


"""
class TestValidateProteins:
    @pytest.fixture(autouse=True)
    def _init_qtable(self, example_qtable):
        self.qtable = example_qtable

    def test_valid_column_is_added(self):
        self.qtable.data = self.qtable.data.drop(columns="Valid")
        msreport.analyze.validate_proteins(self.qtable, remove_contaminants=False)
        data_columns = self.qtable.data.columns.to_list()
        assert "Valid" in data_columns

    @pytest.mark.parametrize(
        "min_peptides, expected_valid", [(0, 3), (1, 3), (2, 2), (3, 0)]
    )
    def test_validate_with_min_peptides(self, min_peptides, expected_valid):
        msreport.analyze.validate_proteins(
            self.qtable, remove_contaminants=False, min_peptides=min_peptides
        )
        assert expected_valid == self.qtable.data["Valid"].sum()
"""


def test_generate_html_sequence_map():
    # Note that this only tests for one specific example
    html_string = msreport.export._generate_html_sequence_map(
        sequence="aBBaCCCaaaaa",
        covered_regions=[(1, 2), (4, 6)],
        coverage_color="#FF0000",
        highlights={0: "#FF00FF"},
        column_length=3,
        row_length=6,
    )
    expected = (
        '<FONT COLOR="#606060">'
        '<FONT COLOR="#000000"> 1   </FONT>'
        '<FONT COLOR="#FF00FF"><u>a</u></FONT>'
        '<FONT COLOR="#FF0000">BB</FONT>'
        " a"
        '<FONT COLOR="#FF0000">CC</FONT>'
        "<br>"
        '<FONT COLOR="#000000"> 7   </FONT>'
        '<FONT COLOR="#FF0000">C</FONT>'
        "aa aaa"
        "</FONT>"
    )
    assert html_string == expected


@pytest.mark.parametrize(
    "coverage_mask, expected_boundaries",
    [
        ([0, 1, 0, 1, 1, 0], [(1, 1), (3, 4)]),
        ([1, 0, 0, 0, 1, 1], [(0, 0), (4, 5)]),
        ([0, 0, 0, 0, 1, 1], [(4, 5)]),
        ([0, 0, 0, 0, 0, 0], []),
        ([1, 1, 1, 1, 1, 1], [(0, 5)]),
    ],
)
def test_intensities_in_logspace(coverage_mask, expected_boundaries):
    boundaries = msreport.export._find_covered_region_boundaries(coverage_mask)
    assert boundaries == expected_boundaries
