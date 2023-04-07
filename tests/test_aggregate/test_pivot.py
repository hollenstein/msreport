import numpy as np
import pandas as pd
import pytest

import msreport.aggregate.pivot as PIVOT


@pytest.fixture
def example_data():
    simple_table = pd.DataFrame(
        {
            "ID": ["A", "B", "C", "B", "C", "D"],
            "Sample": ["S1", "S1", "S1", "S2", "S2", "S2"],
            "Quant": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        }
    )
    complex_table = pd.DataFrame(
        {
            "ID_1": ["A", "A", "B", "A", "A", "B"],
            "ID_2": ["1", "2", "1", "1", "2", "1"],
            "Sample": ["S1", "S1", "S1", "S2", "S2", "S2"],
            "Quant": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        }
    )
    example_data = {"simple_table": simple_table, "complex_table": complex_table}
    return example_data


class TestPivotColumn:
    def test_correct_index_created(self, example_data):
        long_table = example_data["simple_table"]
        pivot_table = PIVOT.pivot_column(
            long_table,
            index="ID",
            group_by="Sample",
            values="Quant",
        )
        assert len(pivot_table.index) == len(set(long_table["ID"]))
        assert set(pivot_table.index) == set(long_table["ID"])

    def test_correct_multi_index_created(self, example_data):
        long_table = example_data["complex_table"]
        pivot_table = PIVOT.pivot_column(
            long_table,
            index=["ID_1", "ID_2"],
            group_by="Sample",
            values="Quant",
        )
        unique_ids = set(zip(long_table["ID_1"], long_table["ID_2"]))
        assert len(pivot_table.index) == len(unique_ids)
        assert set(pivot_table.index) == unique_ids

    def test_correct_columns_created(self, example_data):
        long_table = example_data["simple_table"]
        pivot_table = PIVOT.pivot_column(
            long_table,
            index="ID",
            group_by="Sample",
            values="Quant",
        )
        assert pivot_table.columns.tolist() == ["Quant S1", "Quant S2"]

    def test_pivoted_table_as_expected(self, example_data):
        long_table = example_data["simple_table"]
        pivot_table = PIVOT.pivot_column(
            long_table,
            index="ID",
            group_by="Sample",
            values="Quant",
        )
        expected_table = pd.DataFrame(
            index=["A", "B", "C", "D"],
            data={
                "Quant S1": [1.0, 1.0, 1.0, np.nan],
                "Quant S2": [np.nan, 2.0, 2.0, 2.0],
            },
        )
        assert pivot_table.equals(expected_table)

    def test_pivoted_table_with_multi_index_as_expected(self, example_data):
        long_table = example_data["complex_table"]
        pivot_table = PIVOT.pivot_column(
            long_table,
            index=["ID_1", "ID_2"],
            group_by="Sample",
            values="Quant",
        )
        expected_table = pd.DataFrame(
            data={
                "ID_1": ["A", "A", "B"],
                "ID_2": ["1", "2", "1"],
                "Quant S1": [1.0, 1.0, 1.0],
                "Quant S2": [2.0, 2.0, 2.0],
            },
        ).set_index(["ID_1", "ID_2"])
        assert pivot_table.equals(expected_table)
