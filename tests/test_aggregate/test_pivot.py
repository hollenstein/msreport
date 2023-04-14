import numpy as np
import pandas as pd
import pytest

import msreport.aggregate.pivot as PIVOT


@pytest.fixture
def example_data():
    simple_table = pd.DataFrame(
        {
            "ID": ["A", "B", "C", "B", "C", "D"],
            "Annotation": ["A", "B", "C", "B", "C", "D"],
            "Sample": ["S1", "S1", "S1", "S2", "S2", "S2"],
            "Quant": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        }
    )
    complex_table = pd.DataFrame(
        {
            "ID_1": ["A", "A", "B", "A", "A", "B"],
            "ID_2": ["1", "2", "1", "1", "2", "1"],
            "Annotation": ["A1", "A2", "B1", "A1", "A2", "B1"],
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


class TestJoinUnique:
    def test_column_values_as_expected(self, example_data):
        long_table = example_data["complex_table"]
        output_table = PIVOT.join_unique(long_table, index="ID_1", values="Annotation")
        expected_table = pd.DataFrame(
            data={
                "ID_1": ["A", "B"],
                "Annotation": ["A1;A2", "B1"],
            },
        ).set_index("ID_1")
        assert output_table.equals(expected_table)

    def test_column_values_as_expected_with_multi_index(self, example_data):
        long_table = example_data["complex_table"]
        output_table = PIVOT.join_unique(
            long_table, index=["ID_1", "ID_2"], values="Annotation"
        )
        expected_table = pd.DataFrame(
            data={
                "ID_1": ["A", "A", "B"],
                "ID_2": ["1", "2", "1"],
                "Annotation": ["A1", "A2", "B1"],
            },
        ).set_index(["ID_1", "ID_2"])
        assert output_table.equals(expected_table)


class TestPivotTable:
    def test_pivoted_table_from_single_index_looks_as_expected(self, example_data):
        long_table = example_data["simple_table"]
        output_table = PIVOT.pivot_table(
            long_table,
            index="ID",
            group_by="Sample",
            annotation_columns=["Annotation"],
            pivoting_columns=["Quant"],
        )
        expected_table = pd.DataFrame(
            data={
                "ID": ["A", "B", "C", "D"],
                "Annotation": ["A", "B", "C", "D"],
                "Quant S1": [1.0, 1.0, 1.0, np.nan],
                "Quant S2": [np.nan, 2.0, 2.0, 2.0],
            },
        ).set_index("ID")
        expected_table.reset_index(inplace=True)
        assert output_table.equals(expected_table)

    def test_pivoted_table_from_multi_index_looks_as_expected(self, example_data):
        long_table = example_data["complex_table"]
        output_table = PIVOT.pivot_table(
            long_table,
            index=["ID_1", "ID_2"],
            group_by="Sample",
            annotation_columns=["Annotation"],
            pivoting_columns=["Quant"],
        )
        expected_table = pd.DataFrame(
            data={
                "ID_1": ["A", "A", "B"],
                "ID_2": ["1", "2", "1"],
                "Annotation": ["A1", "A2", "B1"],
                "Quant S1": [1.0, 1.0, 1.0],
                "Quant S2": [2.0, 2.0, 2.0],
            },
        ).set_index(["ID_1", "ID_2"])
        expected_table.reset_index(inplace=True)
        assert output_table.equals(expected_table)
