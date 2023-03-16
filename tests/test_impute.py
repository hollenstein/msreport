import numpy as np
import pandas as pd
import pytest

import msreport.impute


class TestFixedValueImputer:
    @pytest.fixture(autouse=True)
    def _init_table(self):
        self.table = pd.DataFrame(
            {
                "A": [10, 10, 10, np.nan],
                "B": [5, np.nan, 5, np.nan],
                "C": [3, 3, 3, 3],
            }
        )
        self.imputed_positions = [(3, "A"), (1, "B"), (3, "B")]

    def are_all_values_imputed(self, table: pd.DataFrame) -> bool:
        number_missing_values = table.isnull().to_numpy().sum()
        return number_missing_values == 0

    def test_impute_with_constant_strategy(self):
        fill_value = 1
        imputer = msreport.impute.FixedValueImputer(
            strategy="constant", fill_value=fill_value
        )
        imputer.fit(self.table)
        imputed_table = imputer.transform_table(self.table)

        assert self.are_all_values_imputed(imputed_table)
        for pos, col in self.imputed_positions:
            assert imputed_table.loc[pos, col] == 1

    def test_impute_with_below_strategy_and_local(self):
        imputer = msreport.impute.FixedValueImputer(strategy="below", local=True)
        imputer.fit(self.table)
        imputed_table = imputer.transform_table(self.table)

        assert self.are_all_values_imputed(imputed_table)
        for pos, col in self.imputed_positions:
            minimal_column_value = self.table[col].min()
            assert imputed_table.loc[pos, col] < minimal_column_value

    def test_impute_with_below_strategy_and_not_local(self):
        imputer = msreport.impute.FixedValueImputer(strategy="below", local=False)
        imputer.fit(self.table)
        imputed_table = imputer.transform_table(self.table)

        assert self.are_all_values_imputed(imputed_table)
        minimal_array_value = np.nanmin(self.table.to_numpy().flatten())
        for pos, col in self.imputed_positions:
            assert imputed_table.loc[pos, col] < minimal_array_value
