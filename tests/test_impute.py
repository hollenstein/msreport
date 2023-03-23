import numpy as np
import pandas as pd
import pytest

import msreport.impute


def _all_values_imputed(table: pd.DataFrame) -> bool:
    number_missing_values = table.isnull().to_numpy().sum()
    return number_missing_values == 0


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

    def test_impute_with_constant_strategy(self):
        fill_value = 1
        imputer = msreport.impute.FixedValueImputer(
            strategy="constant", fill_value=fill_value
        )
        imputer.fit(self.table)
        imputed_table = imputer.transform(self.table)

        assert _all_values_imputed(imputed_table)
        for pos, col in self.imputed_positions:
            assert imputed_table.loc[pos, col] == 1

    def test_impute_with_below_strategy_and_column_wise(self):
        imputer = msreport.impute.FixedValueImputer(strategy="below", column_wise=True)
        imputer.fit(self.table)
        imputed_table = imputer.transform(self.table)

        assert _all_values_imputed(imputed_table)
        for pos, col in self.imputed_positions:
            minimal_column_value = self.table[col].min()
            assert imputed_table.loc[pos, col] < minimal_column_value

    def test_impute_with_below_strategy_and_not_column_wise(self):
        imputer = msreport.impute.FixedValueImputer(strategy="below", column_wise=False)
        imputer.fit(self.table)
        imputed_table = imputer.transform(self.table)

        assert _all_values_imputed(imputed_table)
        minimal_array_value = np.nanmin(self.table.to_numpy().flatten())
        for pos, col in self.imputed_positions:
            assert imputed_table.loc[pos, col] < minimal_array_value


class TestGaussianImputer:
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

    def test_correct_imputation_with_a_seed_value(self):
        mu, sigma, seed = 0, 1, 0
        np.random.seed(seed)
        expected_random_values = np.random.normal(mu, sigma, 3)
        imputer = msreport.impute.GaussianImputer(mu=mu, sigma=sigma, seed=seed)
        imputer.fit(self.table)
        imputed_table = imputer.transform(self.table)

        assert _all_values_imputed(imputed_table)
        for expected, (pos, col) in zip(expected_random_values, self.imputed_positions):
            observed = imputed_table.loc[pos, col]
            assert observed == expected

    def test_imputing_twice_with_a_seed_value_is_identical(self):
        mu, sigma, seed = 0, 1, 0
        imputer = msreport.impute.GaussianImputer(mu=mu, sigma=sigma, seed=seed)
        imputer.fit(self.table)
        imputed_table_1 = imputer.transform(self.table)
        imputed_table_2 = imputer.transform(self.table)
        assert imputed_table_1.equals(imputed_table_2)

    def test_imputing_twice_with_no_seed_value_is_different(self):
        mu, sigma, seed = 0, 1, None
        imputer = msreport.impute.GaussianImputer(mu=mu, sigma=sigma, seed=seed)
        imputer.fit(self.table)
        imputed_table_1 = imputer.transform(self.table)
        imputed_table_2 = imputer.transform(self.table)
        assert not imputed_table_1.equals(imputed_table_2)


class TestPerseusImputer:
    @pytest.fixture(autouse=True)
    def _init_table(self):
        self.table = pd.DataFrame(
            {
                "A": [10, 7.5, 5, np.nan],
                "B": [5, np.nan, 5, np.nan],
                "C": [3, 3, 3, 3],
            }
        )
        self.imputed_positions = [(3, "A"), (1, "B"), (3, "B")]

    def test_all_values_are_imputed(self):
        median_downshift, std_width, column_wise, seed = 0, 1, True, 0
        imputer = msreport.impute.PerseusImputer(
            median_downshift=median_downshift,
            std_width=std_width,
            column_wise=column_wise,
            seed=seed,
        )
        imputer.fit(self.table)
        imputed_table = imputer.transform(self.table)

        assert _all_values_imputed(imputed_table)
