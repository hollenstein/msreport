from __future__ import annotations
import abc
import itertools
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import statsmodels.nonparametric.smoothers_lowess

import msreport.helper


class BaseSampleNormalizer(abc.ABC):
    """Base class for all sample normalizers."""

    @abc.abstractmethod
    def fit(self, table: pd.DataFrame) -> BaseSampleNormalizer:
        pass

    @abc.abstractmethod
    def is_fitted(self) -> bool:
        pass

    @abc.abstractmethod
    def transform(self, sample: str, values: Iterable) -> Iterable:
        pass

    @abc.abstractmethod
    def transform_table(self, table: pd.DataFrame) -> pd.DataFrame:
        pass


class FixedValueNormalizer(BaseSampleNormalizer):
    """Normalization by a constant normalization factor for each sample.

    Expects log transformed intensity values.
    """

    def __init__(self, center_function: Callable, comparison: str):
        """Inits a FixedValueNormalizer.

        Args:
            center_function: A function that accepts a sequence of values and
                returns a center value such as the median.
            comparison: "paired" or "reference"
        """
        if comparison not in ["paired", "reference"]:
            raise ValueError(
                f'"comparison" = {comparison} not allowed. '
                'Must be either "paired" or "reference".'
            )
        self._comparison_mode = comparison
        self._fit_function = center_function
        self._sample_fits = None

    def fit(self, table: pd.DataFrame) -> BaseSampleNormalizer:
        if self._comparison_mode == "paired":
            self._fit_with_paired_samples(table)
        elif self._comparison_mode == "reference":
            self._fit_with_pseudo_reference(table)
        return self

    def is_fitted(self) -> bool:
        return self._sample_fits is not None

    def transform(self, sample: str, values: Iterable) -> np.ndarray:
        data = np.array(values, dtype=float)
        mask = np.isfinite(data)
        data[mask] = data[mask] - self._sample_fits[sample]
        return data

    def transform_table(self, table: pd.DataFrame) -> pd.DataFrame:
        _table = table.copy()
        for sample in table.columns:
            _table[sample] = self.transform(sample, _table[sample])
        return _table

    def _fit_with_paired_samples(self, table: pd.DataFrame) -> None:
        samples = table.columns.tolist()
        num_samples = len(samples)
        sample_combinations = list(itertools.combinations(range(num_samples), 2))
        ratio_table = np.full((num_samples, num_samples), np.nan)
        for i, j in sample_combinations:
            ratios = table[samples[i]] - table[samples[j]]
            ratios = ratios[np.isfinite(ratios)]
            center_value = self._fit_function(ratios)
            ratio_table[i, j] = center_value
        profile = msreport.helper.solve_ratio_matrix(ratio_table)
        self._sample_fits = dict(zip(samples, profile))

    def _fit_with_pseudo_reference(self, table: pd.DataFrame) -> None:
        ref_mask = table.isna().sum(axis=1) == 0
        ref_values = table[ref_mask].mean(axis=1)
        samples = table.columns.tolist()

        self._sample_fits = {}
        for sample in samples:
            sample_values = table.loc[ref_mask, sample]
            sample_fit = self._fit_function(sample_values - ref_values)
            self._sample_fits[sample] = sample_fit


class ValueDependentNormalizer(BaseSampleNormalizer):
    """Normalization with a value dependent fit for each sample.

    Expects log transformed intensity values.
    """

    def __init__(self, fit_function: Callable):
        """Inits a ValueDependentNormalizer.

        Args:
            fit_function: A function that accepts a sequence of values and returns numpy
                array with two columns. The first column contains the values and the
                second column the associated deviations.
        """
        self._sample_fits = None
        self._fit_function = fit_function

    def fit(self, table: pd.DataFrame) -> BaseSampleNormalizer:
        self._fit_with_pseudo_reference(table)
        return self

    def is_fitted(self) -> bool:
        return self._sample_fits is not None

    def transform(self, sample: str, values: Iterable) -> np.ndarray:
        data = np.array(values, dtype=float)
        mask = np.isfinite(data)

        sample_fit = self._sample_fits[sample]
        fit_values, fit_deviations = [np.array(i) for i in zip(*sample_fit)]
        data[mask] = data[mask] - np.interp(data[mask], fit_values, fit_deviations)
        return data

    def transform_table(self, table: pd.DataFrame) -> pd.DataFrame:
        _table = table.copy()
        for sample in table.columns:
            _table[sample] = self.transform(sample, _table[sample])
        return _table

    def _fit_with_pseudo_reference(self, table: pd.DataFrame) -> None:
        ref_mask = table.isna().sum(axis=1) == 0
        ref_values = table[ref_mask].mean(axis=1)
        samples = table.columns.tolist()

        self._sample_fits = {}
        for sample in samples:
            sample_values = table.loc[ref_mask, sample]
            sample_fit = self._fit_function(sample_values, ref_values)
            self._sample_fits[sample] = sample_fit


class MedianNormalizer(FixedValueNormalizer):
    def __init__(self):
        super(MedianNormalizer, self).__init__(
            center_function=np.median, comparison="paired"
        )


class ModeNormalizer(FixedValueNormalizer):
    def __init__(self):
        super(ModeNormalizer, self).__init__(
            center_function=msreport.helper.mode, comparison="paired"
        )


class LowessNormalizer(ValueDependentNormalizer):
    def __init__(self):
        super(LowessNormalizer, self).__init__(fit_function=_value_dependent_fit_lowess)


def _value_dependent_fit_lowess(
    values: np.ndarray, reference_values: np.ndarray
) -> np.ndarray:
    """Uses lowess to calculate a fit between the values and their deviation
    from the reference_values.
    """
    delta_span_percentage = 0.05
    delta = (reference_values.max() - reference_values.min()) * delta_span_percentage
    deviations = values - reference_values
    return statsmodels.nonparametric.smoothers_lowess.lowess(
        deviations, values, delta=delta, it=5
    )
