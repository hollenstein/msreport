import abc
import itertools
from typing import Callable, Iterable

import numpy as np
import pandas as pd

import msreport.helper


class BaseSampleNormalizer(abc.ABC):
    """ Base class for all sample normalizers."""
    @abc.abstractmethod
    def fit(self, matrix: pd.DataFrame) -> 'Self':
        pass

    @abc.abstractmethod
    def transform(self, sample: str, values: Iterable) -> Iterable:
        pass

    @abc.abstractmethod
    def transform_matrix(self, matrix: pd.DataFrame) -> pd.DataFrame:
        pass


class FixedValueNormalizer(BaseSampleNormalizer):
    """ Expects log transformed intensity values. """

    def __init__(self, center_function: Callable, fit: str = 'paired'):
        if fit not in ['paired', 'reference']:
            raise ValueError(
                f'"fit" = {fit} not allowed. '
                'Must be either "paired" or "reference".'
            )
        self._fitting_mode = fit
        self._center_function = center_function
        self._sample_norm_values = None

    def fit(self, matrix: pd.DataFrame) -> BaseSampleNormalizer:
        if self._fitting_mode == 'paired':
            self._fit_with_paired_samples(matrix)
        elif self._fitting_mode == 'reference':
            self._fit_with_pseudo_reference(matrix)
        return self

    def transform(self, sample: str, values: Iterable) -> np.ndarray:
        data = np.array(values, dtype=float)
        mask = np.isfinite(data)
        data[mask] = data[mask] - self._sample_norm_values[sample]
        return data

    def transform_matrix(self, matrix: pd.DataFrame) -> pd.DataFrame:
        _matrix = matrix.copy()
        for sample in matrix.columns:
            _matrix[sample] = self.transform(sample, _matrix[sample])
        return _matrix

    def _fit_with_paired_samples(self, matrix: pd.DataFrame) -> None:
        samples = matrix.columns.tolist()
        num_samples = len(samples)
        sample_combinations = list(
            itertools.combinations(range(num_samples), 2)
        )
        ratio_matrix = np.full((num_samples, num_samples), np.nan)
        for i, j in sample_combinations:
            ratios = matrix[samples[i]] - matrix[samples[j]]
            ratios = ratios[np.isfinite(ratios)]
            center_value = self._center_function(ratios)
            ratio_matrix[i, j] = center_value
        profile = msreport.helper.solve_ratio_matrix(ratio_matrix)
        self._sample_norm_values = dict(zip(samples, profile))

    def _fit_with_pseudo_reference(self, matrix: pd.DataFrame) -> None:
        ref_mask = (matrix.isna().sum(axis=1) == 0)
        ref_values = matrix[ref_mask].mean(axis=1)
        samples = matrix.columns.tolist()

        self._sample_norm_values = {}
        for sample in samples:
            sample_values = matrix.loc[ref_mask, sample]
            norm_value = self._center_function(sample_values - ref_values)
            self._sample_norm_values[sample] = norm_value
