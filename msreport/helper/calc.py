from typing import Iterable

import numpy as np
import pyteomics.mass
import pyteomics.parser
import scipy.optimize
import scipy.stats


def mode(values: Iterable) -> float:
    """Calculate the mode by using kernel-density estimation.

    Args:
        values: Sequence of values for which the mode will be estimated, only finite
            values are used for the calculation.

    Returns:
        The estimated mode. If no finite values are present, returns nan.
    """
    values = np.asarray(values)
    finite_values = values[np.isfinite(values)]
    if len(finite_values) == 0:
        return np.nan
    elif len(np.unique(finite_values)) == 1:
        return np.unique(finite_values)[0]

    kde = scipy.stats.gaussian_kde(finite_values)

    def _minimum_function(x):
        return -kde(x)[0]

    min_slice, max_sclice = np.percentile(finite_values, (2, 98))
    slice_step = 0.2
    brute_optimize_result = scipy.optimize.brute(
        _minimum_function, [slice(min_slice, max_sclice + slice_step, slice_step)]
    )
    rough_minimum = brute_optimize_result[0]

    local_optimize_result = scipy.optimize.minimize(
        _minimum_function, x0=rough_minimum, method="BFGS"
    )
    fine_minimum = local_optimize_result.x[0]
    return fine_minimum


def calculate_tryptic_ibaq_peptides(protein_sequence: str) -> int:
    """Calculates the number of tryptic iBAQ peptides.

    The number of iBAQ peptides is calculated as the number of tryptic peptides with a
    length between 7 and 30 amino acids. Multiple peptides with the same sequence are
    counted multiple times.

    Args:
        protein_sequence: Amino acid sequence of a protein.

    Returns:
        Number of tryptic iBAQ peptides for the given protein sequence.
    """
    cleavage_rule = "[KR]"
    missed_cleavage = 0
    min_length = 7
    max_length = 30

    digestion_products = pyteomics.parser.icleave(
        protein_sequence,
        cleavage_rule,
        missed_cleavages=missed_cleavage,
        min_length=min_length,
        max_length=max_length,
        regex=True,
    )
    ibaq_peptides = [sequence for index, sequence in digestion_products]
    return len(ibaq_peptides)


def calculate_monoisotopic_mass(protein_sequence: str) -> float:
    """Calculates the monoisotopic mass of the protein sequence in Dalton.

    Note that there is an opinionated behaviour for non-standard amino acids code. "O"
    is Pyrrolysine, "U" is Selenocysteine, "B" is treated as "N", "Z" is treated as "Q",
    and "X" is ignored.

    Args:
        protein_sequence: Amino acid sequence of a protein.

    Returns:
        Monoisotopic mass in Dalton.
    """
    sequence = protein_sequence.replace("B", "N").replace("Z", "Q").replace("X", "")
    return pyteomics.mass.fast_mass(sequence)


def make_coverage_mask(
    protein_length: int, peptide_positions: list[(int, int)]
) -> np.array:
    """Returns a Boolean array with True for positions present in 'peptide_positions'.

    Args:
        protein_length: The number of amino acids in the protein sequence.
        peptide_positions: List of peptide start and end positions.

    Returns:
        A 1-dimensional Boolean array with length equal to 'protein_length'.
    """
    coverage_mask = np.zeros(protein_length, dtype="bool")
    for start, end in peptide_positions:
        coverage_mask[start - 1 : end] = True
    return coverage_mask


def calculate_sequence_coverage(
    protein_length: int, peptide_positions: list[(int, int)], ndigits: int = 1
) -> np.array:
    """Calculates the protein sequence coverage given a list of peptide positions.

    Args:
        protein_length: The number of amino acids in the protein sequence.
        peptide_positions: List of peptide start and end positions.
        ndigits: Optional, number of decimal places for rounding the sequence coverage.

    Returns:
        Sequence coverage in percent, with values ranging from 0 to 100.
    """
    coverage_mask = make_coverage_mask(protein_length, peptide_positions)
    coverage = round(coverage_mask.sum() / protein_length * 100, ndigits)
    return coverage
