from collections import defaultdict as ddict
from typing import Optional, Union

import numpy as np


class Peptide:
    """Representation of a peptide sequence identified by mass spectrometry."""

    def __init__(
        self,
        modified_sequence: str,
        localization_probabilities: Optional[dict] = None,
        protein_position: Optional[int] = None,
    ):
        plain_sequence, modifications = parse_modified_sequence(
            modified_sequence, "[", "]"
        )

        self.plain_sequence = plain_sequence
        self.modified_sequence = modified_sequence
        self.localization_probabilities = localization_probabilities
        self.protein_position = protein_position

        self.modification_positions = ddict(list)
        self.modified_residues = {}
        for position, mod_tag in modifications:
            self.modification_positions[mod_tag].append(position)
            self.modified_residues[position] = mod_tag

    def make_modified_sequence(self, include: Optional[list] = None) -> str:
        """Returns a modified sequence string.

        Args:
            include: Optional, list of modifications that are included in the modified
                sequence string. By default all modifications are added.

        Returns:
            A modified sequence string where modified amino acids are indicated by
            square brackets containing a modification tag. For example
            "PEPT[phospho]IDE"
        """
        if include is None:
            return self.modified_sequence

        selected_modifications = []
        for position, mod_tag in self.modified_residues.items():
            if mod_tag in include:
                selected_modifications.append((position, mod_tag))
        return modify_peptide(self.plain_sequence, selected_modifications)

    def count_modification(self, modification: str) -> int:
        """Returns how often the a specified modification occurs."""
        if modification not in self.modification_positions:
            return 0
        return len(self.modification_positions[modification])

    def isoform_probability(self, modification: str) -> Union[float, None]:
        """Calculates the isoform probability for a given modification.

        Returns:
            The isoform probability for the combination of the assigned modification
            sites. Calculated as the product of the single modification localization
            probabilities. If no localization exist for the specified 'modification',
            None is returned.
        """
        probabilities = []
        for site in self.list_modified_peptide_sites(modification):
            probabilities.append(self.get_peptide_site_probability(site))
        if None in probabilities:
            return None
        return float(np.prod(probabilities))

    def get_peptide_site_probability(self, position: int) -> Optional[float]:
        """Return the modification localization probability of the peptide position.

        Args:
            position: Peptide position which modification localization probability is
                returned.

        Returns:
            Localization probability between 0 and 1. Returns None if the specified
            position does not contain a modification or if no localization probability
            is available.
        """
        return self._get_site_probability(position, is_protein_position=False)

    def get_protein_site_probability(self, position: int) -> Optional[float]:
        """Return the modification localization probability of the protein position.

        Args:
            position: Protein position which modification localization probability is
                returned.

        Returns:
            Localization probability between 0 and 1. Returns None if the specified
            position does not contain a modification or if no localization probability
            is available.
        """
        return self._get_site_probability(position, is_protein_position=True)

    def list_modified_peptide_sites(self, modification: str) -> list[int]:
        """Returns a list of peptide positions containing the specified modification."""
        return self._list_modified_sites(modification, use_protein_position=False)

    def list_modified_protein_sites(self, modification: str) -> list[int]:
        """Returns a list of protein positions containing the specified modification."""
        return self._list_modified_sites(modification, use_protein_position=True)

    def _get_site_probability(
        self, position: int, is_protein_position: bool
    ) -> Optional[float]:
        """Return the modification localization probability of the peptide position.

        Args:
            position: Position which modification localization probability is returned.
            is_protein_position: If True, the specified position is a protein position,
                if False its a peptide position.

        Returns:
            Localization probability between 0 and 1. Returns None if the specified
            position does not contain a modification or if no localization probability
            is available.
        """
        if is_protein_position and self.protein_position is not None:
            position = position - self.protein_position + 1

        if self.localization_probabilities is None:
            return None
        if position not in self.modified_residues:
            return None

        modification = self.modified_residues[position]
        try:
            probability = self.localization_probabilities[modification][position]
        except KeyError:
            probability = None
        return probability

    def _list_modified_sites(
        self, modification: str, use_protein_position: bool
    ) -> list[int]:
        """Returns a list of positions containint the specified modification.

        Args:
            modification: Sites containing this modification are extracted.
            use_protein_position: If True, the returned sites are protein positions and
                if False, peptide positions are returnd.

        Returns:
            A list of modified positions
        """
        if modification not in self.modification_positions:
            return []

        modified_sites = self.modification_positions[modification]
        if use_protein_position and self.protein_position is not None:
            modified_sites = [i + self.protein_position - 1 for i in modified_sites]
        return modified_sites


def parse_modified_sequence(
    modified_sequence: str,
    tag_open: str,
    tag_close: str,
) -> tuple[str, list[tuple[int, str]]]:
    """Returns the plain sequence and a list of modification positions and tags.

    Args:
        modified_sequence: Peptide sequence containing modifications.
        tag_open: Symbol that indicates the beginning of a modification tag, e.g. "[".
        tag_close: Symbol that indicates the end of a modification tag, e.g. "]".

    Returns:
        A tuple containing the plain sequence as a string and a sorted list of
        modification tuples, each containing the position and modification tag
        (excluding the tag_open and tag_close symbols).
    """
    start_counter = 0
    tags = []
    plain_sequence = ""
    for position, char in enumerate(modified_sequence):
        if char == tag_open:
            start_counter += 1
            if start_counter == 1:
                start_position = position
        elif char == tag_close:
            start_counter -= 1
            if start_counter == 0:
                tags.append((start_position, position))
        elif start_counter == 0:
            plain_sequence += char

    modifications = []
    last_position = 0
    for tag_start, tag_end in tags:
        mod_position = tag_start - last_position
        modification = modified_sequence[tag_start + 1 : tag_end]
        modifications.append((mod_position, modification))
        last_position += tag_end - tag_start + 1
    return plain_sequence, sorted(modifications)


def modify_peptide(
    sequence: str,
    modifications: list[tuple[int, str]],
    tag_open: str = "[",
    tag_close: str = "]",
) -> str:
    """Returns a string containing the modifications within the peptide sequence.

    Returns:
        Modified sequence. For example "PEPT[phospho]IDE", for sequence = "PEPTIDE" and
        modifications = [(4, "phospho")]
    """
    last_pos = 0
    modified_sequence = ""
    for pos, mod in sorted(modifications):
        tag = mod.join((tag_open, tag_close))
        modified_sequence += sequence[last_pos:pos] + tag
        last_pos = pos
    modified_sequence += sequence[last_pos:]
    return modified_sequence


def make_localization_string(
    localization_probabilities: dict, decimal_places: int = 3
) -> str:
    """Generates a site localization probability string.

    Args:
        localization_probabilities: A dictionary in the form
            {"modification tag": {position: probability}}, where positions are integers
            and probabilitiesa are floats ranging from 0 to 1.
        decimal_places: Number of decimal places used for the probabilities, default 3.

    Returns:
            A site localization probability string according to the MsReport convention.
            Multiple modifications entries are separted by ";". Each modification entry
            consist of a modification tag and site probabilities, separated by "@". The
            site probability entries consist of f"{position}:{probability}" strings, and
            multiple probability entries are separted by ",".

            For example "15.9949@11:1.000;79.9663@3:0.200,4:0.800"
    """
    modification_strings = []
    for modification, probabilities in localization_probabilities.items():
        localization_strings = []
        for position, probability in probabilities.items():
            probability_string = f"{probability:.{decimal_places}f}"
            localization_strings.append(f"{position}:{probability_string}")
        localization_string = ",".join(localization_strings)
        modification_strings.append(f"{modification}@{localization_string}")
    localization_string = ";".join(modification_strings)
    return localization_string


def read_localization_string(localization_string: str) -> dict[str, dict[int, float]]:
    """Converts a site localization probability string into a dictionary.

    Args:
        localization_string: A site localization probability string according to the
            MsReport convention. Can contain information about multiple modifications,
            which are separted by ";". Each modification entry consist of a modification
            tag and site probabilities, separated by "@". The site probability entries
            consist of f"{peptide position}:{localization probability}" strings, and
            multiple entries are separted by ",".
            For example "15.9949@11:1.000;79.9663@3:0.200,4:0.800"

    Returns:
        A dictionary in the form {"modification tag": {position: probability}}, where
        positions are integers and probabilitiesa are floats ranging from 0 to 1.
    """
    localization: dict[str, dict[int, float]] = {}
    if localization_string == "":
        return localization

    for modification_entry in localization_string.split(";"):
        modification, site_entries = modification_entry.split("@")
        site_probabilities = {}
        for site_entry in site_entries.split(","):
            position, probability = site_entry.split(":")
            site_probabilities[int(position)] = float(probability)
        localization[modification] = site_probabilities
    return localization
