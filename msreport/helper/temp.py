from typing import Iterable, Union
import re
import warnings

import pathlib
import pyteomics.fasta


class Protein:
    def __init__(self, sequence, header, info):
        self.sequence = sequence
        self.fastaHeader = header
        self.headerInfo = info

    def length(self):
        return len(self.sequence)


class ProteinDatabase:
    def __init__(self):
        self.proteins = {}
        self._fasta_parsers = _get_updated_fasta_parsers()

    def add_fasta(self, fasta_path):
        for header, sequence in parse_fasta(fasta_path):
            self.add_fasta_entry(header, sequence)

    def add_fasta_entry(self, header, sequence):
        header_info = pyteomics.fasta.parse(header, parsers=self._fasta_parsers)
        self.proteins[header_info["id"]] = Protein(sequence, header, header_info)

    def __getitem__(self, key: str):
        """Evaluation of self.proteins[key]"""
        return self.proteins[key]

    def __setitem__(self, key: str, value: Protein):
        """Item assignment of self.proteins[key]"""
        self.proteins[key] = value

    def __contains__(self, key: str) -> bool:
        """True if key is in the info axis of self.proteins"""
        return key in self.proteins


class SimpleUniProtLikeMixin(pyteomics.fasta.FlavoredMixin):
    header_pattern = r"^(\w+)\|([-\w]+)\|([-\w]+)"
    header_group = 2

    def parser(self, header):
        db, ID, entry = re.match(self.header_pattern, header).groups()
        info = {"db": db, "id": ID, "entry": entry}
        try:
            gid, taxon = entry.split("_")
            info.update({"gene_id": gid, "taxon": taxon})
        except ValueError:
            pass
        return info


def parse_fasta(fasta_path):
    with open(fasta_path, "r") as file:
        fasta_text = file.read()

    if not fasta_text.startswith("\n"):
        fasta_text = "\n" + fasta_text
    for block in fasta_text.split("\n>")[1:]:
        lines = block.split("\n")
        header = lines[0].strip()
        sequence = "".join(lines[1:]).replace(" ", "")
        yield header, sequence


def import_protein_database(
    fasta_path: Union[str, pathlib.Path, Iterable[Union[str, pathlib.Path]]],
) -> ProteinDatabase:
    """Generates a protein database from one or a list of fasta files.

    Args:
        fasta_path: Path to a fasta file, or a list of paths. The path can be either a
            string or a pathlib.Path instance.

    Returns:
        A protein database containing entries from the parsed fasta files.
    """
    database = ProteinDatabase()
    paths = [fasta_path] if isinstance(fasta_path, (str, pathlib.Path)) else fasta_path
    for path in paths:
        database.add_fasta(path)
    return database


def importProteinDatabase(
    fasta_path: Union[str, pathlib.Path, Iterable[Union[str, pathlib.Path]]],
) -> ProteinDatabase:
    """Generates a protein database from one or a list of fasta files."""
    warnings.warn(
        "This function is deprecated, use import_protein_database instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return import_protein_database(fasta_path)


def extract_modifications(
    peptide: str,
    tag_open: str,
    tag_close: str,
) -> list[tuple[int, str]]:
    """Returns a list of modification positions and strings.

    Args:
        peptide: Peptide sequence containing modifications
        tag_open: Symbol that indicates the beginning of a modification tag, e.g. "[".
        tag_close: Symbol that indicates the end of a modification tag, e.g. "]".

    Returns:
        A sorted list of modification tuples, containing position and modification
        string (excluding the tag_open and tag_close strings).
    """
    start_counter = 0
    tags = []
    for position, char in enumerate(peptide):
        if char == tag_open:
            start_counter += 1
            if start_counter == 1:
                start_position = position
        elif char == tag_close:
            start_counter -= 1
            if start_counter == 0:
                tags.append((start_position, position))

    modifications = []
    last_position = 0
    for tag_start, tag_end in tags:
        mod_position = tag_start - last_position
        modification = peptide[tag_start + 1 : tag_end]
        modifications.append((mod_position, modification))
        last_position += tag_end - tag_start + 1
    return sorted(modifications)


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


def extract_window_around_position(protein_sequence: str, position: int) -> str:
    """Extracts a window around the specified position in the protein sequence.

    Args:
        protein_sequence: The input protein sequence string.
        position: The position in the protein sequence to extract the window around.
            Position is one-indexed, which means that the first amino acid position 1.

    Returns:
        A string containing the window +/- 5 characters around the specified position.
        If the position is too close to the beginning or the end of the
        'protein_sequence', the window is padded with '-' to ensure there are five
        characters before and after the position.

    Example:
        >>> protein_sequence = "ABCDEFGHIJKLM"
        >>> extract_window_around_position(protein_sequence, 7)
        'BCDEFGHIJKL'
        >>> extract_window_around_position(protein_sequence, 1)
        '-----ABCDEF'
        >>> extract_window_around_position(protein_sequence, 13)
        'HIJKLM-----'
    """
    # TODO: Not tested
    extension = 5
    ond_index_correction = -1
    _position = position + ond_index_correction
    gap_filler = "-"

    gap_to_end = len(protein_sequence) - (_position + 1)
    gap_to_start = _position
    left_pad = extension - gap_to_start if gap_to_start < extension else 0
    left_right = extension - gap_to_end if gap_to_end < extension else 0

    window_start = max(_position - extension, 0)
    window_end = min(_position + extension, len(protein_sequence))
    window = protein_sequence[window_start : window_end + 1]
    window = "".join([gap_filler * left_pad, window, gap_filler * left_right])
    return window


def _get_updated_fasta_parsers() -> dict[str, pyteomics.fasta.FlavoredMixin]:
    """Returns a dictionary of fasta header parsers.

    Copies the _std_mixins fasta parser dictionary from pyteomcis and adds the
    SimpleUniProtLikeMixin parser.
    """
    parsers = pyteomics.fasta._std_mixins.copy()
    parsers["simple"] = SimpleUniProtLikeMixin
    return parsers
