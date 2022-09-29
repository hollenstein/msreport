from typing import Optional

import pyteomics.fasta


class ProteinDatabase:
    def __init__(self):
        self.proteins = {}

    def add_fasta(self, fasta_path):
        for header, sequence in parse_fasta(fasta_path):
            header_info = pyteomics.fasta.parse(header)
            self.proteins[header_info["id"]] = Protein(sequence, header, header_info)

    def __getitem__(self, key):
        return self.proteins[key]


class Protein:
    def __init__(self, sequence, header, info):
        self.sequence = sequence
        self.fastaHeader = header
        self.headerInfo = info

    def length(self):
        return len(self.sequence)


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


def importProteinDatabase(
    fasta_path: str,
    database: Optional[ProteinDatabase] = None,
) -> ProteinDatabase:
    if database is None:
        database = ProteinDatabase()
    database.add_fasta(fasta_path)
    return database


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
        A list of modification tuples, containing position and modification string
        (excluding the tag_open and tag_close strings).
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
    return modifications


def modify_peptide(
    sequence: str,
    modifications: list[tuple[int, str]],
    tag_open: str = "[",
    tag_close: str = "]",
) -> str:
    """Returns a string containing the modifications within the peptide sequence.

    Returns:
        Modified sequence, e.g. "PEPT[phospho]IDE" for sequence = "PEPTIDE" and
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
