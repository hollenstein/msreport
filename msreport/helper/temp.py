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
