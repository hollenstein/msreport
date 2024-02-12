import pyteomics.fasta
import re
import pathlib
from typing import Union, Iterable


class Protein:
    def __init__(self, sequence, header, info):
        self.sequence = sequence
        self.fastaHeader = header
        self.headerInfo = info


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


def _get_updated_fasta_parsers() -> dict[str, pyteomics.fasta.FlavoredMixin]:
    """Returns a dictionary of fasta header parsers.

    Copies the _std_mixins fasta parser dictionary from pyteomcis and adds the
    SimpleUniProtLikeMixin parser.
    """
    parsers = pyteomics.fasta._std_mixins.copy()
    parsers["simple"] = SimpleUniProtLikeMixin
    return parsers
