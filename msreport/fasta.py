"""Functionalities for import and access to protein sequence databases from FASTA files.

This module serves as an interface to the `profasta` library, offering a convenient way
to generate a `profasta.db.ProteinDatabase` from one or multiple FASTA files. It
supports custom FASTA header parsing through a configurable header parser.
"""

import pathlib
from typing import Iterable

from profasta.db import ProteinDatabase


def import_protein_database(
    fasta_path: str | pathlib.Path | Iterable[str | pathlib.Path],
    header_parser: str = "uniprot",
) -> ProteinDatabase:
    """Generates a protein database from one or a list of fasta files.

    Args:
        fasta_path: Path to a fasta file, or a list of paths. The path can be either a
            string or a pathlib.Path instance.
        header_parser: Allows specifying the name of the parser to use for parsing the
            FASTA headers. The specified parser must be registered in the global parser
            registry. By default a strict uniprot parser is used.

    Returns:
        A protein database containing entries from the parsed fasta files.
    """
    database = ProteinDatabase()
    paths = [fasta_path] if isinstance(fasta_path, (str, pathlib.Path)) else fasta_path
    for path in paths:
        if isinstance(path, pathlib.Path):
            path = path.as_posix()
        database.add_fasta(path, header_parser=header_parser, overwrite=True)
    return database
