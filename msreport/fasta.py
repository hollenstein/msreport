import pathlib
from typing import Iterable, Union


from profasta.db import ProteinDatabase


def import_protein_database(
    fasta_path: Union[str, pathlib.Path, Iterable[Union[str, pathlib.Path]]],
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
        database.add_fasta(path, header_parser=header_parser, overwrite=True)
    return database
