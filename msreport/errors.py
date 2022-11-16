class MsreportError(Exception):
    ...


class ProteinsNotInFastaWarning(UserWarning):
    """Warning raised when queried proteins are absent from a FASTA file."""
