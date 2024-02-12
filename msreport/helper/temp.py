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
