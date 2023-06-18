def parse_modified_sequence(
    modified_sequence: str,
    tag_open: str,
    tag_close: str,
) -> tuple[str, list]:
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
            site probability entries consist of
            f"{peptide position}:{localization probability}" strings, and multiple
            entries are separted by ",".

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


def read_localization_string(localization_string: str) -> dict:
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
    localization = {}
    for modification_entry in localization_string.split(";"):
        modification, site_entries = modification_entry.split("@")
        site_probabilities = {}
        for site_entry in site_entries.split(","):
            position, probability = site_entry.split(":")
            site_probabilities[int(position)] = float(probability)
        localization[modification] = site_probabilities
    return localization
