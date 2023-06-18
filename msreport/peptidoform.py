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
