import sys
from typing import Any, Callable

import pandas as pd
from panphon.distance import Distance

from utils import df2tsv, get_data

# TODO: barbacone


NONE_COGNATE_GROUP = 0


def process_cognates(
    cognates: dict, lang1: str, lang2: str, current_group_number: int
) -> tuple[dict, int]:
    lang1_cognate_group = cognates.get(lang1)
    lang2_cognate_group = cognates.get(lang2)

    if lang1_cognate_group is None and lang2_cognate_group is None:
        cognates[lang1] = current_group_number
        cognates[lang2] = current_group_number
        current_group_number += 1
        return cognates, current_group_number
    if lang1_cognate_group is None:
        cognates[lang1] = cognates[lang2]
        return cognates, current_group_number
    if lang2_cognate_group is None:
        cognates[lang2] = cognates[lang1]
        return cognates, current_group_number

    for lang, cognate_group in cognates.items():
        if cognate_group == lang2_cognate_group:
            cognates[lang] = lang1_cognate_group
    return cognates, current_group_number


def process_non_cognates(
    cognates: dict, lang1: str, lang2: str, current_group_number: int
) -> tuple[dict, int]:
    lang1_cognate_group = cognates.get(lang1)
    lang2_cognate_group = cognates.get(lang2)

    if lang1_cognate_group is None:
        cognates[lang1] = current_group_number
        current_group_number += 1
    if lang2_cognate_group is None:
        cognates[lang2] = current_group_number
        current_group_number += 1
    return cognates, current_group_number


def get_cognate_groups(
    none_cognate_groups: dict, distances: dict[tuple, float], threshold: float
):
    group_number = 1
    cognates = {}
    for (lang1, lang2), distance in distances.items():
        if distance > threshold:
            cognates, group_number = process_non_cognates(
                cognates, lang1, lang2, group_number
            )
        else:
            cognates, group_number = process_cognates(
                cognates, lang1, lang2, group_number
            )
    return cognates | none_cognate_groups


def get_cognates(
    forms: pd.DataFrame, get_distance: Callable, threshold: float
) -> pd.Series | pd.DataFrame:
    """
    Return groups of cognates.

    :param forms: List of forms to analyze
    :param Callable get_distance: Function to calculate distance between forms
    :param float threshold: Threshold for cutting off cognate groups. If the
        distance is above the threshold, then put the form into a separate
        cognate group
    :return: The list of cognate groups. Cognate group is a list forms
    """
    estimated_cognate_groups = []
    for _, forms_row in forms.iterrows():
        row = forms_row.to_dict()
        sense = row.pop("sense")

        none_cognate_groups = {}

        distances = {}
        for lang1, form1 in row.items():
            for lang2, form2 in row.items():
                if form1 is None:
                    none_cognate_groups[lang1] = NONE_COGNATE_GROUP
                if form2 is None:
                    none_cognate_groups[lang2] = NONE_COGNATE_GROUP

                if (
                    (form1 is None or form2 is None)
                    or lang1 == lang2
                    or (lang1, lang2) in distances.keys()
                    or (lang2, lang1) in distances.keys()
                ):
                    continue

                distances[(lang1, lang2)] = get_distance(form1, form2)

        cognate_groups = get_cognate_groups(none_cognate_groups, distances, threshold)
        estimated_cognate_groups.append({"sense": sense} | cognate_groups)

    estimated_cognate_groups = pd.DataFrame(estimated_cognate_groups)
    estimated_cognate_groups = estimated_cognate_groups[forms.columns]
    return estimated_cognate_groups


def relative_distance(get_distance: Callable) -> Callable:
    def get_relative_distance(f1, f2):
        edit_distance = get_distance(f1, f2)
        longest_form_len = max(len(f1), len(f2))
        return edit_distance / longest_form_len

    return get_relative_distance


if __name__ == "__main__":
    forms_filename = sys.argv[1]
    destination_path = sys.argv[2]

    forms = get_data(forms_filename)
    dst = Distance()
    relative_edit_distance = relative_distance(dst.fast_levenshtein_distance)

    estimated_cognates = get_cognates(forms, relative_edit_distance, threshold=0.85)
    df2tsv(estimated_cognates, destination_path)
