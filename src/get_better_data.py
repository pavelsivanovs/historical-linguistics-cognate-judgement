from pairwise_comparison import NONE_COGNATE_GROUP

import sys

import pandas as pd


BAD_LANGUAGE = "Russian"


def get_better_data():
    forms_filepath = sys.argv[1]
    cognacy_filepath = sys.argv[2]
    forms = pd.read_csv(forms_filepath, delimiter="\t")
    cognacy = pd.read_csv(cognacy_filepath, delimiter="\t")

    if BAD_LANGUAGE in forms.columns:
        forms = forms.drop(columns=[BAD_LANGUAGE])
    if BAD_LANGUAGE in cognacy.columns:
        cognacy = cognacy.drop(columns=[BAD_LANGUAGE])

    assert cognacy.size == forms.size
    cognacy[forms.isna()] = NONE_COGNATE_GROUP

    unnamed_col = "Unnamed: 0"
    if unnamed_col in forms.columns:
        forms.rename(columns={"Unnamed: 0": "sense"}, inplace=True)
    if unnamed_col in cognacy.columns:
        cognacy.rename(columns={"Unnamed: 0": "sense"}, inplace=True)

    forms.to_csv(forms_filepath, sep="\t", index=False)
    cognacy.to_csv(cognacy_filepath, sep="\t", index=False)


if __name__ == "__main__":
    get_better_data()
