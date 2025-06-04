import numpy as np
import pandas as pd


def get_data(filename: str) -> pd.DataFrame:
    """
    Read the data from the forms TSV file.
    """
    df = pd.read_csv(filename, delimiter="\t")
    df.replace({np.nan: None}, inplace=True)
    return df


def df2tsv(df: pd.DataFrame, path: str):
    df.to_csv(path, sep="\t", index=False)
