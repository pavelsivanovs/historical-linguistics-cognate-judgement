from pairwise_comparison import NONE_COGNATE_GROUP
from utils import df2tsv, get_data

import os
import sys

import bcubed
import pandas as pd


def convert_number_to_set(x: int):
    a = set()
    a.add(str(x))
    return a


def get_bcubed_metrics(source_row, gold_row) -> tuple[float, float, float]:
    # bcubed library needs dict values to be sets
    source_dict = (
        source_row[source_row != NONE_COGNATE_GROUP]
        .apply(lambda x: convert_number_to_set(x))
        .to_dict()
    )
    gold_dict = (
        gold_row[gold_row != NONE_COGNATE_GROUP]
        .apply(lambda x: convert_number_to_set(x))
        .to_dict()
    )

    precision = bcubed.precision(source_dict, gold_dict)
    recall = bcubed.recall(source_dict, gold_dict)
    fscore = bcubed.fscore(precision, recall)
    return precision, recall, fscore


def evaluate(source, gold):
    rows = []
    precisions, recalls, fscores = [], [], []

    for source_tuple, gold_tuple in zip(source.itertuples(), gold.itertuples()):
        source_row = pd.Series(source_tuple._asdict())
        gold_row = pd.Series(gold_tuple._asdict())

        non_cognate_cols = ["Index", "sense"]
        source_row = source_row.drop(labels=non_cognate_cols)
        gold_row = gold_row.drop(labels=non_cognate_cols)

        precision, recall, fscore = get_bcubed_metrics(source_row, gold_row)
        precisions.append(precision)
        recalls.append(recall)
        fscores.append(fscore)

        row = pd.Series({"sense": source_tuple.sense})
        row = pd.concat(
            [
                row,
                source_row.add_prefix("source_"),
                gold_row.add_prefix("gold_"),
            ]
        )
        row["precision"] = precision
        row["recall"] = recall
        row["fscore"] = fscore
        rows.append(row)

    source_cols = source.drop(columns=["sense"]).add_prefix("source_").columns
    gold_cols = gold.drop(columns=["sense"]).add_prefix("gold_").columns
    columns = [
        "sense",
        *source_cols,
        *gold_cols,
        "precision",
        "recall",
        "fscore",
    ]
    evaluation_df = pd.DataFrame(rows, columns=columns)

    average_precision = sum(precisions) / len(precisions)
    average_recall = sum(recalls) / len(recalls)
    average_fscore = sum(fscores) / len(fscores)

    return (average_precision, average_recall, average_fscore), evaluation_df


if __name__ == "__main__":
    source_filepath = sys.argv[1]
    gold_filepath = sys.argv[2]
    source = get_data(source_filepath)
    gold = get_data(gold_filepath)
    (p, r, f1), evaluation = evaluate(source, gold)

    print(f"Average precision: {p:.2%}")
    print(f"Average recall: {r:.2%}")
    print(f"Average F-score: {f1:.2%}")

    directory = os.path.dirname(source_filepath)
    filename = os.path.basename(source_filepath)
    evaluation_filepath = os.path.join(directory, f"evaluation_{filename}")
    print(f"Printing evaluation data into the file {evaluation_filepath}")
    df2tsv(evaluation, evaluation_filepath)
