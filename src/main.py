from matplotlib.ticker import PercentFormatter
from tqdm import tqdm
from evaluate import evaluate
from pairwise_comparison import (
    get_cognates,
    relative_distance,
)
from utils import df2tsv, get_data

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from panphon.distance import Distance


def test_every_cent(forms, gold, get_relative_distance):
    precisions, recalls, fscores = [], [], []
    best_threshold, best_fscore = 0, 0
    for threshold in tqdm(np.arange(0.01, 1, 0.01)):
        estimated_cognates = get_cognates(
            forms, get_relative_distance, threshold=threshold
        )
        (p, r, f1), _ = evaluate(estimated_cognates, gold)

        if f1 > best_fscore:
            best_fscore = f1
            best_threshold = threshold

        precisions.append(p)
        recalls.append(r)
        fscores.append(f1)
    return precisions, recalls, fscores, best_threshold


def make_plot(precisions, recalls, fscores, best_threshold):
    x = np.arange(0.01, 1, 0.01)

    plt.figure(figsize=(8, 6))
    plt.plot(x, precisions, label="Precision")
    plt.plot(x, recalls, label="Recall")
    plt.plot(x, fscores, label="F-score")

    plt.axvline(
        best_threshold,
        color="red",
        linestyle="--",
        label=f"Best threshold: {best_threshold:.2f}",
    )

    plt.xlabel("Threshold")
    plt.ylabel("Percentage")
    plt.title(
        "Precision, Recall, F-score vs Threshold for EA Data \n Using Relative Edit Distance with Phonetic Features"
    )
    plt.legend()

    plt.tight_layout()
    plt.savefig("ea_phonetic_feature_distance.png")
    plt.close()


# def make_distribution_plot(precisions, recalls, fscores, title, path):
def make_distribution_plot(fscores, title, path):
    plt.figure(figsize=(8, 4))

    xbins = np.arange(0, 1.01, 0.2)

    plt.hist(
        fscores,
        bins=xbins,
        density=True,
        histtype="bar",
        label=["F-score"],
        color=["green"],
    )

    plt.gca().xaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=10))

    plt.title(title)
    # plt.legend(prop={"size": 15})
    # plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


if __name__ == "__main__":
    # forms = get_data(sys.argv[1])
    # gold = get_data(sys.argv[2])

    # dst = Distance()
    # relative_edit_distance = relative_distance(
    #     dst.hamming_feature_edit_distance
    # )
    # precisions, recalls, fscores, best_threshold = test_every_cent(
    #     forms, gold, relative_edit_distance
    # )
    # make_plot(precisions, recalls, fscores, best_threshold)

    # estimated_cognates = get_cognates(
    #     forms, relative_edit_distance, threshold=best_threshold
    # )
    # (p, r, f1), evaluation = evaluate(estimated_cognates, gold)
    # print(f"Precision: {p:.2%}; Recall: {r:.2%}; F-score: {f1:.2%}")
    # df2tsv(evaluation, f"ea_data_phonetic_feature_{best_threshold}.tsv")

    evaluation_file = sys.argv[1]
    evaluation = get_data(evaluation_file)

    precisions = evaluation["precision"]
    recalls = evaluation["recall"]
    fscores = evaluation["fscore"]

    make_distribution_plot(
        # precisions,
        # recalls,
        fscores,
        title=f"F-score Distribution for EA Data Using Relative Edit Distance \n with Phonetic Features (threshold=0.09; n={len(fscores)})",
        path="img/ea_phonetic_features_distribution.png",
    )
