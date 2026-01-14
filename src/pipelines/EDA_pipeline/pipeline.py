import pandas as pd
from pathlib import Path

from src.load_config import load_config
from .nodes import (
    load_fake_data,
    load_real_data,
    dataset_overview,
    null_summary,
    add_length_features,
    length_statistics,
    plot_length_boxplot,
    plot_length_histogram,
    subject_distribution,
    subject_distribution_by_label
)


def run_eda():
    cfg = load_config()
    paths = cfg["paths"]

    fake = load_fake_data(paths["raw_fake"])
    true = load_real_data(paths["raw_true"])

    fake["label"] = 0
    true["label"] = 1

    df = pd.concat([fake, true], ignore_index=True)

    overview = dataset_overview(df)
    nulls = null_summary(df)

    print("=== DATASET OVERVIEW ===")
    print(overview)
    print("\n=== NULL SUMMARY ===")
    print(nulls)

    df = add_length_features(df)

    text_len_stats = length_statistics(df, "text_len")
    title_len_stats = length_statistics(df, "title_len")

    print("\n=== TEXT LENGTH STATS ===")
    print(text_len_stats)

    print("\n=== TITLE LENGTH STATS ===")
    print(title_len_stats)

    subjects = subject_distribution(df)
    print("\n=== SUBJECT DISTRIBUTION ===")
    print(subjects.head(10))

    subjects_by_label = subject_distribution_by_label(df)
    print("\n=== SUBJECT DISTRIBUTION BY LABEL ===")
    print(subjects_by_label)

    output_dir = Path("docs/EDA")

    plot_length_boxplot(
        df,
        column="text_len",
        output_dir=output_dir,
        filename="text_length_boxplot.png",
    )

    plot_length_boxplot(
        df,
        column="title_len",
        output_dir=output_dir,
        filename="title_length_boxplot.png",
    )

    plot_length_histogram(
        df,
        column="text_len",
        label=1,
        output_dir=output_dir,
        filename="fake_text_length_hist.png",
    )

    plot_length_histogram(
        df,
        column="text_len",
        label=0,
        output_dir=output_dir,
        filename="true_text_length_hist.png",
    )




