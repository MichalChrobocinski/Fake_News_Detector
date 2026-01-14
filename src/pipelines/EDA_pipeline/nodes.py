import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_fake_data(path):
    return pd.read_csv(path)

def load_real_data(path):
    return pd.read_csv(path)

def dataset_overview(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }

def null_summary(df: pd.DataFrame) -> pd.Series:
    return df.isnull().sum()

def add_length_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text_len"] = df["text"].astype(str).str.len()
    df["title_len"] = df["title"].astype(str).str.len()
    df["text_word_count"] = df["text"].astype(str).str.split().str.len()
    return df

def length_statistics(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return df.groupby("label")[column].describe()

def plot_length_boxplot(
    df: pd.DataFrame,
    column: str,
    output_dir: Path,
    filename: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    df.boxplot(column=column, by="label")
    plt.title(f"{column} by class (0=True, 1=Fake)")
    plt.suptitle("")
    plt.ylabel(column)
    plt.savefig(output_dir / filename)
    plt.close()

def plot_length_histogram(
    df: pd.DataFrame,
    column: str,
    label: int,
    output_dir: Path,
    filename: str,
    bins: int = 50,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    df[df["label"] == label][column].hist(bins=bins)
    plt.title(f"{column} distribution | label={label}")
    plt.xlabel(column)
    plt.ylabel("count")
    plt.savefig(output_dir / filename)
    plt.close()

def subject_distribution(df: pd.DataFrame) -> pd.Series:
    return df["subject"].value_counts()

def subject_distribution_by_label(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["label", "subject"])
          .size()
          .reset_index(name="count")
          .sort_values(["label", "count"], ascending=[True, False])
    )



