import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def load_fake_data(path):
    return pd.read_csv(path)

def load_real_data(path):
    return pd.read_csv(path)

def add_label(df: pd.DataFrame, label: int) -> pd.DataFrame:
    df = df.copy()
    df["label"] = label
    return df

def merge_datasets(fake_df: pd.DataFrame, true_df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([fake_df, true_df], axis=0, ignore_index=True)

def drop_unused_columns(
    df: pd.DataFrame,
    columns_to_drop: list = ["title", "subject", "date"]
) -> pd.DataFrame:
    df = df.copy()
    return df.drop(columns=columns_to_drop, errors="ignore")

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def apply_text_cleaning(
    df: pd.DataFrame,
    text_column: str = "text"
) -> pd.DataFrame:
    df = df.copy()
    df[text_column] = df[text_column].astype(str).apply(clean_text)
    return df

def split_data(
    df: pd.DataFrame,
    text_column: str = "text",
    label_column: str = "label",
    random_state: int = 42
):
    X = df[text_column]
    y = df[label_column]

    # 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        stratify=y,
        random_state=random_state
    )

    # z 30% robimy 15% val i 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def tfidf_vectorize(
    X_train,
    X_val,
    X_test,
    max_features: int = 5000
):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer
