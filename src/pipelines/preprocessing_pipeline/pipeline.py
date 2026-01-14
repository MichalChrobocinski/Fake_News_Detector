from .nodes import *
from src.load_config import load_config
import joblib
import os

def run_preprocessing(
    output_dir: str = "data/02_primary/"
):
    cfg = load_config()
    paths = cfg["paths"]

    fake_df = load_fake_data(paths["raw_fake"])
    true_df = load_real_data(paths["raw_true"])

    fake_df = add_label(fake_df, label=0)
    true_df = add_label(true_df, label=1)

    df = merge_datasets(fake_df, true_df)

    df = drop_unused_columns(df)

    df = apply_text_cleaning(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer = tfidf_vectorize(
        X_train, X_val, X_test)

    joblib.dump(X_train_tfidf, os.path.join(output_dir, "X_train_tfidf.pkl"))
    joblib.dump(X_val_tfidf, os.path.join(output_dir, "X_val_tfidf.pkl"))
    joblib.dump(X_test_tfidf, os.path.join(output_dir, "X_test_tfidf.pkl"))

    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_val.to_csv(os.path.join(output_dir, "y_val.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    joblib.dump(vectorizer, os.path.join(output_dir, "tfidf_vectorizer.pkl"))
