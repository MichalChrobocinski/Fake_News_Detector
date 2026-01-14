import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from src.load_config import load_config

def train():
    cfg = load_config()
    paths = cfg["paths"]

    MODEL_DIR = "data/03_model/"
    EVAL_DIR = "data/04_evaluation/"

    X_train = joblib.load(paths["primary_train"])
    X_test = joblib.load(paths["primary_test"])

    y_train = pd.read_csv(paths["primary_train_pred"]).values.ravel()
    y_test = pd.read_csv(paths["primary_test_pred"]).values.ravel()

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Logistic Regression accuracy:", acc)
    print(report)

    joblib.dump(model, MODEL_DIR + "logistic_regression.pkl")

    with open(EVAL_DIR + "logistic_regression.txt", "w") as f:
        f.write(f"Accuracy: {acc}\n\n")
        f.write(report)

    # === TEST EWALUACYJNY, NIE JEST POTRZEBNY ALE PRZYDA SIE W DOKUMENTACJI ===
    DATA_DIR = "data/02_primary/"
    MODEL_DIR = "data/03_model/"

    X_val = joblib.load(DATA_DIR + "X_val_tfidf.pkl")
    X_test = joblib.load(DATA_DIR + "X_test_tfidf.pkl")

    y_val = pd.read_csv(DATA_DIR + "y_val.csv").values.ravel()
    y_test = pd.read_csv(DATA_DIR + "y_test.csv").values.ravel()

    model = joblib.load(MODEL_DIR + "logistic_regression.pkl")
    vectorizer = joblib.load(DATA_DIR + "tfidf_vectorizer.pkl")

    y_val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print("Validation Accuracy:", val_acc)
    print("\nClassification Report (Validation):\n", classification_report(y_val, y_val_pred))
    print("Confusion Matrix (Validation):\n", confusion_matrix(y_val, y_val_pred))

    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print("\nTest Accuracy:", test_acc)
    print("\nClassification Report (Test):\n", classification_report(y_test, y_test_pred))
    print("Confusion Matrix (Test):\n", confusion_matrix(y_test, y_test_pred))

    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]  # dla klasy 1 vs 0

    # 20 najbardziej True news
    top_true_idx = np.argsort(coefs)[-20:]
    top_true_words = feature_names[top_true_idx]

    # 20 najbardziej Fake news
    top_fake_idx = np.argsort(coefs)[:20]
    top_fake_words = feature_names[top_fake_idx]

    print("\nTop 20 True news words:", top_true_words)
    print("Top 20 Fake news words:", top_fake_words)

