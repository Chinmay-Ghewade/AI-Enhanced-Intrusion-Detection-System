"""
train.py
--------
Milestone 4: Model Selection and Training.

Trains several classic ML algorithms on the preprocessed NSL-KDD data,
compares them with cross-validation, and saves the best-performing
model to models/best_model.pkl.

Run:
    python src/train.py
"""

import os
import time
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from constants import RANDOM_STATE
from preprocessing import build_datasets

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")


MODELS = {
    "logistic_regression": LogisticRegression(
        max_iter=500, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "decision_tree": DecisionTreeClassifier(
        max_depth=20, random_state=RANDOM_STATE
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=100, max_depth=20, random_state=RANDOM_STATE, n_jobs=-1
    ),
}


def main():
    train_path = os.path.join(DATA_DIR, "KDDTrain+.txt")
    test_path = os.path.join(DATA_DIR, "KDDTest+.txt")

    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, encoders, scaler, feature_cols = \
        build_datasets(train_path, test_path)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(encoders, os.path.join(MODELS_DIR, "label_encoders.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_columns.pkl"))

    results = []
    best_model_name, best_model, best_score = None, None, -1

    for name, model in MODELS.items():
        print(f"\nTraining {name} ...")
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring="f1_macro")
        mean_cv = cv_scores.mean()
        train_acc = model.score(X_train, y_train)

        print(f"  train_time={train_time:.1f}s  train_acc={train_acc:.4f}  "
              f"cv_f1_macro={mean_cv:.4f}")

        results.append({
            "model": name, "train_time_sec": round(train_time, 2),
            "train_accuracy": round(train_acc, 4),
            "cv_f1_macro": round(mean_cv, 4),
        })

        joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.pkl"))

        if mean_cv > best_score:
            best_score, best_model_name, best_model = mean_cv, name, model

    pd.DataFrame(results).to_csv(
        os.path.join(MODELS_DIR, "training_results.csv"), index=False)

    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    with open(os.path.join(MODELS_DIR, "best_model.txt"), "w") as f:
        f.write(best_model_name)

    print(f"\nBest model: {best_model_name} (cv_f1_macro={best_score:.4f})")
    print("Saved all models + training_results.csv to:", MODELS_DIR)


if __name__ == "__main__":
    main()
