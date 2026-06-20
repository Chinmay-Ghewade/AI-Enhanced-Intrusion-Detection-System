"""
predict.py
----------
Milestone 6 & 7: Real-time Monitoring/Analysis and Automated Classification.

A minimal "real-time" scoring demo: loads the saved model + scaler +
encoders and classifies new network-traffic records (NSL-KDD format,
without the label/difficulty columns) as normal / DoS / Probe / R2L / U2R.

Usage:
    python src/predict.py                  # demo on a few sample rows
    python src/predict.py path/to/file.csv  # score every row in a csv
"""

import os
import sys
import joblib
import pandas as pd

from constants import COLUMN_NAMES, CATEGORICAL_COLUMNS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

FEATURE_COLUMN_NAMES = COLUMN_NAMES[:-2]  # drop label + difficulty


def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    encoders = joblib.load(os.path.join(MODELS_DIR, "label_encoders.pkl"))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
    return model, scaler, encoders, feature_cols


def score(df: pd.DataFrame) -> pd.Series:
    """Classify raw traffic rows (must contain the 41 NSL-KDD feature
    columns). Returns a Series of predicted attack_category labels."""
    model, scaler, encoders, feature_cols = load_artifacts()

    df = df.copy()
    for col in CATEGORICAL_COLUMNS:
        df[col] = encoders[col].transform(df[col].astype(str))

    X = scaler.transform(df[feature_cols])
    preds = model.predict(X)
    return pd.Series(preds, name="predicted_attack_category")


def main():
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        df = pd.read_csv(input_path, header=None, names=FEATURE_COLUMN_NAMES)
    else:
        # Demo: grab a handful of rows straight out of the test set.
        test_path = os.path.join(DATA_DIR, "KDDTest+.txt")
        df = pd.read_csv(test_path, header=None, names=COLUMN_NAMES).head(10)
        df = df[FEATURE_COLUMN_NAMES]
        print("(No input file given — running demo on 10 sample rows from "
              "KDDTest+.txt)\n")

    predictions = score(df)
    result = df.assign(predicted_attack_category=predictions)
    print(result[["protocol_type", "service", "flag",
                   "predicted_attack_category"]].to_string(index=False))


if __name__ == "__main__":
    main()
