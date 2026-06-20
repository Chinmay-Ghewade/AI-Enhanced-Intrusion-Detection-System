"""
preprocessing.py
-----------------
Milestone 2 & 3: Data Collection/Preparation and Feature Engineering.

Loads the raw NSL-KDD text files, cleans them, encodes categorical
features, maps the ~39 raw attack labels down to 5 categories
(normal, DoS, Probe, R2L, U2R), and scales numeric features.

Run directly to generate processed train/test CSVs in data/processed/:
    python src/preprocessing.py
"""

import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from constants import COLUMN_NAMES, CATEGORICAL_COLUMNS, ATTACK_CATEGORY_MAP

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_raw(path: str) -> pd.DataFrame:
    """Load a raw NSL-KDD file (no header row) into a labeled DataFrame."""
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES)
    return df


def add_attack_category(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the raw attack label into one of 5 categories."""
    df = df.copy()
    df["attack_category"] = df["label"].map(ATTACK_CATEGORY_MAP)
    # Any label not seen during mapping (rare/unknown attack types in the
    # test set) is treated conservatively as an attack rather than dropped.
    df["attack_category"] = df["attack_category"].fillna("DoS")
    return df


def encode_categoricals(train_df: pd.DataFrame, test_df: pd.DataFrame, encoders=None):
    """Label-encode protocol_type / service / flag using shared encoders
    fit on the union of train+test so unseen test categories don't crash."""
    encoders = encoders or {}
    for col in CATEGORICAL_COLUMNS:
        if col not in encoders:
            le = LabelEncoder()
            le.fit(pd.concat([train_df[col], test_df[col]]).astype(str))
            encoders[col] = le
        train_df[col] = encoders[col].transform(train_df[col].astype(str))
        test_df[col] = encoders[col].transform(test_df[col].astype(str))
    return train_df, test_df, encoders


def build_datasets(train_path: str, test_path: str):
    """Full preprocessing pipeline. Returns X_train, X_test, y_train, y_test
    (y = attack_category) plus the fitted encoders/scaler so train.py and
    a future real-time scoring service can reuse them."""
    train_df = add_attack_category(load_raw(train_path))
    test_df = add_attack_category(load_raw(test_path))

    # Drop columns not used as model input.
    drop_cols = ["label", "difficulty", "attack_category"]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    train_df, test_df, encoders = encode_categoricals(train_df, test_df)

    X_train, y_train = train_df[feature_cols], train_df["attack_category"]
    X_test, y_test = test_df[feature_cols], test_df["attack_category"]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

    return X_train_scaled, X_test_scaled, y_train.reset_index(drop=True), \
        y_test.reset_index(drop=True), encoders, scaler, feature_cols


def main():
    train_path = os.path.join(DATA_DIR, "KDDTrain+.txt")
    test_path = os.path.join(DATA_DIR, "KDDTest+.txt")

    X_train, X_test, y_train, y_test, encoders, scaler, feature_cols = \
        build_datasets(train_path, test_path)

    processed_dir = os.path.join(DATA_DIR, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    X_train.assign(attack_category=y_train).to_csv(
        os.path.join(processed_dir, "train_processed.csv"), index=False)
    X_test.assign(attack_category=y_test).to_csv(
        os.path.join(processed_dir, "test_processed.csv"), index=False)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(encoders, os.path.join(MODELS_DIR, "label_encoders.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_columns.pkl"))

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print("Class distribution (train):")
    print(y_train.value_counts())
    print("\nSaved processed data to:", processed_dir)
    print("Saved scaler/encoders/feature_columns to:", MODELS_DIR)


if __name__ == "__main__":
    main()
