"""
evaluate.py
-----------
Milestone 5 & 9: Model Evaluation, Optimization and Testing/Validation.

Loads the best trained model, scores it on the (unseen) NSL-KDD test
set, and writes a metrics report + confusion matrix image to reports/.

Run:
    python src/evaluate.py
"""

import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score
)

from constants import ATTACK_CATEGORIES
from preprocessing import build_datasets

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    train_path = os.path.join(DATA_DIR, "KDDTrain+.txt")
    test_path = os.path.join(DATA_DIR, "KDDTest+.txt")

    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, *_ = build_datasets(train_path, test_path)

    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    with open(os.path.join(MODELS_DIR, "best_model.txt")) as f:
        model_name = f.read().strip()

    print(f"Evaluating model: {model_name}")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    report_text = classification_report(y_test, y_pred, labels=ATTACK_CATEGORIES)

    print(f"Test accuracy: {acc:.4f}")
    print(f"Test macro F1: {f1_macro:.4f}")
    print(report_text)

    # Save text report.
    report_path = os.path.join(REPORTS_DIR, "evaluation_report.md")
    with open(report_path, "w") as f:
        f.write(f"# Model Evaluation Report\n\n")
        f.write(f"**Model:** {model_name}\n\n")
        f.write(f"**Test Accuracy:** {acc:.4f}\n\n")
        f.write(f"**Test Macro F1-score:** {f1_macro:.4f}\n\n")
        f.write("## Classification Report\n\n")
        f.write("```\n" + report_text + "\n```\n")

    # Confusion matrix figure.
    cm = confusion_matrix(y_test, y_pred, labels=ATTACK_CATEGORIES)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=ATTACK_CATEGORIES, yticklabels=ATTACK_CATEGORIES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()

    print("\nSaved report to:", report_path)
    print("Saved confusion matrix to:", cm_path)


if __name__ == "__main__":
    main()
