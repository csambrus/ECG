#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def detect_prob_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("prob_")]


def plot_roc_generic(csv_path: Path, out_path: Path):
    df = pd.read_csv(csv_path)

    prob_cols = detect_prob_columns(df)
    assert len(prob_cols) > 1, "No probability columns found"

    class_names = [c.replace("prob_", "") for c in prob_cols]

    y_true = df["y_true"].values
    y_proba = df[prob_cols].values

    n_classes = len(prob_cols)

    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    plt.figure(figsize=(8, 6))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--")

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC – {csv_path.stem}")
    plt.legend()
    plt.grid()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    csv = Path("outputs/incart_cnn/cnn_test_predictions.csv")
    out = csv.parent / "roc_curves.png"

    plot_roc_generic(csv, out)