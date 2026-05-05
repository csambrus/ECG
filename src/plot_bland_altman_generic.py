#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def detect_prob_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("prob_")]


def get_confidence(df: pd.DataFrame) -> np.ndarray:
    prob_cols = detect_prob_columns(df)
    return df[prob_cols].max(axis=1).values


def bland_altman_generic(csv1: Path, csv2: Path, out_path: Path):
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)

    assert len(df1) == len(df2), "Datasets must match in length"

    c1 = get_confidence(df1)
    c2 = get_confidence(df2)

    mean = (c1 + c2) / 2
    diff = c1 - c2

    m = np.mean(diff)
    sd = np.std(diff)

    plt.figure(figsize=(6, 6))
    plt.scatter(mean, diff, alpha=0.3)

    plt.axhline(m, linestyle="--", label="mean")
    plt.axhline(m + 1.96 * sd, linestyle="--")
    plt.axhline(m - 1.96 * sd, linestyle="--")

    plt.xlabel("Mean confidence")
    plt.ylabel("Difference")
    plt.title(f"Bland–Altman\n{csv1.stem} vs {csv2.stem}")
    plt.legend()
    plt.grid()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    csv1 = Path("outputs/incart_cnn/cnn_test_predictions.csv")
    csv2 = Path("outputs/incart_featurext/test_predictions.csv")

    out = Path("outputs/compare/bland_altman.png")

    bland_altman_generic(csv1, csv2, out)