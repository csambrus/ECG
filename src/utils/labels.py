from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable


# ---------------------------------------------------------------------
# AAMI classes (FIX, deterministic!)
# ---------------------------------------------------------------------
AAMI_CLASSES = ["N", "S", "V", "F", "Q"]

AAMI_TO_INT = {c: i for i, c in enumerate(AAMI_CLASSES)}
INT_TO_AAMI = {i: c for c, i in AAMI_TO_INT.items()}


# ---------------------------------------------------------------------
# 1. Raw MIT-BIH → AAMI mapping
# ---------------------------------------------------------------------
def map_aami_labels(symbols: Iterable[str]) -> np.ndarray:
    """
    Nyers MIT-BIH annotációk → AAMI osztályok.

    Returns
    -------
    np.ndarray of str (["N","S","V","F","Q"])
    """
    mapping = {
        # N
        "N": "N",
        "L": "N",
        "R": "N",
        "e": "N",
        "j": "N",

        # S
        "A": "S",
        "a": "S",
        "J": "S",
        "S": "S",

        # V
        "V": "V",
        "E": "V",

        # F
        "F": "F",

        # Q (catch-all)
        "/": "Q",
        "f": "Q",
        "Q": "Q",
        "?": "Q",
    }

    symbols_arr = np.asarray(symbols, dtype=str)

    return np.asarray(
        [mapping.get(sym, "Q") for sym in symbols_arr],
        dtype=str,
    )


# ---------------------------------------------------------------------
# 2. AAMI → integer encoding
# ---------------------------------------------------------------------
def encode_aami_labels(labels: Iterable[str]) -> np.ndarray:
    """
    AAMI label → integer (0–4)

    N=0, S=1, V=2, F=3, Q=4
    """
    labels_arr = np.asarray(labels, dtype=str)

    try:
        return np.asarray(
            [AAMI_TO_INT[l] for l in labels_arr],
            dtype=int,
        )
    except KeyError as e:
        raise ValueError(f"Unknown AAMI label: {e}")


# ---------------------------------------------------------------------
# 3. Integer → AAMI
# ---------------------------------------------------------------------
def decode_aami_labels(y: Iterable[int]) -> np.ndarray:
    """
    Integer → AAMI label
    """
    y_arr = np.asarray(y, dtype=int)

    try:
        return np.asarray(
            [INT_TO_AAMI[int(i)] for i in y_arr],
            dtype=str,
        )
    except KeyError as e:
        raise ValueError(f"Unknown class index: {e}")


# ---------------------------------------------------------------------
# 4. Distribution (QC + log)
# ---------------------------------------------------------------------
def compute_label_distribution(
    labels: Iterable[str] | Iterable[int],
    as_percent: bool = True,
) -> pd.DataFrame:
    """
    Label distribution táblázat.

    Works with:
    - string labels ("N")
    - encoded labels (0–4)
    """
    labels_arr = np.asarray(labels)

    # ha int → decode
    if np.issubdtype(labels_arr.dtype, np.integer):
        labels_arr = decode_aami_labels(labels_arr)

    counts = pd.Series(labels_arr).value_counts().reindex(AAMI_CLASSES, fill_value=0)

    df = pd.DataFrame({
        "count": counts,
    })

    if as_percent:
        total = counts.sum()
        df["percent"] = (counts / total * 100).round(2)

    return df


# ---------------------------------------------------------------------
# 5. Pretty print (log)
# ---------------------------------------------------------------------
def print_label_distribution(labels: Iterable[str] | Iterable[int]) -> None:
    df = compute_label_distribution(labels, as_percent=True)

    print("\nLabel distribution:")
    print(df.to_string())


# ---------------------------------------------------------------------
# 6. Class weights (ML)
# ---------------------------------------------------------------------
def compute_class_weights(y: Iterable[int]) -> dict[int, float]:
    """
    Class weights balanced módon.

    Input:
    - encoded labels (0–4)
    """
    y_arr = np.asarray(y, dtype=int)

    classes = np.arange(len(AAMI_CLASSES))

    counts = np.bincount(y_arr, minlength=len(classes))
    total = counts.sum()

    weights = {
        cls: total / (len(classes) * count) if count > 0 else 0.0
        for cls, count in enumerate(counts)
    }

    return weights
