from __future__ import annotations

import numpy as np
import pandas as pd


def extract_basic_features(beat: np.ndarray) -> dict[str, float]:
    """
    Egyszerű, beat-szintű numerikus feature-ök.

    Feltételezés:
    - a beat már preprocessált
    - tipikusan z-score normalizált
    """
    beat = np.asarray(beat, dtype=float)

    diff1 = np.diff(beat) if len(beat) >= 2 else np.array([0.0])
    diff2 = np.diff(diff1) if len(diff1) >= 2 else np.array([0.0])

    center_idx = len(beat) // 2
    local_radius = min(10, center_idx)
    local_start = max(0, center_idx - local_radius)
    local_end = min(len(beat), center_idx + local_radius + 1)
    local_seg = beat[local_start:local_end]

    features = {
        "mean": float(np.mean(beat)),
        "std": float(np.std(beat)),
        "min": float(np.min(beat)),
        "max": float(np.max(beat)),
        "ptp": float(np.ptp(beat)),  # peak-to-peak
        "median": float(np.median(beat)),
        "q25": float(np.quantile(beat, 0.25)),
        "q75": float(np.quantile(beat, 0.75)),
        "abs_mean": float(np.mean(np.abs(beat))),
        "energy": float(np.sum(beat ** 2)),
        "rms": float(np.sqrt(np.mean(beat ** 2))),
        "diff1_std": float(np.std(diff1)),
        "diff1_abs_mean": float(np.mean(np.abs(diff1))),
        "diff2_std": float(np.std(diff2)),
        "center_value": float(beat[center_idx]),
        "local_mean": float(np.mean(local_seg)),
        "local_std": float(np.std(local_seg)),
        "local_max": float(np.max(local_seg)),
        "local_min": float(np.min(local_seg)),
    }

    return features


def extract_feature_dataframe(X: np.ndarray) -> pd.DataFrame:
    """
    Beat-mátrixból feature DataFrame-et készít.

    Parameters
    ----------
    X : np.ndarray
        shape = [n_beats, beat_len]

    Returns
    -------
    pd.DataFrame
        shape = [n_beats, n_features]
    """
    rows = [extract_basic_features(beat) for beat in X]
    return pd.DataFrame(rows)