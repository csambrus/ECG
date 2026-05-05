from __future__ import annotations

import numpy as np
import pandas as pd


def compute_padding_fraction(beat: np.ndarray, pad_value: float = 0.0) -> float:
    beat = np.asarray(beat)
    return float(np.mean(beat == pad_value))


def compute_noise_score(beat: np.ndarray) -> float:
    beat = np.asarray(beat, dtype=float)
    if len(beat) < 2:
        return 0.0
    return float(np.std(np.diff(beat)))


def compute_qrs_activity_score(
    beat: np.ndarray,
    center_idx: int,
    window_radius: int = 12,
) -> float:
    beat = np.asarray(beat, dtype=float)

    start = max(0, center_idx - window_radius)
    end = min(len(beat), center_idx + window_radius + 1)

    seg = beat[start:end]
    if len(seg) < 2:
        return 0.0

    return float(np.std(np.diff(seg)))


def check_single_beat_quality(
    beat: np.ndarray,
    center_idx: int,
    pad_value: float = 0.0,
    max_padding_fraction: float = 0.10,
    min_std: float = 0.05,
    max_abs_amplitude: float = 8.0,
    max_noise_score: float = 1.5,
    min_qrs_activity_score: float = 0.05,
) -> tuple[bool, dict]:
    """
    Beat minőségellenőrzés egyszerű szabályokkal.

    Feltételezés:
    - a beat már preprocessált
    - opcionálisan z-score normalizált
    """
    beat = np.asarray(beat, dtype=float)

    std = float(np.std(beat))
    max_abs = float(np.max(np.abs(beat))) if len(beat) > 0 else 0.0
    padding_fraction = compute_padding_fraction(beat, pad_value=pad_value)
    noise_score = compute_noise_score(beat)
    qrs_activity_score = compute_qrs_activity_score(beat, center_idx=center_idx)

    reasons = []

    if padding_fraction > max_padding_fraction:
        reasons.append("too_much_padding")

    if std < min_std:
        reasons.append("too_flat")

    if max_abs > max_abs_amplitude:
        reasons.append("extreme_amplitude")

    if noise_score > max_noise_score:
        reasons.append("too_noisy")

    if qrs_activity_score < min_qrs_activity_score:
        reasons.append("low_qrs_activity")

    is_valid = len(reasons) == 0

    metrics = {
        "padding_fraction": padding_fraction,
        "std": std,
        "max_abs_amplitude": max_abs,
        "noise_score": noise_score,
        "qrs_activity_score": qrs_activity_score,
        "is_valid": is_valid,
        "reasons": ",".join(reasons) if reasons else "",
    }

    return is_valid, metrics


def filter_beats_by_quality(
    beats: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray | None = None,
    beat_pre_samples: int = 100,
    pad_value: float = 0.0,
    max_padding_fraction: float = 0.10,
    min_std: float = 0.05,
    max_abs_amplitude: float = 8.0,
    max_noise_score: float = 1.5,
    min_qrs_activity_score: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, pd.DataFrame]:
    beats = np.asarray(beats)
    labels = np.asarray(labels)

    if centers is not None:
        centers = np.asarray(centers)

    keep_mask = []
    rows = []

    for i, beat in enumerate(beats):
        is_valid, metrics = check_single_beat_quality(
            beat=beat,
            center_idx=beat_pre_samples,
            pad_value=pad_value,
            max_padding_fraction=max_padding_fraction,
            min_std=min_std,
            max_abs_amplitude=max_abs_amplitude,
            max_noise_score=max_noise_score,
            min_qrs_activity_score=min_qrs_activity_score,
        )

        keep_mask.append(is_valid)
        row = {"beat_index": i, "label": labels[i], **metrics}
        if centers is not None:
            row["center_sample"] = int(centers[i])
        rows.append(row)

    keep_mask = np.asarray(keep_mask, dtype=bool)
    qc_df = pd.DataFrame(rows)

    beats_f = beats[keep_mask]
    labels_f = labels[keep_mask]
    centers_f = centers[keep_mask] if centers is not None else None

    return beats_f, labels_f, centers_f, qc_df
