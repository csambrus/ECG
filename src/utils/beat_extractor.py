from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

from src.config import BEAT_POST_SEC, BEAT_PRE_SEC, DEFAULT_FS
from src.utils.dataset_loader import ECGRecord
from src.utils.signals import (
    get_preprocessed_multichannel_signal,
    get_preprocessed_signal,
    zscore_signal,
)


@dataclass
class CropInfo:
    start: int
    end: int
    src_start: int
    src_end: int
    dst_start: int
    dst_end: int
    pad_left: int
    pad_right: int
    is_truncated: bool


def safe_crop_1d(
    x: np.ndarray,
    start: int,
    end: int,
    pad_value: float = 0.0,
) -> tuple[np.ndarray, CropInfo]:
    target_len = end - start
    if target_len <= 0:
        raise ValueError(f"Invalid crop length: start={start}, end={end}")

    out = np.full(target_len, pad_value, dtype=float)

    src_start = max(start, 0)
    src_end = min(end, len(x))

    dst_start = src_start - start
    dst_end = dst_start + max(0, src_end - src_start)

    if src_end > src_start:
        out[dst_start:dst_end] = x[src_start:src_end]

    pad_left = max(0, -start)
    pad_right = max(0, end - len(x))

    info = CropInfo(
        start=start,
        end=end,
        src_start=src_start,
        src_end=src_end,
        dst_start=dst_start,
        dst_end=dst_end,
        pad_left=pad_left,
        pad_right=pad_right,
        is_truncated=(pad_left > 0 or pad_right > 0),
    )
    return out, info


def extract_beats(
    rec: ECGRecord,
    channel: int = 0,
    fs: float = DEFAULT_FS,
    pre_sec: float = BEAT_PRE_SEC,
    post_sec: float = BEAT_POST_SEC,
    normalize: bool = True,
    allowed_symbols: Optional[set[str]] = None,
    edge_policy: Literal["skip", "pad"] = "skip",
    pad_value: float = 0.0,
    preprocessing_config: Optional[dict] = None,
    dataset: str | None = None,
):
    if edge_policy not in {"skip", "pad"}:
        raise ValueError("edge_policy must be 'skip' or 'pad'")

    fs = float(fs)

    sig = get_preprocessed_signal(
        rec,
        channel=channel,
        preprocessing_config=preprocessing_config,
        dataset=dataset,
        target_fs=fs,
    )

    pre_samples = int(pre_sec * fs)
    post_samples = int(post_sec * fs)
    beat_len = pre_samples + post_samples

    scale = fs / float(rec.fs)

    beats: list[np.ndarray] = []
    labels: list[str] = []
    centers: list[int] = []
    meta_rows: list[dict] = []

    for i, (sample_orig, symbol) in enumerate(zip(rec.annotation_samples, rec.annotation_symbols)):
        if allowed_symbols is not None and symbol not in allowed_symbols:
            continue

        sample_resampled = int(round(float(sample_orig) * scale))

        start = sample_resampled - pre_samples
        end = sample_resampled + post_samples

        beat, info = safe_crop_1d(sig, start, end, pad_value=pad_value)

        if info.is_truncated and edge_policy == "skip":
            meta_rows.append({
                "record_id": getattr(rec, "record_id", None),
                "ann_index": i,
                "symbol": symbol,
                "center_sample_original": int(sample_orig),
                "center_sample": int(sample_resampled),
                "start": start,
                "end": end,
                "pad_left": info.pad_left,
                "pad_right": info.pad_right,
                "is_truncated": True,
                "kept": False,
                "reason": "edge_truncated",
            })
            continue

        if normalize:
            if info.is_truncated:
                valid = beat[info.dst_start:info.dst_end]
                if len(valid) > 1:
                    valid_z = zscore_signal(valid)
                    beat_norm = np.full_like(beat, pad_value, dtype=float)
                    beat_norm[info.dst_start:info.dst_end] = valid_z
                    beat = beat_norm
            else:
                beat = zscore_signal(beat)

        beats.append(beat)
        labels.append(symbol)
        centers.append(sample_resampled)

        meta_rows.append({
            "record_id": getattr(rec, "record_id", None),
            "ann_index": i,
            "symbol": symbol,
            "center_sample_original": int(sample_orig),
            "center_sample": int(sample_resampled),
            "start": start,
            "end": end,
            "pad_left": info.pad_left,
            "pad_right": info.pad_right,
            "is_truncated": info.is_truncated,
            "kept": True,
            "reason": "ok" if not info.is_truncated else "kept_with_padding",
        })

    if not beats:
        return (
            np.empty((0, beat_len), dtype=float),
            np.empty((0,), dtype=str),
            np.empty((0,), dtype=int),
            pd.DataFrame(meta_rows),
        )

    beats_arr = np.vstack(beats)
    labels_arr = np.asarray(labels)
    centers_arr = np.asarray(centers, dtype=int)
    meta_df = pd.DataFrame(meta_rows)

    return beats_arr, labels_arr, centers_arr, meta_df


def _zscore_multichannel(beat: np.ndarray) -> np.ndarray:
    out = beat.astype(float, copy=True)
    for ch in range(out.shape[1]):
        x = out[:, ch]
        mu = float(np.mean(x))
        sd = float(np.std(x))
        out[:, ch] = x - mu if sd < 1e-8 else (x - mu) / sd
    return out


def extract_beats_multichannel(
    rec: ECGRecord,
    channels: Optional[list[int]] = None,
    fs: float = DEFAULT_FS,
    pre_sec: float = BEAT_PRE_SEC,
    post_sec: float = BEAT_POST_SEC,
    normalize: bool = True,
    allowed_symbols: Optional[set[str]] = None,
    edge_policy: Literal["skip", "pad"] = "skip",
    pad_value: float = 0.0,
    preprocessing_config: Optional[dict] = None,
    dataset: str | None = None,
):
    if edge_policy not in {"skip", "pad"}:
        raise ValueError("edge_policy must be 'skip' or 'pad'")

    fs = float(fs)
    signal = get_preprocessed_multichannel_signal(
        rec,
        preprocessing_config=preprocessing_config,
        dataset=dataset,
        target_fs=fs,
    )

    if channels is not None:
        signal = signal[:, channels]

    pre_samples = int(pre_sec * fs)
    post_samples = int(post_sec * fs)
    beat_len = pre_samples + post_samples
    n_channels = int(signal.shape[1])

    scale = fs / float(rec.fs)

    beats: list[np.ndarray] = []
    labels: list[str] = []
    centers: list[int] = []
    meta_rows: list[dict] = []

    for i, (sample_orig, symbol) in enumerate(zip(rec.annotation_samples, rec.annotation_symbols)):
        if allowed_symbols is not None and symbol not in allowed_symbols:
            meta_rows.append({
                "ann_index": i,
                "symbol": str(symbol),
                "center_sample_original": int(sample_orig),
                "center_sample": int(round(float(sample_orig) * scale)),
                "start": None,
                "end": None,
                "pad_left": 0,
                "pad_right": 0,
                "is_truncated": False,
                "kept": False,
                "reason": "filtered_symbol",
            })
            continue

        sample_resampled = int(round(float(sample_orig) * scale))
        start = sample_resampled - pre_samples
        end = sample_resampled + post_samples

        left_oob = max(0, -start)
        right_oob = max(0, end - signal.shape[0])
        is_truncated = left_oob > 0 or right_oob > 0

        if is_truncated and edge_policy == "skip":
            meta_rows.append({
                "ann_index": i,
                "symbol": str(symbol),
                "center_sample_original": int(sample_orig),
                "center_sample": int(sample_resampled),
                "start": int(start),
                "end": int(end),
                "pad_left": int(left_oob),
                "pad_right": int(right_oob),
                "is_truncated": True,
                "kept": False,
                "reason": "edge_truncated",
            })
            continue

        if edge_policy == "pad":
            beat = np.full((beat_len, n_channels), pad_value, dtype=float)
            src_start = max(0, start)
            src_end = min(signal.shape[0], end)
            dst_start = src_start - start
            dst_end = dst_start + max(0, src_end - src_start)
            if src_end > src_start:
                beat[dst_start:dst_end, :] = signal[src_start:src_end, :]
        else:
            beat = signal[start:end, :].astype(float, copy=False)

        if normalize:
            beat = _zscore_multichannel(beat)

        beats.append(beat.astype(np.float32, copy=False))
        labels.append(str(symbol))
        centers.append(int(sample_resampled))

        meta_rows.append({
            "ann_index": i,
            "symbol": str(symbol),
            "center_sample_original": int(sample_orig),
            "center_sample": int(sample_resampled),
            "start": int(start),
            "end": int(end),
            "pad_left": int(left_oob),
            "pad_right": int(right_oob),
            "is_truncated": bool(is_truncated),
            "kept": True,
            "reason": "ok" if not is_truncated else "kept_with_padding",
        })

    if not beats:
        return (
            np.empty((0, beat_len, n_channels), dtype=np.float32),
            np.empty((0,), dtype=object),
            np.empty((0,), dtype=np.int64),
            pd.DataFrame(meta_rows),
        )

    return (
        np.asarray(beats, dtype=np.float32),
        np.asarray(labels, dtype=object),
        np.asarray(centers, dtype=np.int64),
        pd.DataFrame(meta_rows),
    )