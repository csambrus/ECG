from __future__ import annotations

from fractions import Fraction
from typing import Optional

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample_poly

from src.config import (
    DEFAULT_FS,
    INCART_GAIN_NORMALIZATION,
    RESAMPLE_TO_DEFAULT_FS,
)


def zscore_signal(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < eps:
        return x - mu
    return (x - mu) / sigma


def butter_bandpass_filter(
    x: np.ndarray,
    fs: float,
    lowcut: float,
    highcut: float,
    order: int = 3,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)

    if fs <= 0:
        raise ValueError("Az fs pozitív kell legyen.")

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    if not 0 < low < high < 1:
        raise ValueError(f"Invalid bandpass range: lowcut={lowcut}, highcut={highcut}, fs={fs}")

    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, x)


def remove_baseline_wander(
    x: np.ndarray,
    fs: float,
    cutoff: float = 0.5,
    order: int = 3,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)

    if fs <= 0:
        raise ValueError("Az fs pozitív kell legyen.")

    nyq = 0.5 * fs
    wn = cutoff / nyq

    if not 0 < wn < 1:
        raise ValueError(f"Invalid high-pass cutoff: cutoff={cutoff}, fs={fs}")

    b, a = butter(order, wn, btype="high")
    return filtfilt(b, a, x)


def notch_filter(
    x: np.ndarray,
    fs: float,
    freq: float = 50.0,
    q: float = 30.0,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)

    if fs <= 0:
        raise ValueError("Az fs pozitív kell legyen.")
    if freq <= 0:
        raise ValueError("A notch frekvencia pozitív kell legyen.")
    if q <= 0:
        raise ValueError("A notch Q pozitív kell legyen.")

    nyq = 0.5 * fs
    w0 = freq / nyq

    if not 0 < w0 < 1:
        raise ValueError(f"Invalid notch frequency: freq={freq}, fs={fs}")

    b, a = iirnotch(w0=w0, Q=q)
    return filtfilt(b, a, x)


def preprocess_signal(
    x: np.ndarray,
    fs: float,
    config: Optional[dict] = None,
) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()

    if config is None:
        return x

    if config.get("remove_baseline", False):
        x = remove_baseline_wander(
            x,
            fs=fs,
            cutoff=float(config.get("baseline_cutoff", 0.5)),
            order=int(config.get("baseline_order", 3)),
        )

    if config.get("use_bandpass", False):
        x = butter_bandpass_filter(
            x,
            fs=fs,
            lowcut=float(config.get("lowcut", 0.5)),
            highcut=float(config.get("highcut", 40.0)),
            order=int(config.get("bandpass_order", 3)),
        )

    if config.get("use_notch", False):
        x = notch_filter(
            x,
            fs=fs,
            freq=float(config.get("notch_freq", 50.0)),
            q=float(config.get("notch_q", 30.0)),
        )

    if config.get("normalize_signal", False):
        x = zscore_signal(x)

    return x


def infer_dataset_name_from_record(rec) -> str:
    rid = str(getattr(rec, "record_id", "")).strip().upper()
    if rid.startswith("I"):
        return "incart"
    return "mitbih"


def normalize_incart_gain_1d(
    x: np.ndarray,
    cfg: Optional[dict] = None,
) -> np.ndarray:
    cfg = cfg or INCART_GAIN_NORMALIZATION
    if not cfg.get("enabled", False):
        return np.asarray(x, dtype=float)

    x = np.asarray(x, dtype=float).copy()
    eps = float(cfg.get("eps", 1e-6))
    target_scale = float(cfg.get("target_scale", 1.0))
    method = str(cfg.get("method", "robust_mad")).lower()

    if method == "robust_mad":
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        robust_sigma = 1.4826 * mad
        denom = robust_sigma
    elif method == "median_abs":
        denom = float(np.median(np.abs(x)))
    else:
        raise ValueError(f"Ismeretlen INCART gain normalization method: {method}")

    if abs(denom) < eps:
        return x

    return x / denom * target_scale


def maybe_apply_dataset_gain_normalization(
    x: np.ndarray,
    dataset: str | None = None,
) -> np.ndarray:
    ds = str(dataset or "").lower().strip()
    x = np.asarray(x, dtype=float)

    if ds != "incart":
        return x

    if x.ndim == 1:
        return normalize_incart_gain_1d(x)

    if x.ndim == 2:
        out = np.empty_like(x, dtype=float)
        for ch in range(x.shape[1]):
            out[:, ch] = normalize_incart_gain_1d(x[:, ch])
        return out

    raise ValueError(f"Unsupported signal shape for gain normalization: {x.shape}")


def resample_signal(
    x: np.ndarray,
    orig_fs: float,
    target_fs: float,
    axis: int = 0,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)

    if orig_fs <= 0 or target_fs <= 0:
        raise ValueError("orig_fs és target_fs pozitív kell legyen.")

    if abs(orig_fs - target_fs) < 1e-12:
        return x.copy()

    ratio = Fraction(target_fs / orig_fs).limit_denominator(1000)
    up = ratio.numerator
    down = ratio.denominator

    return resample_poly(x, up=up, down=down, axis=axis)


def get_preprocessed_signal(
    rec,
    channel: int = 0,
    preprocessing_config: Optional[dict] = None,
    dataset: str | None = None,
    target_fs: float | None = None,
) -> np.ndarray:
    sig = np.asarray(rec.signal, dtype=float)
    ds = (dataset or infer_dataset_name_from_record(rec)).lower().strip()

    if sig.ndim == 1:
        x = sig
    elif sig.ndim == 2:
        if channel < 0 or channel >= sig.shape[1]:
            raise IndexError(
                f"Channel index out of range: channel={channel}, n_channels={sig.shape[1]}"
            )
        x = sig[:, channel]
    else:
        raise ValueError(f"Unsupported signal shape: {sig.shape}")

    x = maybe_apply_dataset_gain_normalization(x, dataset=ds)
    x = preprocess_signal(x, fs=float(rec.fs), config=preprocessing_config)

    effective_target_fs = float(target_fs) if target_fs is not None else (
        float(DEFAULT_FS) if RESAMPLE_TO_DEFAULT_FS else float(rec.fs)
    )

    x = resample_signal(x, orig_fs=float(rec.fs), target_fs=effective_target_fs, axis=0)
    return np.asarray(x, dtype=float)


def get_preprocessed_multichannel_signal(
    rec,
    preprocessing_config: Optional[dict] = None,
    dataset: str | None = None,
    target_fs: float | None = None,
) -> np.ndarray:
    sig = np.asarray(rec.signal, dtype=float)
    ds = (dataset or infer_dataset_name_from_record(rec)).lower().strip()

    if sig.ndim == 1:
        sig = sig[:, np.newaxis]
    elif sig.ndim != 2:
        raise ValueError(f"Unsupported signal shape: {sig.shape}")

    sig = maybe_apply_dataset_gain_normalization(sig, dataset=ds)

    channels = []
    for ch in range(sig.shape[1]):
        x = preprocess_signal(sig[:, ch], fs=float(rec.fs), config=preprocessing_config)
        channels.append(x)

    out = np.stack(channels, axis=1)

    effective_target_fs = float(target_fs) if target_fs is not None else (
        float(DEFAULT_FS) if RESAMPLE_TO_DEFAULT_FS else float(rec.fs)
    )

    out = resample_signal(out, orig_fs=float(rec.fs), target_fs=effective_target_fs, axis=0)
    return np.asarray(out, dtype=float)