from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Alap validálás
# ---------------------------------------------------------------------
def _validate_1d_signal(x: np.ndarray, name: str = "signal") -> np.ndarray:
    """
    Ellenőrzi, hogy a bemenet nem üres, 1D NumPy tömb legyen.
    """
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError(f"A(z) {name} legyen 1D. Kapott shape: {x.shape}")

    if x.size == 0:
        raise ValueError(f"A(z) {name} üres.")

    return x


# ---------------------------------------------------------------------
# RR intervallumok annotációból
# ---------------------------------------------------------------------
def compute_rr_intervals_from_annotations(
    annotation_samples: np.ndarray,
    fs: float,
) -> np.ndarray:
    """
    RR intervallumok számítása annotált R-csúcs mintapozíciókból.
    """
    annotation_samples = np.asarray(annotation_samples, dtype=int)

    if fs <= 0:
        raise ValueError("Az fs pozitív kell legyen.")

    if annotation_samples.ndim != 1:
        raise ValueError(
            f"Az annotation_samples legyen 1D. Kapott shape: {annotation_samples.shape}"
        )

    if annotation_samples.size < 2:
        return np.array([], dtype=float)

    rr = np.diff(annotation_samples) / float(fs)
    return rr


# ---------------------------------------------------------------------
# Beat alapstatisztikák
# ---------------------------------------------------------------------
def _safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size > 0 else np.nan


def _safe_std(x: np.ndarray) -> float:
    return float(np.std(x, ddof=0)) if x.size > 0 else np.nan


def _safe_max_abs_diff(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(x))))


def _safe_curvature_std(x: np.ndarray) -> float:
    if x.size < 3:
        return 0.0

    diff1 = np.diff(x)
    diff2 = np.diff(diff1)

    if diff2.size == 0:
        return 0.0

    return float(np.std(diff2, ddof=0))


def _safe_skewness(x: np.ndarray) -> float:
    if x.size == 0:
        return np.nan

    mean_val = np.mean(x)
    std_val = np.std(x, ddof=0)

    if std_val == 0:
        return 0.0

    z = (x - mean_val) / std_val
    return float(np.mean(z**3))


def _safe_kurtosis(x: np.ndarray) -> float:
    if x.size == 0:
        return np.nan

    mean_val = np.mean(x)
    std_val = np.std(x, ddof=0)

    if std_val == 0:
        return 0.0

    z = (x - mean_val) / std_val
    return float(np.mean(z**4))


def _count_zero_crossings(x: np.ndarray) -> int:
    if x.size < 2:
        return 0

    signs = np.sign(x)
    zero_crossings = np.sum(np.diff(signs) != 0)
    return int(zero_crossings)


def _qrs_width_approximation(beat: np.ndarray, fs: float) -> float:
    """
    Egyszerű közelítő QRS-szélesség feature.
    """
    if beat.size == 0:
        return np.nan

    if fs <= 0:
        raise ValueError("Az fs pozitív kell legyen.")

    max_abs = np.max(np.abs(beat))
    if max_abs == 0:
        return 0.0

    threshold = 0.5 * max_abs
    mask = np.abs(beat) > threshold
    width_sec = np.sum(mask) / float(fs)

    return float(width_sec)


# ---------------------------------------------------------------------
# Egyetlen beat feature extraction
# ---------------------------------------------------------------------
def extract_morphology_features(
    beat: np.ndarray,
    center_idx: int | None = None,
    rr_prev: float | None = None,
    rr_next: float | None = None,
    fs: float = 360.0,
) -> dict[str, float]:
    """
    Egyetlen beatből morfológiai és egyszerű ritmus-feature-ök kinyerése.
    """
    beat = _validate_1d_signal(beat, name="beat")

    if fs <= 0:
        raise ValueError("Az fs pozitív kell legyen.")

    if center_idx is None:
        center_idx = len(beat) // 2

    if not (0 <= center_idx < len(beat)):
        raise ValueError(
            f"Érvénytelen center_idx={center_idx}, beat hossza={len(beat)}"
        )

    r_amp = beat[center_idx]
    max_amp = np.max(beat)
    min_amp = np.min(beat)
    ptp = max_amp - min_amp

    max_slope = _safe_max_abs_diff(beat)
    curvature = _safe_curvature_std(beat)

    energy = float(np.sum(beat**2))
    rms = float(np.sqrt(np.mean(beat**2)))

    qrs_width_approx = _qrs_width_approximation(beat, fs=fs)

    mean_val = _safe_mean(beat)
    std_val = _safe_std(beat)
    skewness = _safe_skewness(beat)
    kurtosis = _safe_kurtosis(beat)

    zero_crossings = _count_zero_crossings(beat)

    features = {
        "rr_prev": float(rr_prev) if rr_prev is not None else np.nan,
        "rr_next": float(rr_next) if rr_next is not None else np.nan,
        "r_amp": float(r_amp),
        "max_amp": float(max_amp),
        "min_amp": float(min_amp),
        "ptp": float(ptp),
        "max_slope": float(max_slope),
        "curvature": float(curvature),
        "qrs_width_approx": float(qrs_width_approx),
        "energy": float(energy),
        "rms": float(rms),
        "mean": float(mean_val),
        "std": float(std_val),
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
        "zero_crossings": float(zero_crossings),
    }

    return features


# ---------------------------------------------------------------------
# Közös meta helper
# ---------------------------------------------------------------------
def _build_common_meta(
    record_id: str | int | None,
    beat_index: int,
    beat_labels: np.ndarray | list[Any] | None,
    i: int,
    beat_len: int,
    center_idx: int,
    center_sample: int | None = None,
) -> dict[str, Any]:
    record_id_str = str(record_id) if record_id is not None else None
    record_beat_index = (
        f"{record_id_str}_{beat_index}" if record_id_str is not None else None
    )

    return {
        "record_id": record_id_str,
        "beat_index": int(beat_index),
        "record_beat_index": record_beat_index,
        "label": beat_labels[i] if beat_labels is not None else None,
        "beat_len": int(beat_len),
        "center_idx": int(center_idx),
        "beat_center_sample": int(center_sample) if center_sample is not None else None,
    }


# ---------------------------------------------------------------------
# Beat-szintű feature DataFrame építés
# ---------------------------------------------------------------------
def build_morphology_feature_dataframe(
    beats: np.ndarray,
    annotation_samples: np.ndarray | None,
    fs: float,
    beat_labels: np.ndarray | list[Any] | None = None,
    record_id: str | int | None = None,
    center_indices: np.ndarray | list[int] | None = None,
    beat_indices: np.ndarray | list[int] | None = None,
) -> pd.DataFrame:
    """
    Beat-szintű feature DataFrame építése.

    Ha beat_indices meg van adva, azt használjuk stabil beat_index-ként.
    Ha nincs, akkor 0..n_beats-1.
    """
    beats = np.asarray(beats, dtype=float)

    if beats.ndim != 2:
        raise ValueError(
            f"A beats tömb legyen 2D, shape=(n_beats, beat_len). Kapott: {beats.shape}"
        )

    if fs <= 0:
        raise ValueError("Az fs pozitív kell legyen.")

    n_beats = beats.shape[0]

    if beat_labels is not None and len(beat_labels) != n_beats:
        raise ValueError("A beat_labels hossza nem egyezik a beat-ek számával.")

    if center_indices is not None and len(center_indices) != n_beats:
        raise ValueError("A center_indices hossza nem egyezik a beat-ek számával.")

    if beat_indices is not None and len(beat_indices) != n_beats:
        raise ValueError("A beat_indices hossza nem egyezik a beat-ek számával.")

    rr = None
    if annotation_samples is not None:
        rr = compute_rr_intervals_from_annotations(annotation_samples, fs)

    if beat_indices is None:
        beat_indices = np.arange(n_beats, dtype=int)
    else:
        beat_indices = np.asarray(beat_indices, dtype=int)

    rows: list[dict[str, Any]] = []

    for i in range(n_beats):
        beat = beats[i]

        if center_indices is None:
            center_idx = len(beat) // 2
        else:
            center_idx = int(center_indices[i])

        rr_prev = None
        rr_next = None

        if rr is not None:
            if i > 0 and (i - 1) < len(rr):
                rr_prev = float(rr[i - 1])
            if i < len(rr):
                rr_next = float(rr[i])

        feats = extract_morphology_features(
            beat=beat,
            center_idx=center_idx,
            rr_prev=rr_prev,
            rr_next=rr_next,
            fs=fs,
        )

        row = {
            **_build_common_meta(
                record_id=record_id,
                beat_index=int(beat_indices[i]),
                beat_labels=beat_labels,
                i=i,
                beat_len=len(beat),
                center_idx=center_idx,
                center_sample=None,
            ),
            **feats,
        }

        rows.append(row)

    return pd.DataFrame(rows)


def build_morphology_feature_dataframe_from_full_annotations(
    beats: np.ndarray,
    beat_centers: np.ndarray,
    full_annotation_samples: np.ndarray,
    fs: float,
    beat_labels: np.ndarray | list[Any] | None = None,
    record_id: str | int | None = None,
    center_indices: np.ndarray | list[int] | None = None,
    beat_indices: np.ndarray | list[int] | None = None,
) -> pd.DataFrame:
    """
    Beat-szintű morphology feature DataFrame építése úgy, hogy az RR kontextust
    a teljes annotációs sorozatból számoljuk.

    Ha beat_indices meg van adva, azt használjuk stabil beat_index-ként.
    """
    beats = np.asarray(beats, dtype=float)
    beat_centers = np.asarray(beat_centers, dtype=int)
    full_annotation_samples = np.asarray(full_annotation_samples, dtype=int)

    if beats.ndim != 2:
        raise ValueError(
            f"A beats tömb legyen 2D, shape=(n_beats, beat_len). Kapott: {beats.shape}"
        )

    if beat_centers.ndim != 1:
        raise ValueError(f"A beat_centers legyen 1D. Kapott shape: {beat_centers.shape}")

    if full_annotation_samples.ndim != 1:
        raise ValueError(
            "A full_annotation_samples legyen 1D. "
            f"Kapott shape: {full_annotation_samples.shape}"
        )

    if fs <= 0:
        raise ValueError("Az fs pozitív kell legyen.")

    n_beats = beats.shape[0]

    if len(beat_centers) != n_beats:
        raise ValueError(
            "A beat_centers hossza nem egyezik a beat-ek számával. "
            f"n_beats={n_beats}, len(beat_centers)={len(beat_centers)}"
        )

    if beat_labels is not None and len(beat_labels) != n_beats:
        raise ValueError("A beat_labels hossza nem egyezik a beat-ek számával.")

    if center_indices is not None and len(center_indices) != n_beats:
        raise ValueError("A center_indices hossza nem egyezik a beat-ek számával.")

    if beat_indices is not None and len(beat_indices) != n_beats:
        raise ValueError("A beat_indices hossza nem egyezik a beat-ek számával.")

    if beat_indices is None:
        beat_indices = np.arange(n_beats, dtype=int)
    else:
        beat_indices = np.asarray(beat_indices, dtype=int)

    sample_to_full_index = {
        int(sample): idx for idx, sample in enumerate(full_annotation_samples)
    }

    rows: list[dict[str, Any]] = []

    for i in range(n_beats):
        beat = beats[i]
        center_sample = int(beat_centers[i])

        if center_indices is None:
            center_idx = len(beat) // 2
        else:
            center_idx = int(center_indices[i])

        rr_prev = None
        rr_next = None

        full_idx = sample_to_full_index.get(center_sample)

        if full_idx is not None:
            if full_idx > 0:
                rr_prev = (
                    (full_annotation_samples[full_idx] - full_annotation_samples[full_idx - 1])
                    / float(fs)
                )

            if full_idx < len(full_annotation_samples) - 1:
                rr_next = (
                    (full_annotation_samples[full_idx + 1] - full_annotation_samples[full_idx])
                    / float(fs)
                )

        feats = extract_morphology_features(
            beat=beat,
            center_idx=center_idx,
            rr_prev=rr_prev,
            rr_next=rr_next,
            fs=fs,
        )

        row = {
            **_build_common_meta(
                record_id=record_id,
                beat_index=int(beat_indices[i]),
                beat_labels=beat_labels,
                i=i,
                beat_len=len(beat),
                center_idx=center_idx,
                center_sample=center_sample,
            ),
            **feats,
        }

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Egyszerű minőségi flag-ek
# ---------------------------------------------------------------------
def add_morphology_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robusztus morphology quality flag-ek.

    Mindig létrehozza az oszlopokat:
      - has_rr_context_prev
      - has_rr_context_next
      - rr_prev_plausible
      - rr_next_plausible
      - qrs_width_plausible
    """

    df = df.copy()

    n = len(df)

    # ------------------------------------------------------------------
    # Kötelező oszlopok inicializálása
    # ------------------------------------------------------------------
    df["has_rr_context_prev"] = False
    df["has_rr_context_next"] = False

    df["rr_prev_plausible"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
    df["rr_next_plausible"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
    df["qrs_width_plausible"] = pd.Series(pd.NA, index=df.index, dtype="boolean")

    # ------------------------------------------------------------------
    # RR prev
    # ------------------------------------------------------------------
    if "rr_prev" in df.columns:
        rr_prev = pd.to_numeric(df["rr_prev"], errors="coerce")

        has_prev = rr_prev.notna() & np.isfinite(rr_prev)
        df["has_rr_context_prev"] = has_prev

        df.loc[has_prev, "rr_prev_plausible"] = (
            rr_prev.loc[has_prev].between(0.24, 3.0, inclusive="both")
        )

    # ------------------------------------------------------------------
    # RR next
    # ------------------------------------------------------------------
    if "rr_next" in df.columns:
        rr_next = pd.to_numeric(df["rr_next"], errors="coerce")

        has_next = rr_next.notna() & np.isfinite(rr_next)
        df["has_rr_context_next"] = has_next

        df.loc[has_next, "rr_next_plausible"] = (
            rr_next.loc[has_next].between(0.24, 3.0, inclusive="both")
        )

    # ------------------------------------------------------------------
    # QRS width
    # ------------------------------------------------------------------
    if "qrs_width_approx" in df.columns:
        qrs = pd.to_numeric(df["qrs_width_approx"], errors="coerce")

        has_qrs = qrs.notna() & np.isfinite(qrs)

        df.loc[has_qrs, "qrs_width_plausible"] = (
            qrs.loc[has_qrs].between(0.02, 0.30, inclusive="both")
        )

    return df