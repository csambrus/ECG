from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wfdb

from src.config import get_ds_par

@dataclass
class MITBIHRecord:
    record_id: str
    signal: np.ndarray
    fs: float
    sig_names: list[str]
    annotation_samples: np.ndarray
    annotation_symbols: np.ndarray


def load_metadata(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_record_from_dir(
    record_id: str,
    data_dir: str | Path | None = None,
    verbose: int = 0,
) -> MITBIHRecord:
    data_dir = Path(data_dir or get_ds_par("mitbih", "raw_dir"))
    record_path = data_dir / str(record_id)

    rec = wfdb.rdrecord(str(record_path))
    ann = wfdb.rdann(str(record_path), "atr")

    signal = np.asarray(rec.p_signal, dtype=float)
    fs = float(rec.fs)
    sig_names = list(rec.sig_name)

    annotation_samples = np.asarray(ann.sample)
    annotation_symbols = np.asarray(ann.symbol, dtype=object)

    if verbose >= 1:
        print(
            f"[load_record] {record_id} | "
            f"shape={signal.shape}, fs={fs}, "
            f"ch={len(sig_names)}, ann={len(annotation_samples)}"
        )

    if verbose >= 2:
        unique, counts = np.unique(annotation_symbols, return_counts=True)
        dist = dict(zip(unique, counts))
        print(f"  channels: {sig_names}")
        print(f"  first 5 symbols: {annotation_symbols[:5].tolist()}")
        print(f"  symbol dist: {dist}")

    return MITBIHRecord(
        record_id=str(record_id),
        signal=signal,
        fs=fs,
        sig_names=sig_names,
        annotation_samples=annotation_samples,
        annotation_symbols=annotation_symbols,
    )


def cast_record_id(values: pd.Series | np.ndarray | list[str]) -> pd.Series:
    return pd.Series(values, copy=False).astype("string").str.strip()


def cast_beat_meta_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    required = {"record_id", "beat_index", "record_beat_index"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Hiányzó beat meta oszlop(ok): {sorted(missing)}")

    out["record_id"] = cast_record_id(out["record_id"])
    out["beat_index"] = pd.to_numeric(out["beat_index"], errors="raise").astype("int64")
    out["record_beat_index"] = out["record_beat_index"].astype("string")

    if "center_sample" in out.columns:
        out["center_sample"] = pd.to_numeric(out["center_sample"], errors="raise").astype("int64")

    if "label_raw" in out.columns:
        out["label_raw"] = out["label_raw"].astype("string")

    if "label_aami" in out.columns:
        out["label_aami"] = out["label_aami"].astype("string")

    return out


def assert_same_length(X: np.ndarray, y: np.ndarray, beat_meta_df: pd.DataFrame) -> None:
    if len(X) != len(y) or len(X) != len(beat_meta_df):
        raise ValueError(
            "Az NPZ tartalma inkonzisztens: "
            f"len(X)={len(X)}, len(y)={len(y)}, len(beat_meta)={len(beat_meta_df)}"
        )

def validate_record_beat_index(
    df: pd.DataFrame,
    df_name: str,
    record_col: str = "record_id",
    beat_col: str = "beat_index",
    key_col: str = "record_beat_index",
) -> None:
    """
    Ellenőrzi, hogy a record_beat_index:
    1. létezik
    2. nem hiányzik
    3. pontosan record_id + "_" + beat_index formában van képezve
    4. egyedi
    """
    required = {record_col, beat_col, key_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Hiányzó kulcsoszlop(ok) a(z) {df_name} táblában: {sorted(missing)}"
        )

    if df.empty:
        return

    record_series = pd.Series(df[record_col], copy=False).astype("string").str.strip()
    beat_series = pd.to_numeric(df[beat_col], errors="raise").astype("int64")
    key_series = pd.Series(df[key_col], copy=False).astype("string")

    if record_series.isna().any():
        raise ValueError(f"Hiányzó {record_col} érték a(z) {df_name} táblában.")
    if beat_series.isna().any():
        raise ValueError(f"Hiányzó {beat_col} érték a(z) {df_name} táblában.")
    if key_series.isna().any():
        raise ValueError(f"Hiányzó {key_col} érték a(z) {df_name} táblában.")

    expected = record_series.astype(str) + "_" + beat_series.astype(str)
    mismatch_mask = key_series != expected

    if mismatch_mask.any():
        bad = pd.DataFrame(
            {
                record_col: record_series[mismatch_mask],
                beat_col: beat_series[mismatch_mask],
                key_col: key_series[mismatch_mask],
                "expected_record_beat_index": expected[mismatch_mask],
            }
        ).head(10)

        raise ValueError(
            f"Hibás {key_col} képzés a(z) {df_name} táblában. "
            f"Elvárt forma: record_id + '_' + beat_index.\n"
            f"Példák:\n{bad}"
        )

    dup_mask = key_series.duplicated(keep=False)
    if dup_mask.any():
        bad = pd.DataFrame(
            {
                record_col: record_series[dup_mask],
                beat_col: beat_series[dup_mask],
                key_col: key_series[dup_mask],
            }
        ).head(10)

        raise ValueError(
            f"Duplikált {key_col} a(z) {df_name} táblában.\n"
            f"Példák:\n{bad}"
        )

def load_npz_split(
    path: str | Path,
    *,
    return_meta: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    path = Path(path)
    data = np.load(path, allow_pickle=True)

    required = {"X", "y", "record_id", "beat_index", "record_beat_index"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(
            f"Az NPZ nem kompatibilis az új sémával: {path}\n"
            f"Hiányzó mező(k): {sorted(missing)}\n"
            f"Elérhető mezők: {sorted(data.files)}"
        )

    X = data["X"]
    y = data["y"]

    meta_dict: dict[str, Any] = {
        "record_id": data["record_id"],
        "beat_index": data["beat_index"],
        "record_beat_index": data["record_beat_index"],
    }

    if "center_sample" in data.files:
        meta_dict["center_sample"] = data["center_sample"]
    if "label_raw" in data.files:
        meta_dict["label_raw"] = data["label_raw"]
    if "label_aami" in data.files:
        meta_dict["label_aami"] = data["label_aami"]

    beat_meta_df = pd.DataFrame(meta_dict)
    beat_meta_df = cast_beat_meta_schema(beat_meta_df)

    assert_same_length(X, y, beat_meta_df)
    validate_record_beat_index(beat_meta_df, str(path))

    if return_meta:
        return X, y, beat_meta_df

    return X, y, beat_meta_df["record_beat_index"].astype(str).to_numpy(dtype=object)