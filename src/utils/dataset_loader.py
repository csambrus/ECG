from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wfdb

from src.config import get_ds_par


INCART_REQUIRED_EXTENSIONS = [".dat", ".hea", ".atr"]


@dataclass
class ECGRecord:
    record_id: str
    signal: np.ndarray
    fs: float
    sig_names: list[str]
    annotation_samples: np.ndarray
    annotation_symbols: np.ndarray



def load_record(
    dataset: str,
    record_name: str,
    data_dir: Optional[str | Path] = None,
    **kwargs: Any,
):
    """
    Dataset-agnosztikus rekordbetöltő dispatcher.

    Parameters
    ----------
    dataset : str
        Dataset neve, pl. "mitbih", "incart".
    record_name : str
        Rekord neve / azonosítója, pl. "100".
    data_dir : Optional[str | Path]
        Opcionális adatkönyvtár. Ha None, akkor a dataset-specifikus loader
        a saját default könyvtárát használja.
    **kwargs : Any
        További dataset-specifikus paraméterek továbbadása.

    Returns
    -------
    Any
        A dataset-specifikus loader által visszaadott rekord objektum.

    Raises
    ------
    ValueError
        Ha a dataset nem támogatott.
    """
    ds = dataset.strip().lower()

    if ds == "mitbih":
        from src.utils.mitbih_loader import load_record_from_dir

        if data_dir is None:
            return load_record_from_dir(record_id=record_name, **kwargs)
        return load_record_from_dir(
            record_id=record_name,
            data_dir=Path(data_dir),
            **kwargs,
        )

    if ds == "incart":
        from src.utils.incart_loader import load_incart_record

        if data_dir is None:
            return load_incart_record(record_id=record_name, **kwargs)
        return load_incart_record(
            record_id=record_name,
            data_dir=Path(data_dir),
            **kwargs,
        )

    raise ValueError(
        f"Nem támogatott dataset: {dataset!r}. "
        f"Támogatottak jelenleg: ['mitbih', 'incart']"
    )


def load_metadata(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_wfdb_record(
    record_id: str,
    data_dir: str | Path,
    verbose: int = 0,
    log_prefix: str = "load_record",
) -> ECGRecord:
    data_dir = Path(data_dir)
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
            f"[{log_prefix}] {record_id} | "
            f"shape={signal.shape}, fs={fs}, "
            f"ch={len(sig_names)}, ann={len(annotation_samples)}"
        )

    if verbose >= 2:
        unique, counts = np.unique(annotation_symbols, return_counts=True)
        dist = dict(zip(unique, counts))
        print(f"  channels: {sig_names}")
        print(f"  first 5 symbols: {annotation_symbols[:5].tolist()}")
        print(f"  symbol dist: {dist}")

    return ECGRecord(
        record_id=str(record_id),
        signal=signal,
        fs=fs,
        sig_names=sig_names,
        annotation_samples=annotation_samples,
        annotation_symbols=annotation_symbols,
    )


def load_mitbih_record(
    record_id: str,
    data_dir: str | Path | None = None,
    verbose: int = 0,
) -> ECGRecord:
    return _load_wfdb_record(
        record_id=record_id,
        data_dir=data_dir or get_ds_par("mitbih", "raw_dir"),
        verbose=verbose,
        log_prefix="load_mitbih_record",
    )


def is_incart_record_available(
    record_id: str,
    data_dir: str | Path | None = None,
) -> bool:
    data_dir = Path(data_dir or get_ds_par("incart", "raw_dir"))
    return all(
        (data_dir / f"{record_id}{ext}").exists()
        for ext in INCART_REQUIRED_EXTENSIONS
    )


def extract_incart_zip(
    zip_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    overwrite: bool = False,
) -> None:
    zip_path = Path(zip_path or get_ds_par("incart", "incartdb_zip"))
    data_dir = Path(data_dir or get_ds_par("incart", "raw_dir"))

    if not zip_path.exists():
        raise FileNotFoundError(f"INCART zip file not found: {zip_path}")

    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"[EXTRACT] {zip_path} -> {data_dir}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            member_path = Path(member)

            if member.endswith("/"):
                continue

            if member_path.suffix.lower() not in {".dat", ".hea", ".atr"}:
                continue

            target_path = data_dir / member_path.name

            if target_path.exists() and not overwrite:
                continue

            with zf.open(member) as src, open(target_path, "wb") as dst:
                dst.write(src.read())

    print("[OK] INCART zip extracted")


def download_incart_record(
    record_id: str,
    data_dir: str | Path | None = None,
) -> None:
    data_dir = Path(data_dir or get_ds_par("incart", "raw_dir"))
    data_dir.mkdir(parents=True, exist_ok=True)

    if is_incart_record_available(record_id, data_dir=data_dir):
        print(f"[SKIP] {record_id} already available")
        return

    print(f"[DOWNLOAD] {record_id}")
    wfdb.dl_database(
        "incartdb",
        dl_dir=str(data_dir),
        records=[record_id],
    )

    if is_incart_record_available(record_id, data_dir=data_dir):
        print(f"[OK] {record_id}")
    else:
        print(f"[WARN] {record_id} incomplete download")


def ensure_incart_record_available(
    record_id: str,
    data_dir: str | Path | None = None,
    zip_path: str | Path | None = None,
) -> None:
    data_dir = Path(data_dir or get_ds_par("incart", "raw_dir"))

    if is_incart_record_available(record_id, data_dir=data_dir):
        return

    effective_zip = Path(zip_path or get_ds_par("incart", "incartdb_zip"))

    if effective_zip.exists():
        extract_incart_zip(zip_path=effective_zip, data_dir=data_dir)

        if is_incart_record_available(record_id, data_dir=data_dir):
            return

        print(f"[WARN] Record {record_id} still missing after zip extraction")

    download_incart_record(record_id=record_id, data_dir=data_dir)


def ensure_all_incart_available(
    data_dir: str | Path | None = None,
    zip_path: str | Path | None = None,
    record_ids: list[str] | None = None,
) -> None:
    data_dir = Path(data_dir or get_ds_par("incart", "raw_dir"))
    record_ids = record_ids or get_ds_par("incart", "records")

    missing = [
        r for r in record_ids
        if not is_incart_record_available(r, data_dir=data_dir)
    ]
    if not missing:
        return

    effective_zip = Path(zip_path or get_ds_par("incart", "incartdb_zip"))

    if effective_zip.exists():
        extract_incart_zip(zip_path=effective_zip, data_dir=data_dir)
        missing = [
            r for r in record_ids
            if not is_incart_record_available(r, data_dir=data_dir)
        ]

    for record_id in missing:
        download_incart_record(record_id=record_id, data_dir=data_dir)


def load_incart_record(
    record_id: str,
    data_dir: str | Path | None = None,
    verbose: int = 0,
    ensure_available: bool = True,
    zip_path: str | Path | None = None,
) -> ECGRecord:
    data_dir = Path(data_dir or get_ds_par("incart", "raw_dir"))

    if ensure_available:
        ensure_incart_record_available(
            record_id=record_id,
            data_dir=data_dir,
            zip_path=zip_path,
        )

    return _load_wfdb_record(
        record_id=record_id,
        data_dir=data_dir,
        verbose=verbose,
        log_prefix="load_incart_record",
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
        out["center_sample"] = pd.to_numeric(
            out["center_sample"], errors="raise"
        ).astype("int64")

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


def load_record_by_dataset(
    dataset: str,
    record_name: str,
    verbose: int = 0,
    ensure_available: bool = True,
    zip_path: str | Path | None = None,
):
    dataset = dataset.lower().strip()

    if dataset == "mitbih":
        return load_mitbih_record(
            record_id=record_name,
            verbose=verbose,
        )

    if dataset == "incart":
        return load_incart_record(
            record_id=record_name,
            verbose=verbose,
            ensure_available=ensure_available,
            zip_path=zip_path,
        )

    raise ValueError(f"Unknown dataset: {dataset}")