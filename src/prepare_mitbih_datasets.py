#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    BEAT_POST_SEC,
    BEAT_PRE_SEC,
    BEAT_QUALITY_CONFIG,
    DEFAULT_FS,
    PREPROCESSING_CONFIG,
    encode_target_labels,
    get_ds_par,
    get_label_mapping,
    get_label_mode,
    get_label_mode_classes,
    remap_aami_labels,
)
from src.utils.beat_extractor import extract_beats
from src.utils.beat_quality import filter_beats_by_quality
from src.utils.dataset_loader import load_mitbih_record, validate_record_beat_index
from src.utils.labels import compute_label_distribution, map_aami_labels

VALID_BEAT_SYMBOLS = {
    "N", "L", "R", "A", "a", "J", "S", "V", "F", "e", "j", "E", "/", "f", "Q"
}
RANDOM_STATE = 42

BEAT_INDEX_COLUMNS = ["record_id", "beat_index"]
BEAT_INDEX_DTYPES = {"record_id": "string", "beat_index": "int64"}
BEAT_META_DTYPES = {
    "record_id": "string",
    "beat_index": "int64",
    "record_beat_index": "string",
    "center_sample": "int64",
    "label_raw": "string",
    "label_aami": "string",
    "label_target": "string",
}


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

    for col in ["center_sample"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="raise").astype("int64")
    for col in ["label_raw", "label_aami", "label_target", "split"]:
        if col in out.columns:
            out[col] = out[col].astype("string")
    return out


def finalize_beat_meta(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    out = cast_beat_meta_schema(df)
    out["split"] = split_name
    validate_record_beat_index(out, f"{split_name}_beat_meta")
    return out


def get_n_extract_skipped(extract_meta: pd.DataFrame) -> int:
    if extract_meta is None or extract_meta.empty or "kept" not in extract_meta.columns:
        return 0
    return int((~extract_meta["kept"]).sum())


def build_dataset(record_list: list[str]):
    X_parts: list[np.ndarray] = []
    y_target_parts: list[np.ndarray] = []
    beat_meta_parts: list[pd.DataFrame] = []

    record_qc_rows: list[dict] = []
    all_beat_qc: list[pd.DataFrame] = []

    fs = int(DEFAULT_FS)
    pre_samples = int(BEAT_PRE_SEC * fs)

    for record_name in record_list:
        rec = load_mitbih_record(record_id=record_name)

        raw_beats, raw_labels, raw_centers, extract_meta = extract_beats(
            rec,
            channel=0,
            fs=fs,
            pre_sec=BEAT_PRE_SEC,
            post_sec=BEAT_POST_SEC,
            normalize=True,
            allowed_symbols=VALID_BEAT_SYMBOLS,
            edge_policy="skip",
            preprocessing_config=PREPROCESSING_CONFIG,
            dataset="mitbih",
        )

        n_extract_skipped = get_n_extract_skipped(extract_meta)

        if len(raw_beats) == 0:
            record_qc_rows.append({
                "record_id": str(record_name),
                "n_beats_after_extract": 0,
                "n_beats_before_quality_filter": 0,
                "n_beats_after_quality_filter": 0,
                "n_removed_by_quality_filter": 0,
                "fraction_removed_by_quality_filter": np.nan,
                "n_after_label_filter": 0,
                "n_extract_skipped": n_extract_skipped,
            })
            continue

        if not extract_meta.empty:
            extract_meta = extract_meta.copy()
            extract_meta["record_id"] = str(record_name)
            extract_meta["stage"] = "extract"
            all_beat_qc.append(extract_meta)

        kept_extract_meta = extract_meta.loc[extract_meta["kept"] == True].copy().reset_index(drop=True)
        raw_beat_index = kept_extract_meta["ann_index"].to_numpy(dtype=np.int64)

        beats_qf, labels_qf, centers_qf, beat_qc_df = filter_beats_by_quality(
            beats=raw_beats,
            labels=raw_labels,
            centers=raw_centers,
            beat_pre_samples=pre_samples,
            **BEAT_QUALITY_CONFIG,
        )

        beat_qc_df = beat_qc_df.copy()
        beat_qc_df["record_id"] = str(record_name)
        beat_qc_df["stage"] = "quality"
        all_beat_qc.append(beat_qc_df)

        quality_keep_mask = beat_qc_df["is_valid"].to_numpy(dtype=bool)
        beat_index_qf = raw_beat_index[quality_keep_mask]

        aami_labels = np.asarray(map_aami_labels(labels_qf), dtype=object)
        target_labels = remap_aami_labels(aami_labels)
        y_target = encode_target_labels(target_labels)

        beat_meta_df = pd.DataFrame({
            "record_id": pd.Series([str(record_name)] * len(beats_qf), dtype="string"),
            "beat_index": np.asarray(beat_index_qf, dtype=np.int64),
            "record_beat_index": pd.Series([f"{record_name}_{idx}" for idx in beat_index_qf], dtype="string"),
            "center_sample": np.asarray(centers_qf, dtype=np.int64),
            "label_raw": pd.Series(labels_qf, dtype="string"),
            "label_aami": pd.Series(aami_labels, dtype="string"),
            "label_target": pd.Series(target_labels, dtype="string"),
        })
        beat_meta_df = cast_beat_meta_schema(beat_meta_df)
        validate_record_beat_index(beat_meta_df, f"beat_meta_{record_name}")

        X_parts.append(beats_qf)
        y_target_parts.append(y_target)
        beat_meta_parts.append(beat_meta_df)

        record_qc_rows.append({
            "record_id": str(record_name),
            "n_beats_after_extract": int(len(raw_beats)),
            "n_beats_before_quality_filter": int(len(raw_beats)),
            "n_beats_after_quality_filter": int(len(beats_qf)),
            "n_removed_by_quality_filter": int(len(raw_beats) - len(beats_qf)),
            "fraction_removed_by_quality_filter": float((len(raw_beats) - len(beats_qf)) / max(len(raw_beats), 1)),
            "n_after_label_filter": int(len(beats_qf)),
            "n_extract_skipped": n_extract_skipped,
        })

    beat_len = int((BEAT_PRE_SEC + BEAT_POST_SEC) * fs)

    if not X_parts:
        empty_X = np.empty((0, beat_len), dtype=float)
        empty_y = np.empty((0,), dtype=np.int64)
        empty_meta = cast_beat_meta_schema(pd.DataFrame({
            "record_id": pd.Series([], dtype="string"),
            "beat_index": pd.Series([], dtype="int64"),
            "record_beat_index": pd.Series([], dtype="string"),
            "center_sample": pd.Series([], dtype="int64"),
            "label_raw": pd.Series([], dtype="string"),
            "label_aami": pd.Series([], dtype="string"),
            "label_target": pd.Series([], dtype="string"),
        }))
        return empty_X, empty_y, empty_meta, pd.DataFrame(record_qc_rows), pd.DataFrame()

    X = np.vstack(X_parts).astype(np.float32)
    y = np.concatenate(y_target_parts).astype(np.int64)
    beat_meta_df = pd.concat(beat_meta_parts, ignore_index=True)
    beat_meta_df = cast_beat_meta_schema(beat_meta_df)
    validate_record_beat_index(beat_meta_df, "beat_meta_df")

    record_qc_df = pd.DataFrame(record_qc_rows)
    if not record_qc_df.empty:
        record_qc_df["record_id"] = cast_record_id(record_qc_df["record_id"])

    beat_qc_df = pd.concat(all_beat_qc, ignore_index=True) if all_beat_qc else pd.DataFrame()
    if not beat_qc_df.empty and "record_id" in beat_qc_df.columns:
        beat_qc_df["record_id"] = cast_record_id(beat_qc_df["record_id"])

    return X, y, beat_meta_df, record_qc_df, beat_qc_df


def split_records() -> tuple[list[str], list[str], list[str]]:
    train_records, test_records = train_test_split(
        get_ds_par("mitbih", "records"),
        test_size=0.20,
        random_state=RANDOM_STATE,
    )
    train_records, val_records = train_test_split(
        train_records,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )
    return list(train_records), list(val_records), list(test_records)


def save_split_npz(out_path: Path, X: np.ndarray, y: np.ndarray, beat_meta_df: pd.DataFrame) -> None:
    beat_meta_df = cast_beat_meta_schema(beat_meta_df)
    validate_record_beat_index(beat_meta_df, out_path.name)

    np.savez_compressed(
        out_path,
        X=X.astype(np.float32),
        y=y.astype(np.int64),
        record_id=beat_meta_df["record_id"].astype(str).to_numpy(dtype=object),
        beat_index=beat_meta_df["beat_index"].to_numpy(dtype=np.int64),
        record_beat_index=beat_meta_df["record_beat_index"].astype(str).to_numpy(dtype=object),
        center_sample=beat_meta_df["center_sample"].to_numpy(dtype=np.int64),
        label_raw=beat_meta_df["label_raw"].astype(str).to_numpy(dtype=object),
        label_aami=beat_meta_df["label_aami"].astype(str).to_numpy(dtype=object),
        label_target=beat_meta_df["label_target"].astype(str).to_numpy(dtype=object),
    )


def save_distribution_csv(labels_or_y: np.ndarray, split_name: str, out_path: Path, class_names: list[str]) -> None:
    if np.issubdtype(np.asarray(labels_or_y).dtype, np.integer):
        labels = np.asarray([class_names[int(i)] for i in labels_or_y], dtype=object)
    else:
        labels = np.asarray(labels_or_y, dtype=object)

    counts = pd.Series(labels).value_counts().reindex(class_names, fill_value=0)
    total = counts.sum()
    df = pd.DataFrame({
        "label": counts.index,
        "count": counts.values,
        "percent": ((counts / total) * 100).round(2) if total > 0 else 0.0,
        "split": split_name,
    })
    df.to_csv(out_path, index=False)


def build_metadata(split_data: dict[str, dict[str, object]]) -> dict:
    return {
        "source_dataset": "mitbih",
        "display_name": get_ds_par("mitbih", "display_name"),
        "label_mode": get_label_mode(),
        "class_names": get_label_mode_classes(),
        "label_mapping": get_label_mapping(),
        "fs": int(get_ds_par("mitbih", "fs")),
        "beat_pre_sec": BEAT_PRE_SEC,
        "beat_post_sec": BEAT_POST_SEC,
        "beat_pre_samples": int(BEAT_PRE_SEC * DEFAULT_FS),
        "beat_post_samples": int(BEAT_POST_SEC * DEFAULT_FS),
        "beat_length": int((BEAT_PRE_SEC + BEAT_POST_SEC) * DEFAULT_FS),
        "random_state": RANDOM_STATE,
        "preprocessing_config": PREPROCESSING_CONFIG,
        "beat_quality_config": BEAT_QUALITY_CONFIG,
        "beat_index_columns": BEAT_INDEX_COLUMNS,
        "beat_index_dtypes": BEAT_INDEX_DTYPES,
        "beat_meta_dtypes": BEAT_META_DTYPES,
        "train_records": split_data["train"]["records"],
        "val_records": split_data["val"]["records"],
        "test_records": split_data["test"]["records"],
        "train_shape": list(split_data["train"]["X"].shape),
        "val_shape": list(split_data["val"]["X"].shape),
        "test_shape": list(split_data["test"]["X"].shape),
    }


def prepare_mitbih_datasets() -> None:
    outdir = Path(get_ds_par("mitbih", "beats_dir"))
    outdir.mkdir(parents=True, exist_ok=True)

    train_records, val_records, test_records = split_records()

    X_train, y_train, meta_train, qc_train, bqc_train = build_dataset(train_records)
    X_val, y_val, meta_val, qc_val, bqc_val = build_dataset(val_records)
    X_test, y_test, meta_test, qc_test, bqc_test = build_dataset(test_records)

    split_data = {
        "train": {"X": X_train, "y": y_train, "beat_meta": finalize_beat_meta(meta_train, "train"), "record_qc": qc_train, "beat_qc": bqc_train, "records": train_records},
        "val": {"X": X_val, "y": y_val, "beat_meta": finalize_beat_meta(meta_val, "val"), "record_qc": qc_val, "beat_qc": bqc_val, "records": val_records},
        "test": {"X": X_test, "y": y_test, "beat_meta": finalize_beat_meta(meta_test, "test"), "record_qc": qc_test, "beat_qc": bqc_test, "records": test_records},
    }

    for split_name in ["train", "val", "test"]:
        save_split_npz(outdir / f"{split_name}.npz", split_data[split_name]["X"], split_data[split_name]["y"], split_data[split_name]["beat_meta"])
        split_data[split_name]["beat_meta"].to_csv(outdir / f"{split_name}_beat_meta.csv", index=False)

    record_qc_df = pd.concat([qc_train.assign(split="train"), qc_val.assign(split="val"), qc_test.assign(split="test")], ignore_index=True)
    record_qc_df.to_csv(outdir / "record_quality_summary.csv", index=False)

    beat_qc_tables = []
    for split_name in ["train", "val", "test"]:
        df = split_data[split_name]["beat_qc"]
        if not df.empty:
            beat_qc_tables.append(df.assign(split=split_name))
    beat_qc_df = pd.concat(beat_qc_tables, ignore_index=True) if beat_qc_tables else pd.DataFrame()
    beat_qc_df.to_csv(outdir / "beat_quality_details.csv", index=False)

    class_names = get_label_mode_classes()
    save_distribution_csv(y_train, "train", outdir / "train_label_distribution.csv", class_names)
    save_distribution_csv(y_val, "val", outdir / "val_label_distribution.csv", class_names)
    save_distribution_csv(y_test, "test", outdir / "test_label_distribution.csv", class_names)
    save_distribution_csv(np.concatenate([y_train, y_val, y_test]), "global", outdir / "global_label_distribution.csv", class_names)

    with open(outdir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(build_metadata(split_data), f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved to: {outdir}")


if __name__ == "__main__":
    prepare_mitbih_datasets()