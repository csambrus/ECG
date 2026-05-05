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
    DEFAULT_FS,
    PREPROCESSING_CONFIG,
    encode_target_labels,
    get_ds_par,
    get_label_mapping,
    get_label_mode,
    get_label_mode_classes,
    remap_aami_labels,
)
from src.utils.beat_extractor import extract_beats_multichannel
from src.utils.dataset_loader import load_mitbih_record, validate_record_beat_index
from src.utils.labels import map_aami_labels

RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE_FROM_TRAIN = 0.20

VALID_BEAT_SYMBOLS = {
    "N", "L", "R", "A", "a", "J", "S", "V", "F", "e", "j", "E", "/", "f", "Q"
}


def save_split_npz(path: Path, X: np.ndarray, y: np.ndarray, beat_meta_df: pd.DataFrame) -> None:
    np.savez_compressed(
        path,
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


def build_dataset_for_records(record_ids: list[str]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    all_beats: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_meta: list[pd.DataFrame] = []

    fs = int(DEFAULT_FS)

    for record_id in record_ids:
        print(f"[BUILD-2CH] {record_id}")

        rec = load_mitbih_record(record_id=record_id)

        beats, raw_symbols, beat_centers, extract_meta = extract_beats_multichannel(
            rec=rec,
            channels=[0, 1],
            fs=fs,
            pre_sec=BEAT_PRE_SEC,
            post_sec=BEAT_POST_SEC,
            normalize=True,
            allowed_symbols=VALID_BEAT_SYMBOLS,
            edge_policy="skip",
            preprocessing_config=PREPROCESSING_CONFIG,
            dataset="mitbih",
        )

        if len(raw_symbols) == 0:
            print(f"[WARN] {record_id}: no extracted beats")
            continue

        kept_meta = extract_meta.loc[extract_meta["kept"] == True].copy().reset_index(drop=True)
        beat_index = (
            kept_meta["ann_index"].to_numpy(dtype=np.int64)
            if "ann_index" in kept_meta.columns
            else np.arange(len(raw_symbols), dtype=np.int64)
        )

        aami_labels = np.asarray(map_aami_labels(raw_symbols), dtype=object)
        target_labels = remap_aami_labels(aami_labels)
        y = encode_target_labels(target_labels)

        meta_df = pd.DataFrame(
            {
                "record_id": pd.Series([str(record_id)] * len(beats), dtype="string"),
                "beat_index": beat_index,
                "record_beat_index": pd.Series(
                    [f"{record_id}_{i}" for i in beat_index],
                    dtype="string",
                ),
                "center_sample": np.asarray(beat_centers, dtype=np.int64),
                "label_raw": pd.Series(np.asarray(raw_symbols, dtype=object), dtype="string"),
                "label_aami": pd.Series(aami_labels, dtype="string"),
                "label_target": pd.Series(target_labels, dtype="string"),
            }
        )
        validate_record_beat_index(meta_df, f"mitbih2ch_meta_{record_id}")

        all_beats.append(beats.astype(np.float32))
        all_y.append(y.astype(np.int64))
        all_meta.append(meta_df)

    if not all_beats:
        raise ValueError("Nem sikerült 2 csatornás beat-eket előállítani a MIT-BIH rekordokból.")

    X = np.concatenate(all_beats, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0).astype(np.int64)
    beat_meta_df = pd.concat(all_meta, ignore_index=True)

    if len(X) != len(y) or len(X) != len(beat_meta_df):
        raise ValueError(
            "Hossz-inkonzisztencia a MIT-BIH 2ch datasetben: "
            f"len(X)={len(X)}, len(y)={len(y)}, len(meta)={len(beat_meta_df)}"
        )

    return X, y, beat_meta_df


def summarize_split(
    split_name: str,
    X: np.ndarray,
    y: np.ndarray,
    beat_meta_df: pd.DataFrame,
    records: list[str],
) -> dict:
    class_names = get_label_mode_classes()
    labels = np.asarray([class_names[int(v)] for v in y], dtype=object)
    label_counts = pd.Series(labels).value_counts().reindex(class_names, fill_value=0)

    return {
        "split": split_name,
        "n_records": int(len(records)),
        "records": list(records),
        "n_beats": int(len(y)),
        "X_shape": list(X.shape),
        "label_distribution": {str(k): int(v) for k, v in label_counts.items()},
        "n_unique_record_beat_index": int(beat_meta_df["record_beat_index"].nunique()),
    }


def save_metadata(
    train_records: list[str],
    val_records: list[str],
    test_records: list[str],
    out_dir: Path,
) -> None:
    beat_pre_samples = int(DEFAULT_FS * BEAT_PRE_SEC)
    beat_post_samples = int(DEFAULT_FS * BEAT_POST_SEC)

    metadata = {
        "source_dataset": "mitbih",
        "display_name": get_ds_par("mitbih", "display_name"),
        "label_mode": get_label_mode(),
        "class_names": get_label_mode_classes(),
        "label_mapping": get_label_mapping(),
        "n_channels": 2,
        "channel_indices": [0, 1],
        "fs_original": int(get_ds_par("mitbih", "fs")),
        "fs": int(DEFAULT_FS),
        "beat_pre_sec": float(BEAT_PRE_SEC),
        "beat_post_sec": float(BEAT_POST_SEC),
        "beat_pre_samples": int(beat_pre_samples),
        "beat_post_samples": int(beat_post_samples),
        "beat_length": int(beat_pre_samples + beat_post_samples),
        "preprocessing_config": PREPROCESSING_CONFIG,
        "train_records": train_records,
        "val_records": val_records,
        "test_records": test_records,
        "cnn_variant": "2ch",
        "input_kind": "2ch",
    }

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def prepare_mitbih_datasets_2ch() -> None:
    out_dir = Path(get_ds_par("mitbih", "cnn2_data_dir"))
    out_dir.mkdir(parents=True, exist_ok=True)

    train_records, test_records = train_test_split(
        get_ds_par("mitbih", "records"),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    train_records, val_records = train_test_split(
        train_records,
        test_size=VAL_SIZE_FROM_TRAIN,
        random_state=RANDOM_STATE,
    )

    train_records = list(train_records)
    val_records = list(val_records)
    test_records = list(test_records)

    X_train, y_train, meta_train = build_dataset_for_records(train_records)
    X_val, y_val, meta_val = build_dataset_for_records(val_records)
    X_test, y_test, meta_test = build_dataset_for_records(test_records)

    save_split_npz(out_dir / "train.npz", X_train, y_train, meta_train)
    save_split_npz(out_dir / "val.npz", X_val, y_val, meta_val)
    save_split_npz(out_dir / "test.npz", X_test, y_test, meta_test)

    meta_train.to_csv(out_dir / "train_beat_meta.csv", index=False)
    meta_val.to_csv(out_dir / "val_beat_meta.csv", index=False)
    meta_test.to_csv(out_dir / "test_beat_meta.csv", index=False)

    split_summary = {
        "train": summarize_split("train", X_train, y_train, meta_train, train_records),
        "val": summarize_split("val", X_val, y_val, meta_val, val_records),
        "test": summarize_split("test", X_test, y_test, meta_test, test_records),
    }

    with open(out_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2, ensure_ascii=False)

    records_df = pd.DataFrame(
        {
            "split": ["train"] * len(train_records)
            + ["val"] * len(val_records)
            + ["test"] * len(test_records),
            "record_id": train_records + val_records + test_records,
        }
    )
    records_df.to_csv(out_dir / "record_split.csv", index=False)

    save_metadata(train_records, val_records, test_records, out_dir)
    print(f"[OK] Saved 2ch MIT-BIH dataset to: {out_dir}")


if __name__ == "__main__":
    prepare_mitbih_datasets_2ch()