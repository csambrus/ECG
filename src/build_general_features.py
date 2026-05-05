#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_ds_par
from src.utils.features_basic import extract_feature_dataframe
from src.utils.dataset_loader import load_metadata, load_npz_split, validate_record_beat_index


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def invert_mapping(mapping: dict[str, int]) -> dict[int, str]:
    return {int(v): str(k) for k, v in mapping.items()}


def decode_labels(y: np.ndarray, inv_mapping: dict[int, str]) -> np.ndarray:
    return np.asarray([inv_mapping[int(v)] for v in y], dtype=object)


def cast_record_id(values: pd.Series | np.ndarray | list[str]) -> pd.Series:
    return pd.Series(values, copy=False).astype("string").str.strip()


def cast_beat_feature_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    required = {"record_id", "beat_index", "record_beat_index", "y"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Hiányzó kötelező oszlop(ok): {sorted(missing)}")

    out["record_id"] = cast_record_id(out["record_id"])
    out["beat_index"] = pd.to_numeric(out["beat_index"], errors="raise").astype("int64")
    out["record_beat_index"] = out["record_beat_index"].astype("string")
    out["y"] = pd.to_numeric(out["y"], errors="raise").astype("int64")

    if "label" in out.columns:
        out["label"] = out["label"].astype("string")
    if "split" in out.columns:
        out["split"] = out["split"].astype("string")
    if "label_raw" in out.columns:
        out["label_raw"] = out["label_raw"].astype("string")
    if "label_aami" in out.columns:
        out["label_aami"] = out["label_aami"].astype("string")

    return out


def build_split_feature_table(
    split_name: str,
    X: np.ndarray,
    y: np.ndarray,
    beat_meta_df: pd.DataFrame,
    inv_mapping: dict[int, str],
) -> pd.DataFrame:
    feat_df = extract_feature_dataframe(X)

    if len(feat_df) != len(beat_meta_df):
        raise ValueError(
            f"Feature és meta sorok száma eltér a(z) {split_name} splitben: "
            f"{len(feat_df)} vs {len(beat_meta_df)}"
        )

    feat_df["split"] = split_name
    feat_df["record_id"] = cast_record_id(beat_meta_df["record_id"])
    feat_df["beat_index"] = pd.to_numeric(beat_meta_df["beat_index"], errors="raise").astype("int64")
    feat_df["record_beat_index"] = beat_meta_df["record_beat_index"].astype("string")
    feat_df["y"] = y.astype(np.int64)
    feat_df["label"] = pd.Series(decode_labels(y, inv_mapping), dtype="string")
    feat_df["sample_index"] = np.arange(len(feat_df), dtype=np.int64)

    if "center_sample" in beat_meta_df.columns:
        feat_df["center_sample"] = pd.to_numeric(
            beat_meta_df["center_sample"], errors="raise"
        ).astype("int64")

    if "label_raw" in beat_meta_df.columns:
        feat_df["label_raw"] = beat_meta_df["label_raw"].astype("string")

    if "label_aami" in beat_meta_df.columns:
        feat_df["label_aami"] = beat_meta_df["label_aami"].astype("string")

    feat_df = cast_beat_feature_schema(feat_df)
    validate_record_beat_index(feat_df, f"{split_name}_general_features")

    meta_cols = [
        c for c in [
            "split",
            "record_id",
            "beat_index",
            "record_beat_index",
            "sample_index",
            "center_sample",
            "y",
            "label",
            "label_raw",
            "label_aami",
        ]
        if c in feat_df.columns
    ]
    feature_cols = [c for c in feat_df.columns if c not in meta_cols]
    return feat_df[meta_cols + feature_cols]


def summarize_split(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [
        c for c in numeric_cols if c not in {"y", "sample_index", "beat_index", "center_sample"}
    ]

    label_counts = (
        df["label"].value_counts(dropna=False).sort_index().to_dict()
        if "label" in df.columns
        else {}
    )

    missing_by_column = df.isna().sum().to_dict()

    return {
        "n_rows": int(len(df)),
        "n_records": int(df["record_id"].nunique()) if "record_id" in df.columns else 0,
        "n_numeric_features": int(len(numeric_cols)),
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "missing_by_column": {str(k): int(v) for k, v in missing_by_column.items()},
    }


def save_manifest(
    dataset: str,
    features_dir: Path,
    metadata: dict,
    split_files: dict[str, str],
    split_summaries: dict[str, dict],
) -> None:
    manifest_path = features_dir / "manifest_general_features.json"

    manifest = {
        "artifact_type": f"{dataset}_general_features",
        "created_at": now_iso(),
        "source_dataset": metadata.get("source_dataset", dataset),
        "beat_pre_samples": metadata.get("beat_pre_samples"),
        "beat_post_samples": metadata.get("beat_post_samples"),
        "beat_length": metadata.get("beat_length"),
        "label_mapping": metadata.get("label_mapping"),
        "beat_index_columns": metadata.get("beat_index_columns"),
        "beat_index_dtypes": metadata.get("beat_index_dtypes"),
        "beat_meta_dtypes": metadata.get("beat_meta_dtypes"),
        "train_records": metadata.get("train_records"),
        "val_records": metadata.get("val_records"),
        "test_records": metadata.get("test_records"),
        "feature_files": split_files,
        "split_summaries": split_summaries,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    report_input = {
        "created_at": now_iso(),
        "feature_family": "general",
        "dataset": dataset,
        "manifest_file": str(manifest_path),
        "split_files": split_files,
        "split_summaries": split_summaries,
    }

    with open(features_dir / "report_input_general_features.json", "w", encoding="utf-8") as f:
        json.dump(report_input, f, ensure_ascii=False, indent=2)


def build_general_features(dataset: str = "mitbih") -> None:

    dataset = dataset.lower().strip()

    dataset_dir = Path(get_ds_par(dataset, "beats_dir"))
    features_dir = Path(get_ds_par(dataset, "features_general_dir"))

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Hiányzó dataset_dir: {dataset_dir}")

    features_dir.mkdir(parents=True, exist_ok=True)

    print("DATA DIR:", dataset_dir)
    print("OUT DIR:", features_dir)

    metadata_path = dataset_dir / "metadata.json"
    metadata = load_metadata(metadata_path) if metadata_path.exists() else {}

    label_mapping = metadata.get("label_mapping")
    if not label_mapping:
        raise ValueError(f"Hiányzik a label_mapping a metadata.json-ból: {metadata_path}")

    inv_mapping = invert_mapping(label_mapping)

    split_files: dict[str, str] = {}
    split_summaries: dict[str, dict] = {}

    for split_name in ["train", "val", "test"]:
        X, y, beat_meta_df = load_npz_split(
            dataset_dir / f"{split_name}.npz",
            return_meta=True,
        )

        df = build_split_feature_table(
            split_name=split_name,
            X=X,
            y=y,
            beat_meta_df=beat_meta_df,
            inv_mapping=inv_mapping,
        )

        out_csv = features_dir / f"{split_name}_general_features.csv"
        df.to_csv(out_csv, index=False)

        split_files[split_name] = str(out_csv)
        split_summaries[split_name] = summarize_split(df)

        print(f"[OK] Saved: {out_csv}")

    save_manifest(
        dataset=dataset,
        features_dir=features_dir,
        metadata=metadata,
        split_files=split_files,
        split_summaries=split_summaries,
    )

    print(f"[OK] Manifest: {features_dir / 'manifest_general_features.json'}")
    print(f"[OK] Report input: {features_dir / 'report_input_general_features.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build general ECG feature tables.")
    parser.add_argument("dataset", type=str, nargs="?", default="mitbih")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_general_features(
        dataset=args.dataset,
    )