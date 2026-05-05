#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.beat_extractor import extract_beats
from src.config import (
    BEAT_POST_SEC,
    BEAT_PRE_SEC,
    PREPROCESSING_CONFIG,
    PREFERRED_LEADS,
    get_ds_par,
)
from src.utils.features_morphology import (
    add_morphology_quality_flags,
    build_morphology_feature_dataframe_from_full_annotations,
)
from src.utils.labels import map_aami_labels
from src.utils.dataset_loader import (
    load_record_by_dataset,
    load_npz_split,
    validate_record_beat_index,
)

VALID_BEAT_SYMBOLS = {
    "N", "L", "R", "A", "a", "J", "S", "V", "F", "e", "j", "E", "/", "f", "Q"
}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_prepare_metadata(beats_dir: Path) -> dict:
    meta_path = beats_dir / "metadata.json"
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def cast_record_id(values: pd.Series | np.ndarray | list[str]) -> pd.Series:
    return pd.Series(values, copy=False).astype("string").str.strip()


def cast_morphology_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    required = {"record_id", "beat_index", "record_beat_index"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Hiányzó morphology kulcs(oszlop): {sorted(missing)}")

    out["record_id"] = cast_record_id(out["record_id"])
    out["beat_index"] = pd.to_numeric(out["beat_index"], errors="raise").astype("int64")
    out["record_beat_index"] = out["record_beat_index"].astype("string")

    if "label" in out.columns:
        out["label"] = out["label"].astype("string")
    if "raw_symbol" in out.columns:
        out["raw_symbol"] = out["raw_symbol"].astype("string")
    if "lead_name" in out.columns:
        out["lead_name"] = out["lead_name"].astype("string")
    if "split" in out.columns:
        out["split"] = out["split"].astype("string")
    if "source_dataset" in out.columns:
        out["source_dataset"] = out["source_dataset"].astype("string")

    return out


def select_best_lead(
    signal: np.ndarray,
    sig_names: list[str] | tuple[str, ...] | None,
) -> tuple[np.ndarray, int, str]:
    signal = np.asarray(signal, dtype=float)

    if signal.ndim == 1:
        return signal, 0, "lead0"

    if signal.ndim != 2:
        raise ValueError(f"Érvénytelen signal shape: {signal.shape}")

    if sig_names is None:
        return signal[:, 0], 0, "lead0"

    sig_names = list(sig_names)

    for preferred in PREFERRED_LEADS:
        if preferred in sig_names:
            idx = sig_names.index(preferred)
            return signal[:, idx], idx, preferred

    return signal[:, 0], 0, str(sig_names[0])


def resolve_record_source_dataset(
    record_id: str,
    dataset: str,
    source_dataset: str | None = None,
    record_source_map: dict[str, str] | None = None,
) -> str:
    rid = str(record_id).strip()

    if record_source_map is not None and rid in record_source_map:
        return str(record_source_map[rid]).lower()

    if source_dataset is not None:
        return source_dataset.lower().strip()

    if dataset in {"mitbih", "incart"}:
        return dataset

    if rid.startswith("I"):
        return "incart"
    if rid.isdigit():
        return "mitbih"

    raise ValueError(f"Nem dönthető el a rekord forrás-datasetje: {rid}")


def process_single_record(
    dataset: str,
    record_id: str,
    source_dataset: str | None = None,
    record_source_map: dict[str, str] | None = None,
    verbose:bool = False
) -> pd.DataFrame:
    raw_dataset = resolve_record_source_dataset(
        record_id=record_id,
        dataset=dataset,
        source_dataset=source_dataset,
        record_source_map=record_source_map,
    )
    if(verbose):
        print(f"[INFO] Processing {dataset} / {record_id} (raw={raw_dataset})...")

    rec = load_record_by_dataset(raw_dataset, record_name=record_id)

    signal = np.asarray(rec.signal, dtype=float)
    fs = float(rec.fs)
    sig_names = getattr(rec, "sig_names", None)

    _, lead_index, lead_name = select_best_lead(signal=signal, sig_names=sig_names)

    beats, raw_symbols, centers, extract_meta = extract_beats(
        rec=rec,
        channel=lead_index,
        fs=fs,
        pre_sec=BEAT_PRE_SEC,
        post_sec=BEAT_POST_SEC,
        normalize=True,
        allowed_symbols=VALID_BEAT_SYMBOLS,
        preprocessing_config=PREPROCESSING_CONFIG,
    )

    if len(beats) == 0:
        return pd.DataFrame()

    kept_extract_meta = extract_meta.loc[extract_meta["kept"] == True].copy().reset_index(drop=True)
    beat_index = kept_extract_meta["ann_index"].to_numpy(dtype=np.int64)

    aami_labels = map_aami_labels(raw_symbols)
    aami_labels = np.asarray(aami_labels, dtype=object).reshape(-1)

    df = build_morphology_feature_dataframe_from_full_annotations(
        beats=beats,
        beat_centers=centers,
        full_annotation_samples=rec.annotation_samples,
        fs=fs,
        beat_labels=aami_labels,
        record_id=record_id,
        center_indices=None,
        beat_indices=beat_index,
    )

    if "beat_index" not in df.columns:
        raise ValueError(f"Hiányzik a beat_index oszlop a morphology dataframe-ből: {record_id}")

    df["record_id"] = cast_record_id(df["record_id"])
    df["source_dataset"] = str(raw_dataset)
    df["raw_symbol"] = pd.Series(np.asarray(raw_symbols, dtype=object), dtype="string")
    df["lead_index"] = int(lead_index)
    df["lead_name"] = str(lead_name)
    df["fs"] = float(fs)
    df["beat_pre_sec"] = float(BEAT_PRE_SEC)
    df["beat_post_sec"] = float(BEAT_POST_SEC)

    if not extract_meta.empty and "center" in extract_meta.columns:
        kept_meta = extract_meta.loc[extract_meta["kept"] == True].copy().reset_index(drop=True)
        if len(kept_meta) == len(df):
            df["crop_start"] = kept_meta["start"].to_numpy()
            df["crop_end"] = kept_meta["end"].to_numpy()
            df["pad_left"] = kept_meta["pad_left"].to_numpy()
            df["pad_right"] = kept_meta["pad_right"].to_numpy()
            df["is_truncated"] = kept_meta["is_truncated"].to_numpy()

    df = add_morphology_quality_flags(df)
    df = cast_morphology_schema(df)
    validate_record_beat_index(df, f"morphology_record_{record_id}")

    meta_cols = [
        c for c in [
            "record_id",
            "beat_index",
            "record_beat_index",
            "source_dataset",
            "label",
            "raw_symbol",
            "lead_index",
            "lead_name",
            "fs",
            "beat_pre_sec",
            "beat_post_sec",
            "crop_start",
            "crop_end",
            "pad_left",
            "pad_right",
            "is_truncated",
        ]
        if c in df.columns
    ]
    other_cols = [c for c in df.columns if c not in meta_cols]
    return df[meta_cols + other_cols]


def infer_records_from_split_npz(npz_path: Path) -> list[str]:
    _, _, beat_meta_df = load_npz_split(npz_path, return_meta=True)
    if beat_meta_df is None or "record_id" not in beat_meta_df.columns:
        raise ValueError(f"A splitből nem nyerhető ki record_id: {npz_path}")
    return (
        beat_meta_df["record_id"]
        .astype("string")
        .dropna()
        .drop_duplicates()
        .tolist()
    )


def build_split_table(
    dataset: str,
    split_name: str,
    record_list: list[str],
    source_dataset: str | None = None,
    record_source_map: dict[str, str] | None = None,
    verbose:bool = False
) -> pd.DataFrame:
    all_dfs: list[pd.DataFrame] = []

    for record_id in record_list:
        try:
            df_rec = process_single_record(
                dataset=dataset,
                record_id=str(record_id),
                source_dataset=source_dataset,
                record_source_map=record_source_map,
                verbose = verbose
            )
            if not df_rec.empty:
                df_rec["split"] = split_name
                all_dfs.append(df_rec)
        except Exception as exc:
            print(f"[ERROR] Record {record_id} failed: {exc}")

    if not all_dfs:
        raise ValueError(
       	    f"A(z) {split_name} splitben nem keletkezett egyetlen morphology rekord sem."
        )

    out = pd.concat(all_dfs, ignore_index=True)
    out = cast_morphology_schema(out)
    validate_record_beat_index(out, f"{split_name}_morphology")

    meta_cols = [
        c for c in [
            "split",
            "record_id",
            "beat_index",
            "record_beat_index",
            "source_dataset",
            "label",
            "raw_symbol",
            "lead_index",
            "lead_name",
            "fs",
            "beat_pre_sec",
            "beat_post_sec",
            "crop_start",
            "crop_end",
            "pad_left",
            "pad_right",
            "is_truncated",
        ]
        if c in out.columns
    ]
    other_cols = [c for c in out.columns if c not in meta_cols]
    return out[meta_cols + other_cols]


def build_qc_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({"metric": ["n_rows"], "value": [0]})

    summary = []
    summary.append({"metric": "n_rows", "value": len(df)})
    summary.append({"metric": "n_records", "value": df["record_id"].nunique()})
    summary.append({"metric": "n_labels", "value": df["label"].nunique()})

    if "source_dataset" in df.columns:
        for ds_name, count in df["source_dataset"].value_counts(dropna=False).sort_index().items():
            summary.append({"metric": f"source_dataset_count_{ds_name}", "value": int(count)})

    for col in [
        "has_rr_context_prev",
        "has_rr_context_next",
        "rr_prev_plausible",
        "rr_next_plausible",
        "qrs_width_plausible",
    ]:
        if col in df.columns:
            summary.append({"metric": f"{col}_true", "value": int(df[col].sum())})
            summary.append({"metric": f"{col}_false", "value": int((~df[col]).sum())})

    label_counts = df["label"].value_counts(dropna=False).sort_index()
    for label, count in label_counts.items():
        summary.append({"metric": f"label_count_{label}", "value": int(count)})

    return pd.DataFrame(summary)


def summarize_split(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n_rows": 0, "n_records": 0, "label_counts": {}, "missing_by_column": {}}

    out = {
        "n_rows": int(len(df)),
        "n_records": int(df["record_id"].nunique()),
        "label_counts": {
            str(k): int(v)
            for k, v in df["label"].value_counts(dropna=False).sort_index().to_dict().items()
        },
        "missing_by_column": {
            str(k): int(v)
            for k, v in df.isna().sum().to_dict().items()
        },
    }

    if "source_dataset" in df.columns:
        out["source_dataset_counts"] = {
            str(k): int(v)
            for k, v in df["source_dataset"].value_counts(dropna=False).sort_index().to_dict().items()
        }

    return out


def build_morphology_features(
    dataset: str = "mitbih",
    source_dataset: str | None = None,
    verbose:bool = False
) -> None:

    dataset = dataset.lower().strip()

    beats_dir = Path(get_ds_par(dataset, "beats_dir"))
    features_dir = Path(get_ds_par(dataset, "features_morphology_dir"))

    if not beats_dir.exists():
        raise FileNotFoundError(f"Hiányzó beats_dir: {beats_dir}")

    features_dir.mkdir(parents=True, exist_ok=True)

    print("DATA DIR:", beats_dir)
    print("OUT DIR:", features_dir)

    metadata = load_prepare_metadata(beats_dir)

    record_source_map = metadata.get("record_source_map")
    if isinstance(record_source_map, dict):
        record_source_map = {str(k): str(v).lower() for k, v in record_source_map.items()}
    else:
        record_source_map = None

    effective_source_dataset = (
        source_dataset
        or metadata.get("source_dataset")
    )
    if effective_source_dataset is not None:
        effective_source_dataset = str(effective_source_dataset).lower()

    split_files: dict[str, str] = {}
    split_qc_files: dict[str, str] = {}
    split_summaries: dict[str, dict] = {}

    for split_name in ["train", "val", "test"]:
        npz_path = beats_dir / f"{split_name}.npz"

        if split_name in metadata and isinstance(metadata.get(split_name), list):
            record_list = metadata[split_name]
        elif f"{split_name}_records" in metadata:
            record_list = metadata[f"{split_name}_records"]
        else:
            record_list = infer_records_from_split_npz(npz_path)

        df = build_split_table(
            dataset=dataset,
            split_name=split_name,
            record_list=[str(r) for r in record_list],
            source_dataset=effective_source_dataset,
            record_source_map=record_source_map,
            verbose = verbose
        )

        out_csv = features_dir / f"{split_name}_morphology_features.csv"
        qc_csv = features_dir / f"{split_name}_morphology_features_qc_summary.csv"

        df.to_csv(out_csv, index=False)
        build_qc_summary(df).to_csv(qc_csv, index=False)

        split_files[split_name] = str(out_csv)
        split_qc_files[split_name] = str(qc_csv)
        split_summaries[split_name] = summarize_split(df)

        print(f"[OK] Saved: {out_csv}")
        print(f"[OK] Saved: {qc_csv}")

    manifest = {
        "artifact_type": f"{dataset}_morphology_features",
        "created_at": now_iso(),
        "source_dataset": metadata.get("source_dataset", dataset),
        "record_source_map_present": bool(record_source_map),
        "beat_pre_samples": metadata.get("beat_pre_samples"),
        "beat_post_samples": metadata.get("beat_post_samples"),
        "beat_length": metadata.get("beat_length"),
        "label_mapping": metadata.get("label_mapping"),
        "beat_index_columns": metadata.get("beat_index_columns"),
        "beat_index_dtypes": metadata.get("beat_index_dtypes"),
        "beat_meta_dtypes": metadata.get("beat_meta_dtypes"),
        "preprocessing_config": PREPROCESSING_CONFIG,
        "train_records": metadata.get("train_records"),
        "val_records": metadata.get("val_records"),
        "test_records": metadata.get("test_records"),
        "feature_files": split_files,
        "qc_files": split_qc_files,
        "split_summaries": split_summaries,
    }

    manifest_path = features_dir / "manifest_morphology_features.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    report_input = {
        "created_at": now_iso(),
        "feature_family": "morphology",
        "dataset": dataset,
        "manifest_file": str(manifest_path),
        "split_files": split_files,
        "qc_files": split_qc_files,
        "split_summaries": split_summaries,
    }

    report_input_path = features_dir / "report_input_morphology_features.json"
    with open(report_input_path, "w", encoding="utf-8") as f:
        json.dump(report_input, f, ensure_ascii=False, indent=2)

    print(f"[OK] Manifest: {manifest_path}")
    print(f"[OK] Report input: {report_input_path}")




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build morphology ECG feature tables.")
    parser.add_argument("dataset", type=str, nargs="?", default="mitbih")
    parser.add_argument("--source-dataset", type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    build_morphology_features(
        dataset=args.dataset,
        source_dataset=args.source_dataset,
    )