#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from src.config import get_ds_par
from src.utils.dataset_loader import load_metadata, validate_record_beat_index
from src.utils.reporting import Reporter


def load_feature_split_csv(path: Path, split_name: str, family_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Hiányzó feature fájl: {path} (split={split_name}, family={family_name})")

    if path.stat().st_size == 0:
        raise ValueError(f"Üres feature CSV: {path} (split={split_name}, family={family_name})")

    try:
        df = pd.read_csv(path, low_memory=False)
    except EmptyDataError as exc:
        raise ValueError(f"A CSV üres vagy nem parse-olható: {path}") from exc

    if "rr_next_plausible" in df.columns:
        df["rr_next_plausible"] = pd.to_numeric(df["rr_next_plausible"], errors="coerce")

    return df


def normalize_general_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = {"record_beat_index", "label", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"A general feature táblából hiányzik: {sorted(missing)}")

    df["record_beat_index"] = df["record_beat_index"].astype("string")
    if "record_id" in df.columns:
        df["record_id"] = df["record_id"].astype("string")
    if "beat_index" in df.columns:
        df["beat_index"] = pd.to_numeric(df["beat_index"], errors="raise").astype("int64")
    df["label"] = df["label"].astype("string")
    df["y"] = pd.to_numeric(df["y"], errors="raise").astype("int64")

    validate_record_beat_index(df, "general_df")
    return df


def normalize_morphology_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = {"record_beat_index"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"A morphology feature táblából hiányzik: {sorted(missing)}")

    df["record_beat_index"] = df["record_beat_index"].astype("string")
    if "record_id" in df.columns:
        df["record_id"] = df["record_id"].astype("string")
    if "beat_index" in df.columns:
        df["beat_index"] = pd.to_numeric(df["beat_index"], errors="raise").astype("int64")

    validate_record_beat_index(df, "morphology_df")
    return df


def merge_general_and_morphology(general_df: pd.DataFrame, morphology_df: pd.DataFrame | None) -> pd.DataFrame:
    general_df = normalize_general_columns(general_df)

    if morphology_df is None:
        return general_df

    morphology_df = normalize_morphology_columns(morphology_df)

    morph_non_feature_cols = {
        "split", "record_id", "beat_index", "record_beat_index", "label",
        "source_dataset", "beat_center_sample", "raw_symbol", "lead_name",
        "lead_index", "fs", "beat_pre_samples", "beat_post_samples",
        "beat_pre_sec", "beat_post_sec", "crop_start", "crop_end",
        "pad_left", "pad_right", "is_truncated",
    }

    morph_feature_cols = [c for c in morphology_df.columns if c not in morph_non_feature_cols]
    morph_subset = morphology_df[["record_beat_index", *morph_feature_cols]].copy()

    overlap_cols = (set(general_df.columns) & set(morph_subset.columns)) - {"record_beat_index"}
    if overlap_cols:
        morph_subset = morph_subset.rename(columns={c: f"morph_{c}" for c in overlap_cols})

    merged = general_df.merge(morph_subset, on="record_beat_index", how="left", validate="one_to_one")
    return merged


def feature_columns_from_df(df: pd.DataFrame) -> list[str]:
    non_feature_cols = {
        "split", "record_id", "record_beat_index", "beat_index", "sample_index",
        "center_sample", "y", "label", "label_raw", "label_aami", "label_target",
    }

    feature_cols = []
    for c in df.columns:
        if c in non_feature_cols:
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            feature_cols.append(c)
    return feature_cols


def save_confusion_matrix_plot(reporter: Reporter, cm: np.ndarray, class_names: list[str], figure_name: str, title: str) -> None:
    with reporter.figure(figure_name, figsize=(8, 8)) as (fig, ax):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
        ax.set_title(title)


def evaluate_model(
    reporter: Reporter,
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    split_name: str,
    model_name: str,
    class_names: list[str],
    out_dir: Path,
    model_label: str,
) -> dict[str, float | int | str]:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    metrics = {
        "model": f"{model_name}_{model_label}",
        "split": split_name,
        "n_samples": int(len(y)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
    }

    labels = np.arange(len(class_names))
    report = classification_report(
        y, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0
    )
    pd.DataFrame(report).transpose().to_csv(out_dir / f"{model_name}_{split_name}_classification_report.csv")

    cm = confusion_matrix(y, y_pred, labels=labels)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(out_dir / f"{model_name}_{split_name}_confusion_matrix.csv")

    save_confusion_matrix_plot(
        reporter=reporter,
        cm=cm,
        class_names=class_names,
        figure_name=f"{model_name}_{split_name}_confusion_matrix",
        title=f"{model_name} - {split_name}",
    )

    pred_df = pd.DataFrame({"y_true": y.astype(np.int64), "y_pred": y_pred.astype(np.int64)})
    if y_proba is not None:
        for i, class_name in enumerate(class_names):
            pred_df[f"prob_{class_name}"] = y_proba[:, i]
    pred_df.to_csv(out_dir / f"{model_name}_{split_name}_predictions.csv", index=False)

    return metrics


def load_split_tables(split_name: str, general_features_dir: Path, morph_features_dir: Path | None) -> tuple[pd.DataFrame, np.ndarray, list[str], dict]:
    general_path = general_features_dir / f"{split_name}_general_features.csv"
    morph_path = morph_features_dir / f"{split_name}_morphology_features.csv" if morph_features_dir and morph_features_dir.exists() else None

    general_df = load_feature_split_csv(general_path, split_name, "general")
    morphology_df = load_feature_split_csv(morph_path, split_name, "morphology") if morph_path is not None and morph_path.exists() else None

    merged_df = merge_general_and_morphology(general_df, morphology_df)

    if "y" not in merged_df.columns or "label" not in merged_df.columns:
        raise ValueError(f"A {split_name} táblában nincs 'y' vagy 'label' oszlop.")

    feature_cols = feature_columns_from_df(merged_df)
    if not feature_cols:
        raise ValueError(f"Nincs használható numerikus feature a {split_name} splitben.")

    X = merged_df[feature_cols].copy()
    y = merged_df["y"].to_numpy(dtype=np.int64)

    merge_info = {
        "general_feature_file": str(general_path),
        "morphology_feature_file": str(morph_path) if morph_path is not None and morph_path.exists() else None,
        "n_rows_after_merge": int(len(merged_df)),
        "n_feature_columns": int(len(feature_cols)),
        "feature_columns": feature_cols,
    }

    return X, y, feature_cols, merge_info


def labels_from_dataset_metadata(dataset: str) -> list[str]:
    beats_dir = Path(get_ds_par(dataset, "beats_dir"))
    meta = load_metadata(beats_dir / "metadata.json")
    mapping = meta.get("label_mapping")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Hiányzik vagy üres a beats metadata.label_mapping.")
    return [k for k, _ in sorted(mapping.items(), key=lambda kv: int(kv[1]))]


def save_missing_summary(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, out_path: Path) -> None:
    pd.DataFrame({
        "feature": X_train.columns,
        "train_n_missing": X_train.isna().sum().values,
        "val_n_missing": X_val.isna().sum().values,
        "test_n_missing": X_test.isna().sum().values,
    }).to_csv(out_path, index=False)


def compute_balanced_class_weight_dict(y: np.ndarray, class_names: list[str]) -> dict[int, float]:
    classes = np.arange(len(class_names))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


def save_label_reports(y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, class_weights: dict[int, float], class_names: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_one(y: np.ndarray, name: str) -> None:
        labels = np.asarray([class_names[int(i)] for i in y], dtype=object)
        counts = pd.Series(labels).value_counts().reindex(class_names, fill_value=0)
        total = counts.sum()
        df = pd.DataFrame({
            "label": counts.index,
            "count": counts.values,
            "percent": ((counts / total) * 100).round(2) if total > 0 else 0.0,
        })
        df.to_csv(out_dir / f"label_dist_{name}.csv", index=False)

    save_one(y_train, "train")
    save_one(y_val, "val")
    save_one(y_test, "test")

    class_weights_df = pd.DataFrame({
        "class_index": list(class_weights.keys()),
        "class_name": [class_names[i] for i in class_weights.keys()],
        "class_weight": list(class_weights.values()),
    })
    class_weights_df.to_csv(out_dir / "class_weights.csv", index=False)


def build_models() -> dict[str, Pipeline]:
    return {
        "logreg": Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")),
        ]),
        "rf": Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1)),
        ]),
    }


def align_feature_schema(X: pd.DataFrame, target_columns: list[str]) -> pd.DataFrame:
    X = X.copy()
    missing_cols = [c for c in target_columns if c not in X.columns]
    extra_cols = [c for c in X.columns if c not in target_columns]

    for col in missing_cols:
        X[col] = np.nan
    if extra_cols:
        X = X.drop(columns=extra_cols)

    return X[target_columns]


def train_featurext(dataset: str = "mitbih", results_name: str | None = None) -> None:
    dataset = dataset.lower().strip()

    general_features_dir = Path(get_ds_par(dataset, "features_general_dir"))
    morph_features_dir = Path(get_ds_par(dataset, "features_morphology_dir"))
    results_dir = Path(get_ds_par(dataset, "featurext_dir"))

    if results_name:
        results_dir = results_dir / results_name

    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = results_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    reporter = Reporter(results_dir)

    X_train, y_train, train_feature_cols, train_info = load_split_tables("train", general_features_dir, morph_features_dir)
    X_val, y_val, val_feature_cols, val_info = load_split_tables("val", general_features_dir, morph_features_dir)
    X_test, y_test, test_feature_cols, test_info = load_split_tables("test", general_features_dir, morph_features_dir)

    X_val = align_feature_schema(X_val, train_feature_cols)
    X_test = align_feature_schema(X_test, train_feature_cols)

    class_names = labels_from_dataset_metadata(dataset)

    save_missing_summary(X_train, X_val, X_test, results_dir / "feature_missing_summary.csv")

    class_weights = compute_balanced_class_weight_dict(y_train, class_names)
    save_label_reports(y_train, y_val, y_test, class_weights, class_names, reports_dir)

    models = build_models()
    all_metrics = []
    model_label = dataset

    for model_name, model in models.items():
        print(f"\nTraining model: {model_name}")
        model.fit(X_train, y_train)

        for split_name, X_split, y_split in [
            ("train", X_train, y_train),
            ("val", X_val, y_val),
            ("test", X_test, y_test),
        ]:
            metrics = evaluate_model(
                reporter=reporter,
                model=model,
                X=X_split,
                y=y_split,
                split_name=split_name,
                model_name=model_name,
                class_names=class_names,
                out_dir=results_dir,
                model_label=model_label,
            )
            all_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(results_dir / "metrics_summary.csv", index=False)

    run_summary = {
        "dataset": dataset,
        "dataset_type": "exported_feature_tables",
        "general_features_dir": str(general_features_dir),
        "morph_features_dir": str(morph_features_dir),
        "results_dir": str(results_dir),
        "reports_dir": str(reports_dir),
        "class_names": class_names,
        "class_weights": class_weights,
        "feature_columns": train_feature_cols,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "train_info": train_info,
        "val_info": val_info,
        "test_info": test_info,
        "metrics_summary_file": str(results_dir / "metrics_summary.csv"),
    }

    with open(results_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    print(metrics_df)
    print(f"[OK] Saved to: {results_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train feature-based ECG models.")
    parser.add_argument("dataset", type=str, nargs="?", default="mitbih")
    parser.add_argument("--results-name", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_featurext(dataset=args.dataset, results_name=args.results_name)