#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import AAMI_COLOR_MAP, get_ds_par
from src.utils.labels import AAMI_CLASSES
from src.utils.dataset_loader import load_npz_split, validate_record_beat_index
from src.utils.reporting import Reporter

RANDOM_STATE = 42
MAX_SAMPLES_PER_SPLIT = 3000

TSNE_N_COMPONENTS = 2
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000
TSNE_INIT = "pca"
TSNE_LEARNING_RATE = "auto"

CLASS_NAMES = {i: cls for i, cls in enumerate(AAMI_CLASSES)}
CLASS_COLORS = {i: AAMI_COLOR_MAP[cls] for i, cls in enumerate(AAMI_CLASSES)}

def cast_record_beat_index_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    required = {"record_beat_index"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Hiányzó kulcs(oszlop): {sorted(missing)}")

    out["record_beat_index"] = out["record_beat_index"].astype("string")

    if "y" in out.columns:
        out["y"] = pd.to_numeric(out["y"], errors="raise").astype("int64")

    return out


def load_general_features(split_name: str, general_features_dir: Path) -> pd.DataFrame:
    path = general_features_dir / f"{split_name}_general_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Hiányzó general feature file: {path}")

    df = pd.read_csv(path, dtype={"record_beat_index": "string"})
    df = cast_record_beat_index_df(df)

    required = {"record_beat_index", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"A {split_name} general feature táblából hiányzik: {sorted(missing)}")

    validate_record_beat_index(df, f"{split_name}_general_features")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    non_feature_cols = {
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
        "npz_row",
    }
    return [
        c for c in df.columns
        if c not in non_feature_cols and pd.api.types.is_numeric_dtype(df[c])
    ]


def stratified_subsample_indices(
    y: np.ndarray,
    max_samples: int,
    random_state: int = 42,
) -> np.ndarray:
    n = len(y)
    if n <= max_samples:
        return np.arange(n)

    rng = np.random.default_rng(random_state)

    classes, counts = np.unique(y, return_counts=True)
    proportions = counts / counts.sum()

    target_counts = np.floor(proportions * max_samples).astype(int)
    target_counts = np.maximum(target_counts, 1)

    diff = max_samples - target_counts.sum()

    if diff > 0:
        order = np.argsort(-counts)
        for i in range(diff):
            target_counts[order[i % len(order)]] += 1
    elif diff < 0:
        order = np.argsort(target_counts)[::-1]
        i = 0
        while target_counts.sum() > max_samples and i < 100000:
            idx = order[i % len(order)]
            if target_counts[idx] > 1:
                target_counts[idx] -= 1
            i += 1

    selected_indices = []

    for cls, k in zip(classes, target_counts):
        cls_idx = np.where(y == cls)[0]
        k = min(k, len(cls_idx))
        chosen = rng.choice(cls_idx, size=k, replace=False)
        selected_indices.append(chosen)

    selected_indices = np.concatenate(selected_indices)
    rng.shuffle(selected_indices)
    return selected_indices


def fit_tsne(X: np.ndarray) -> np.ndarray:
    tsne = TSNE(
        n_components=TSNE_N_COMPONENTS,
        perplexity=TSNE_PERPLEXITY,
        learning_rate=TSNE_LEARNING_RATE,
        init=TSNE_INIT,
        random_state=RANDOM_STATE,
        max_iter=TSNE_N_ITER,
    )
    return tsne.fit_transform(X)


def plot_side_by_side_tsne(
    reporter: Reporter,
    X_emb_featurext: np.ndarray,
    X_emb_cnn: np.ndarray,
    y: np.ndarray,
    split_name: str,
) -> None:
    figure_name = f"tsne_featurext_vs_cnn_{split_name}"

    with reporter.figure(figure_name, figsize=(16, 7)) as (fig, _):
        axes = fig.subplots(1, 2)

        panels = [
            (axes[0], X_emb_featurext, f"Featurext t-SNE - {split_name}"),
            (axes[1], X_emb_cnn, f"CNN input-space t-SNE - {split_name}"),
        ]

        classes = np.unique(y)

        for ax, X_emb, title in panels:
            for cls in classes:
                cls = int(cls)
                idx = y == cls
                label = f"{cls} ({CLASS_NAMES.get(cls, str(cls))})"

                ax.scatter(
                    X_emb[idx, 0],
                    X_emb[idx, 1],
                    s=10,
                    alpha=0.7,
                    label=label,
                    color=CLASS_COLORS.get(cls, "black"),
                )

            ax.set_title(title)
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")

        axes[1].legend(markerscale=2)
        fig.tight_layout()


def run_for_split(
    split_name: str,
    reporter: Reporter,
    dataset_dir: Path,
    general_features_dir: Path,
    results_dir: Path,
    max_samples_per_split: int,
) -> None:
    print("=" * 60)
    print(f"Processing split: {split_name}")

    X_beats, y_npz, npz_index_df = load_npz_split(dataset_dir / f"{split_name}.npz", return_meta=True)
    feat_df = load_general_features(split_name, general_features_dir)

    npz_index_df = npz_index_df.copy()
    npz_index_df["y"] = np.asarray(y_npz, dtype=np.int64)
    npz_index_df["npz_row"] = np.arange(len(npz_index_df), dtype=np.int64)
    npz_index_df = cast_record_beat_index_df(npz_index_df)
    validate_record_beat_index(npz_index_df, f"{split_name}_npz_index")

    merged = npz_index_df.merge(
        feat_df,
        on=["record_beat_index", "y"],
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        raise ValueError(f"Nincs közös minta a(z) {split_name} splitben a merge után.")

    feature_cols = get_feature_columns(merged)
    if not feature_cols:
        raise ValueError(f"Nincs használható feature oszlop a(z) {split_name} splitben.")

    npz_rows = merged["npz_row"].to_numpy(dtype=np.int64)
    y = merged["y"].to_numpy(dtype=np.int64)

    X_cnn = X_beats[npz_rows]
    if X_cnn.ndim > 2:
        X_cnn = X_cnn.reshape(X_cnn.shape[0], -1)

    X_featurext = merged[feature_cols].copy()

    preprocess = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    X_featurext = preprocess.fit_transform(X_featurext)

    sel_idx = stratified_subsample_indices(
        y=y,
        max_samples=max_samples_per_split,
        random_state=RANDOM_STATE,
    )

    X_featurext_sub = X_featurext[sel_idx]
    X_cnn_sub = X_cnn[sel_idx]
    y_sub = y[sel_idx]
    merged_sub = merged.iloc[sel_idx].reset_index(drop=True)

    print(f"Merged sample count: {len(y)}")
    print(f"Subsampled count:    {len(y_sub)}")

    X_emb_featurext = fit_tsne(X_featurext_sub)
    X_emb_cnn = fit_tsne(X_cnn_sub)

    plot_side_by_side_tsne(
        reporter=reporter,
        X_emb_featurext=X_emb_featurext,
        X_emb_cnn=X_emb_cnn,
        y=y_sub,
        split_name=split_name,
    )

    emb_df = pd.DataFrame(
        {
            "record_beat_index": merged_sub["record_beat_index"].astype("string"),
            "y": y_sub,
            "class_name": [CLASS_NAMES[int(v)] for v in y_sub],
            "featurext_tsne_1": X_emb_featurext[:, 0],
            "featurext_tsne_2": X_emb_featurext[:, 1],
            "cnn_tsne_1": X_emb_cnn[:, 0],
            "cnn_tsne_2": X_emb_cnn[:, 1],
        }
    )

    out_csv = results_dir / f"tsne_featurext_vs_cnn_{split_name}.csv"
    emb_df.to_csv(out_csv, index=False)

    print(f"Saved plot: {reporter.out_dir / f'tsne_featurext_vs_cnn_{split_name}.png'}")
    print(f"Saved table: {out_csv}")



def plot_tsne_featurext_vs_cnn(
    dataset: str = "mitbih",
    max_samples_per_split: int = MAX_SAMPLES_PER_SPLIT,
) -> None:

    dataset = dataset.lower().strip()

    resolved_dataset_dir = Path(get_ds_par(dataset, "beats_dir"))
    resolved_general_dir = Path(get_ds_par(dataset, "features_general_dir"))
    resolved_results_dir = Path(get_ds_par(dataset, "tsne_dir"))

    if not resolved_dataset_dir.exists():
        raise FileNotFoundError(f"Hiányzó dataset_dir: {resolved_dataset_dir}")
    if not resolved_general_dir.exists():
        raise FileNotFoundError(f"Hiányzó general_features_dir: {resolved_general_dir}")

    resolved_results_dir.mkdir(parents=True, exist_ok=True)

    reporter = Reporter(resolved_results_dir)

    print("Starting featurext vs CNN t-SNE comparison...")
    print(f"Dataset dir: {resolved_dataset_dir}")
    print(f"Feature dir: {resolved_general_dir}")
    print(f"Results dir: {resolved_results_dir}")

    for split_name in ["train", "val", "test"]:
        run_for_split(
            split_name,
            reporter=reporter,
            dataset_dir=resolved_dataset_dir,
            general_features_dir=resolved_general_dir,
            results_dir=resolved_results_dir,
            max_samples_per_split=max_samples_per_split,
        )

    print("Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare featurext vs CNN input-space using t-SNE.")
    parser.add_argument("dataset", nargs="?", default="mitbih")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--general-features-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--max-samples-per-split", type=int, default=MAX_SAMPLES_PER_SPLIT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_tsne_featurext_vs_cnn(
        dataset=args.dataset,
        data_dir=args.data_dir,
        general_features_dir=args.general_features_dir,
        results_dir=args.results_dir,
        max_samples_per_split=args.max_samples_per_split,
    )