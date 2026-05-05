#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE

from src.config import AAMI_COLOR_MAP, get_ds_par
from src.utils.labels import AAMI_CLASSES
from src.utils.dataset_loader import load_npz_split
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


def flatten_features(X: np.ndarray) -> np.ndarray:
    if X.ndim == 2:
        return X
    return X.reshape(X.shape[0], -1)


def stratified_subsample(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(y)
    if n <= max_samples:
        return X, y

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

    return X[selected_indices], y[selected_indices]


def describe_labels(y: np.ndarray) -> None:
    classes, counts = np.unique(y, return_counts=True)
    print("Label distribution:")
    for cls, cnt in zip(classes, counts):
        name = CLASS_NAMES.get(int(cls), str(cls))
        print(f"  {cls} ({name}): {cnt}")


def plot_tsne_scatter(
    reporter: Reporter,
    X_embedded: np.ndarray,
    y: np.ndarray,
    title: str,
    figure_name: str,
) -> None:
    with reporter.figure(figure_name, figsize=(10, 8)) as (fig, ax):
        classes = np.unique(y)

        for cls in classes:
            cls = int(cls)
            idx = y == cls
            label = f"{cls} ({CLASS_NAMES.get(cls, str(cls))})"

            ax.scatter(
                X_embedded[idx, 0],
                X_embedded[idx, 1],
                s=10,
                alpha=0.7,
                label=label,
                color=CLASS_COLORS.get(cls, "black"),
            )

        ax.set_title(title)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(markerscale=2)
        fig.tight_layout()


def run_tsne_for_split(
    npz_path: Path,
    split_name: str,
    reporter: Reporter,
    dataset: str,
    max_samples_per_split: int,
) -> None:
    print("=" * 60)
    print(f"Processing dataset: {dataset}")
    print(f"Processing split:   {split_name}")
    print(f"Input file:         {npz_path}")

    X, y, _ = load_npz_split(npz_path)

    print(f"Original X shape: {X.shape}")
    print(f"Original y shape: {y.shape}")
    describe_labels(y)

    X = flatten_features(X)
    print(f"Flattened X shape: {X.shape}")

    X_sub, y_sub = stratified_subsample(
        X,
        y,
        max_samples=max_samples_per_split,
        random_state=RANDOM_STATE,
    )

    print(f"Subsampled X shape: {X_sub.shape}")
    print(f"Subsampled y shape: {y_sub.shape}")
    describe_labels(y_sub)

    print("Running t-SNE...")

    tsne = TSNE(
        n_components=TSNE_N_COMPONENTS,
        perplexity=TSNE_PERPLEXITY,
        learning_rate=TSNE_LEARNING_RATE,
        init=TSNE_INIT,
        random_state=RANDOM_STATE,
        max_iter=TSNE_N_ITER,
    )

    X_embedded = tsne.fit_transform(X_sub)

    figure_name = f"tsne_{split_name}"
    plot_tsne_scatter(
        reporter=reporter,
        X_embedded=X_embedded,
        y=y_sub,
        title=f"{dataset} t-SNE - {split_name}",
        figure_name=figure_name,
    )

    print(f"Saved plot: {reporter.out_dir / f'{figure_name}.png'}")


def find_split_files(dataset_dir: Path) -> dict[str, Path]:
    candidates = {
        "train": dataset_dir / "train.npz",
        "val": dataset_dir / "val.npz",
        "test": dataset_dir / "test.npz",
    }

    missing = [name for name, path in candidates.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Nem találom a várt split fájlokat.\n"
            f"Keresett könyvtár: {dataset_dir}\n"
            f"Hiányzik: {missing}\n"
            "Várt fájlnevek: train.npz, val.npz, test.npz"
        )

    return candidates


def plot_dataset_tsne(
    dataset: str,
    max_samples_per_split: int = MAX_SAMPLES_PER_SPLIT,
) -> None:
    dataset = dataset.lower().strip()

    dataset_dir = Path(get_ds_par(dataset, "beats_dir"))
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Hiányzó dataset_dir: {dataset_dir}")

    results_dir = Path(get_ds_par(dataset, "tsne_dir"))
    results_dir.mkdir(parents=True, exist_ok=True)

    reporter = Reporter(results_dir)

    print(f"Starting t-SNE visualization for {dataset} splits...")
    print(f"[INFO] Dataset dir: {dataset_dir}")
    print(f"[INFO] Results dir: {results_dir}")

    split_files = find_split_files(dataset_dir)

    for split_name, npz_path in split_files.items():
        run_tsne_for_split(
            npz_path=npz_path,
            split_name=split_name,
            reporter=reporter,
            dataset=dataset,
            max_samples_per_split=max_samples_per_split,
        )

    print("Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot t-SNE for dataset splits.")
    parser.add_argument("dataset")
    parser.add_argument("--max-samples-per-split", type=int, default=MAX_SAMPLES_PER_SPLIT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_dataset_tsne(
        dataset=args.dataset,
        max_samples_per_split=args.max_samples_per_split,
    )