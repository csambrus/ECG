#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from src.config import get_ds_par

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_npz_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Hiányzó NPZ fájl: {path}")

    data = np.load(path, allow_pickle=True)

    if "X" not in data or "y" not in data:
        raise KeyError(f"Az NPZ nem tartalmazza a szükséges 'X' és 'y' kulcsokat: {path}")

    X = data["X"]
    y = data["y"]
    return X, y


def flatten_beats(X: np.ndarray) -> np.ndarray:
    if X.ndim < 2:
        raise ValueError(f"X túl alacsony dimenziójú: shape={X.shape}")
    return X.reshape(X.shape[0], -1)


def sample_indices(
    n: int,
    max_samples: int | None,
    random_state: int,
) -> np.ndarray:
    idx = np.arange(n)

    if max_samples is None or max_samples >= n:
        return idx

    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(idx, size=max_samples, replace=False))


def build_dataframe(
    emb: np.ndarray,
    dataset_labels: np.ndarray,
    class_labels: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tsne_1": emb[:, 0],
            "tsne_2": emb[:, 1],
            "dataset": dataset_labels,
            "class_label": class_labels,
        }
    )


def plot_by_dataset(df: pd.DataFrame, out_path: Path, title: str) -> None:
    plt.figure(figsize=(10, 8))

    for dataset in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == dataset]
        plt.scatter(
            sub["tsne_1"],
            sub["tsne_2"],
            s=10,
            alpha=0.6,
            label=dataset,
        )

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_by_class(df: pd.DataFrame, out_path: Path, title: str) -> None:
    plt.figure(figsize=(10, 8))

    for class_name in sorted(df["class_label"].astype(str).unique()):
        sub = df[df["class_label"].astype(str) == str(class_name)]
        plt.scatter(
            sub["tsne_1"],
            sub["tsne_2"],
            s=10,
            alpha=0.6,
            label=str(class_name),
        )

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="Class", loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_cross_dataset_tsne(
    source_dataset: str | None = None,
    target_dataset: str | None = None,
    split: str = "test",
    results_dir: str | Path | None = None,
    max_per_dataset: int | None = 3000,
    random_state: int = 42,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    source_path: str | Path | None = None,
    target_path: str | Path | None = None,
) -> None:
    split = split.lower().strip()

    if split not in {"train", "val", "test"}:
        raise ValueError(f"Ismeretlen split: {split}")

    if results_dir is None:
        results_dir = get_ds_par("cross_dataset", "tsne_dir")
    else:
        results_dir = Path(results_dir)

    results_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Input path feloldás
    # ---------------------------------------------------------
    if source_path is None:
        if source_dataset is None:
            raise ValueError("source_dataset vagy source_path megadása kötelező.")
        source_dataset = source_dataset.lower().strip()
        source_path = Path(get_ds_par(source_dataset, "beats_dir")) / f"{split}.npz"
    else:
        source_path = Path(source_path)
        if source_dataset is None:
            source_dataset = source_path.parent.name

    if target_path is None:
        if target_dataset is None:
            raise ValueError("target_dataset vagy target_path megadása kötelező.")
        target_dataset = target_dataset.lower().strip()
        target_path = Path(get_ds_par(target_dataset, "beats_dir")) / f"{split}.npz"
    else:
        target_path = Path(target_path)
        if target_dataset is None:
            target_dataset = target_path.parent.name

    print(f"[INFO] Source: {source_dataset} -> {source_path}")
    print(f"[INFO] Target: {target_dataset} -> {target_path}")

    # ---------------------------------------------------------
    # Adatok betöltése
    # ---------------------------------------------------------
    X_src, y_src = load_npz_xy(source_path)
    X_tgt, y_tgt = load_npz_xy(target_path)

    src_idx = sample_indices(len(X_src), max_per_dataset, random_state)
    tgt_idx = sample_indices(len(X_tgt), max_per_dataset, random_state)

    X_src = X_src[src_idx]
    y_src = y_src[src_idx]

    X_tgt = X_tgt[tgt_idx]
    y_tgt = y_tgt[tgt_idx]

    print(f"[INFO] Sampled source shape: X={X_src.shape}, y={y_src.shape}")
    print(f"[INFO] Sampled target shape: X={X_tgt.shape}, y={y_tgt.shape}")

    # ---------------------------------------------------------
    # Flatten + szigorú shape ellenőrzés
    # ---------------------------------------------------------
    X_src_flat = flatten_beats(X_src)
    X_tgt_flat = flatten_beats(X_tgt)

    if X_src_flat.shape[1] != X_tgt_flat.shape[1]:
        raise ValueError(
            "A két dataset flattenelt dimenziója eltér, ezért a közös t-SNE nem számolható.\n"
            f"  {source_dataset}: {X_src_flat.shape}\n"
            f"  {target_dataset}: {X_tgt_flat.shape}\n\n"
            "Használj előre alignolt datasetet, vagy adj meg explicit "
            "source_path / target_path paramétereket kompatibilis NPZ fájlokra."
        )

    # ---------------------------------------------------------
    # Közös t-SNE input
    # ---------------------------------------------------------
    X_all = np.vstack([X_src_flat, X_tgt_flat])
    y_all = np.concatenate([y_src, y_tgt])

    dataset_labels = np.array(
        [source_dataset] * len(X_src_flat) + [target_dataset] * len(X_tgt_flat),
        dtype=object,
    )

    print(f"[INFO] Combined shape for t-SNE: {X_all.shape}")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    emb = tsne.fit_transform(X_all)

    df = build_dataframe(
        emb=emb,
        dataset_labels=dataset_labels,
        class_labels=y_all,
    )

    prefix = f"tsne_{source_dataset}_vs_{target_dataset}_{split}"

    csv_path = results_dir / f"{prefix}.csv"
    png_dataset_path = results_dir / f"{prefix}_by_dataset.png"
    png_class_path = results_dir / f"{prefix}_by_class.png"
    meta_path = results_dir / f"{prefix}_meta.json"

    df.to_csv(csv_path, index=False)

    plot_by_dataset(
        df=df,
        out_path=png_dataset_path,
        title=f"t-SNE by dataset ({source_dataset} vs {target_dataset}, {split})",
    )

    plot_by_class(
        df=df,
        out_path=png_class_path,
        title=f"t-SNE by class ({source_dataset} vs {target_dataset}, {split})",
    )

    metadata = {
        "created_at": now_iso(),
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "split": split,
        "source_path": str(source_path),
        "target_path": str(target_path),
        "source_n": int(len(X_src)),
        "target_n": int(len(X_tgt)),
        "combined_n": int(len(X_all)),
        "flattened_dim": int(X_all.shape[1]),
        "max_per_dataset": max_per_dataset,
        "random_state": random_state,
        "perplexity": perplexity,
        "n_iter": n_iter,
        "outputs": {
            "csv": str(csv_path),
            "png_by_dataset": str(png_dataset_path),
            "png_by_class": str(png_class_path),
        },
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("[OK] Kész.")
    print(f"[OUT] CSV:         {csv_path}")
    print(f"[OUT] PNG dataset: {png_dataset_path}")
    print(f"[OUT] PNG class:   {png_class_path}")
    print(f"[OUT] META:        {meta_path}")




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Két dataset közös t-SNE vizualizációja."
    )
    parser.add_argument("source_dataset")
    parser.add_argument("target_dataset")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-per-dataset", type=int, default=3000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--n-iter", type=int, default=1000)
    parser.add_argument("--results-dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    plot_cross_dataset_tsne(
        source_dataset=args.source_dataset,
        target_dataset=args.target_dataset,
        split=args.split,
        results_dir=args.results_dir,
        max_per_dataset=args.max_per_dataset,
        random_state=args.random_state,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
    )