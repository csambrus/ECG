from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from src.config import get_ds_par


RANDOM_STATE = 42
ALIGN_MODE = "crop"


def save_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_npz_with_meta(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Hiányzó fájl: {path}")

    data = np.load(path, allow_pickle=True)
    out = {k: data[k] for k in data.files}

    required = {"X", "y", "record_id", "beat_index", "record_beat_index"}
    missing = required - set(out.keys())
    if missing:
        raise ValueError(f"Az input NPZ nem kompatibilis: {path}\nHiányzó mező(k): {sorted(missing)}")

    n = len(out["y"])
    for k, v in out.items():
        if len(v) != n and k not in {"X"}:
            raise ValueError(f"Hosszhiba az NPZ-ben: {path}\n{k}: len={len(v)}\ny: len={n}")

    return out


def load_metadata(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset_splits(base_dir: Path) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    splits = {
        "train": load_npz_with_meta(base_dir / "train.npz"),
        "val": load_npz_with_meta(base_dir / "val.npz"),
        "test": load_npz_with_meta(base_dir / "test.npz"),
    }
    meta = load_metadata(base_dir / "metadata.json")
    return splits, meta


def save_split_npz_with_meta(path: Path, split: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **split)


def align(X: np.ndarray, target_len: int, mode: str) -> np.ndarray:
    if X.shape[1] == target_len:
        return X

    if mode == "crop":
        return X[:, :target_len]

    if mode == "pad":
        pad = target_len - X.shape[1]
        if pad < 0:
            raise ValueError(f"Pad módnál a target_len nem lehet kisebb: {target_len} < {X.shape[1]}")
        if X.ndim == 2:
            return np.pad(X, ((0, 0), (0, pad)))
        if X.ndim == 3:
            return np.pad(X, ((0, 0), (0, pad), (0, 0)))
        raise ValueError(f"Nem támogatott dimenzió pad módhoz: {X.ndim}")

    raise ValueError("Shape mismatch és align_mode='error'")


def align_all(datasets: list[dict[str, np.ndarray]], mode: str) -> list[dict[str, np.ndarray]]:
    lengths = [d["X"].shape[1] for d in datasets]
    if len(set(lengths)) == 1:
        return datasets

    if mode == "error":
        raise ValueError(f"Shape mismatch a datasetek között. Hosszak: {lengths}")

    target = min(lengths) if mode == "crop" else max(lengths)

    aligned = []
    for d in datasets:
        out = dict(d)
        out["X"] = align(d["X"], target, mode)
        aligned.append(out)
    return aligned


def concat(*datasets: dict[str, np.ndarray], align_mode: str = ALIGN_MODE) -> dict[str, np.ndarray]:
    aligned = align_all(list(datasets), align_mode)
    keys = aligned[0].keys()
    out = {}
    for k in keys:
        out[k] = np.concatenate([d[k] for d in aligned], axis=0)
    return out


def take_rows(d: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for k, v in d.items():
        if k == "X":
            out[k] = v[idx]
        else:
            out[k] = v[idx]
    return out


def ensure_same_label_space(meta_a: dict[str, Any], meta_b: dict[str, Any]) -> None:
    if meta_a.get("label_mapping") != meta_b.get("label_mapping"):
        raise ValueError(
            "A két dataset label_mapping-je eltér. "
            "Futtasd ugyanazzal a LABEL_MODE-dal mindkét prepare pipeline-t."
        )


def build_metadata(
    name: str,
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    align_mode: str,
    source_meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset": name,
        "align_mode": align_mode,
        "label_mode": source_meta.get("label_mode"),
        "class_names": source_meta.get("class_names"),
        "label_mapping": source_meta.get("label_mapping"),
        "fs": source_meta.get("fs"),
        "splits": {
            "train": {"shape": list(train["X"].shape)},
            "val": {"shape": list(val["X"].shape)},
            "test": {"shape": list(test["X"].shape)},
        },
    }


def save_all(
    out_dir: Path,
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    name: str,
    align_mode: str,
    source_meta: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_split_npz_with_meta(out_dir / "train.npz", train)
    save_split_npz_with_meta(out_dir / "val.npz", val)
    save_split_npz_with_meta(out_dir / "test.npz", test)
    save_json(out_dir / "metadata.json", build_metadata(name, train, val, test, align_mode, source_meta))


def build_cross_test(
    mitbih: dict[str, dict[str, np.ndarray]],
    incart: dict[str, dict[str, np.ndarray]],
    align_mode: str,
    source_meta: dict[str, Any],
) -> None:
    train, val, test = align_all([mitbih["train"], mitbih["val"], incart["test"]], align_mode)
    out_dir = Path(get_ds_par("cross_test", "beats_dir"))
    save_all(out_dir, train, val, test, "cross_test", align_mode, source_meta)


def build_mixed(
    mitbih: dict[str, dict[str, np.ndarray]],
    incart: dict[str, dict[str, np.ndarray]],
    align_mode: str,
    source_meta: dict[str, Any],
) -> None:
    all_data = concat(
        mitbih["train"], mitbih["val"], mitbih["test"],
        incart["train"], incart["val"], incart["test"],
        align_mode=align_mode,
    )

    idx = np.arange(len(all_data["y"]))
    idx_tr, idx_tmp = train_test_split(
        idx,
        test_size=0.3,
        stratify=all_data["y"],
        random_state=RANDOM_STATE,
    )
    idx_val, idx_te = train_test_split(
        idx_tmp,
        test_size=0.5,
        stratify=all_data["y"][idx_tmp],
        random_state=RANDOM_STATE,
    )

    train = take_rows(all_data, idx_tr)
    val = take_rows(all_data, idx_val)
    test = take_rows(all_data, idx_te)

    out_dir = Path(get_ds_par("mixed", "beats_dir"))
    save_all(out_dir, train, val, test, "mixed", align_mode, source_meta)


def build_domain_generalization(
    mitbih: dict[str, dict[str, np.ndarray]],
    incart: dict[str, dict[str, np.ndarray]],
    align_mode: str,
    source_meta: dict[str, Any],
) -> None:
    train = mitbih["train"]
    val = concat(mitbih["val"], incart["val"], align_mode=align_mode)
    test = incart["test"]
    train, val, test = align_all([train, val, test], align_mode)

    out_dir = Path(get_ds_par("domain_generalization", "beats_dir"))
    save_all(out_dir, train, val, test, "domain_generalization", align_mode, source_meta)


def prepare_cross_dataset_splits(
    mitbih_dir: str | Path | None = None,
    incart_dir: str | Path | None = None,
    align_mode: str = ALIGN_MODE,
) -> None:
    mitbih_dir = Path(mitbih_dir) if mitbih_dir is not None else Path(get_ds_par("mitbih", "beats_dir"))
    incart_dir = Path(incart_dir) if incart_dir is not None else Path(get_ds_par("incart", "beats_dir"))

    mitbih, meta_mitbih = load_dataset_splits(mitbih_dir)
    incart, meta_incart = load_dataset_splits(incart_dir)

    ensure_same_label_space(meta_mitbih, meta_incart)

    build_cross_test(mitbih, incart, align_mode, meta_mitbih)
    build_mixed(mitbih, incart, align_mode, meta_mitbih)
    build_domain_generalization(mitbih, incart, align_mode, meta_mitbih)

    print("✔️ Kész.")