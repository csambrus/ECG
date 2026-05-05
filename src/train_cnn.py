#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight

from src.config import BATCH_SIZE, CNN_AUGMENTATION_CONFIG, EPOCHS, get_ds_par
from src.utils.dataset_loader import load_metadata, load_npz_split
from src.utils.reporting import Reporter

SEED = 42
PATIENCE = 6
VAL_MACRO_F1_EVERY = 2

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"[WARN] memory growth beállítás sikertelen: {e}")

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")
    USE_MIXED_PRECISION = True
except Exception:
    USE_MIXED_PRECISION = False


def normalize_variant(variant: str | None) -> str | None:
    if variant is None:
        return None
    normalized = str(variant).strip().lower()
    allowed = {"1ch", "2ch","12ch", "multichannel"}
    if normalized not in allowed:
        raise ValueError(f"Ismeretlen CNN variant: {variant!r}. Megengedett: {sorted(allowed)}")
    return normalized


def labels_from_metadata(metadata: dict) -> list[str]:
    mapping = metadata.get("label_mapping")
    if isinstance(mapping, dict) and mapping:
        return [k for k, _ in sorted(mapping.items(), key=lambda kv: int(kv[1]))]
    raise ValueError("Hiányzik vagy üres a metadata.label_mapping.")


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def ensure_cnn_input_shape(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        return X[..., np.newaxis]
    if X.ndim == 3:
        return X
    raise ValueError(f"Váratlan X shape: {X.shape}")


def infer_input_kind_from_array(X: np.ndarray) -> str:
    if X.ndim == 2:
        return "1ch"
    if X.ndim == 3 and X.shape[-1] == 1:
        return "1ch"
    if X.ndim == 3 and X.shape[-1] > 1:
        return "multichannel"
    return "unknown"


def compute_class_weights(y: np.ndarray, n_classes: int) -> dict[int, float]:
    classes = np.arange(n_classes)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(cls): float(w) for cls, w in zip(classes, weights)}


def build_cnn_model(input_shape: tuple[int, ...], n_classes: int) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv1D(32, 7, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    x = tf.keras.layers.Conv1D(64, 5, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    x = tf.keras.layers.Conv1D(128, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv1D(128, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    outputs = tf.keras.layers.Dense(n_classes, activation="softmax", dtype="float32")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def add_random_baseline_drift(x: tf.Tensor) -> tf.Tensor:
    batch = tf.shape(x)[0]
    steps = tf.shape(x)[1]
    channels = tf.shape(x)[2]

    t = tf.linspace(0.0, 1.0, steps)
    t = tf.reshape(t, (1, -1, 1))

    amp = tf.random.uniform(
        shape=(batch, 1, 1),
        minval=float(CNN_AUGMENTATION_CONFIG["baseline_drift_amplitude_min"]),
        maxval=float(CNN_AUGMENTATION_CONFIG["baseline_drift_amplitude_max"]),
        dtype=x.dtype,
    )
    phase = tf.random.uniform(shape=(batch, 1, 1), minval=0.0, maxval=2.0 * np.pi, dtype=x.dtype)
    cycles = tf.random.uniform(
        shape=(batch, 1, 1),
        minval=float(CNN_AUGMENTATION_CONFIG["baseline_drift_cycles_min"]),
        maxval=float(CNN_AUGMENTATION_CONFIG["baseline_drift_cycles_max"]),
        dtype=x.dtype,
    )

    drift = amp * tf.sin(2.0 * np.pi * cycles * t + phase)
    drift = tf.tile(drift, [1, 1, channels])
    return x + drift


def augment_ecg_batch(x: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    cfg = CNN_AUGMENTATION_CONFIG

    if cfg.get("use_scale", False):
        scales = tf.random.uniform(
            shape=(tf.shape(x)[0], 1, 1),
            minval=float(cfg["scale_min"]),
            maxval=float(cfg["scale_max"]),
            dtype=x.dtype,
        )
        x = x * scales

    if cfg.get("use_gaussian_noise", False):
        noise = tf.random.normal(
            shape=tf.shape(x),
            mean=0.0,
            stddev=float(cfg["noise_std"]),
            dtype=x.dtype,
        )
        x = x + noise

    if cfg.get("use_shift", False):
        shift = tf.random.uniform(
            shape=(),
            minval=-int(cfg["shift_max"]),
            maxval=int(cfg["shift_max"]) + 1,
            dtype=tf.int32,
        )
        x = tf.roll(x, shift=shift, axis=1)

    if cfg.get("use_baseline_drift", False):
        x = add_random_baseline_drift(x)

    return x, y


def make_tf_dataset(X: np.ndarray, y: np.ndarray, batch_size: int, training: bool = False) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.cache()

    if training:
        ds = ds.shuffle(buffer_size=min(len(y), 10000), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=False)

    if training and CNN_AUGMENTATION_CONFIG.get("enabled", False):
        ds = ds.map(augment_ecg_batch, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


class ValidationMacroF1Callback(tf.keras.callbacks.Callback):
    def __init__(self, X_val: np.ndarray, y_val: np.ndarray, batch_size: int = BATCH_SIZE, every_n_epochs: int = 1) -> None:
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.batch_size = batch_size
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.best_macro_f1: float | None = None
        self.best_epoch: int | None = None
        self.last_macro_f1: float | None = None

    def _compute_macro_f1(self) -> float:
        y_prob = self.model.predict(self.X_val, batch_size=self.batch_size, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        return float(f1_score(self.y_val, y_pred, average="macro", zero_division=0))

    def on_epoch_end(self, epoch, logs=None) -> None:
        logs = logs or {}
        epoch_number = epoch + 1
        should_compute = (self.last_macro_f1 is None or epoch_number % self.every_n_epochs == 0)

        if should_compute:
            val_macro_f1 = self._compute_macro_f1()
            self.last_macro_f1 = val_macro_f1
            if self.best_macro_f1 is None or val_macro_f1 > self.best_macro_f1:
                self.best_macro_f1 = val_macro_f1
                self.best_epoch = epoch_number
            print(f" — val_macro_f1: {val_macro_f1:.6f}")
        else:
            val_macro_f1 = float(self.last_macro_f1)
            print(f" — val_macro_f1: {val_macro_f1:.6f} (előző érték megtartva)")

        logs["val_macro_f1"] = float(val_macro_f1)


def plot_training_history(reporter: Reporter, history: tf.keras.callbacks.History) -> None:
    hist = history.history

    if "loss" in hist and "val_loss" in hist:
        with reporter.figure("training_loss", figsize=(8, 5)) as (fig, ax):
            ax.plot(hist["loss"], label="train_loss")
            ax.plot(hist["val_loss"], label="val_loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("CNN training loss")
            ax.legend()

    if "accuracy" in hist and "val_accuracy" in hist:
        with reporter.figure("training_accuracy", figsize=(8, 5)) as (fig, ax):
            ax.plot(hist["accuracy"], label="train_accuracy")
            ax.plot(hist["val_accuracy"], label="val_accuracy")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.set_title("CNN training accuracy")
            ax.legend()

    if "val_macro_f1" in hist:
        with reporter.figure("validation_macro_f1", figsize=(8, 5)) as (fig, ax):
            ax.plot(hist["val_macro_f1"], label="val_macro_f1")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Macro F1")
            ax.set_title("CNN validation macro-F1")
            ax.legend()


def prepare_prediction_df(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    beat_meta_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = pd.DataFrame(y_prob, columns=[f"prob_{c}" for c in class_names])

    if beat_meta_df is not None and not beat_meta_df.empty:
        meta = beat_meta_df.copy()
        for col in ["record_id", "record_beat_index", "label_raw", "label_aami", "label_target"]:
            if col in meta.columns:
                df[col] = meta[col].astype("string").to_numpy()
        for col in ["beat_index"]:
            if col in meta.columns:
                df[col] = pd.to_numeric(meta[col], errors="raise").astype("int64").to_numpy()

    df["y_true"] = np.asarray(y_true, dtype=np.int64)
    df["y_pred"] = np.asarray(y_pred, dtype=np.int64)
    return df


def evaluate_split(
    reporter: Reporter,
    model: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    beat_meta_df: pd.DataFrame | None,
    split_name: str,
    class_names: list[str],
    out_dir: Path,
    model_label: str,
    batch_size: int = BATCH_SIZE,
) -> dict[str, float | int | str]:
    y_prob = model.predict(X, batch_size=batch_size, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    metrics = {
        "model": model_label,
        "split": split_name,
        "n_samples": int(len(y)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, y_pred, average="macro", zero_division=0)),
    }

    labels = np.arange(len(class_names))

    report = classification_report(
        y, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0
    )
    pd.DataFrame(report).transpose().to_csv(out_dir / f"cnn_{split_name}_classification_report.csv")

    cm = confusion_matrix(y, y_pred, labels=labels)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(out_dir / f"cnn_{split_name}_confusion_matrix.csv")

    with reporter.figure(f"cnn_{split_name}_confusion_matrix", figsize=(8, 8)) as (fig, ax):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
        ax.set_title(f"CNN - {split_name}")

    pred_df = prepare_prediction_df(y_prob, y_true=y, y_pred=y_pred, class_names=class_names, beat_meta_df=beat_meta_df)
    pred_df.to_csv(out_dir / f"cnn_{split_name}_predictions.csv", index=False)

    return metrics


def history_to_serializable_dict(history: tf.keras.callbacks.History) -> dict[str, list[float]]:
    return {key: [float(v) for v in values] for key, values in history.history.items()}


def train_cnn(
    dataset: str = "mitbih",
    data_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
    results_name: str | None = None,
    variant: str | None = None,
    batch_size: int = BATCH_SIZE,
) -> None:
    dataset = dataset.lower().strip()
    variant = normalize_variant(variant)

    if data_dir is None:
        if dataset == "incart" and variant in {"12ch", "multichannel"}:
            data_dir = Path(get_ds_par(dataset, "cnn12_data_dir"))
        else:
            data_dir = Path(get_ds_par(dataset, "cnn_data_dir"))
    else:
        data_dir = Path(data_dir)

    if results_dir is None:
        if dataset == "incart" and variant in {"12ch", "multichannel"}:
            results_dir = Path(get_ds_par(dataset, "cnn12_dir"))
            resolved_variant = "12ch"
        else:
            results_dir = Path(get_ds_par(dataset, "cnn_dir"))
            resolved_variant = "1ch"
    else:
        results_dir = Path(results_dir)
        resolved_variant = variant or "1ch"

    if results_name:
        results_dir = results_dir / results_name

    if not data_dir.exists():
        raise FileNotFoundError(f"Hiányzó data_dir: {data_dir}")

    results_dir.mkdir(parents=True, exist_ok=True)

    reporter = Reporter(results_dir)
    set_seed(SEED)

    X_train_raw, y_train, meta_train = load_npz_split(data_dir / "train.npz", return_meta=True)
    X_val_raw, y_val, meta_val = load_npz_split(data_dir / "val.npz", return_meta=True)
    X_test_raw, y_test, meta_test = load_npz_split(data_dir / "test.npz", return_meta=True)

    metadata = load_metadata(data_dir / "metadata.json")
    class_names = labels_from_metadata(metadata)

    X_train = ensure_cnn_input_shape(X_train_raw)
    X_val = ensure_cnn_input_shape(X_val_raw)
    X_test = ensure_cnn_input_shape(X_test_raw)

    n_classes = len(class_names)
    input_shape = X_train.shape[1:]
    input_kind = infer_input_kind_from_array(X_train)

    class_weights = compute_class_weights(y_train, n_classes=n_classes)

    train_ds = make_tf_dataset(X_train, y_train, batch_size=batch_size, training=True)
    val_ds = make_tf_dataset(X_val, y_val, batch_size=batch_size, training=False)

    model = build_cnn_model(input_shape=input_shape, n_classes=n_classes)

    val_macro_f1_cb = ValidationMacroF1Callback(X_val, y_val, batch_size=batch_size, every_n_epochs=VAL_MACRO_F1_EVERY)

    callbacks = [
        val_macro_f1_cb,
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.5, patience=2, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(results_dir / "best_model.keras"),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    plot_training_history(reporter, history)

    history_dict = history_to_serializable_dict(history)
    pd.DataFrame.from_dict(history_dict, orient="index").transpose().to_csv(results_dir / "training_history.csv", index=False)
    with open(results_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history_dict, f, ensure_ascii=False, indent=2)

    all_metrics = []
    model_label = f"cnn_{dataset}_{resolved_variant}"

    for split_name, X_split, y_split, meta_split in [
        ("train", X_train, y_train, meta_train),
        ("val", X_val, y_val, meta_val),
        ("test", X_test, y_test, meta_test),
    ]:
        metrics = evaluate_split(
            reporter=reporter,
            model=model,
            X=X_split,
            y=y_split,
            beat_meta_df=meta_split,
            split_name=split_name,
            class_names=class_names,
            out_dir=results_dir,
            model_label=model_label,
            batch_size=batch_size,
        )
        all_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(results_dir / "metrics_summary.csv", index=False)

    class_weights_df = pd.DataFrame({
        "class_index": list(class_weights.keys()),
        "class_weight": list(class_weights.values()),
        "class_name": [class_names[i] for i in class_weights.keys()],
    })
    class_weights_df.to_csv(results_dir / "class_weights.csv", index=False)

    run_summary = {
        "dataset": dataset,
        "variant": resolved_variant,
        "label_mode": metadata.get("label_mode"),
        "class_names": class_names,
        "label_mapping": metadata.get("label_mapping"),
        "input_kind": input_kind,
        "input_shape_train": list(X_train.shape),
        "input_shape_val": list(X_val.shape),
        "input_shape_test": list(X_test.shape),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "class_weights": class_weights,
        "seed": SEED,
        "batch_size": batch_size,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "augmentation_config": CNN_AUGMENTATION_CONFIG,
        "use_mixed_precision": USE_MIXED_PRECISION,
        "best_epoch_by_val_macro_f1": val_macro_f1_cb.best_epoch,
        "best_val_macro_f1": val_macro_f1_cb.best_macro_f1,
    }

    with open(results_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    print(metrics_df)
    print(f"[OK] Saved outputs to: {results_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN on ECG splits.")
    parser.add_argument("dataset", type=str, nargs="?", default="mitbih")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--results-name", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None, choices=["1ch", "12ch", "multichannel"])
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_cnn(
        dataset=args.dataset,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        results_name=args.results_name,
        variant=args.variant,
        batch_size=args.batch_size,
    )