#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.beat_extractor import extract_beats
from src.config import (
    BEAT_POST_SEC,
    BEAT_PRE_SEC,
    PREPROCESSING_CONFIG,
    QC_RECORD_MODE,
    QC_MAX_RECORDS,
    QC_RANDOM_SEED,
    QC_VERBOSE_LOAD,
    get_ds_par,
)
from src.utils.dataset_loader import load_record
from src.utils.labels import map_aami_labels
from src.utils.reporting import Reporter
from src.utils.signals import get_preprocessed_signal

PLOT_SEGMENT_SAMPLES = 2500
MAX_BEATS_TO_OVERLAY = 20

def detect_rr_outliers(ann_samples: np.ndarray, fs: float) -> dict:
    if len(ann_samples) < 3:
        return {
            "rr_mean_sec": np.nan,
            "rr_std_sec": np.nan,
            "rr_too_short_n": 0,
            "rr_too_long_n": 0,
            "rr_z_outlier_n": 0,
        }

    rr = np.diff(ann_samples) / fs
    rr_mean = float(np.mean(rr))
    rr_std = float(np.std(rr))

    rr_too_short_n = int(np.sum(rr < 0.30))
    rr_too_long_n = int(np.sum(rr > 2.00))

    if rr_std == 0:
        rr_z_outlier_n = 0
    else:
        rr_z = np.abs((rr - rr_mean) / rr_std)
        rr_z_outlier_n = int(np.sum(rr_z > 4.0))

    return {
        "rr_mean_sec": rr_mean,
        "rr_std_sec": rr_std,
        "rr_too_short_n": rr_too_short_n,
        "rr_too_long_n": rr_too_long_n,
        "rr_z_outlier_n": rr_z_outlier_n,
    }


def plot_signal_segment(
    reporter: Reporter,
    dataset_display_name: str,
    record_name: str,
    sig_1d: np.ndarray,
    ann_samples: np.ndarray,
    ann_symbols: np.ndarray,
    fs: float,
    n_samples: int = PLOT_SEGMENT_SAMPLES,
) -> None:
    seg = sig_1d[:n_samples]
    ann_mask = ann_samples < n_samples
    seg_ann = ann_samples[ann_mask]
    seg_sym = ann_symbols[ann_mask]

    t = np.arange(len(seg)) / fs

    with reporter.figure(f"{record_name}_segment", figsize=(14, 4)) as (fig, ax):
        ax.plot(t, seg, linewidth=1)

        if len(seg_ann) > 0:
            ax.scatter(seg_ann / fs, sig_1d[seg_ann], s=18)
            for sample, sym in zip(seg_ann, seg_sym):
                ax.text(sample / fs, sig_1d[sample], str(sym), fontsize=8)

        ax.set_title(f"{dataset_display_name} {record_name} - first segment with annotations")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        fig.tight_layout()


def plot_overlay_beats(
    reporter: Reporter,
    dataset: str,
    dataset_display_name: str,
    record_name: str,
    beats: np.ndarray,
    beat_pre_sec: float = BEAT_PRE_SEC,
    max_beats: int = MAX_BEATS_TO_OVERLAY,
) -> None:
    if len(beats) == 0:
        return

    fs = get_ds_par(dataset, "fs")
    beat_pre_samples = int(beat_pre_sec * fs)
    n = min(max_beats, len(beats))

    with reporter.figure(f"{record_name}_overlay_beats", figsize=(10, 6)) as (fig, ax):
        for i in range(n):
            ax.plot(beats[i], alpha=0.5, linewidth=1)

        ax.axvline(beat_pre_samples, linestyle="--", linewidth=1)
        ax.set_title(f"{dataset_display_name} {record_name} - overlay of {n} extracted beats")
        ax.set_xlabel("Sample in beat window")
        ax.set_ylabel("Normalized amplitude")
        fig.tight_layout()

def plot_label_distribution(
    reporter: Reporter,
    dataset_display_name: str,
    record_name: str,
    labels: np.ndarray,
) -> None:
    if len(labels) == 0:
        return

    vc = pd.Series(labels).value_counts().sort_values(ascending=False)

    with reporter.figure(f"{record_name}_label_distribution", figsize=(14, 5)) as (fig, ax):
        vc.plot(kind="bar", ax=ax, width=0.6)

        ax.set_title(f"{dataset_display_name} {record_name} - beat symbol distribution")
        ax.set_xlabel("Annotation symbol")
        ax.set_ylabel("Count")

        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        # ritkítás ha sok label van
        if len(vc) > 20:
            step = len(vc) // 20
            ax.set_xticks(range(0, len(vc), step))

        ax.margins(x=0.01)

        fig.tight_layout()


def plot_global_distribution(
    reporter: Reporter,
    series: pd.Series,
    figure_name: str,
    title: str,
    xlabel: str,
    figsize: tuple[float, float],
) -> None:
    if series.empty:
        return

    with reporter.figure(figure_name, figsize=figsize) as (fig, ax):
        series.value_counts().plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        fig.tight_layout()


def check_record(dataset: str, record_name: str) -> tuple[dict, pd.DataFrame]:
    rec = load_record(dataset, record_name)

    sig_1d = get_preprocessed_signal(
        rec,
        channel=0,
        preprocessing_config=PREPROCESSING_CONFIG,
    )
    ann_samples = rec.annotation_samples

    beats, beat_labels, beat_centers, extract_meta = extract_beats(
        rec,
        channel=0,
        fs=get_ds_par(dataset, "fs"),
        pre_sec=BEAT_PRE_SEC,
        post_sec=BEAT_POST_SEC,
        normalize=True,
        preprocessing_config=PREPROCESSING_CONFIG,
    )
    aami_labels = map_aami_labels(beat_labels)

    signal_std = float(np.std(sig_1d))
    signal_min = float(np.min(sig_1d))
    signal_max = float(np.max(sig_1d))
    signal_range = signal_max - signal_min
    noise_level = float(np.std(np.diff(sig_1d))) if len(sig_1d) > 1 else np.nan

    rr_info = detect_rr_outliers(ann_samples, rec.fs)

    abs_sig = np.abs(sig_1d)
    amp_threshold = float(np.mean(abs_sig) + 6 * np.std(abs_sig))
    amp_outlier_n = int(np.sum(abs_sig > amp_threshold))

    n_extract_skipped = (
        int((~extract_meta["kept"]).sum())
        if not extract_meta.empty and "kept" in extract_meta.columns
        else 0
    )

    n_channels = int(rec.signal.shape[1]) if rec.signal.ndim > 1 else 1

    summary = {
        "dataset": dataset,
        "record": record_name,
        "n_samples": int(rec.signal.shape[0]),
        "n_channels": n_channels,
        "fs": float(rec.fs),
        "n_annotations": int(len(ann_samples)),
        "n_extracted_beats": int(len(beat_labels)),
        "n_extract_skipped": n_extract_skipped,
        "signal_mean": float(np.mean(sig_1d)),
        "signal_std": signal_std,
        "signal_min": signal_min,
        "signal_max": signal_max,
        "signal_range": signal_range,
        "noise_level": noise_level,
        "has_nan": bool(np.isnan(rec.signal).any()),
        "has_inf": bool(np.isinf(rec.signal).any()),
        "flatline": bool(signal_std < 1e-3),
        "amp_outlier_n": amp_outlier_n,
        **rr_info,
    }

    kept_extract_meta = extract_meta.loc[extract_meta["kept"] == True].copy().reset_index(drop=True)
    beat_index = kept_extract_meta["ann_index"].to_numpy(dtype=np.int64)

    beat_df = pd.DataFrame(
        {
            "dataset": dataset,
            "record": record_name,
            "beat_index": beat_index,
            "record_beat_index": [f"{record_name}_{i}" for i in beat_index],
            "sample": beat_centers,
            "symbol": beat_labels,
            "aami": aami_labels,
        }
    )

    return summary, beat_df


def plot_record_qc_row(
    reporter: Reporter,
    dataset: str,
    dataset_display_name: str,
    record_name: str,
    sig_1d: np.ndarray,
    ann_samples: np.ndarray,
    ann_symbols: np.ndarray,
    fs: float,
    beats: np.ndarray,
    beat_labels: np.ndarray,
    n_samples: int = PLOT_SEGMENT_SAMPLES,
    max_beats: int = MAX_BEATS_TO_OVERLAY,
) -> None:
    seg = sig_1d[:n_samples]
    ann_mask = ann_samples < n_samples
    seg_ann = ann_samples[ann_mask]
    seg_sym = ann_symbols[ann_mask]
    t = np.arange(len(seg)) / fs

    beat_pre_samples = int(BEAT_PRE_SEC * get_ds_par(dataset, "fs"))
    n_overlay = min(max_beats, len(beats))

    vc = pd.Series(beat_labels).value_counts().sort_values(ascending=False) if len(beat_labels) else pd.Series(dtype=int)

    with reporter.figure(f"{record_name}_qc_row", figsize=(15, 3.8)) as (fig, _):
        axes = fig.subplots(1, 3)

        # 1. segment
        axes[0].plot(t, seg, linewidth=1)
        if len(seg_ann) > 0:
            axes[0].scatter(seg_ann / fs, sig_1d[seg_ann], s=12)
            for sample, sym in zip(seg_ann, seg_sym):
                axes[0].text(sample / fs, sig_1d[sample], str(sym), fontsize=6)
        axes[0].set_title(f"{record_name} segment")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Amp.")

        # 2. overlay beats
        if len(beats) > 0:
            for i in range(n_overlay):
                axes[1].plot(beats[i], alpha=0.5, linewidth=0.8)
            axes[1].axvline(beat_pre_samples, linestyle="--", linewidth=1)
        axes[1].set_title(f"{record_name} overlay beats")
        axes[1].set_xlabel("Sample")
        axes[1].set_ylabel("Norm. amp.")

        # 3. label distribution
        if not vc.empty:
            vc.plot(kind="bar", ax=axes[2])
        axes[2].set_title(f"{record_name} label dist.")
        axes[2].set_xlabel("Symbol")
        axes[2].set_ylabel("Count")

        fig.suptitle(f"{dataset_display_name} - {record_name}", fontsize=11)
        fig.tight_layout()



def qc_dataset_ext(dataset: str) -> None:
    dataset = dataset.lower()
    dataset_display_name = str(get_ds_par(dataset, "display_name"))

    qc_dir = Path(get_ds_par(dataset, "qc_dir"))
    qc_dir.mkdir(parents=True, exist_ok=True)

    fig_dir = Path(qc_dir) / "graphs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    interim_dir = Path(get_ds_par(dataset, "interim_dir"))
    interim_dir.mkdir(parents=True, exist_ok=True)

    reporter = Reporter(fig_dir)

    all_summary: list[dict] = []
    all_beats: list[pd.DataFrame] = []

    print(f"Running extended QC on {dataset_display_name}...")

    records = list(get_ds_par(dataset, "records"))

    # ------------------------------------------------------------
    # QC plot record selection
    # ------------------------------------------------------------
    qc_record_mode = str(QC_RECORD_MODE).lower().strip()
    qc_max_records = int(QC_MAX_RECORDS)
    qc_random_seed = int(QC_RANDOM_SEED)
    qc_verbose_load = int(QC_VERBOSE_LOAD)

    if qc_max_records < 1:
        qc_max_records = 1

    if qc_record_mode not in {"all", "head", "random"}:
        print(f"[WARN] Unknown qc_record_mode={qc_record_mode!r}; fallback to 'all'")
        qc_record_mode = "all"

    if qc_record_mode == "all":
        plot_records = records
    elif qc_record_mode == "head":
        plot_records = records[:qc_max_records]
    else:  # random
        rng = np.random.default_rng(qc_random_seed)
        n_pick = min(qc_max_records, len(records))
        plot_records = sorted(rng.choice(records, size=n_pick, replace=False).tolist())

    plot_record_set = set(plot_records)

    print(f"[INFO] Total records: {len(records)}")
    print(
        f"[INFO] QC plot mode: {qc_record_mode} | "
        f"plotting {len(plot_records)} record(s)"
    )
    if qc_record_mode == "random":
        print(f"[INFO] QC random seed: {qc_random_seed}")

    # ------------------------------------------------------------
    # Full numeric QC for ALL records
    # Plot only for selected subset
    # ------------------------------------------------------------
    for i, record_name in enumerate(records, start=1):
        try:
            print(f"[QC {i}/{len(records)}] {record_name}")

            summary, beat_df = check_record(dataset, record_name)
            all_summary.append(summary)
            all_beats.append(beat_df)

            if record_name not in plot_record_set:
                continue

            rec = load_record(dataset, record_name, verbose=qc_verbose_load)

            sig_1d = get_preprocessed_signal(
                rec,
                channel=0,
                preprocessing_config=PREPROCESSING_CONFIG,
            )
            ann_samples = rec.annotation_samples
            ann_symbols = rec.annotation_symbols

            beats, beat_labels, _, _ = extract_beats(
                rec,
                channel=0,
                fs=get_ds_par(dataset, "fs"),
                pre_sec=BEAT_PRE_SEC,
                post_sec=BEAT_POST_SEC,
                normalize=True,
                preprocessing_config=PREPROCESSING_CONFIG,
            )

            plot_record_qc_row(
                reporter=reporter,
                dataset=dataset,
                dataset_display_name=dataset_display_name,
                record_name=record_name,
                sig_1d=sig_1d,
                ann_samples=ann_samples,
                ann_symbols=ann_symbols,
                fs=rec.fs,
                beats=beats,
                beat_labels=beat_labels,
            )

        except Exception as e:
            print(f"[ERROR] {record_name}: {e}")
            all_summary.append(
                {
                    "dataset": dataset,
                    "record": record_name,
                    "error": str(e),
                }
            )

    summary_df = pd.DataFrame(all_summary)
    summary_df.to_csv(qc_dir / "qc_summary.csv", index=False)

    beat_df = pd.concat(all_beats, ignore_index=True) if all_beats else pd.DataFrame()
    beat_df.to_csv(interim_dir / "beat_annotations.csv", index=False)

    if not beat_df.empty:
        dup_mask = beat_df.duplicated(subset=["record_beat_index"], keep=False)
        if dup_mask.any():
            print("[WARN] Duplicate record_beat_index found in QC beat table.")
            beat_df.loc[dup_mask].to_csv(
                qc_dir / "duplicate_record_beat_index.csv",
                index=False,
            )

        raw_counts = (
            beat_df.groupby(["record", "symbol"])
            .size()
            .reset_index(name="n")
            .sort_values(["record", "n"], ascending=[True, False])
        )
        raw_counts.to_csv(
            interim_dir / f"{dataset}_beat_symbol_counts_by_record.csv",
            index=False,
        )

        aami_counts = (
            beat_df.groupby(["record", "aami"])
            .size()
            .reset_index(name="n")
            .sort_values(["record", "n"], ascending=[True, False])
        )
        aami_counts.to_csv(
            interim_dir / f"{dataset}_aami_counts_by_record.csv",
            index=False,
        )

        global_raw = beat_df["symbol"].value_counts().rename_axis("symbol").reset_index(name="n")
        global_raw.to_csv(
            interim_dir / f"{dataset}_beat_symbol_counts_global.csv",
            index=False,
        )

        global_aami = beat_df["aami"].value_counts().rename_axis("aami").reset_index(name="n")
        global_aami.to_csv(
            interim_dir / f"{dataset}_aami_counts_global.csv",
            index=False,
        )

        plot_global_distribution(
            reporter=reporter,
            series=beat_df["symbol"],
            figure_name="global_beat_symbol_distribution",
            title=f"{dataset_display_name} global beat symbol distribution",
            xlabel="Annotation symbol",
            figsize=(10, 4),
        )

        plot_global_distribution(
            reporter=reporter,
            series=beat_df["aami"],
            figure_name="global_aami_distribution",
            title=f"{dataset_display_name} global AAMI distribution",
            xlabel="AAMI class",
            figsize=(8, 4),
        )

    error_mask = (
        summary_df["error"].notna()
        if "error" in summary_df.columns
        else pd.Series(False, index=summary_df.index)
    )
    problem_mask = error_mask.copy()

    for col in ["has_nan", "has_inf", "flatline"]:
        if col in summary_df.columns:
            problem_mask = problem_mask | (summary_df[col] == True)

    for col in ["amp_outlier_n", "rr_too_short_n", "rr_too_long_n", "rr_z_outlier_n"]:
        if col in summary_df.columns:
            problem_mask = problem_mask | (summary_df[col] >= 1)

    problems_df = summary_df.loc[problem_mask].copy()
    problems_df.to_csv(qc_dir / "problems.csv", index=False)

    print("\nSaved:")
    print(f"- {qc_dir / 'qc_summary.csv'}")
    print(f"- {interim_dir / 'beat_annotations.csv'}")
    print(f"- {qc_dir / 'problems.csv'}")
    print(f"- figures in {fig_dir}")
    print(f"- plotted records: {len(plot_records)} / {len(records)}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extended QC for ECG dataset.")
    parser.add_argument("dataset", help="Dataset name, e.g. mitbih or incart")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qc_dataset_ext(args.dataset)


if __name__ == "__main__":
    main()
