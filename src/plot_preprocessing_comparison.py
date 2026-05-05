#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.config import PREPROCESSING_CONFIG, get_ds_par
from src.utils.reporting import Reporter
from src.utils.signals import get_preprocessed_signal, remove_baseline_wander
from src.utils.dataset_loader import load_record

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Raw / baseline-corrected / fully preprocessed ECG comparison plot"
    )
    parser.add_argument(
        "dataset",
        choices=["mitbih", "incart"],
        help="Melyik adatbázisból töltsük a rekordot.",
    )
    parser.add_argument(
        "--record-name",
        required=True,
        help="Rekord neve / azonosítója, pl. 100 vagy I01",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Melyik csatornát használjuk. Alapértelmezés: 0",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=2500,
        help="Hány mintát ábrázoljunk a jel elejéről. Alapértelmezés: 2500",
    )
    return parser.parse_args()

def get_channel_signal(signal: np.ndarray, channel: int) -> np.ndarray:
    sig = np.asarray(signal, dtype=float)

    if sig.ndim == 1:
        if channel != 0:
            raise IndexError("1D jel esetén csak a 0. csatorna érvényes.")
        return sig

    if sig.ndim != 2:
        raise ValueError(f"Nem támogatott signal shape: {sig.shape}")

    if channel < 0 or channel >= sig.shape[1]:
        raise IndexError(
            f"Érvénytelen channel index: {channel}, elérhető csatornák: 0..{sig.shape[1]-1}"
        )

    return sig[:, channel]


def plot_preprocessing_comparison(
    dataset: str,
    record_name: str,
    channel: int = 0,
    n_samples: int = 2500,
) -> None:

    out_dir = get_ds_par(dataset, "results_dir") / "preprocessing_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    reporter = Reporter(out_dir)

    rec = load_record(dataset, record_name=record_name)

    raw_sig = get_channel_signal(rec.signal, channel=channel)

    baseline_corrected_sig = remove_baseline_wander(
        raw_sig,
        fs=rec.fs,
        cutoff=PREPROCESSING_CONFIG.get("baseline_cutoff", 0.5),
        order=PREPROCESSING_CONFIG.get("baseline_order", 3),
    )

    filtered_sig = get_preprocessed_signal(
        rec,
        channel=channel,
        preprocessing_config=PREPROCESSING_CONFIG,
    )

    n_samples = min(int(n_samples), len(raw_sig))
    raw_seg = raw_sig[:n_samples]
    corrected_seg = baseline_corrected_sig[:n_samples]
    filtered_seg = filtered_sig[:n_samples]
    baseline_seg = raw_seg - corrected_seg

    t = np.arange(n_samples) / float(rec.fs)

    figure_name = f"{dataset}_{record_name}_ch{channel}_preprocessing_comparison"

    with reporter.figure(figure_name, figsize=(14, 10)) as (fig, _):
        axes = fig.subplots(3, 1)

        axes[0].plot(t, raw_seg, linewidth=1, label="Raw")
        axes[0].plot(t, baseline_seg, linewidth=1, label="Estimated baseline")
        axes[0].legend()
        axes[0].set_title(
            f"{dataset.upper()} {record_name} - raw signal and estimated baseline"
        )
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Amplitude")

        axes[1].plot(t, corrected_seg, linewidth=1)
        axes[1].set_title("After baseline removal")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Amplitude")

        axes[2].plot(t, filtered_seg, linewidth=1)
        axes[2].set_title("After full preprocessing")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_ylabel("Amplitude")

        fig.tight_layout()

    print(f"[OK] Saved figure to: {reporter.out_dir / f'{figure_name}.png'}")


def main() -> None:
    args = parse_args()
    plot_preprocessing_comparison(
        dataset=args.dataset,
        record_name=args.record_name,
        channel=args.channel,
        n_samples=args.n_samples,
    )


if __name__ == "__main__":
    main()