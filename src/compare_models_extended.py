#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_ds_par
from src.utils.reporting import Reporter

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None


METRICS_TO_COMPARE = [
    "accuracy",
    "macro_f1",
    "weighted_f1",
]

RANK_METRIC = "macro_f1"
RANK_SPLIT = "val"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def get_default_metrics_path(dataset: str, family: str) -> Path:
    dataset = dataset.lower().strip()
    family = family.lower().strip()

    if dataset in {"mitbih", "incart"}:
        if family == "featurext":
            return Path(get_ds_par(dataset, "featurext_dir")) / "metrics_summary.csv"
        if family == "cnn":
            return Path(get_ds_par(dataset, "cnn_dir")) / "metrics_summary.csv"
        if family in {"cnn12", "cnn12ch"}:
            return Path(get_ds_par(dataset, "cnn12_dir")) / "metrics_summary.csv"

    metrics_root = get_ds_par("cross_dataset", "metrics_dir")
    return Path(metrics_root) / f"{dataset}_{family}" / "metrics_summary.csv"


def get_default_results_dir(
    left_name: str,
    right_name: str,
    dataset: str | None = None,
    results_subdir: str | None = None,
) -> Path:
    compare_root = get_ds_par("cross_dataset", "compare_dir")
    compare_root.mkdir(parents=True, exist_ok=True)

    if results_subdir is not None:
        results_dir = compare_root / results_subdir
    elif dataset is not None:
        results_dir = compare_root / f"{dataset}_{left_name}_vs_{right_name}"
    else:
        results_dir = compare_root / f"{left_name}_vs_{right_name}"

    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def load_metrics(path: Path, source_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Hiányzó metrics fájl: {path}")

    df = pd.read_csv(path)

    required = {"model", "split", *METRICS_TO_COMPARE}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Hiányzó oszlop(ok) a metrics fájlban {path}: {sorted(missing)}"
        )

    df = df.copy()
    df["source"] = source_name
    return df


def select_best_row(
    df: pd.DataFrame,
    rank_metric: str = RANK_METRIC,
    rank_split: str = RANK_SPLIT,
) -> tuple[pd.DataFrame, str]:
    split_df = df[df["split"] == rank_split].copy()
    if split_df.empty:
        raise ValueError(f"A metrics fájlban nincs {rank_split} split.")

    best_idx = split_df[rank_metric].idxmax()
    best_model = str(split_df.loc[best_idx, "model"])

    return df[df["model"] == best_model].copy(), best_model


def build_comparison(
    left_metrics_path: Path,
    right_metrics_path: Path,
    left_name: str,
    right_name: str,
    select_best_left: bool = False,
    select_best_right: bool = False,
) -> tuple[pd.DataFrame, str | None, str | None]:
    left_df = load_metrics(left_metrics_path, source_name=left_name)
    right_df = load_metrics(right_metrics_path, source_name=right_name)

    left_selected_model = None
    right_selected_model = None

    if select_best_left:
        left_df, left_selected_model = select_best_row(left_df)

    if select_best_right:
        right_df, right_selected_model = select_best_row(right_df)

    merged = left_df.merge(
        right_df,
        on="split",
        suffixes=(f"_{left_name}", f"_{right_name}"),
        how="inner",
        validate="one_to_one",
    )

    rows = []
    for _, row in merged.iterrows():
        out = {
            "split": row["split"],
            f"{left_name}_model": row[f"model_{left_name}"],
            f"{right_name}_model": row[f"model_{right_name}"],
        }

        for metric in METRICS_TO_COMPARE:
            lv = float(row[f"{metric}_{left_name}"])
            rv = float(row[f"{metric}_{right_name}"])

            out[f"{metric}_{left_name}"] = lv
            out[f"{metric}_{right_name}"] = rv
            out[f"{metric}_delta_{right_name}_minus_{left_name}"] = rv - lv
            out[f"{metric}_mean_{left_name}_{right_name}"] = (lv + rv) / 2.0

        rows.append(out)

    return pd.DataFrame(rows), left_selected_model, right_selected_model


def build_long_table(
    compare_df: pd.DataFrame,
    left_name: str,
    right_name: str,
) -> pd.DataFrame:
    rows = []

    for _, row in compare_df.iterrows():
        split_name = row["split"]

        rows.append(
            {
                "split": split_name,
                "family": left_name,
                "variant": left_name,
                "model": row[f"{left_name}_model"],
                "accuracy": row[f"accuracy_{left_name}"],
                "macro_f1": row[f"macro_f1_{left_name}"],
                "weighted_f1": row[f"weighted_f1_{left_name}"],
            }
        )

        rows.append(
            {
                "split": split_name,
                "family": right_name,
                "variant": right_name,
                "model": row[f"{right_name}_model"],
                "accuracy": row[f"accuracy_{right_name}"],
                "macro_f1": row[f"macro_f1_{right_name}"],
                "weighted_f1": row[f"weighted_f1_{right_name}"],
            }
        )

    return pd.DataFrame(rows)


def build_leaderboard(long_df: pd.DataFrame) -> pd.DataFrame:
    board = long_df[long_df["split"] == RANK_SPLIT].copy()
    board = board.sort_values(RANK_METRIC, ascending=False).reset_index(drop=True)
    board.insert(0, "rank", np.arange(1, len(board) + 1))
    return board


def plot_metric_comparison(
    reporter: Reporter,
    compare_df: pd.DataFrame,
    metric: str,
    left_name: str,
    right_name: str,
) -> None:
    x_labels = compare_df["split"].tolist()
    x = np.arange(len(x_labels))
    width = 0.38

    with reporter.figure(f"compare_{metric}", figsize=(8, 5)) as (fig, ax):
        ax.bar(
            x - width / 2,
            compare_df[f"{metric}_{left_name}"],
            width=width,
            label=left_name,
        )
        ax.bar(
            x + width / 2,
            compare_df[f"{metric}_{right_name}"],
            width=width,
            label=right_name,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel(metric)
        ax.set_title(f"{left_name} vs {right_name} - {metric}")
        ax.legend()
        fig.tight_layout()


def plot_bland_altman(
    reporter: Reporter,
    compare_df: pd.DataFrame,
    metric: str,
    left_name: str,
    right_name: str,
) -> None:
    mean_vals = compare_df[f"{metric}_mean_{left_name}_{right_name}"].to_numpy(dtype=float)
    diff_vals = compare_df[f"{metric}_delta_{right_name}_minus_{left_name}"].to_numpy(dtype=float)

    mean_diff = np.mean(diff_vals)
    sd_diff = np.std(diff_vals, ddof=1) if len(diff_vals) > 1 else 0.0

    loa_upper = mean_diff + 1.96 * sd_diff
    loa_lower = mean_diff - 1.96 * sd_diff

    with reporter.figure(f"bland_altman_{metric}", figsize=(7, 5)) as (fig, ax):
        ax.scatter(mean_vals, diff_vals, s=50)

        for _, row in compare_df.iterrows():
            ax.text(
                row[f"{metric}_mean_{left_name}_{right_name}"],
                row[f"{metric}_delta_{right_name}_minus_{left_name}"],
                str(row["split"]),
                fontsize=9,
                ha="left",
                va="bottom",
            )

        ax.axhline(mean_diff, linestyle="--", label=f"Mean diff = {mean_diff:.4f}")
        ax.axhline(loa_upper, linestyle=":", label=f"+1.96 SD = {loa_upper:.4f}")
        ax.axhline(loa_lower, linestyle=":", label=f"-1.96 SD = {loa_lower:.4f}")

        ax.set_xlabel(f"Mean of {left_name} and {right_name} ({metric})")
        ax.set_ylabel(f"{right_name} - {left_name} ({metric})")
        ax.set_title(f"Bland-Altman: {left_name} vs {right_name} - {metric}")
        ax.legend()
        fig.tight_layout()


def maybe_run_stat_test(
    compare_df: pd.DataFrame,
    left_name: str,
    right_name: str,
) -> pd.DataFrame:
    rows = []

    for metric in METRICS_TO_COMPARE:
        left_vals = compare_df[f"{metric}_{left_name}"].to_numpy(dtype=float)
        right_vals = compare_df[f"{metric}_{right_name}"].to_numpy(dtype=float)

        row = {
            "metric": metric,
            "n_pairs": int(len(left_vals)),
            "test": None,
            "statistic": None,
            "p_value": None,
            "note": None,
        }

        if len(left_vals) < 5:
            row["note"] = (
                "Nagyon kevés pár áll rendelkezésre "
                "(tipikusan csak train/val/test), "
                "ezért a stat teszt gyenge és óvatosan értelmezendő."
            )

        if wilcoxon is None:
            row["note"] = "scipy nincs elérve, Wilcoxon teszt nem futott."
            rows.append(row)
            continue

        try:
            stat = wilcoxon(right_vals, left_vals, zero_method="wilcox", alternative="two-sided")
            row["test"] = "wilcoxon"
            row["statistic"] = float(stat.statistic)
            row["p_value"] = float(stat.pvalue)
            if row["note"] is None:
                row["note"] = "Opcionális Wilcoxon signed-rank teszt."
        except Exception as exc:
            row["note"] = f"A stat teszt nem futott le: {exc}"

        rows.append(row)

    return pd.DataFrame(rows)


def compare_models_extended(
    left_metrics_path: str | Path,
    right_metrics_path: str | Path,
    left_name: str,
    right_name: str,
    results_subdir: str | None = None,
    dataset: str | None = None,
    results_dir: str | Path | None = None,
    select_best_left: bool = False,
    select_best_right: bool = False,
) -> None:
    left_metrics_path = Path(left_metrics_path)
    right_metrics_path = Path(right_metrics_path)

    if results_dir is not None:
        resolved_results_dir = Path(results_dir)
        resolved_results_dir.mkdir(parents=True, exist_ok=True)
    else:
        resolved_results_dir = get_default_results_dir(
            left_name=left_name,
            right_name=right_name,
            dataset=dataset,
            results_subdir=results_subdir,
        )

    reporter = Reporter(resolved_results_dir)

    compare_df, left_selected_model, right_selected_model = build_comparison(
        left_metrics_path=left_metrics_path,
        right_metrics_path=right_metrics_path,
        left_name=left_name,
        right_name=right_name,
        select_best_left=select_best_left,
        select_best_right=select_best_right,
    )

    long_df = build_long_table(compare_df, left_name=left_name, right_name=right_name)
    leaderboard_df = build_leaderboard(long_df)

    comparison_csv = resolved_results_dir / "comparison_table.csv"
    long_csv = resolved_results_dir / "comparison_long.csv"
    leaderboard_csv = resolved_results_dir / "leaderboard.csv"

    compare_df.to_csv(comparison_csv, index=False)
    long_df.to_csv(long_csv, index=False)
    leaderboard_df.to_csv(leaderboard_csv, index=False)

    for metric in METRICS_TO_COMPARE:
        plot_metric_comparison(
            reporter=reporter,
            compare_df=compare_df,
            metric=metric,
            left_name=left_name,
            right_name=right_name,
        )
        plot_bland_altman(
            reporter=reporter,
            compare_df=compare_df,
            metric=metric,
            left_name=left_name,
            right_name=right_name,
        )

    stat_df = maybe_run_stat_test(compare_df, left_name=left_name, right_name=right_name)
    stat_test_csv = resolved_results_dir / "stat_test.csv"
    stat_df.to_csv(stat_test_csv, index=False)

    summary = {
        "created_at": now_iso(),
        "compare_type": "models_extended",
        "dataset": dataset,
        "left_name": left_name,
        "right_name": right_name,
        "left_selected_model": left_selected_model,
        "right_selected_model": right_selected_model,
        "rank_metric": RANK_METRIC,
        "rank_split": RANK_SPLIT,
        "left_metrics_file": str(left_metrics_path),
        "right_metrics_file": str(right_metrics_path),
        "comparison_table_file": str(comparison_csv),
        "comparison_long_file": str(long_csv),
        "leaderboard_file": str(leaderboard_csv),
        "stat_test_file": str(stat_test_csv),
        "metrics_compared": METRICS_TO_COMPARE,
    }

    with open(resolved_results_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[OK] Saved:")
    print(f"  - {comparison_csv}")
    print(f"  - {long_csv}")
    print(f"  - {leaderboard_csv}")
    print(f"  - {stat_test_csv}")
    print(f"  - {resolved_results_dir / 'run_summary.json'}")

    print(f"\nLeaderboard ({RANK_SPLIT} {RANK_METRIC}):")
    print(leaderboard_df)
    print("\nComparison table:")
    print(compare_df)


def compare_dataset_models_extended(
    dataset: str,
    left_family: str,
    right_family: str,
    results_subdir: str | None = None,
    select_best_left: bool = False,
    select_best_right: bool = False,
) -> None:
    dataset = dataset.lower().strip()
    left_metrics_path = get_default_metrics_path(dataset, left_family)
    right_metrics_path = get_default_metrics_path(dataset, right_family)

    compare_models_extended(
        left_metrics_path=left_metrics_path,
        right_metrics_path=right_metrics_path,
        left_name=left_family,
        right_name=right_family,
        results_subdir=results_subdir,
        dataset=dataset,
        select_best_left=select_best_left,
        select_best_right=select_best_right,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extended model comparison.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--left-family", default=None)
    parser.add_argument("--right-family", default=None)
    parser.add_argument("--left-metrics-path", default=None)
    parser.add_argument("--right-metrics-path", default=None)
    parser.add_argument("--left-name", default=None)
    parser.add_argument("--right-name", default=None)
    parser.add_argument("--results-subdir", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--select-best-left", action="store_true")
    parser.add_argument("--select-best-right", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.left_metrics_path and args.right_metrics_path and args.left_name and args.right_name:
        compare_models_extended(
            left_metrics_path=args.left_metrics_path,
            right_metrics_path=args.right_metrics_path,
            left_name=args.left_name,
            right_name=args.right_name,
            results_subdir=args.results_subdir,
            dataset=args.dataset,
            results_dir=args.results_dir,
            select_best_left=args.select_best_left,
            select_best_right=args.select_best_right,
        )
    elif args.dataset and args.left_family and args.right_family:
        compare_dataset_models_extended(
            dataset=args.dataset,
            left_family=args.left_family,
            right_family=args.right_family,
            results_subdir=args.results_subdir,
            select_best_left=args.select_best_left,
            select_best_right=args.select_best_right,
        )
    else:
        raise SystemExit(
            "Add meg vagy:\n"
            "1) --dataset --left-family --right-family\n"
            "vagy\n"
            "2) --left-metrics-path --right-metrics-path --left-name --right-name"
        )