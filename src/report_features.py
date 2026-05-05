#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from src.config import get_ds_par
from src.utils.reporting import Reporter


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def normalize_label_series(series: pd.Series) -> pd.Series:
    """
    A label oszlop elemeit garantáltan hash-elhető skalár értékekké alakítja.
    """
    return series.apply(
        lambda x: x.item() if isinstance(x, np.ndarray) and np.asarray(x).size == 1 else x
    )


def numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    cols = df.select_dtypes(include=[np.number]).columns.tolist()

    exclude = {
        "y",
        "sample_index",
        "beat_index",
        "beat_idx",
        "center_idx",
        "center_sample",
        "lead_index",
        "fs",
        "beat_pre_samples",
        "beat_post_samples",
        "beat_pre_sec",
        "beat_post_sec",
        "crop_start",
        "crop_end",
        "pad_left",
        "pad_right",
    }

    return [c for c in cols if c not in exclude]


def safe_read_csv(csv_path: Path) -> pd.DataFrame | None:
    """
    Biztonságos CSV-beolvasás:
    - hiányzó fájl -> None
    - 0 bájtos fájl -> None
    - EmptyDataError -> None
    """
    if not csv_path.exists():
        print(f"[WARN] Hiányzó feature fájl, kihagyva: {csv_path}")
        return None

    try:
        if csv_path.stat().st_size == 0:
            print(f"[WARN] Üres feature fájl (0 bájt), kihagyva: {csv_path}")
            return None
    except OSError as exc:
        print(f"[WARN] Nem olvasható fájlméret, kihagyva: {csv_path} | {exc}")
        return None

    try:
        df = pd.read_csv(csv_path, low_memory=False)

        if "rr_next_plausible" in df.columns:
            df["rr_next_plausible"] = pd.to_numeric(
                df["rr_next_plausible"],
                errors="coerce",
            )

        return df

    except EmptyDataError:
        print(f"[WARN] CSV üres vagy nincs benne parse-olható oszlop, kihagyva: {csv_path}")
        return None


def save_missing_report(
    reporter: Reporter,
    df: pd.DataFrame,
    prefix: str,
) -> Path:
    out_name = f"{prefix}_missing_values"

    miss = df.isna().sum().rename("n_missing").reset_index()
    miss.columns = ["column", "n_missing"]
    miss["fraction_missing"] = miss["n_missing"] / max(len(df), 1)

    reporter.save_df(miss, out_name, index=False, print_df=False)
    return reporter.out_dir / f"{out_name}.csv"


def save_label_means(
    reporter: Reporter,
    df: pd.DataFrame,
    prefix: str,
) -> Path | None:
    if "label" not in df.columns:
        return None

    num_cols = numeric_feature_cols(df)
    if not num_cols:
        return None

    out_name = f"{prefix}_label_means"

    df = df.copy()
    df["label"] = normalize_label_series(df["label"])

    grp = df.groupby("label")[num_cols].mean().reset_index()
    reporter.save_df(grp, out_name, index=False, print_df=False)

    return reporter.out_dir / f"{out_name}.csv"


def save_correlations(
    reporter: Reporter,
    df: pd.DataFrame,
    prefix: str,
) -> Path | None:
    num_cols = numeric_feature_cols(df)
    if len(num_cols) < 2:
        return None

    out_name = f"{prefix}_correlation_matrix"
    corr = df[num_cols].corr()
    reporter.save_df(corr.reset_index(), out_name, index=False, print_df=False)

    return reporter.out_dir / f"{out_name}.csv"


def plot_label_distribution(
    reporter: Reporter,
    df: pd.DataFrame,
    prefix: str,
) -> Path | None:
    if "label" not in df.columns or df.empty:
        return None

    df = df.copy()
    df["label"] = normalize_label_series(df["label"])

    vc = df["label"].value_counts().sort_index()
    figure_name = f"{prefix}_label_distribution"

    with reporter.figure(figure_name, figsize=(10, 4.5)) as (fig, ax):
        vc.plot(kind="bar", ax=ax, width=0.65)

        ax.set_title(f"{prefix} label distribution")
        ax.set_xlabel("Label")
        ax.set_ylabel("Count")

        ax.tick_params(axis="x", labelrotation=0, labelsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.margins(x=0.02)

        fig.tight_layout()

    return reporter.out_dir / f"{figure_name}.png"

def plot_feature_histograms(
    reporter: Reporter,
    df: pd.DataFrame,
    prefix: str,
    max_features: int = 6,
) -> list[str]:
    out_files: list[str] = []
    num_cols = numeric_feature_cols(df)

    for col in num_cols[:max_features]:
        figure_name = f"{prefix}_hist_{col}"

        with reporter.figure(figure_name, figsize=(9, 4.8)) as (fig, ax):
            series = df[col].dropna()
            if len(series) > 0:
                n_unique = series.nunique()

                # Diszkrét / kevés különböző érték esetén levegősebb oszlopdiagram
                if n_unique <= 25:
                    vc = series.value_counts().sort_index()
                    vc.plot(kind="bar", ax=ax, width=0.65)
                    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
                    ax.margins(x=0.02)
                else:
                    ax.hist(series, bins=40, rwidth=0.9)
                    ax.tick_params(axis="x", labelrotation=0, labelsize=9)

                ax.grid(axis="y", linestyle="--", alpha=0.35)

            ax.set_title(f"{prefix} - {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            fig.tight_layout()

        out_files.append(str(reporter.out_dir / f"{figure_name}.png"))

    return out_files

def build_family_report(
    family_name: str,
    family_dir: Path,
    reporter: Reporter,
) -> dict:
    family_report: dict = {
        "family": family_name,
        "created_at": now_iso(),
        "family_dir": str(family_dir),
        "splits": {},
    }

    for split_name in ["train", "val", "test"]:
        csv_path = family_dir / f"{split_name}_{family_name}_features.csv"

        df = safe_read_csv(csv_path)
        if df is None:
            continue

        print(f"[INFO] Reporting {family_name}/{split_name}: {csv_path} | shape={df.shape}")

        missing_csv = save_missing_report(reporter, df, f"{family_name}_{split_name}")
        label_means_csv = save_label_means(reporter, df, f"{family_name}_{split_name}")
        corr_csv = save_correlations(reporter, df, f"{family_name}_{split_name}")
        label_dist_png = plot_label_distribution(reporter, df, f"{family_name}_{split_name}")
        hist_pngs = plot_feature_histograms(reporter, df, f"{family_name}_{split_name}")

        n_records = int(df["record_id"].nunique()) if "record_id" in df.columns else 0

        family_report["splits"][split_name] = {
            "n_rows": int(len(df)),
            "n_records": n_records,
            "n_columns": int(df.shape[1]),
            "feature_file": str(csv_path),
            "missing_report_csv": str(missing_csv),
            "label_means_csv": str(label_means_csv) if label_means_csv else None,
            "correlation_csv": str(corr_csv) if corr_csv else None,
            "label_distribution_png": str(label_dist_png) if label_dist_png else None,
            "histogram_pngs": hist_pngs,
        }

    return family_report


def report_features(dataset: str = "mitbih") -> None:
    dataset = dataset.lower().strip()

    features_general_dir = Path(get_ds_par(dataset, "features_general_dir"))
    features_morphology_dir = Path(get_ds_par(dataset, "features_morphology_dir"))
    report_dir = Path(get_ds_par(dataset, "feature_report_dir"))

    if not features_general_dir.exists():
        raise FileNotFoundError(f"Hiányzó features_general_dir: {features_general_dir}")
    if not features_morphology_dir.exists():
        raise FileNotFoundError(f"Hiányzó features_morphology_dir: {features_morphology_dir}")

    report_dir.mkdir(parents=True, exist_ok=True)

    reporter = Reporter(report_dir)

    print(f"[INFO] dataset={dataset}")
    print(f"[INFO] features_general_dir={features_general_dir}")
    print(f"[INFO] features_morphology_dir={features_morphology_dir}")
    print(f"[INFO] report_dir={report_dir}")

    general_report = build_family_report("general", features_general_dir, reporter=reporter)
    morphology_report = build_family_report("morphology", features_morphology_dir, reporter=reporter)

    combined = {
        "created_at": now_iso(),
        "report_type": f"{dataset}_feature_report",
        "dataset": dataset,
        "general": general_report,
        "morphology": morphology_report,
    }

    out_json = report_dir / "report_input_features.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    rows = []
    for family_key in ["general", "morphology"]:
        fam = combined[family_key]
        for split_name, info in fam["splits"].items():
            rows.append(
                {
                    "dataset": dataset,
                    "family": family_key,
                    "split": split_name,
                    "n_rows": info["n_rows"],
                    "n_records": info["n_records"],
                    "n_columns": info["n_columns"],
                    "feature_file": info["feature_file"],
                }
            )

    summary_df = pd.DataFrame(rows)
    reporter.save_df(summary_df, "feature_report_summary", index=False, print_df=False)

    print(f"[OK] Feature report JSON: {out_json}")
    print(f"[OK] Feature report summary: {report_dir / 'feature_report_summary.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dataset feature reports.")
    parser.add_argument("dataset", type=str, nargs="?", default="mitbih")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report_features(dataset=args.dataset)
    