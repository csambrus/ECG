#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize


def read_predictions_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Hiányzó prediction CSV: {path}")

    df = pd.read_csv(path)
    required = {"y_true", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Hiányzó kötelező oszlop(ok) a prediction CSV-ben: {sorted(missing)} | {path}")

    return df


def detect_prob_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("prob_")]


def detect_class_names(df: pd.DataFrame) -> list[str]:
    prob_cols = detect_prob_columns(df)
    if prob_cols:
        return [c.removeprefix("prob_") for c in prob_cols]

    n_classes = int(max(df["y_true"].max(), df["y_pred"].max()) + 1)
    return [str(i) for i in range(n_classes)]


def save_roc_plot(
    df: pd.DataFrame,
    out_path: str | Path,
    title: str | None = None,
) -> dict[str, float]:
    prob_cols = detect_prob_columns(df)
    if not prob_cols:
        raise ValueError("ROC görbéhez szükségesek a prob_* oszlopok.")

    class_names = [c.removeprefix("prob_") for c in prob_cols]
    y_true = df["y_true"].to_numpy(dtype=np.int64)
    y_proba = df[prob_cols].to_numpy(dtype=float)

    n_classes = len(prob_cols)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    auc_by_class: dict[str, float] = {}

    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = float(auc(fpr, tpr))
        auc_by_class[class_name] = roc_auc
        plt.plot(fpr, tpr, label=f"{class_name} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title or "ROC curves")
    plt.legend()
    plt.grid(True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    return auc_by_class


def ensure_same_order_or_key(
    left: pd.DataFrame,
    right: pd.DataFrame,
    join_key: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if join_key is not None:
        if join_key not in left.columns or join_key not in right.columns:
            raise ValueError(f"A join kulcs hiányzik az egyik prediction CSV-ből: {join_key}")
        left2 = left.sort_values(join_key).reset_index(drop=True)
        right2 = right.sort_values(join_key).reset_index(drop=True)
        if not left2[join_key].equals(right2[join_key]):
            raise ValueError(f"A join kulcs szerinti sorok nem illeszkednek: {join_key}")
        return left2, right2

    if len(left) != len(right):
        raise ValueError("A két prediction CSV hossza eltér, join_key nélkül nem hasonlítható össze.")

    return left.reset_index(drop=True), right.reset_index(drop=True)


def confidence_series(df: pd.DataFrame) -> pd.Series:
    prob_cols = detect_prob_columns(df)
    if prob_cols:
        return df[prob_cols].max(axis=1)

    return pd.Series(np.ones(len(df), dtype=float))


def save_bland_altman_plot(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    out_path: str | Path,
    left_name: str,
    right_name: str,
    join_key: str | None = None,
) -> dict[str, float]:
    left_df, right_df = ensure_same_order_or_key(left_df, right_df, join_key=join_key)

    left_conf = confidence_series(left_df).to_numpy(dtype=float)
    right_conf = confidence_series(right_df).to_numpy(dtype=float)

    mean_vals = (left_conf + right_conf) / 2.0
    diff_vals = left_conf - right_conf

    mean_diff = float(np.mean(diff_vals))
    std_diff = float(np.std(diff_vals, ddof=1)) if len(diff_vals) > 1 else 0.0
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    plt.figure(figsize=(6, 6))
    plt.scatter(mean_vals, diff_vals, alpha=0.35)
    plt.axhline(mean_diff, linestyle="--", label=f"mean={mean_diff:.4f}")
    plt.axhline(loa_upper, linestyle="--", label=f"+1.96 SD={loa_upper:.4f}")
    plt.axhline(loa_lower, linestyle="--", label=f"-1.96 SD={loa_lower:.4f}")
    plt.xlabel("Mean confidence")
    plt.ylabel(f"Difference ({left_name} - {right_name})")
    plt.title(f"Bland–Altman\n{left_name} vs {right_name}")
    plt.legend()
    plt.grid(True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "loa_upper": loa_upper,
        "loa_lower": loa_lower,
        "n": int(len(diff_vals)),
    }


def summarize_prediction_csv(df: pd.DataFrame, source_name: str) -> dict:
    class_names = detect_class_names(df)
    return {
        "source_name": source_name,
        "n_rows": int(len(df)),
        "class_names": class_names,
        "has_probabilities": bool(detect_prob_columns(df)),
        "y_true_distribution": df["y_true"].value_counts().sort_index().to_dict(),
        "y_pred_distribution": df["y_pred"].value_counts().sort_index().to_dict(),
    }


def derive_output_dir(
    predictions_csv: str | Path,
    out_dir: str | Path | None,
) -> Path:
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    pred_path = Path(predictions_csv)
    derived = pred_path.parent / "evaluation"
    derived.mkdir(parents=True, exist_ok=True)
    return derived


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dataset-agnosztikus értékelő script prediction CSV-khez."
    )
    parser.add_argument(
        "--predictions-csv",
        required=True,
        help="Elsődleges prediction CSV (ROC és summary ehhez készül).",
    )
    parser.add_argument(
        "--compare-csv",
        default=None,
        help="Másodlagos prediction CSV Bland–Altman összehasonlításhoz.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Kimeneti mappa. Ha nincs megadva, a prediction CSV mellé evaluation/ mappa jön létre.",
    )
    parser.add_argument(
        "--title-prefix",
        default="",
        help="Opcionális cím prefix az ábrákhoz.",
    )
    parser.add_argument(
        "--left-name",
        default="left_model",
        help="Bland–Altman bal oldali modell neve.",
    )
    parser.add_argument(
        "--right-name",
        default="right_model",
        help="Bland–Altman jobb oldali modell neve.",
    )
    parser.add_argument(
        "--join-key",
        default=None,
        help="Opcionális közös kulcs a két prediction CSV illesztéséhez (pl. record_beat_index).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    primary_csv = Path(args.predictions_csv)
    out_dir = derive_output_dir(primary_csv, args.out_dir)

    primary_df = read_predictions_csv(primary_csv)
    summary = {
        "primary": summarize_prediction_csv(primary_df, source_name=primary_csv.name),
        "outputs": {},
    }

    roc_path = out_dir / "roc_curves.png"
    roc_json_path = out_dir / "roc_auc_summary.json"

    try:
        roc_title = f"{args.title_prefix} ROC".strip()
        auc_by_class = save_roc_plot(primary_df, roc_path, title=roc_title or None)
        roc_json_path.write_text(
            json.dumps(auc_by_class, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary["outputs"]["roc_plot"] = str(roc_path)
        summary["outputs"]["roc_auc_summary"] = str(roc_json_path)
    except ValueError as exc:
        summary["outputs"]["roc_plot"] = None
        summary["outputs"]["roc_auc_summary"] = None
        summary["roc_warning"] = str(exc)

    if args.compare_csv:
        compare_csv = Path(args.compare_csv)
        compare_df = read_predictions_csv(compare_csv)

        ba_path = out_dir / "bland_altman.png"
        ba_json_path = out_dir / "bland_altman_summary.json"

        ba_summary = save_bland_altman_plot(
            primary_df,
            compare_df,
            out_path=ba_path,
            left_name=args.left_name,
            right_name=args.right_name,
            join_key=args.join_key,
        )
        ba_json_path.write_text(
            json.dumps(ba_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        summary["compare"] = summarize_prediction_csv(compare_df, source_name=compare_csv.name)
        summary["outputs"]["bland_altman_plot"] = str(ba_path)
        summary["outputs"]["bland_altman_summary"] = str(ba_json_path)

    summary_path = out_dir / "evaluation_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] Evaluation summary: {summary_path}")
    if summary["outputs"].get("roc_plot"):
        print(f"[OK] ROC plot: {summary['outputs']['roc_plot']}")
    if summary["outputs"].get("bland_altman_plot"):
        print(f"[OK] Bland–Altman plot: {summary['outputs']['bland_altman_plot']}")


if __name__ == "__main__":
    main()
