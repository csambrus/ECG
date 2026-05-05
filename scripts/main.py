#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

from src.config import OUTPUT_DIR

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / ".pipeline_state"
STATE_DIR.mkdir(exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------
# Pipeline step típusok
# ---------------------------------------------------------------------

#PipelineArgs = list[Any] | dict[str, Any] | None
#PipelineStep = str | tuple[str, PipelineArgs]

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

from src.download_mitbih_paralel import download_mitbih_paralel
from src.qc_dataset_ext import qc_dataset_ext
from src.plot_preprocessing_comparison import plot_preprocessing_comparison
from src.prepare_mitbih_datasets import prepare_mitbih_datasets
from src.build_general_features import build_general_features
from src.build_morphology_features import build_morphology_features
from src.report_features import report_features
from src.train_featurext import train_featurext
from src.plot_dataset_tsne import plot_dataset_tsne
from src.plot_tsne_featurext_vs_cnn import plot_tsne_featurext_vs_cnn
from src.train_cnn import train_cnn

from src.download_incart_paralel import download_incart_paralel
from src.prepare_incart_datasets import prepare_incart_datasets
from src.prepare_incart_datasets_12ch import prepare_incart_datasets_12ch

from src.prepare_cross_dataset_splits import prepare_cross_dataset_splits
from src.plot_cross_dataset_tsne import plot_cross_dataset_tsne
from src.compare_models_extended import compare_models_extended


PipelineArgs = list[Any] | dict[str, Any] | None
PipelineStep = Callable[..., Any] | tuple[Callable[..., Any], PipelineArgs]

# ---------------------------------------------------------------------
# PIPELINE BLOKKOK
# ---------------------------------------------------------------------

MITBIH_PIPELINE: list[PipelineStep] = [
    download_mitbih_paralel,
    (qc_dataset_ext, {"dataset": "mitbih"}),
    (plot_preprocessing_comparison, {"dataset": "mitbih", "record_name": "100"}),
    prepare_mitbih_datasets,
    (build_general_features, {"dataset": "mitbih"}),
    (build_morphology_features, {"dataset": "mitbih"}),
    (report_features, {"dataset": "mitbih"}),
    (train_featurext, {"dataset": "mitbih"}),
    (plot_dataset_tsne, {"dataset": "mitbih"}),
    (plot_tsne_featurext_vs_cnn, {"dataset": "mitbih"}),
    (train_cnn, {"dataset": "mitbih"}),
]

INCART_PIPELINE: list[PipelineStep] = [
    download_incart_paralel,
    (qc_dataset_ext, {"dataset": "incart"}),
    (plot_preprocessing_comparison, {"dataset": "incart", "record_name": "I01"}),
    prepare_incart_datasets,
    (build_general_features, {"dataset": "incart"}),
    (build_morphology_features, {"dataset": "incart"}),
    (report_features, {"dataset": "incart"}),
    (train_featurext, {"dataset": "incart"}),
    (plot_dataset_tsne, {"dataset": "incart"}),
    (plot_tsne_featurext_vs_cnn, {"dataset": "incart"}),
    (train_cnn, {"dataset": "incart"}),
    prepare_incart_datasets_12ch,
    (train_cnn, {"dataset": "incart", "variant": "12ch"}),
]

# ---------------------------------------------------------------------
# CROSS DATASET PREP
# ---------------------------------------------------------------------

CROSS_DATASET_PREP_PIPELINE: list[PipelineStep] = [
    prepare_cross_dataset_splits,
]

# ---------------------------------------------------------------------
# CROSS TEST
# ---------------------------------------------------------------------

CROSS_TEST_PIPELINE: list[PipelineStep] = CROSS_DATASET_PREP_PIPELINE + [
    (
        build_general_features,
        {
            "dataset": "cross_test",
            "data_dir": str(OUTPUT_DIR / "cross_dataset" / "cross_test"),
            "out_dir": str(OUTPUT_DIR / "cross_test_features_general"),
        },
    ),
    (
        build_morphology_features,
        {
            "dataset": "cross_test",
            "beats_dir": str(OUTPUT_DIR / "cross_dataset" / "cross_test"),
            "out_dir": str(OUTPUT_DIR / "cross_test_features_morphology"),
        },
    ),
    (
        report_features,
        {
            "dataset": "cross_test",
            "general_dir": str(OUTPUT_DIR / "cross_test_features_general"),
            "morphology_dir": str(OUTPUT_DIR / "cross_test_features_morphology"),
            "report_dir": str(OUTPUT_DIR / "cross_test_feature_reports"),
        },
    ),
    (
        train_featurext,
        {
            "dataset": "cross_test",
            "general_features_dir": str(OUTPUT_DIR / "cross_test_features_general"),
            "morph_features_dir": str(OUTPUT_DIR / "cross_test_features_morphology"),
            "results_dir": str(OUTPUT_DIR / "cross_test_featurext"),
        },
    ),
    (
        plot_tsne_featurext_vs_cnn,
        {
            "dataset": "cross_test",
            "data_dir": str(OUTPUT_DIR / "cross_dataset" / "cross_test"),
            "general_features_dir": str(OUTPUT_DIR / "cross_test_features_general"),
            "results_dir": str(OUTPUT_DIR / "cross_test_tsne_compare"),
        },
    ),
    (
        train_cnn,
        {
            "dataset": "cross_test",
            "data_dir": str(OUTPUT_DIR / "cross_dataset" / "cross_test"),
            "results_dir": str(OUTPUT_DIR / "cross_test_cnn"),
        },
    ),
]

# ---------------------------------------------------------------------
# MIXED
# ---------------------------------------------------------------------

MIXED_PIPELINE: list[PipelineStep] = CROSS_DATASET_PREP_PIPELINE + [
    (
        build_general_features,
        {
            "dataset": "mixed",
            "data_dir": str(OUTPUT_DIR / "cross_dataset" / "mixed"),
            "out_dir": str(OUTPUT_DIR / "mixed_features_general"),
        },
    ),
    (
        build_morphology_features,
        {
            "dataset": "mixed",
            "beats_dir": str(OUTPUT_DIR / "cross_dataset" / "mixed"),
            "out_dir": str(OUTPUT_DIR / "mixed_features_morphology"),
        },
    ),
    (
        report_features,
        {
            "dataset": "mixed",
            "general_dir": str(OUTPUT_DIR / "mixed_features_general"),
            "morphology_dir": str(OUTPUT_DIR / "mixed_features_morphology"),
            "report_dir": str(OUTPUT_DIR / "mixed_feature_reports"),
        },
    ),
    (
        train_featurext,
        {
            "dataset": "mixed",
            "general_features_dir": str(OUTPUT_DIR / "mixed_features_general"),
            "morph_features_dir": str(OUTPUT_DIR / "mixed_features_morphology"),
            "results_dir": str(OUTPUT_DIR / "mixed_featurext"),
        },
    ),
    (
        plot_tsne_featurext_vs_cnn,
        {
            "dataset": "mixed",
            "data_dir": str(OUTPUT_DIR / "cross_dataset" / "mixed"),
            "general_features_dir": str(OUTPUT_DIR / "mixed_features_general"),
            "results_dir": str(OUTPUT_DIR / "mixed_tsne_compare"),
        },
    ),
    (
        train_cnn,
        {
            "dataset": "mixed",
            "data_dir": str(OUTPUT_DIR / "cross_dataset" / "mixed"),
            "results_dir": str(OUTPUT_DIR / "mixed_cnn"),
        },
    ),
]

# ---------------------------------------------------------------------
# DOMAIN GENERALIZATION
# ---------------------------------------------------------------------

DOMAIN_GENERALIZATION_PIPELINE: list[PipelineStep] = CROSS_DATASET_PREP_PIPELINE + [
    (
        build_general_features,
        {
            "dataset": "domain_generalization",
            "data_dir": str(OUTPUT_DIR / "cross_dataset" / "domain_generalization"),
            "out_dir": str(OUTPUT_DIR / "domain_generalization_features_general"),
        },
    ),
    (
        build_morphology_features,
        {
            "dataset": "domain_generalization",
            "beats_dir": str(OUTPUT_DIR / "cross_dataset" / "domain_generalization"),
            "out_dir": str(OUTPUT_DIR / "domain_generalization_features_morphology"),
        },
    ),
    (
        report_features,
        {
            "dataset": "domain_generalization",
            "general_dir": str(OUTPUT_DIR / "domain_generalization_features_general"),
            "morphology_dir": str(OUTPUT_DIR / "domain_generalization_features_morphology"),
            "report_dir": str(OUTPUT_DIR / "domain_generalization_feature_reports"),
        },
    ),
    (
        train_featurext,
        {
            "dataset": "domain_generalization",
            "general_features_dir": str(OUTPUT_DIR / "domain_generalization_features_general"),
            "morph_features_dir": str(OUTPUT_DIR / "domain_generalization_features_morphology"),
            "results_dir": str(OUTPUT_DIR / "domain_generalization_featurext"),
        },
    ),
    (
        plot_tsne_featurext_vs_cnn,
        {
            "dataset": "domain_generalization",
            "data_dir": str(OUTPUT_DIR / "cross_dataset" / "domain_generalization"),
            "general_features_dir": str(OUTPUT_DIR / "domain_generalization_features_general"),
            "results_dir": str(OUTPUT_DIR / "domain_generalization_tsne_compare"),
        },
    ),
    (
        train_cnn,
        {
            "dataset": "domain_generalization",
            "data_dir": str(OUTPUT_DIR / "cross_dataset" / "domain_generalization"),
            "results_dir": str(OUTPUT_DIR / "domain_generalization_cnn"),
        },
    ),
]

# ---------------------------------------------------------------------
# EGYEDI BLOKKOK
# ---------------------------------------------------------------------

FEATURES_GENERAL_PIPELINE: list[PipelineStep] = [
    (build_general_features, {"dataset": "mitbih"}),
]

FEATURES_MORPHOLOGY_PIPELINE: list[PipelineStep] = [
    (build_morphology_features, {"dataset": "mitbih"}),
]

INCART_FEATURES_GENERAL_PIPELINE: list[PipelineStep] = [
    (build_general_features, {"dataset": "incart"}),
]

INCART_FEATURES_MORPHOLOGY_PIPELINE: list[PipelineStep] = [
    (build_morphology_features, {"dataset": "incart"}),
]

REPORT_FEATURES_PIPELINE: list[PipelineStep] = [
    (report_features, {"dataset": "mitbih"}),
]

REPORT_INCART_FEATURES_PIPELINE: list[PipelineStep] = [
    (report_features, {"dataset": "incart"}),
]

TRAIN_FEATUREXT_PIPELINE: list[PipelineStep] = [
    (train_featurext, {"dataset": "mitbih"}),
]

TRAIN_INCART_FEATUREXT_PIPELINE: list[PipelineStep] = [
    (train_featurext, {"dataset": "incart"}),
]

TRAIN_CNN_PIPELINE: list[PipelineStep] = [
    (train_cnn, {"dataset": "mitbih"}),
]

TRAIN_INCART_CNN_PIPELINE: list[PipelineStep] = [
    (train_cnn, {"dataset": "incart"}),
]

TRAIN_INCART_CNN_12CH_PIPELINE: list[PipelineStep] = [
    (train_cnn, {"dataset": "incart", "variant": "12ch"}),
]

PLOT_PREPROCESSING_MITBIH_PIPELINE: list[PipelineStep] = [
    (plot_preprocessing_comparison, {"dataset": "mitbih", "record_name": "100"}),
]

PLOT_PREPROCESSING_INCART_PIPELINE: list[PipelineStep] = [
    (plot_preprocessing_comparison, {"dataset": "incart", "record_name": "I01"}),
]

PLOT_TSNE_DATASET_MITBIH_PIPELINE: list[PipelineStep] = [
    (plot_dataset_tsne, {"dataset": "mitbih"}),
]

PLOT_TSNE_DATASET_INCART_PIPELINE: list[PipelineStep] = [
    (plot_dataset_tsne, {"dataset": "incart"}),
]

PLOT_TSNE_FEATUREXT_VS_CNN_MITBIH_PIPELINE: list[PipelineStep] = [
    (plot_tsne_featurext_vs_cnn, {"dataset": "mitbih"}),
]

PLOT_TSNE_FEATUREXT_VS_CNN_INCART_PIPELINE: list[PipelineStep] = [
    (plot_tsne_featurext_vs_cnn, {"dataset": "incart"}),
]

PLOT_CROSS_DATASET_TSNE_PIPELINE: list[PipelineStep] = [
    (plot_cross_dataset_tsne, {"dataset_left": "mitbih", "dataset_right": "incart", "split": "test"}),
]

# ---------------------------------------------------------------------
# REPORT BLOKKOK
# ---------------------------------------------------------------------

REPORT_CROSS_TEST_FEATURES_PIPELINE: list[PipelineStep] = [
    (
        report_features,
        {
            "dataset": "cross_test",
            "general_dir": str(OUTPUT_DIR / "cross_test_features_general"),
            "morphology_dir": str(OUTPUT_DIR / "cross_test_features_morphology"),
            "report_dir": str(OUTPUT_DIR / "cross_test_feature_reports"),
        },
    ),
]

REPORT_DOMAIN_GENERALIZATION_FEATURES_PIPELINE: list[PipelineStep] = [
    (
        report_features,
        {
            "dataset": "domain_generalization",
            "general_dir": str(OUTPUT_DIR / "domain_generalization_features_general"),
            "morphology_dir": str(OUTPUT_DIR / "domain_generalization_features_morphology"),
            "report_dir": str(OUTPUT_DIR / "domain_generalization_feature_reports"),
        },
    ),
]

REPORT_MIXED_FEATURES_PIPELINE: list[PipelineStep] = [
    (
        report_features,
        {
            "dataset": "mixed",
            "general_dir": str(OUTPUT_DIR / "mixed_features_general"),
            "morphology_dir": str(OUTPUT_DIR / "mixed_features_morphology"),
            "report_dir": str(OUTPUT_DIR / "mixed_feature_reports"),
        },
    ),
]

# ---------------------------------------------------------------------
# COMPARE
# ---------------------------------------------------------------------

COMPARE_MITBIH_FEATUREXT_VS_CNN_PIPELINE: list[PipelineStep] = [
    (
        compare_models_extended,
        {
            "dataset": "mitbih",
            "left_family": "featurext",
            "right_family": "cnn",
            "results_subdir": "mitbih_featurext_vs_cnn_extended",
            "select_best_left": True,
        },
    ),
]

COMPARE_INCART_FEATUREXT_VS_CNN_PIPELINE: list[PipelineStep] = [
    (
        compare_models_extended,
        {
            "dataset": "incart",
            "left_family": "featurext",
            "right_family": "cnn",
            "results_subdir": "incart_featurext_vs_cnn_extended",
            "select_best_left": True,
        },
    ),
]

COMPARE_INCART_CNN_VS_CNN12_PIPELINE: list[PipelineStep] = [
    (
        compare_models_extended,
        {
            "dataset": "incart",
            "left_family": "cnn",
            "right_family": "cnn12",
            "results_subdir": "incart_cnn_vs_cnn12ch_extended",
        },
    ),
]

COMPARE_ALL_PIPELINE: list[PipelineStep] = (
    COMPARE_MITBIH_FEATUREXT_VS_CNN_PIPELINE
    + COMPARE_INCART_FEATUREXT_VS_CNN_PIPELINE
    + COMPARE_INCART_CNN_VS_CNN12_PIPELINE
)

# ---------------------------------------------------------------------
# PIPELINES
# ---------------------------------------------------------------------

PIPELINES: dict[str, list[PipelineStep]] = {
    "mitbih": MITBIH_PIPELINE,
    "incart": INCART_PIPELINE,

    "cross_dataset_prepare": CROSS_DATASET_PREP_PIPELINE,
    "cross_test": CROSS_TEST_PIPELINE,
    "mixed": MIXED_PIPELINE,
    "domain_generalization": DOMAIN_GENERALIZATION_PIPELINE,

    "feature_general": FEATURES_GENERAL_PIPELINE,
    "feature_morphology": FEATURES_MORPHOLOGY_PIPELINE,
    "incart_feature_general": INCART_FEATURES_GENERAL_PIPELINE,
    "incart_feature_morphology": INCART_FEATURES_MORPHOLOGY_PIPELINE,

    "report_features": REPORT_FEATURES_PIPELINE,
    "report_incart_features": REPORT_INCART_FEATURES_PIPELINE,
    "report_cross_test_features": REPORT_CROSS_TEST_FEATURES_PIPELINE,
    "report_domain_generalization_features": REPORT_DOMAIN_GENERALIZATION_FEATURES_PIPELINE,
    "report_mixed_features": REPORT_MIXED_FEATURES_PIPELINE,

    "train_featurext": TRAIN_FEATUREXT_PIPELINE,
    "train_incart_featurext": TRAIN_INCART_FEATUREXT_PIPELINE,
    "train_cnn": TRAIN_CNN_PIPELINE,
    "train_incart_cnn": TRAIN_INCART_CNN_PIPELINE,
    "train_incart_cnn_12ch": TRAIN_INCART_CNN_12CH_PIPELINE,

    "plot_preprocessing_mitbih": PLOT_PREPROCESSING_MITBIH_PIPELINE,
    "plot_preprocessing_incart": PLOT_PREPROCESSING_INCART_PIPELINE,
    "plot_dataset_tsne_mitbih": PLOT_TSNE_DATASET_MITBIH_PIPELINE,
    "plot_dataset_tsne_incart": PLOT_TSNE_DATASET_INCART_PIPELINE,
    "plot_tsne_featurext_vs_cnn_mitbih": PLOT_TSNE_FEATUREXT_VS_CNN_MITBIH_PIPELINE,
    "plot_tsne_featurext_vs_cnn_incart": PLOT_TSNE_FEATUREXT_VS_CNN_INCART_PIPELINE,
    "plot_cross_dataset_tsne": PLOT_CROSS_DATASET_TSNE_PIPELINE,

    "compare_mitbih_featurext_vs_cnn": COMPARE_MITBIH_FEATUREXT_VS_CNN_PIPELINE,
    "compare_incart_featurext_vs_cnn": COMPARE_INCART_FEATUREXT_VS_CNN_PIPELINE,
    "compare_incart_cnn_vs_cnn12": COMPARE_INCART_CNN_VS_CNN12_PIPELINE,
    "compare_all": COMPARE_ALL_PIPELINE,

    "all": (
        MITBIH_PIPELINE
        + INCART_PIPELINE
        + CROSS_TEST_PIPELINE
        + MIXED_PIPELINE[1:]
        + DOMAIN_GENERALIZATION_PIPELINE[1:]
        + PLOT_CROSS_DATASET_TSNE_PIPELINE
        + COMPARE_ALL_PIPELINE
    ),
}

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# STEP HELPERS
# ---------------------------------------------------------------------

def normalize_step(step: PipelineStep) -> tuple[Callable[..., Any], PipelineArgs]:
    if callable(step):
        return step, None

    func, args = step
    return func, args

def step_display_name(step: PipelineStep) -> str:
    func, args = normalize_step(step)
    func_name = func.__name__

    if args is None:
        return func_name

    if isinstance(args, dict):
        if not args:
            return func_name
        parts = [f"{k}={v!r}" for k, v in args.items()]
        return f"{func_name}({', '.join(parts)})"

    if isinstance(args, list):
        if not args:
            return func_name
        return f"{func_name}({', '.join(repr(x) for x in args)})"

    return func_name


def pipeline_signature(steps: list[PipelineStep]) -> str:
    serialized_steps = [step_display_name(step) for step in steps]
    txt = "\n".join(serialized_steps)
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------
# STATE HELPERS
# ---------------------------------------------------------------------

def state_file_for(pipeline_name: str) -> Path:
    return STATE_DIR / f"{pipeline_name}.json"


def default_state(pipeline_name: str, steps: list[PipelineStep]) -> dict[str, Any]:
    return {
        "pipeline_name": pipeline_name,
        "signature": pipeline_signature(steps),
        "completed_prefix": -1,
    }


def load_state(pipeline_name: str, steps: list[PipelineStep]) -> dict[str, Any]:
    path = state_file_for(pipeline_name)
    if not path.exists():
        return default_state(pipeline_name, steps)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default_state(pipeline_name, steps)

    expected_signature = pipeline_signature(steps)
#    if data.get("signature") != expected_signature:
#        return default_state(pipeline_name, steps)

    completed_prefix = int(data.get("completed_prefix", -1))
    completed_prefix = max(-1, min(completed_prefix, len(steps) - 1))

    return {
        "pipeline_name": pipeline_name,
        "signature": expected_signature,
        "completed_prefix": completed_prefix,
    }


def save_state(pipeline_name: str, steps: list[PipelineStep], completed_prefix: int) -> None:
    path = state_file_for(pipeline_name)
    data = {
        "pipeline_name": pipeline_name,
        "signature": pipeline_signature(steps),
        "completed_prefix": completed_prefix,
        "steps": [step_display_name(step) for step in steps],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def reset_state(pipeline_name: str) -> None:
    path = state_file_for(pipeline_name)
    if path.exists():
        path.unlink()

# ---------------------------------------------------------------------
# RUNNER
# ---------------------------------------------------------------------

def run_step(step: PipelineStep) -> int:
    func, args = normalize_step(step)
    printable = step_display_name(step)

    print(f"\n{'=' * 90}")
    print(f"Futtatás: {printable}")
    print(f"{'=' * 90}\n")

    try:
        if args is None:
            func()
        elif isinstance(args, dict):
            func(**args)
        elif isinstance(args, list):
            func(*args)
        else:
            raise TypeError(f"Nem támogatott step argumentum típus: {type(args)}")

        gc.collect()
        return 0

    except Exception as exc:
        print(f"[ERROR] {printable}: {exc}")
        return 1



def run_pipeline(pipeline_name: str, force_restart: bool = False) -> int:
    if pipeline_name not in PIPELINES:
        print(f"Ismeretlen pipeline: {pipeline_name}")
        print("Elérhető pipeline-ok:")
        for name in PIPELINES:
            print(f"  - {name}")
        return 1

    steps = PIPELINES[pipeline_name]

    if force_restart:
        reset_state(pipeline_name)

    state = load_state(pipeline_name, steps)
    completed_prefix = state["completed_prefix"]

    if completed_prefix >= 0:
        print(f"Korábban sikeres prefix vége: {completed_prefix}")
        print("Átugrandó lépések:")
        for i in range(completed_prefix + 1):
            print(f"  [SKIP STEP: {i+1}/{len(steps)}] {step_display_name(steps[i])}")
    else:
        print("Nincs korábbi sikeres checkpoint.")

    for i, step in enumerate(steps):
        label = step_display_name(step)

        if i <= completed_prefix:
            #print(f"[SKIP STEP: {i+1}/{len(steps)}] {label}")
            continue

        rc = run_step(step)

        if rc == 0:
            completed_prefix = i
            save_state(pipeline_name, steps, completed_prefix)
            print(f"[OK STEP: {i+1}/{len(steps)}] {label}")
        else:
            save_state(pipeline_name, steps, completed_prefix)
            print(f"[FAIL {i+1}/{len(steps)}] {label} (exit code: {rc})")
            print("A pipeline megállt. Következő futáskor ettől a ponttól megy tovább.")
            return rc

    print("\nPipeline sikeresen lefutott.")
    return 0

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline runner checkpoint támogatással")
    parser.add_argument(
        "pipeline",
        choices=PIPELINES.keys(),
        help="Melyik pipeline fusson",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Checkpoint törlése és teljes újrafuttatás",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rc = run_pipeline(args.pipeline, force_restart=args.restart)
    sys.exit(rc)


if __name__ == "__main__":
    main()