from pathlib import Path
from typing import Any
import os
import numpy as np

# ---------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------
# Ha nem a Colab-ban vagyunk:
if os.environ.get("COLAB_GPU") is None:
    PROJECT_ROOT = Path("/workspace/ECG")
else:
    # Ha a Colab-ban vagyunk, akkor a PROJECT_ROOT környezeti változót használjuk.
    PROJECT_ROOT = Path(
        os.environ.get("PROJECT_ROOT") or Path(__file__).resolve().parents[1]
    ).resolve()

# ---------------------------------------------------------------------
# Label-space config
# ---------------------------------------------------------------------
# Options:
# - "aami5"
# - "binary_n_vs_rest"
# - "ternary_n_v_other"
LABEL_MODE = "ternary_n_v_other"

# ---------------------------------------------------------------------
# Alap könyvtárak
# ---------------------------------------------------------------------
DATA_DIR    = PROJECT_ROOT / "data"
RAW_DIR     = DATA_DIR / "raw"
OUTPUT_DIR  = PROJECT_ROOT / "outputs" / LABEL_MODE
INTERIM_DIR = DATA_DIR / "interim" / LABEL_MODE
LOGS_DIR    = PROJECT_ROOT / "logs"

for d in [DATA_DIR, RAW_DIR, INTERIM_DIR, OUTPUT_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Global signal / dataset harmonization
# ---------------------------------------------------------------------
DEFAULT_FS = 360
RESAMPLE_TO_DEFAULT_FS = True

INCART_GAIN_NORMALIZATION = {
    "enabled": True,
    "method": "robust_mad",   # "robust_mad" | "median_abs"
    "target_scale": 1.0,
    "eps": 1e-6,
}

LABEL_MODE_CONFIGS: dict[str, dict[str, Any]] = {
    "aami5": {
        "class_names": ["N", "S", "V", "F", "Q"],
        "map": {
            "N": "N",
            "S": "S",
            "V": "V",
            "F": "F",
            "Q": "Q",
        },
    },
    "binary_n_vs_rest": {
        "class_names": ["N", "ABNORMAL"],
        "map": {
            "N": "N",
            "S": "ABNORMAL",
            "V": "ABNORMAL",
            "F": "ABNORMAL",
            "Q": "ABNORMAL",
        },
    },
    "ternary_n_v_other": {
        "class_names": ["N", "V", "OTHER"],
        "map": {
            "N": "N",
            "V": "V",
            "S": "OTHER",
            "F": "OTHER",
            "Q": "OTHER",
        },
    },
}

# ---------------------------------------------------------------------
# QC / PREPROCESSING / BEAT EXTRACTION
# ---------------------------------------------------------------------


QC_RECORD_MODE = "random"      # "all" | "head" | "random"
QC_MAX_RECORDS = 12
QC_RANDOM_SEED = 42
QC_PLOT_ALL_RECORDS = False
QC_VERBOSE_LOAD = False

# Beat kivágás az R hullám körül (sec)
BEAT_PRE_SEC = 0.25
BEAT_POST_SEC = 0.35

PREPROCESSING_CONFIG = {
    "remove_baseline": True,
    "baseline_cutoff": 0.5,
    "baseline_order": 3,
    "use_bandpass": True,
    "lowcut": 0.5,
    "highcut": 40.0,
    "bandpass_order": 3,
    "use_notch": True,
    "notch_freq": 50.0,
    "notch_q": 30.0,
    "normalize_signal": True,

}

BEAT_QUALITY_CONFIG = {
    "max_padding_fraction": 0.10,
    "min_std": 0.05,
    "max_abs_amplitude": 8.0,
    "max_noise_score": 1.5,
    "min_qrs_activity_score": 0.05,
}

PREFERRED_LEADS = ["MLII", "II"]

# ---------------------------------------------------------------------
# CNN config
# ---------------------------------------------------------------------
BATCH_SIZE = 512
EPOCHS = 20

CNN_AUGMENTATION_CONFIG = {
    "enabled": True,
    "use_scale": True,
    "scale_min": 0.95,
    "scale_max": 1.05,
    "use_gaussian_noise": True,
    "noise_std": 0.01,
    "use_shift": True,
    "shift_max": 3,
    "use_baseline_drift": True,
    "baseline_drift_amplitude_min": 0.02,
    "baseline_drift_amplitude_max": 0.12,
    "baseline_drift_cycles_min": 0.2,
    "baseline_drift_cycles_max": 1.0,
}

# ---------------------------------------------------------------------
# Plot settings
# ---------------------------------------------------------------------

PLOT_DPI = 150
PLOT_SAVE = True
PLOT_SHOW = False
PLOT_INLINE = True

AAMI_COLOR_MAP = {
    "N": "green",
    "S": "purple",
    "V": "red",
    "F": "blue",
    "Q": "black",
}

# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Dataset-specifikus központi config
# ---------------------------------------------------------------------
# MIT-BIH rekord lista
MITBIH_RECORDS = [
    "100","101","102","103","104","105","106","107",
    "108","109","111","112","113","114","115","116",
    "117","118","119","121","122","123","124","200",
    "201","202","203","205","207","208","209","210",
    "212","213","214","215","217","219","220","221",
    "222","223","228","230","231","232","233","234"
]

# INCART rekordok: I01 - I75
INCART_RECORDS = [f"I{i:02d}" for i in range(1, 76)]

DATASET_CONFIG: dict[str, dict[str, object]] = {

    # MAIN DATASET DIRS & CONFIGS
    "mitbih": {
        "display_name": "MIT-BIH Arrhythmia Database",

        "raw_dir": RAW_DIR / "mitbih",
        "interim_dir": INTERIM_DIR,
        "beats_dir": INTERIM_DIR / "mitbih_beats",
        "qc_dir": OUTPUT_DIR / "mitbih_qc",
        "results_dir": OUTPUT_DIR / "mitbih",
        "tsne_dir": OUTPUT_DIR / "mitbih_tsne",
        "featurext_dir": OUTPUT_DIR / "mitbih_featurext",
        "cnn_dir": OUTPUT_DIR / "mitbih_cnn",
        "features_general_dir": INTERIM_DIR / "mitbih_features_general",
        "features_morphology_dir": INTERIM_DIR / "mitbih_features_morphology",
        "feature_report_dir": OUTPUT_DIR / "mitbih_feature_reports",
        "cnn_data_dir": INTERIM_DIR / "mitbih_beats",
        "cnn2_data_dir": INTERIM_DIR / "mitbih_2ch_beats",

        "records": MITBIH_RECORDS,
        "n_records": len(MITBIH_RECORDS),
        "fs": 360,
        "cnn_input_kind": "1ch",
        "default_cnn_variant": "1ch",
        "default_channel_count": 1,
    },
    "incart": {
        "display_name": "INCART 12-lead Arrhythmia Database",

        "raw_dir": RAW_DIR / "incart",
        "interim_dir": INTERIM_DIR,
        "beats_dir": INTERIM_DIR / "incart_beats",
        "beats12_dir": INTERIM_DIR / "incart_12ch_beats",
        "qc_dir": OUTPUT_DIR / "incart_qc",
        "results_dir": OUTPUT_DIR / "incart",
        "tsne_dir": OUTPUT_DIR / "incart_tsne",
        "featurext_dir": OUTPUT_DIR / "incart_featurext",
        "cnn_dir": OUTPUT_DIR / "incart_cnn",
        "cnn12_dir": OUTPUT_DIR / "incart_cnn_12ch",
        "features_general_dir": INTERIM_DIR / "incart_features_general",
        "features_morphology_dir": INTERIM_DIR / "incart_features_morphology",
        "feature_report_dir": OUTPUT_DIR / "incart_feature_reports",
        "cnn_data_dir": INTERIM_DIR / "incart_beats",
        "cnn12_data_dir": INTERIM_DIR / "incart_12ch_beats",

	"records": INCART_RECORDS,
        "n_records": len(INCART_RECORDS),
        "incartdb_zip": "incartdb.zip",
        "fs": 257,
        "cnn_input_kind": "1ch",
        "cnn12_input_kind": "multichannel",
        "default_cnn_variant": "1ch",
        "available_cnn_variants": ["1ch", "12ch"],
        "default_channel_count": 12,
    },

    # CROSS DATASET DIRS
    "cross_test": {
        "beats_dir": INTERIM_DIR / "cross_dataset" / "cross_test",
        "qc_dir": OUTPUT_DIR / "cross_test_qc",
        "results_dir": OUTPUT_DIR / "cross_test",
        "tsne_dir": OUTPUT_DIR / "cross_test_tsne",
        "featurext_dir": OUTPUT_DIR / "cross_test_featurext",
        "cnn_dir": OUTPUT_DIR / "cross_test_cnn",
        "features_general_dir": INTERIM_DIR / "cross_test_features_general",
        "features_morphology_dir": INTERIM_DIR / "cross_test_features_morphology",
        "feature_report_dir": OUTPUT_DIR / "cross_test_feature_reports",
        "cnn_data_dir": INTERIM_DIR / "cross_dataset" / "cross_test",
    },
    "mixed": {
        "beats_dir": INTERIM_DIR / "cross_dataset" / "mixed",
        "qc_dir": OUTPUT_DIR / "mixed_qc",
        "results_dir": OUTPUT_DIR / "mixed",
        "tsne_dir": OUTPUT_DIR / "mixed_tsne",
        "featurext_dir": OUTPUT_DIR / "mixed_featurext",
        "cnn_dir": OUTPUT_DIR / "mixed_cnn",
        "features_general_dir": INTERIM_DIR / "mixed_features_general",
        "features_morphology_dir": INTERIM_DIR / "mixed_features_morphology",
        "feature_report_dir": OUTPUT_DIR / "mixed_feature_reports",
        "cnn_data_dir": INTERIM_DIR / "cross_dataset" / "mixed",
    },
    "domain_generalization": {
        "beats_dir": INTERIM_DIR / "cross_dataset" / "domain_generalization",
        "qc_dir": OUTPUT_DIR / "domain_generalization_qc",
        "results_dir": OUTPUT_DIR / "domain_generalization",
        "tsne_dir": OUTPUT_DIR / "domain_generalization_tsne",
        "featurext_dir": OUTPUT_DIR / "domain_generalization_featurext",
        "cnn_dir": OUTPUT_DIR / "domain_generalization_cnn",
        "features_general_dir": INTERIM_DIR / "domain_generalization_features_general",
        "features_morphology_dir": INTERIM_DIR / "domain_generalization_features_morphology",
        "feature_report_dir": OUTPUT_DIR / "domain_generalization_feature_reports",
        "cnn_data_dir": INTERIM_DIR / "cross_dataset" / "domain_generalization",
    },
    "cross_dataset": {
        "compare_dir": OUTPUT_DIR / "cross_ds_compare",
        "tsne_dir": OUTPUT_DIR / "cross_ds_tsne",
        "metrics_dir": OUTPUT_DIR / "cross_ds_metrics",
    },
}

# ---------------------------------------------------------------------
# Dataset config getters
# ---------------------------------------------------------------------

def normalize_dataset_name(dataset_name: str) -> str:
    name = dataset_name.strip().lower()
    if name not in DATASET_CONFIG:
        allowed = ", ".join(sorted(DATASET_CONFIG))
        raise ValueError(
            f"Ismeretlen dataset: {dataset_name!r}. Megengedett értékek: {allowed}"
        )
    return name

def get_ds_par(dataset_name: str, param_name: str) -> Any:
    ds = normalize_dataset_name(dataset_name)

    if param_name not in DATASET_CONFIG[ds]:
        allowed = ", ".join(sorted(DATASET_CONFIG[ds].keys()))
        raise KeyError(
            f"A(z) {param_name!r} paraméter nincs definiálva a(z) {ds!r} datasethez. "
            f"Elérhető kulcsok: {allowed}"
        )

    return DATASET_CONFIG[ds][param_name]


def get_label_mode() -> str:
    mode = str(LABEL_MODE).strip().lower()
    if mode not in LABEL_MODE_CONFIGS:
        allowed = ", ".join(sorted(LABEL_MODE_CONFIGS))
        raise ValueError(f"Ismeretlen LABEL_MODE: {LABEL_MODE!r}. Megengedett: {allowed}")
    return mode


def get_label_mode_config() -> dict[str, Any]:
    return LABEL_MODE_CONFIGS[get_label_mode()]


def get_label_mode_classes() -> list[str]:
    return list(get_label_mode_config()["class_names"])


def get_label_mapping() -> dict[str, int]:
    classes = get_label_mode_classes()
    return {name: i for i, name in enumerate(classes)}


def remap_aami_labels(aami_labels: list[str] | np.ndarray) -> np.ndarray:
    mapping = get_label_mode_config()["map"]
    arr = np.asarray(aami_labels, dtype=object)
    return np.asarray([mapping[str(x)] for x in arr], dtype=object)


def encode_target_labels(target_labels: list[str] | np.ndarray) -> np.ndarray:
    label_mapping = get_label_mapping()
    arr = np.asarray(target_labels, dtype=object)
    return np.asarray([label_mapping[str(x)] for x in arr], dtype=np.int64)

def get_label_mode_tag() -> str:
    mode = get_label_mode()
    tags = {
        "aami5": "label_aami5",
        "binary_n_vs_rest": "label_n_other",
        "ternary_n_v_other": "label_n_v_other",
    }
    return tags[mode]

