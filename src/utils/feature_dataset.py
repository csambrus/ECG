from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.features_hrv import (
    DEFAULT_RPEAK_CONFIG,
    RPeakDetectionConfig,
    extract_hrv_features,
)


# ---------------------------------------------------------------------
# Segédfüggvény: kiválaszt egy 1D lead-et egy rekordból
# ---------------------------------------------------------------------
def select_lead_from_signal(
    signal: np.ndarray,
    lead: int | None = None,
) -> np.ndarray:
    """
    1D vagy 2D ECG jelből kiválaszt egyetlen 1D lead-et.

    Parameters
    ----------
    signal : np.ndarray
        A bemeneti jel.
        Elfogadott formák:
        - 1D: (n_samples,)
        - 2D: (n_samples, n_leads)
    lead : int | None
        Ha a jel 2D, akkor melyik lead indexet válasszuk.
        Ha None és 2D a jel, akkor az első leadet választjuk.

    Returns
    -------
    np.ndarray
        A kiválasztott 1D jel.

    Raises
    ------
    ValueError
        Ha a signal dimenziója nem 1 vagy 2.
    IndexError
        Ha a lead index kívül esik a tartományon.
    """
    signal = np.asarray(signal, dtype=float)

    # Ha már 1D, akkor nincs mit kiválasztani.
    if signal.ndim == 1:
        return signal

    # Ha 2D, akkor egy konkrét csatornát választunk.
    if signal.ndim == 2:
        if lead is None:
            lead = 0

        if not (0 <= lead < signal.shape[1]):
            raise IndexError(
                f"Érvénytelen lead index: {lead}. "
                f"Elérhető leadek: 0 .. {signal.shape[1] - 1}"
            )

        return signal[:, lead]

    raise ValueError(
        f"A signal csak 1D vagy 2D lehet. Kapott shape: {signal.shape}"
    )


# ---------------------------------------------------------------------
# Rekord-szintű feature-kinyerés
# ---------------------------------------------------------------------
def extract_features_for_record(
    signal: np.ndarray,
    fs: float,
    record_id: str | int,
    lead: int | None = None,
    lead_name: str | None = None,
    label: str | int | None = None,
    dataset: str | None = None,
    rpeak_config: RPeakDetectionConfig = DEFAULT_RPEAK_CONFIG,
) -> dict[str, Any]:
    """
    Egyetlen rekordból számol HRV feature-öket és visszaad egy lapos szótárat.

    Parameters
    ----------
    signal : np.ndarray
        ECG jel. Lehet:
        - 1D: egyetlen lead
        - 2D: több lead, pl. (n_samples, n_leads)
    fs : float
        Mintavételi frekvencia Hz-ben.
    record_id : str | int
        Rekord azonosító.
    lead : int | None
        Ha a signal 2D, akkor melyik lead-et használjuk.
    lead_name : str | None
        Opcionális elnevezés, pl. "MLII", "II", "V5".
    label : str | int | None
        Opcionális címke (klasszifikációs célhoz).
    dataset : str | None
        Opcionális dataset név, pl. "MITBIH", "PTBXL".
    rpeak_config : RPeakDetectionConfig
        R-peak detektálási konfiguráció.

    Returns
    -------
    dict[str, Any]
        Lapos rekord-szintű feature szótár.

    Notes
    -----
    A visszatérő szótár közvetlenül DataFrame-be tehető.
    """
    # 1. Egyetlen 1D lead kiválasztása
    signal_1d = select_lead_from_signal(signal=signal, lead=lead)

    # 2. HRV feature-ök és köztes eredmények számítása
    out = extract_hrv_features(
        signal=signal_1d,
        fs=fs,
        rpeak_config=rpeak_config,
        return_intermediates=True,
    )

    # 3. Rekord szintű metaadatok összeállítása
    row = {
        "record_id": str(record_id),
        "dataset": dataset,
        "fs": float(fs),
        "n_samples": int(len(signal_1d)),
        "lead_index": lead,
        "lead_name": lead_name,
        "label": label,
        "n_rpeaks": int(len(out["peaks"])),
        "n_rr": int(len(out["rr_intervals"])),
        # Kért feature-ök
        "mean_rr": out["mean_rr"],
        "std_rr": out["std_rr"],
        "mean_hr": out["mean_hr"],
        "sdnn": out["sdnn"],
        "rmssd": out["rmssd"],
        "pnn50": out["pnn50"],
    }

    return row


# ---------------------------------------------------------------------
# Több rekordból DataFrame építése
# ---------------------------------------------------------------------
def build_feature_table(
    records: list[dict[str, Any]],
    rpeak_config: RPeakDetectionConfig = DEFAULT_RPEAK_CONFIG,
) -> pd.DataFrame:
    """
    Több rekordból feature táblát készít.

    A `records` lista minden eleme egy dict legyen, például:
    {
        "signal": np.ndarray,
        "fs": 360,
        "record_id": "100",
        "lead": 0,
        "lead_name": "MLII",
        "label": "N",
        "dataset": "MITBIH",
    }

    Parameters
    ----------
    records : list[dict[str, Any]]
        Rekordok listája.
    rpeak_config : RPeakDetectionConfig
        R-peak detektálási konfiguráció.

    Returns
    -------
    pd.DataFrame
        Rekord-szintű feature tábla.

    Notes
    -----
    Ha egy rekordnál hiba történik, azt nem dobjuk el csendben:
    inkább feljegyezzük az error mezőben, hogy a dataset-building
    során látható legyen, melyik rekord problémás.
    """
    rows: list[dict[str, Any]] = []

    for rec in records:
        try:
            row = extract_features_for_record(
                signal=rec["signal"],
                fs=rec["fs"],
                record_id=rec["record_id"],
                lead=rec.get("lead"),
                lead_name=rec.get("lead_name"),
                label=rec.get("label"),
                dataset=rec.get("dataset"),
                rpeak_config=rpeak_config,
            )
            row["error"] = None

        except Exception as exc:
            # Nem állítjuk le az egész buildet egyetlen hibás rekord miatt.
            row = {
                "record_id": str(rec.get("record_id")),
                "dataset": rec.get("dataset"),
                "fs": rec.get("fs"),
                "lead_index": rec.get("lead"),
                "lead_name": rec.get("lead_name"),
                "label": rec.get("label"),
                "n_samples": None,
                "n_rpeaks": None,
                "n_rr": None,
                "mean_rr": np.nan,
                "std_rr": np.nan,
                "mean_hr": np.nan,
                "sdnn": np.nan,
                "rmssd": np.nan,
                "pnn50": np.nan,
                "error": str(exc),
            }

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------
# Egyszerű quality flag-ek
# ---------------------------------------------------------------------
def add_basic_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Egyszerű minőségi jelzőoszlopokat ad a feature táblához.

    Parameters
    ----------
    df : pd.DataFrame
        Feature tábla.

    Returns
    -------
    pd.DataFrame
        Kiegészített DataFrame.

    Notes
    -----
    Ezek csak baseline QC flag-ek, nem klinikai minőségellenőrzés.
    """
    df = df.copy()

    # Volt-e futási hiba?
    df["is_error"] = df["error"].notna()

    # Volt-e elég peak a HRV-hez?
    # Legalább 3 peak -> legalább 2 RR -> rmssd / pnn50 már számolható.
    df["has_min_rpeaks_for_hrv"] = df["n_rpeaks"].fillna(0) >= 3

    # Alapvetően értelmes-e az átlag HR?
    # Csak durva sanity check.
    df["hr_in_plausible_range"] = df["mean_hr"].between(20, 250, inclusive="both")

    # Értelmes-e az átlag RR?
    df["rr_in_plausible_range"] = df["mean_rr"].between(0.24, 3.0, inclusive="both")

    return df