from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


# ---------------------------------------------------------------------
# Adatosztály az R-peak detektálás konfigurációjához
# ---------------------------------------------------------------------
# Azért hasznos, mert így a detektálás paraméterei egy helyen vannak,
# és később könnyű finomhangolni őket külön MIT-BIH vagy PTB-XL esetén is.
@dataclass(frozen=True)
class RPeakDetectionConfig:
    """
    Konfiguráció az egyszerű R-peak detektáláshoz.

    Attributes
    ----------
    bandpass_low_hz : float
        A sávszűrő alsó határfrekvenciája Hz-ben.
    bandpass_high_hz : float
        A sávszűrő felső határfrekvenciája Hz-ben.
    filter_order : int
        Butterworth szűrő rendje.
    min_rr_seconds : float
        Minimális megengedett idő két R-csúcs között másodpercben.
        Ez védi a detektálást attól, hogy ugyanazt a QRS-komplexet
        többször találja meg.
    height_quantile : float
        A csúcsmagasság-küszöb kvantilise a feldolgozott jel alapján.
        Pl. 0.90 azt jelenti, hogy a csúcsokhoz küszöbként a jel 90.
        percentilisét használjuk.
    prominence_factor : float
        A csúcs-prominencia becsléséhez használt szorzó.
    """

    bandpass_low_hz: float = 5.0
    bandpass_high_hz: float = 20.0
    filter_order: int = 3
    min_rr_seconds: float = 0.25
    height_quantile: float = 0.90
    prominence_factor: float = 0.5


# ---------------------------------------------------------------------
# Alapértelmezett konfiguráció
# ---------------------------------------------------------------------
DEFAULT_RPEAK_CONFIG = RPeakDetectionConfig()


# ---------------------------------------------------------------------
# Segédfüggvény: jel előkészítése
# ---------------------------------------------------------------------
def _validate_signal(signal: np.ndarray) -> np.ndarray:
    """
    Ellenőrzi és 1D lebegőpontos NumPy tömbbé alakítja a bemeneti jelet.

    Parameters
    ----------
    signal : np.ndarray
        Nyers ECG jel.

    Returns
    -------
    np.ndarray
        1D float típusú tömb.

    Raises
    ------
    ValueError
        Ha a jel üres vagy nem 1D.
    """
    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 1:
        raise ValueError(f"A signal legyen 1D, kapott shape: {signal.shape}")

    if signal.size == 0:
        raise ValueError("A signal üres.")

    return signal


# ---------------------------------------------------------------------
# Butterworth bandpass szűrő
# ---------------------------------------------------------------------
# Egyszerű, általános QRS-kiemeléshez.
# Nem tökéletes klinikai detektor, de jó baseline.
def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = 3,
) -> np.ndarray:
    """
    Bandpass szűrés Butterworth szűrővel.

    Parameters
    ----------
    signal : np.ndarray
        1D ECG jel.
    fs : float
        Mintavételi frekvencia Hz-ben.
    low_hz : float
        Alsó vágási frekvencia.
    high_hz : float
        Felső vágási frekvencia.
    order : int
        Szűrő rendje.

    Returns
    -------
    np.ndarray
        Szűrt jel.
    """
    if fs <= 0:
        raise ValueError("Az fs pozitív kell legyen.")

    nyquist = 0.5 * fs
    low = low_hz / nyquist
    high = high_hz / nyquist

    if not (0 < low < high < 1):
        raise ValueError(
            f"Érvénytelen szűrési tartomány: low={low_hz}, high={high_hz}, fs={fs}"
        )

    b, a = butter(order, [low, high], btype="bandpass")
    filtered = filtfilt(b, a, signal)
    return filtered


# ---------------------------------------------------------------------
# Egyszerű R-peak detektálás
# ---------------------------------------------------------------------
# A lépések:
# 1. bandpass szűrés
# 2. abs() / energia-jellegű kiemelés
# 3. csúcskeresés find_peaks segítségével
#
# Ez a megoldás nem Pan-Tompkins teljes implementáció,
# de jól olvasható és könnyen adaptálható.
def detect_r_peaks(
    signal: np.ndarray,
    fs: float,
    config: RPeakDetectionConfig = DEFAULT_RPEAK_CONFIG,
) -> np.ndarray:
    """
    R-csúcsok detektálása ECG jelből.

    Parameters
    ----------
    signal : np.ndarray
        1D ECG jel.
    fs : float
        Mintavételi frekvencia Hz-ben.
    config : RPeakDetectionConfig
        R-peak detektálási konfiguráció.

    Returns
    -------
    np.ndarray
        Az R-csúcsok indexei a minták között.

    Notes
    -----
    - Általános baseline megoldás.
    - MIT-BIH és PTB-XL esetén is használható.
    - Erősen zajos vagy atípusos jeleknél később érdemes lehet
      fejlettebb detektorra váltani.
    """
    signal = _validate_signal(signal)

    # QRS-t jobban kiemelő szűrt jel
    filtered = bandpass_filter(
        signal=signal,
        fs=fs,
        low_hz=config.bandpass_low_hz,
        high_hz=config.bandpass_high_hz,
        order=config.filter_order,
    )

    # Egyszerű kiemelés: abszolút érték
    # Ezzel a pozitív/negatív polaritás különbsége kevésbé gond.
    enhanced = np.abs(filtered)

    # Adaptív küszöb:
    # a jel kvantilise alapján. Ez sokszor robusztusabb, mint fix küszöb.
    height_threshold = np.quantile(enhanced, config.height_quantile)

    # Prominencia becslés: a jel szórásának egy része
    prominence_threshold = np.std(enhanced) * config.prominence_factor

    # Minimális távolság mintákban.
    # Példa: 0.25 sec * 360 Hz = 90 minta
    min_distance_samples = max(1, int(config.min_rr_seconds * fs))

    peaks, _ = find_peaks(
        enhanced,
        distance=min_distance_samples,
        height=height_threshold,
        prominence=prominence_threshold,
    )

    return peaks.astype(int)


# ---------------------------------------------------------------------
# RR intervallumok számítása
# ---------------------------------------------------------------------
def compute_rr_intervals(peaks: np.ndarray, fs: float) -> np.ndarray:
    """
    RR intervallumok számítása másodpercben.

    Parameters
    ----------
    peaks : np.ndarray
        R-csúcsok mintabeli indexei.
    fs : float
        Mintavételi frekvencia Hz-ben.

    Returns
    -------
    np.ndarray
        RR intervallumok másodpercben.

    Notes
    -----
    RR_i = (R_i - R_{i-1}) / fs
    """
    peaks = np.asarray(peaks, dtype=int)

    if fs <= 0:
        raise ValueError("Az fs pozitív kell legyen.")

    if peaks.size < 2:
        return np.array([], dtype=float)

    rr_intervals = np.diff(peaks) / float(fs)
    return rr_intervals


# ---------------------------------------------------------------------
# Heart rate sorozat számítása RR-ből
# ---------------------------------------------------------------------
def compute_heart_rate(rr_intervals: np.ndarray) -> np.ndarray:
    """
    Szívfrekvencia számítása bpm-ben az RR intervallumokból.

    Parameters
    ----------
    rr_intervals : np.ndarray
        RR intervallumok másodpercben.

    Returns
    -------
    np.ndarray
        Heart rate értékek bpm-ben.

    Notes
    -----
    HR = 60 / RR
    """
    rr_intervals = np.asarray(rr_intervals, dtype=float)

    if rr_intervals.size == 0:
        return np.array([], dtype=float)

    # Nullával osztás védése
    valid = rr_intervals > 0
    hr = np.full(rr_intervals.shape, np.nan, dtype=float)
    hr[valid] = 60.0 / rr_intervals[valid]
    return hr


# ---------------------------------------------------------------------
# HRV feature-ök kiszámítása RR intervallumokból
# ---------------------------------------------------------------------
def extract_hrv_features_from_rr(rr_intervals: np.ndarray) -> dict[str, float]:
    """
    HRV és kapcsolódó feature-ök számítása RR intervallumokból.

    A visszaadott feature-ök:
    - mean_rr
    - std_rr
    - mean_hr
    - sdnn
    - rmssd
    - pnn50

    Parameters
    ----------
    rr_intervals : np.ndarray
        RR intervallumok másodpercben.

    Returns
    -------
    dict[str, float]
        Feature szótár.

    Notes
    -----
    Definíciók:
    - mean_rr: RR intervallumok átlaga
    - std_rr: RR intervallumok szórása
    - mean_hr: HR átlag
    - sdnn: gyakorlatilag az RR-ek szórása
    - rmssd: egymást követő RR különbségek négyzetes átlagának gyöke
    - pnn50: azon egymást követő RR-különbségek aránya,
             amelyek abszolút értékben > 50 ms
    """
    rr_intervals = np.asarray(rr_intervals, dtype=float)

    # Alapértelmezett kimenet NaN-okkal.
    # Ez azért jó, mert ha túl rövid a jel vagy nincs elég peak,
    # nem dobunk hibát, hanem egyértelműen jelöljük, hogy
    # a feature nem számolható stabilan.
    features = {
        "mean_rr": np.nan,
        "std_rr": np.nan,
        "mean_hr": np.nan,
        "sdnn": np.nan,
        "rmssd": np.nan,
        "pnn50": np.nan,
    }

    if rr_intervals.size == 0:
        return features

    hr = compute_heart_rate(rr_intervals)

    # mean_rr
    features["mean_rr"] = float(np.mean(rr_intervals))

    # std_rr
    features["std_rr"] = float(np.std(rr_intervals, ddof=0))

    # mean_hr
    if np.any(~np.isnan(hr)):
        features["mean_hr"] = float(np.nanmean(hr))

    # sdnn
    # Klasszikusan az NN intervallumok szórása.
    # Itt egyszerűsítve RR-t használunk, baseline célra.
    features["sdnn"] = float(np.std(rr_intervals, ddof=0))

    # RMSSD és pNN50 számításához legalább 2 RR kell,
    # tehát legalább 3 detektált R-csúcs.
    if rr_intervals.size >= 2:
        rr_diff = np.diff(rr_intervals)

        # rmssd
        features["rmssd"] = float(np.sqrt(np.mean(rr_diff**2)))

        # pnn50
        # 50 ms = 0.05 sec
        nn50 = np.sum(np.abs(rr_diff) > 0.05)
        features["pnn50"] = float(nn50 / len(rr_diff))

    return features


# ---------------------------------------------------------------------
# Komplett pipeline: signal -> R-peaks -> RR -> feature-ök
# ---------------------------------------------------------------------
def extract_hrv_features(
    signal: np.ndarray,
    fs: float,
    rpeak_config: RPeakDetectionConfig = DEFAULT_RPEAK_CONFIG,
    return_intermediates: bool = False,
) -> dict[str, Any]:
    """
    HRV feature-ök kinyerése közvetlenül ECG jelből.

    Parameters
    ----------
    signal : np.ndarray
        1D ECG jel.
    fs : float
        Mintavételi frekvencia Hz-ben.
    rpeak_config : RPeakDetectionConfig
        R-peak detektálási konfiguráció.
    return_intermediates : bool
        Ha True, akkor a köztes eredményeket is visszaadja:
        peaks, rr_intervals, hr

    Returns
    -------
    dict[str, Any]
        Alapesetben csak a feature-ök.
        Ha return_intermediates=True, akkor plusz köztes eredmények is.

    Examples
    --------
    >>> out = extract_hrv_features(signal, fs=360)
    >>> out["mean_rr"]
    >>> out["rmssd"]
    """
    signal = _validate_signal(signal)

    peaks = detect_r_peaks(signal=signal, fs=fs, config=rpeak_config)
    rr_intervals = compute_rr_intervals(peaks=peaks, fs=fs)
    hr = compute_heart_rate(rr_intervals)
    features = extract_hrv_features_from_rr(rr_intervals)

    if not return_intermediates:
        return features

    return {
        **features,
        "peaks": peaks,
        "rr_intervals": rr_intervals,
        "hr": hr,
    }