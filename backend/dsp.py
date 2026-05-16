"""
DSP utilities for ECG Test Lab.

Processing pipeline for each filter family:
  1. High-pass  (0.5 Hz)  – baseline wander removal
  2. Notch      (50 Hz)   – power-line interference removal
  3. Low-pass   (40 Hz)   – muscle / EMG noise attenuation

Additional analysis:
  - Welch PSD (power spectral density)
  - FFT frequency spectrum
  - Short-time Fourier Transform (spectrogram)
  - R-peak / heartbeat detection via adaptive threshold
  - Heart rate estimation (BPM)
  - SNR estimation using a Butterworth reference signal

Designed for readability: a DSP course should see each step clearly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import signal

# ── Constants ────────────────────────────────────────────────────────
FS       = 360.0                         # Sampling frequency (Hz)
DATA_DIR = Path(__file__).parent / "data"

NOISE_MAP: Dict[str, str] = {
    "test1": "Normal ECG (clean)",
    "test2": "Baseline wander noise",
    "test3": "50 Hz power-line interference",
    "test4": "Muscle / EMG noise",
    "test5": "Mixed noise",
}

FILTER_INFO: Dict[str, str] = {
    "fir": (
        "FIR pipeline: linear-phase FIR high-pass (0.5 Hz, 301 taps) → "
        "IIR notch at 50 Hz → FIR low-pass (40 Hz, 201 taps). "
        "FIR filters have linear phase response — no phase distortion — "
        "which preserves QRS morphology accurately."
    ),
    "butterworth": (
        "Butterworth pipeline: maximally-flat IIR high-pass (order 3, 0.5 Hz) → "
        "IIR notch at 50 Hz → IIR low-pass (order 4, 40 Hz). "
        "Butterworth filters have the flattest possible passband magnitude response "
        "with zero ripple, making them the most popular choice for general ECG denoising."
    ),
    "chebyshev": (
        "Chebyshev Type I pipeline: sharper-roll-off IIR high-pass (order 3, 0.5 Hz, 0.5 dB ripple) → "
        "IIR notch at 50 Hz → IIR low-pass (order 4, 40 Hz, 0.5 dB ripple). "
        "Chebyshev filters achieve a steeper transition band than Butterworth at the same order, "
        "trading passband ripple for superior attenuation in the stopband."
    ),
}


# ── Signal loading ───────────────────────────────────────────────────

def load_ecg(test_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a local ECG CSV file. Returns (time, ecg) numpy arrays."""
    file_path = DATA_DIR / f"{test_id}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Unknown ECG test: '{test_id}'. Valid IDs: test1–test5")
    df = pd.read_csv(file_path)
    return df["time"].to_numpy(dtype=float), df["ecg"].to_numpy(dtype=float)


# ── Core filtering utilities ─────────────────────────────────────────

def _safe_filtfilt(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Zero-phase forward-backward filtering (filtfilt).
    Falls back to causal lfilter when the signal is too short for filtfilt padding.
    """
    padlen = max(len(b), len(a)) * 3
    if len(x) <= padlen:
        return signal.lfilter(b, a, x)
    return signal.filtfilt(b, a, x)


def apply_notch(x: np.ndarray, fs: float = FS, freq: float = 50.0, q: float = 30.0) -> np.ndarray:
    """IIR notch filter at `freq` Hz with quality factor Q."""
    b, a = signal.iirnotch(w0=freq, Q=q, fs=fs)
    return _safe_filtfilt(b, a, x)


def apply_filter_pipeline(x: np.ndarray, family: str, fs: float = FS) -> np.ndarray:
    """
    Apply the full denoising pipeline for the chosen filter family.

    Pipeline: high-pass → 50 Hz notch → low-pass

    Parameters
    ----------
    x      : raw ECG signal (1-D array)
    family : 'fir' | 'butterworth' | 'chebyshev'
    fs     : sampling frequency in Hz (default 360 Hz)

    Returns
    -------
    filtered signal (same length as x)
    """
    family = family.lower().strip()
    nyq = fs / 2.0  # Nyquist frequency

    if family == "fir":
        # ── FIR pipeline ─────────────────────────────────────────────
        # High-pass: window-sinc, 301 taps, cutoff = 0.5 Hz
        hp_b = signal.firwin(numtaps=301, cutoff=0.5, fs=fs, pass_zero=False)
        y = _safe_filtfilt(hp_b, np.array([1.0]), x)

        # 50 Hz notch (IIR — FIR notch would need hundreds of taps)
        y = apply_notch(y, fs)

        # Low-pass: window-sinc, 201 taps, cutoff = 40 Hz
        lp_b = signal.firwin(numtaps=201, cutoff=40.0, fs=fs, pass_zero=True)
        y = _safe_filtfilt(lp_b, np.array([1.0]), y)
        return y

    elif family == "butterworth":
        # ── Butterworth pipeline ──────────────────────────────────────
        # High-pass: order 3, Wn = 0.5 Hz (maximally flat)
        hp_b, hp_a = signal.butter(3, 0.5 / nyq, btype="highpass")
        y = _safe_filtfilt(hp_b, hp_a, x)

        y = apply_notch(y, fs)

        # Low-pass: order 4, Wn = 40 Hz
        lp_b, lp_a = signal.butter(4, 40.0 / nyq, btype="lowpass")
        y = _safe_filtfilt(lp_b, lp_a, y)
        return y

    elif family == "chebyshev":
        # ── Chebyshev Type I pipeline ─────────────────────────────────
        # High-pass: order 3, 0.5 dB ripple, Wn = 0.5 Hz
        hp_b, hp_a = signal.cheby1(3, 0.5, 0.5 / nyq, btype="highpass")
        y = _safe_filtfilt(hp_b, hp_a, x)

        y = apply_notch(y, fs)

        # Low-pass: order 4, 0.5 dB ripple, Wn = 40 Hz
        lp_b, lp_a = signal.cheby1(4, 0.5, 40.0 / nyq, btype="lowpass")
        y = _safe_filtfilt(lp_b, lp_a, y)
        return y

    else:
        raise ValueError(f"Unknown filter_type '{family}'. Use: fir, butterworth, or chebyshev")


def _butterworth_reference(x: np.ndarray, fs: float = FS) -> np.ndarray:
    """
    Produce a clean reference signal for SNR estimation purposes only.
    Uses the Butterworth pipeline (filter-agnostic baseline).
    """
    nyq = fs / 2.0
    hp_b, hp_a = signal.butter(3, 0.5 / nyq, btype="highpass")
    lp_b, lp_a = signal.butter(4, 40.0 / nyq, btype="lowpass")
    y = _safe_filtfilt(hp_b, hp_a, x)
    y = apply_notch(y, fs)
    y = _safe_filtfilt(lp_b, lp_a, y)
    return y


# ── Analysis functions ────────────────────────────────────────────────

def snr_db(reference: np.ndarray, noisy: np.ndarray) -> float:
    """
    Estimate Signal-to-Noise Ratio in dB.
    SNR = 10 · log10( E[signal²] / E[noise²] )
    where noise = noisy − reference.
    """
    noise = noisy - reference
    signal_power = float(np.mean(reference ** 2)) + 1e-12
    noise_power  = float(np.mean(noise ** 2))    + 1e-12
    return round(10.0 * np.log10(signal_power / noise_power), 2)


def calculate_psd(x: np.ndarray, fs: float = FS) -> Tuple[np.ndarray, np.ndarray]:
    """
    Welch Power Spectral Density estimate.
    nperseg is limited to min(512, signal_length) for robustness.
    """
    freqs, power = signal.welch(x, fs=fs, nperseg=min(512, len(x)))
    return freqs, power


def calculate_fft(x: np.ndarray, fs: float = FS) -> Tuple[np.ndarray, np.ndarray]:
    """
    One-sided FFT magnitude spectrum (mean-centred signal).
    Returns (frequencies, magnitudes).
    """
    centered = x - np.mean(x)
    spectrum  = np.fft.rfft(centered)
    freqs     = np.fft.rfftfreq(len(centered), d=1.0 / fs)
    magnitudes = np.abs(spectrum) / len(centered)
    return freqs, magnitudes


def calculate_spectrogram(
    x: np.ndarray, fs: float = FS, max_freq: float = 120.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Short-Time Fourier Transform (STFT) spectrogram in dB.
    Returns (times, frequencies, power_dB) — all truncated to max_freq.
    """
    nperseg = min(128, len(x) // 8)
    noverlap = nperseg // 2
    freqs, times, Zxx = signal.spectrogram(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    # Trim to max_freq
    mask  = freqs <= max_freq
    freqs = freqs[mask]
    Zxx   = Zxx[mask, :]
    # Convert to dB with noise floor at -80 dB
    power_db = 10.0 * np.log10(np.abs(Zxx) + 1e-10)
    return times, freqs, power_db


def detect_r_peaks(filtered: np.ndarray, fs: float = FS) -> Tuple[np.ndarray, float]:
    """
    Detect R-peaks using a normalised adaptive threshold.

    Steps:
      1. Z-score normalise the signal.
      2. Use scipy.signal.find_peaks with:
         - minimum distance of 300 ms (physiological refractory period)
         - adaptive height threshold at the 88th percentile
         - minimum prominence of 0.35 normalised units

    Returns (peak_indices, heart_rate_bpm).
    """
    z = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-12)
    min_distance = int(0.30 * fs)                    # 300 ms
    height       = max(0.45, float(np.percentile(z, 88)))
    peaks, _     = signal.find_peaks(z, distance=min_distance, height=height, prominence=0.35)

    duration_min = len(filtered) / fs / 60.0         # signal duration in minutes
    bpm = float(len(peaks) / duration_min) if duration_min > 0 else 0.0
    return peaks, round(bpm, 2)


# ── Downsampling helper ───────────────────────────────────────────────

def downsample(values: np.ndarray, max_points: int = 2000) -> List[float]:
    """
    Uniformly subsample an array to at most `max_points` for JSON transport.
    Preserves start and end of the signal.
    """
    if len(values) <= max_points:
        return values.astype(float).round(6).tolist()
    idx = np.linspace(0, len(values) - 1, max_points, dtype=int)
    return values[idx].astype(float).round(6).tolist()


def downsample_2d(matrix: np.ndarray, max_cols: int = 300) -> List[List[float]]:
    """Subsample a 2-D array (e.g. spectrogram Z) along the time axis."""
    if matrix.shape[1] <= max_cols:
        return np.round(matrix.astype(float), 4).tolist()
    col_idx = np.linspace(0, matrix.shape[1] - 1, max_cols, dtype=int)
    return np.round(matrix[:, col_idx].astype(float), 4).tolist()


# ── Main analysis entry point ─────────────────────────────────────────

def analyze_ecg(test_id: str, filter_type: str) -> Dict:
    """
    Full ECG analysis pipeline.

    1. Load signal from CSV
    2. Apply chosen filter family pipeline
    3. Compute SNR before / after
    4. Detect R-peaks and estimate heart rate
    5. Compute PSD, FFT, and spectrogram
    6. Return everything serialisable as JSON

    Parameters
    ----------
    test_id     : 'test1' … 'test5'
    filter_type : 'fir' | 'butterworth' | 'chebyshev'

    Returns
    -------
    dict with all metrics and signal arrays for the frontend
    """
    # ── Load ─────────────────────────────────────────────────────────
    time, original = load_ecg(test_id)

    # ── Filter ───────────────────────────────────────────────────────
    filtered  = apply_filter_pipeline(original, filter_type)
    reference = _butterworth_reference(original)  # clean baseline for SNR

    # ── Metrics ──────────────────────────────────────────────────────
    snr_before = snr_db(reference, original)
    snr_after  = snr_db(reference, filtered)
    peaks, bpm = detect_r_peaks(filtered)

    # ── Frequency analysis ───────────────────────────────────────────
    psd_f, psd_before = calculate_psd(original)
    _,     psd_after  = calculate_psd(filtered)

    fft_f, fft_before = calculate_fft(original)
    _,     fft_after  = calculate_fft(filtered)

    spec_times, spec_freqs, spec_power = calculate_spectrogram(original)

    # ── Peak coordinates (in downsampled time space) ─────────────────
    peak_times  = time[peaks].round(6).tolist()
    peak_values = filtered[peaks].round(6).tolist()

    return {
        # Identity
        "test_id":   test_id,
        "test_name": f"Test {test_id[-1]}",
        "noise_type": NOISE_MAP.get(test_id, "Unknown"),
        "filter_type": filter_type.title(),
        "filter_info": FILTER_INFO.get(filter_type.lower(), "Automatic ECG denoising pipeline."),

        # Scalars
        "sampling_rate":        FS,
        "heart_rate_bpm":       bpm,
        "snr_before_db":        snr_before,
        "snr_after_db":         snr_after,
        "snr_improvement_db":   round(snr_after - snr_before, 2),
        "detected_heartbeats":  int(len(peaks)),

        # Time-domain signals
        "time":             downsample(time),
        "original_signal":  downsample(original),
        "filtered_signal":  downsample(filtered),

        # PSD
        "psd": {
            "frequency": psd_f.round(4).tolist(),
            "before":    psd_before.round(10).tolist(),
            "after":     psd_after.round(10).tolist(),
        },

        # FFT
        "fft": {
            "frequency": fft_f.round(4).tolist(),
            "before":    fft_before.round(10).tolist(),
            "after":     fft_after.round(10).tolist(),
        },

        # Spectrogram (STFT)
        "spectrogram": {
            "times":       spec_times.round(4).tolist(),
            "frequencies": spec_freqs.round(4).tolist(),
            "power":       downsample_2d(spec_power),
        },

        # R-peaks
        "r_peaks": {
            "time":      peak_times,
            "amplitude": peak_values,
        },
    }
