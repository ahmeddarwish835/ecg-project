"""Generate five local ECG-like CSV files for the DSP dashboard."""
from pathlib import Path
import numpy as np
import pandas as pd

FS = 360
DURATION = 10
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(42)
t = np.arange(0, DURATION, 1 / FS)

def synthetic_ecg(t: np.ndarray, bpm: float = 72) -> np.ndarray:
    rr = 60.0 / bpm
    y = np.zeros_like(t)
    beats = np.arange(0.6, t[-1] - 0.2, rr)
    for b in beats:
        # Gaussian waves: P, Q, R, S, T
        y += 0.10 * np.exp(-0.5 * ((t - (b - 0.18)) / 0.035) ** 2)
        y += -0.15 * np.exp(-0.5 * ((t - (b - 0.035)) / 0.012) ** 2)
        y += 1.15 * np.exp(-0.5 * ((t - b) / 0.014) ** 2)
        y += -0.25 * np.exp(-0.5 * ((t - (b + 0.035)) / 0.016) ** 2)
        y += 0.32 * np.exp(-0.5 * ((t - (b + 0.27)) / 0.075) ** 2)
    y += 0.015 * np.sin(2 * np.pi * 0.2 * t)
    return y

clean = synthetic_ecg(t)
baseline = clean + 0.35 * np.sin(2 * np.pi * 0.33 * t)
powerline = clean + 0.16 * np.sin(2 * np.pi * 50 * t)
emg = clean + 0.08 * rng.normal(size=len(t)) + 0.04 * np.sin(2 * np.pi * 95 * t)
mixed = clean + 0.30 * np.sin(2 * np.pi * 0.27 * t) + 0.13 * np.sin(2 * np.pi * 50 * t) + 0.07 * rng.normal(size=len(t))

signals = [clean, baseline, powerline, emg, mixed]
for i, sig in enumerate(signals, 1):
    pd.DataFrame({"time": t, "ecg": sig}).to_csv(DATA_DIR / f"test{i}.csv", index=False)
print(f"Generated {len(signals)} ECG test files in {DATA_DIR}")
