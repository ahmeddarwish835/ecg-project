# 🫀 ECG Test Lab – Interactive ECG Filtering Dashboard

> A full-stack DSP course project demonstrating real-time ECG signal processing and analysis.

---

## 📋 Project Overview

**ECG Test Lab** is an interactive biomedical signal processing dashboard that lets you apply digital filters to ECG signals and visualise the effect in real time. It simulates five ECG test cases with different noise types and allows applying three filter families (FIR, Butterworth, Chebyshev) through an automatic DSP pipeline.

### Tech Stack

| Layer     | Technology                                      |
|-----------|-------------------------------------------------|
| Frontend  | React 18 + Tailwind CSS v3 + Plotly.js          |
| Backend   | Python FastAPI + Uvicorn                        |
| DSP       | NumPy + SciPy                                   |
| Build     | Vite 5                                          |

---

## 🚀 Quick Start

### Prerequisites

- **Node.js** 18+ and **npm** 9+
- **Python** 3.10+

### 1 — Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

API will be available at `http://localhost:8000`  
Swagger docs at `http://localhost:8000/docs`

### 2 — Frontend

```bash
cd frontend
npm install
npm run dev
```

Dashboard will open at `http://localhost:5173`

---

## 📁 Project Structure

```
ecg-test-lab/
│
├── README.md
│
├── backend/
│   ├── main.py             ← FastAPI app and endpoints
│   ├── dsp.py              ← All DSP processing functions
│   ├── generate_data.py    ← Script to regenerate ECG CSV files
│   ├── requirements.txt
│   └── data/
│       ├── test1.csv       ← Normal ECG
│       ├── test2.csv       ← Baseline wander noise
│       ├── test3.csv       ← 50 Hz power-line interference
│       ├── test4.csv       ← Muscle / EMG noise
│       └── test5.csv       ← Mixed noise
│
└── frontend/
    ├── index.html
    ├── vite.config.js
    ├── tailwind.config.js
    ├── postcss.config.js
    ├── package.json
    ├── .env.example
    └── src/
        ├── main.jsx        ← Full React app (single page)
        ├── index.css       ← Tailwind + custom glass styles
        └── lib/
            └── api.js      ← API client (fetch wrapper)
```

---

## 🔬 ECG Test Signals

| ID    | Label  | Noise Type                    |
|-------|--------|-------------------------------|
| test1 | Test 1 | Normal ECG (clean)            |
| test2 | Test 2 | Baseline wander noise (0.33 Hz)|
| test3 | Test 3 | 50 Hz power-line interference  |
| test4 | Test 4 | Muscle / EMG noise (HF)        |
| test5 | Test 5 | Mixed noise (BW + PLI + EMG)  |

All signals are synthesised at **360 Hz** over **10 seconds** using Gaussian ECG waveform components (P, Q, R, S, T waves) at 72 BPM.

---

## ⚙️ Filter Families

### Automatic Pipeline (all families)

```
[High-pass 0.5 Hz] → [Notch 50 Hz] → [Low-pass 40 Hz]
```

| Stage         | Purpose                            |
|---------------|------------------------------------|
| High-pass 0.5 Hz | Remove baseline wander / DC drift |
| Notch 50 Hz   | Suppress power-line interference   |
| Low-pass 40 Hz | Attenuate muscle / EMG noise      |

### Filter Characteristics

| Filter      | Design       | Passband      | Trade-off                        |
|-------------|--------------|---------------|----------------------------------|
| FIR         | Window-sinc  | Linear phase  | Higher order (more coefficients) |
| Butterworth | IIR maximally flat | No ripple | Gradual roll-off             |
| Chebyshev I | IIR          | 0.5 dB ripple | Sharper roll-off than Butterworth|

---

## 📊 Dashboard Features

### Result Metrics Cards
- Selected Test & Noise Type
- Heart Rate (BPM)
- Detected R-Peaks
- SNR Before / After filtering (dB)
- SNR Improvement

### Interactive Tabs

| Tab              | Content                                         |
|------------------|-------------------------------------------------|
| Time Domain      | Original signal, Filtered signal, Comparison   |
| Frequency Domain | PSD (Welch), FFT Spectrum, Spectrogram (STFT)  |
| Heart Analysis   | R-Peak detection visualisation + stats          |
| Filter Info      | DSP pipeline details + SNR explanation         |

---

## 🔌 API Reference

### `GET /`
Health check.

### `GET /tests`
Returns list of available ECG test signals.

### `GET /filters`
Returns list of filter families with descriptions.

### `POST /analyze`

**Request body:**
```json
{
  "test_id": "test1",
  "filter_type": "butterworth"
}
```

**Response:**
```json
{
  "test_id": "test1",
  "test_name": "Test 1",
  "noise_type": "Normal ECG (clean)",
  "filter_type": "Butterworth",
  "filter_info": "...",
  "sampling_rate": 360.0,
  "heart_rate_bpm": 72.0,
  "snr_before_db": 15.3,
  "snr_after_db": 22.1,
  "snr_improvement_db": 6.8,
  "detected_heartbeats": 12,
  "time": [...],
  "original_signal": [...],
  "filtered_signal": [...],
  "psd": { "frequency": [...], "before": [...], "after": [...] },
  "fft": { "frequency": [...], "before": [...], "after": [...] },
  "spectrogram": { "times": [...], "frequencies": [...], "power": [[...]] },
  "r_peaks": { "time": [...], "amplitude": [...] }
}
```

---

## 🎓 DSP Concepts Demonstrated

- **FIR filters**: Linear phase, window-sinc design, `firwin()`
- **IIR filters**: Butterworth (maximally flat), Chebyshev Type I (equiripple)
- **Zero-phase filtering**: `filtfilt()` for no phase distortion
- **Notch filter**: `iirnotch()` for narrowband interference removal
- **Welch PSD**: Power spectral density via Welch's method
- **FFT analysis**: One-sided magnitude spectrum
- **STFT / Spectrogram**: Time-frequency representation
- **R-peak detection**: Adaptive threshold + `find_peaks()`
- **SNR estimation**: Signal-to-noise ratio in dB

---

## 🔄 Regenerating ECG Test Data

If you need to regenerate the CSV files:

```bash
cd backend
python generate_data.py
```

---

## 🛠️ Troubleshooting

**Backend not connecting?**
- Ensure the backend is running on port 8000
- Check CORS: the backend allows `localhost:5173` and `localhost:3000`
- Copy `.env.example` to `.env` in the frontend folder if you changed the port

**Plotly graphs not rendering?**
- Ensure `npm install` completed successfully
- Check browser console for errors

**Python version issues?**
- The project requires Python 3.10+
- Use a virtual environment: `python -m venv .venv && source .venv/bin/activate`
