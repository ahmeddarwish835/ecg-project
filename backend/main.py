"""
ECG Test Lab – FastAPI Backend
==============================

Endpoints:
  GET  /         → health check
  GET  /tests    → list available ECG test signals
  POST /analyze  → run DSP pipeline and return full analysis result

Run with:
  uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from dsp import NOISE_MAP, analyze_ecg, FILTER_INFO

# ── App setup ─────────────────────────────────────────────────────────
app = FastAPI(
    title="ECG Test Lab API",
    description="DSP course project: ECG signal filtering and analysis using FIR, Butterworth, and Chebyshev filters.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # Local development
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",

        # Railway frontend deployment
        "https://alert-essence-production-07b8.up.railway.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request schema ────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    test_id: str
    filter_type: str

    @field_validator("test_id")
    @classmethod
    def validate_test_id(cls, v: str) -> str:
        valid = set(NOISE_MAP.keys())

        if v not in valid:
            raise ValueError(
                f"test_id must be one of {sorted(valid)}"
            )

        return v

    @field_validator("filter_type")
    @classmethod
    def validate_filter_type(cls, v: str) -> str:
        valid = {"fir", "butterworth", "chebyshev"}

        if v.lower() not in valid:
            raise ValueError(
                f"filter_type must be one of {sorted(valid)}"
            )

        return v.lower()

# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Health check — confirms the API is running."""

    return {
        "status": "ok",
        "message": "ECG Test Lab API is running",
        "docs": "/docs",
    }


@app.get("/tests", tags=["Meta"])
def list_tests():
    """Return the list of available ECG test signals."""

    return [
        {
            "id": test_id,
            "label": f"Test {test_id[-1]}",
            "noise_type": noise,
        }
        for test_id, noise in NOISE_MAP.items()
    ]


@app.get("/filters", tags=["Meta"])
def list_filters():
    """Return the list of available filter families with descriptions."""

    return [
        {"id": k, "description": v}
        for k, v in FILTER_INFO.items()
    ]


@app.post("/analyze", tags=["Analysis"])
def analyze(request: AnalyzeRequest):
    """
    Run the full DSP analysis pipeline.

    Body:
        test_id    : 'test1' | 'test2' | 'test3' | 'test4' | 'test5'
        filter_type: 'fir'   | 'butterworth' | 'chebyshev'

    Returns a JSON object with signal arrays, metrics, PSD, FFT,
    spectrogram, and R-peak data ready for Plotly visualisation.
    """

    try:
        return analyze_ecg(
            request.test_id,
            request.filter_type
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc)
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc)
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"DSP analysis failed: {type(exc).__name__}: {exc}",
        ) from exc