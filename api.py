"""
CricAI - FastAPI Backend
Real-time ball-by-ball prediction API.
Run with: uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn

from model_xgboost import CricAIPredictor

app = FastAPI(
    title="CricAI Prediction API",
    description="Ball-by-ball cricket AI prediction engine",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once at startup
predictor = CricAIPredictor()


class BallContext(BaseModel):
    """Input schema for one ball prediction."""
    batter: str = Field(..., example="Kohli")
    bowler: str = Field(..., example="Starc")
    pitch_type: str = Field(..., example="green")
    venue: str = Field(..., example="MCG")
    over: int = Field(..., ge=0, le=19)
    ball: int = Field(..., ge=0, le=5)
    phase: str = Field(..., example="death")
    consecutive_dots: int = Field(0, ge=0)
    wickets_fallen: int = Field(0, ge=0, le=10)
    runs_scored: int = Field(0, ge=0)
    target: int = Field(0, ge=0)
    runs_needed: int = Field(0, ge=0)
    balls_remaining: int = Field(120, ge=0)
    required_rate: float = Field(0.0, ge=0)
    current_rr: float = Field(0.0, ge=0)
    pressure_index: float = Field(0.0, ge=0, le=1)
    bowler_type: str = Field(..., example="pace")
    bowler_economy: float = Field(8.0)
    bowler_wicket_rate: float = Field(0.15)
    batter_avg: float = Field(30.0)
    batter_sr: float = Field(130.0)
    batter_pressure_resist: float = Field(0.70)
    pitch_pace_mult: float = Field(1.0)
    pitch_spin_mult: float = Field(1.0)
    pitch_bounce_var: float = Field(0.5)


class PredictionResponse(BaseModel):
    probabilities: dict
    dominant_outcome: str
    confidence_pct: float
    pressure_index: float
    ai_commentary: str


@app.get("/health")
def health():
    return {"status": "ok", "model": "CricAI v1.0"}


@app.post("/predict", response_model=PredictionResponse)
def predict(ctx: BallContext):
    """Predict next-ball outcome probabilities."""
    try:
        result = predictor.predict(ctx.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pitch-types")
def pitch_types():
    """Return available pitch types and their characteristics."""
    return {
        "green":   {"pace_mult": 1.5, "spin_mult": 0.4, "avg_score": 145},
        "dry":     {"pace_mult": 0.7, "spin_mult": 1.8, "avg_score": 155},
        "flat":    {"pace_mult": 0.8, "spin_mult": 0.7, "avg_score": 185},
        "damp":    {"pace_mult": 1.2, "spin_mult": 0.9, "avg_score": 148},
        "cracked": {"pace_mult": 1.3, "spin_mult": 2.0, "avg_score": 132},
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
