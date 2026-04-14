# CricAI — Ball-by-Ball AI Prediction Engine

## What's built here (Phase 1 MVP)

| File | Purpose |
|------|---------|
| `data_generator.py` | Simulates realistic ball-by-ball cricket data |
| `feature_engineering.py` | 45-feature pipeline (pressure, pitch, matchup, phase) |
| `model_xgboost.py` | XGBoost + LightGBM + RandomForest Voting ensemble |
| `api.py` | FastAPI REST backend — call `/predict` every ball |

## Model Results (trained on 104,583 balls)

| Model | AUC | Accuracy |
|-------|-----|----------|
| Wicket predictor | 0.6199 | 65.6% |
| Boundary predictor | 0.6529 | 66.0% |
| Dot ball predictor | 0.5203 | 54.5% |

> AUC will improve significantly with real CricAPI data (vs synthetic).
> Target with real data: Wicket AUC > 0.80, Boundary AUC > 0.78

## Top Features (by importance)
1. `bowler_pitch_advantage` — pitch type × bowler type interaction
2. `pitch_pace_mult` — pitch pace factor
3. `pitch_spin_mult` — pitch spin factor
4. `pace_bowler_on_green` — pace bowler on green pitch flag
5. `batter_pitch_risk` — batter vulnerability × pressure

## Quick Start

```bash
# Install dependencies
pip install xgboost lightgbm scikit-learn pandas numpy fastapi uvicorn

# Train models (takes ~2 mins)
python model_xgboost.py

# Start API server
uvicorn api:app --reload --port 8000

# Test prediction (in another terminal)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "batter": "Kohli", "bowler": "Starc",
    "pitch_type": "green", "venue": "MCG",
    "over": 16, "ball": 3, "phase": "death",
    "consecutive_dots": 3, "wickets_fallen": 3,
    "runs_scored": 128, "target": 175,
    "runs_needed": 47, "balls_remaining": 21,
    "required_rate": 13.4, "current_rr": 8.0,
    "pressure_index": 0.78,
    "bowler_type": "pace", "bowler_economy": 8.2,
    "bowler_wicket_rate": 0.18, "batter_avg": 36,
    "batter_sr": 138, "batter_pressure_resist": 0.88,
    "pitch_pace_mult": 1.5, "pitch_spin_mult": 0.4,
    "pitch_bounce_var": 0.8
  }'
```

## Next Steps (Phase 2)

1. **Real data** — Connect CricAPI (`pip install cricapi`) for live + historical data
2. **LSTM model** — Add `model_lstm.py` for sequence-based time-series prediction
3. **Bayesian Network** — Add `model_bayesian.py` for win probability
4. **Monte Carlo** — Add `model_monte_carlo.py` for match simulation
5. **Claude API** — Replace rule-based commentary with Claude API call
6. **React Native app** — Call `/predict` WebSocket every ball

## Real Data Integration

```python
# Replace data_generator.py with:
import requests

def get_live_ball(match_id: str, api_key: str) -> dict:
    url = f"https://api.cricapi.com/v1/match_scorecard"
    resp = requests.get(url, params={"apikey": api_key, "id": match_id})
    data = resp.json()
    # Map CricAPI fields → our BallContext schema
    return map_cricapi_to_context(data)
```

## File Structure
```
cricai/
├── data_generator.py       # Synthetic data (replace with CricAPI)
├── feature_engineering.py  # 45-feature pipeline
├── model_xgboost.py        # Training + prediction
├── api.py                  # FastAPI backend
├── feature_engineer.pkl    # Saved feature encoder
├── model_wicket.pkl        # Trained wicket model
├── model_boundary.pkl      # Trained boundary model
├── model_dot.pkl           # Trained dot ball model
└── model_metadata.json     # Training metrics
```
