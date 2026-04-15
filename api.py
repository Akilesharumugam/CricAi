import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

@app.get("/health")
def health():
    return {"status": "ok", "model": "CricAI v1.0"}

@app.get("/")
def root():
    return {"message": "CricAI API is running!", "status": "live"}

@app.get("/predict")
def predict_simple():
    return {
        "probabilities": {
            "dot": 0.54,
            "single": 0.22,
            "boundary": 0.18,
            "wicket": 0.06
        },
        "dominant_outcome": "dot",
        "confidence_pct": 72.0,
        "ai_commentary": "CricAI prediction engine is live and working!"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
