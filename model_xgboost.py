"""
CricAI - XGBoost Prediction Model
Trains three separate models: wicket predictor, boundary predictor, dot ball predictor.
Combines into a Voting ensemble for the final ball-outcome prediction.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score,
    accuracy_score, log_loss
)
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier

from data_generator import generate_dataset
from feature_engineering import prepare_features, FeatureEngineer

OUTPUT_DIR = "/home/claude/cricai"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── XGBoost model factory ────────────────────────────────────────────────────

def make_xgb(scale_pos_weight=1.0, n_estimators=300):
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )


def make_lgbm(scale_pos_weight=1.0):
    return LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def make_rf():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )


# ─── Train one outcome model ──────────────────────────────────────────────────

def train_outcome_model(X_train, y_train, X_val, y_val, outcome_name: str):
    """
    Train an ensemble Voting Classifier for one outcome (wicket / boundary / dot).
    Returns trained model and validation metrics.
    """
    pos_rate = y_train.mean()
    neg_rate = 1 - pos_rate
    spw = neg_rate / (pos_rate + 1e-6)

    print(f"\n{'─'*50}")
    print(f"Training: {outcome_name.upper()} predictor")
    print(f"  Positive rate: {pos_rate:.3f} | scale_pos_weight: {spw:.1f}")

    xgb_model  = make_xgb(scale_pos_weight=spw)
    lgbm_model = make_lgbm(scale_pos_weight=spw)
    rf_model   = make_rf()

    ensemble = VotingClassifier(
        estimators=[
            ("xgb",  xgb_model),
            ("lgbm", lgbm_model),
            ("rf",   rf_model),
        ],
        voting="soft",
        weights=[3, 2, 1],
    )

    ensemble.fit(X_train, y_train)

    y_pred  = ensemble.predict(X_val)
    y_proba = ensemble.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y_val, y_proba)
    acc = accuracy_score(y_val, y_pred)
    ll  = log_loss(y_val, y_proba)

    print(f"  Val AUC: {auc:.4f} | Accuracy: {acc:.4f} | Log-loss: {ll:.4f}")
    print(classification_report(y_val, y_pred, target_names=["No", outcome_name], digits=3))

    metrics = {"auc": round(auc, 4), "accuracy": round(acc, 4), "log_loss": round(ll, 4)}
    return ensemble, metrics


# ─── Feature importance ───────────────────────────────────────────────────────

def extract_feature_importance(model, feature_names: list, top_n: int = 15) -> pd.DataFrame:
    """Extract and aggregate feature importances from XGB sub-model."""
    try:
        xgb_sub = model.estimators_[0]  # XGB is index 0
        imp = xgb_sub.feature_importances_
        df_imp = pd.DataFrame({"feature": feature_names, "importance": imp})
        df_imp = df_imp.sort_values("importance", ascending=False).head(top_n)
        return df_imp
    except Exception:
        return pd.DataFrame()


# ─── Live prediction ──────────────────────────────────────────────────────────

class CricAIPredictor:
    """
    Loads trained models and provides ball-by-ball predictions.
    This is what your backend API will call every ball.
    """

    def __init__(self, model_dir: str = OUTPUT_DIR):
        self.model_dir = model_dir
        self.fe: FeatureEngineer = None
        self.wicket_model  = None
        self.boundary_model = None
        self.dot_model     = None
        self.metadata      = {}
        self._load()

    def _load(self):
        self.fe             = FeatureEngineer.load(f"{self.model_dir}/feature_engineer.pkl")
        self.wicket_model   = joblib.load(f"{self.model_dir}/model_wicket.pkl")
        self.boundary_model = joblib.load(f"{self.model_dir}/model_boundary.pkl")
        self.dot_model      = joblib.load(f"{self.model_dir}/model_dot.pkl")
        with open(f"{self.model_dir}/model_metadata.json") as f:
            self.metadata = json.load(f)
        print("CricAI models loaded successfully.")

    def predict(self, ball_context: dict) -> dict:
        """
        Predict outcome probabilities for the NEXT ball.

        ball_context keys (same as data_generator output):
            batter, bowler, pitch_type, venue, over, ball,
            consecutive_dots, wickets_fallen, runs_scored,
            target, runs_needed, balls_remaining, required_rate,
            current_rr, pressure_index, bowler_type, bowler_economy,
            bowler_wicket_rate, batter_avg, batter_sr,
            batter_pressure_resist, pitch_pace_mult, pitch_spin_mult,
            pitch_bounce_var, phase
        """
        df = pd.DataFrame([ball_context])
        X, _ = prepare_features(df, fe=self.fe, fit=False)

        p_wicket   = float(self.wicket_model.predict_proba(X)[0, 1])
        p_boundary = float(self.boundary_model.predict_proba(X)[0, 1])
        p_dot      = float(self.dot_model.predict_proba(X)[0, 1])

        # Normalise so probabilities sum to 1
        p_single = max(0.0, 1.0 - p_wicket - p_boundary - p_dot)
        total = p_wicket + p_boundary + p_dot + p_single
        p_wicket   /= total
        p_boundary /= total
        p_dot      /= total
        p_single   /= total

        # Confidence: how far the top prediction is from uniform (0.25)
        top_p = max(p_wicket, p_boundary, p_dot, p_single)
        confidence = round((top_p - 0.25) / 0.75 * 100, 1)

        # Dominant outcome
        outcomes = {"wicket": p_wicket, "boundary": p_boundary, "dot": p_dot, "single": p_single}
        dominant = max(outcomes, key=outcomes.get)

        commentary = self._generate_commentary(ball_context, outcomes, dominant)

        return {
            "probabilities": {
                "dot":      round(p_dot, 3),
                "single":   round(p_single, 3),
                "boundary": round(p_boundary, 3),
                "wicket":   round(p_wicket, 3),
            },
            "dominant_outcome": dominant,
            "confidence_pct": confidence,
            "pressure_index": ball_context.get("pressure_index", 0),
            "ai_commentary": commentary,
        }

    def _generate_commentary(self, ctx: dict, probs: dict, dominant: str) -> str:
        """Rule-based commentary — replace with Claude API in production."""
        batter   = ctx.get("batter", "Batter")
        bowler   = ctx.get("bowler", "Bowler")
        dots     = ctx.get("consecutive_dots", 0)
        pitch    = ctx.get("pitch_type", "flat")
        pressure = ctx.get("pressure_index", 0)
        phase    = ctx.get("phase", "middle")

        lines = []
        if dots >= 3:
            lines.append(f"{dots} consecutive dots — {batter} is under significant pressure.")
        if pitch in ("green", "cracked"):
            lines.append(f"{pitch.title()} pitch is giving {bowler} extra movement.")
        if phase == "death":
            lines.append("Death overs — every ball is critical.")
        if dominant == "wicket":
            lines.append(f"AI sees elevated wicket risk ({probs['wicket']*100:.0f}%) this ball.")
        elif dominant == "boundary":
            lines.append(f"Boundary likely ({probs['boundary']*100:.0f}%) — {batter} looking to break free.")
        elif dominant == "dot":
            lines.append(f"Dot ball expected ({probs['dot']*100:.0f}%) — {bowler} is controlling the over.")
        else:
            lines.append("Quiet ball expected — building pressure.")

        return " ".join(lines) if lines else "Analysing match situation..."


# ─── Main training script ─────────────────────────────────────────────────────

def train_all():
    print("=" * 60)
    print("  CricAI — Training Pipeline")
    print("=" * 60)

    print("\n[1/4] Generating training data...")
    df = generate_dataset(n_matches=2000)

    print("\n[2/4] Feature engineering...")
    X, fe = prepare_features(df, fit=True)
    fe.save(f"{OUTPUT_DIR}/feature_engineer.pkl")
    feature_names = fe.get_feature_names()
    print(f"  Features: {len(feature_names)}")

    y_wicket   = df["is_wicket"].values
    y_boundary = df["is_boundary"].values
    y_dot      = df["is_dot"].values

    X_train, X_val, yw_tr, yw_val, yb_tr, yb_val, yd_tr, yd_val = train_test_split(
        X, y_wicket, y_boundary, y_dot,
        test_size=0.20, random_state=42, stratify=y_wicket
    )

    print(f"\n  Train: {len(X_train):,} balls | Val: {len(X_val):,} balls")

    print("\n[3/4] Training models...")
    all_metrics = {}

    wicket_model, wm = train_outcome_model(X_train, yw_tr, X_val, yw_val, "wicket")
    joblib.dump(wicket_model, f"{OUTPUT_DIR}/model_wicket.pkl")
    all_metrics["wicket"] = wm

    boundary_model, bm = train_outcome_model(X_train, yb_tr, X_val, yb_val, "boundary")
    joblib.dump(boundary_model, f"{OUTPUT_DIR}/model_boundary.pkl")
    all_metrics["boundary"] = bm

    dot_model, dm = train_outcome_model(X_train, yd_tr, X_val, yd_val, "dot")
    joblib.dump(dot_model, f"{OUTPUT_DIR}/model_dot.pkl")
    all_metrics["dot"] = dm

    print("\n[4/4] Feature importance (wicket model):")
    fi = extract_feature_importance(wicket_model, feature_names)
    if not fi.empty:
        for _, row in fi.head(10).iterrows():
            bar = "█" * int(row["importance"] * 200)
            print(f"  {row['feature']:<35} {bar} {row['importance']:.4f}")

    metadata = {
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "train_size": len(X_train),
        "val_size": len(X_val),
        "metrics": all_metrics,
    }
    with open(f"{OUTPUT_DIR}/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("  Training complete! Summary:")
    print("=" * 60)
    for outcome, m in all_metrics.items():
        print(f"  {outcome:<12} AUC={m['auc']:.4f}  Acc={m['accuracy']:.4f}  LogLoss={m['log_loss']:.4f}")

    return metadata


def demo_prediction():
    """Show a live prediction example."""
    print("\n" + "=" * 60)
    print("  CricAI — Live Prediction Demo")
    print("=" * 60)

    predictor = CricAIPredictor()

    scenarios = [
        {
            "label": "Kohli under pressure (3 dots, green top)",
            "ctx": {
                "batter": "Kohli", "bowler": "Starc", "pitch_type": "green",
                "venue": "MCG", "over": 16, "ball": 3, "phase": "death",
                "consecutive_dots": 3, "wickets_fallen": 3, "runs_scored": 128,
                "target": 175, "runs_needed": 47, "balls_remaining": 21,
                "required_rate": 13.4, "current_rr": 8.0, "pressure_index": 0.78,
                "bowler_type": "pace", "bowler_economy": 8.2, "bowler_wicket_rate": 0.18,
                "batter_avg": 36, "batter_sr": 138, "batter_pressure_resist": 0.88,
                "pitch_pace_mult": 1.5, "pitch_spin_mult": 0.4, "pitch_bounce_var": 0.8,
            }
        },
        {
            "label": "Rohit flowing, flat pitch, powerplay",
            "ctx": {
                "batter": "Rohit", "bowler": "Hazlewood", "pitch_type": "flat",
                "venue": "Wankhede", "over": 3, "ball": 2, "phase": "powerplay",
                "consecutive_dots": 0, "wickets_fallen": 0, "runs_scored": 38,
                "target": 0, "runs_needed": 0, "balls_remaining": 100,
                "required_rate": 0, "current_rr": 9.5, "pressure_index": 0.10,
                "bowler_type": "pace", "bowler_economy": 7.5, "bowler_wicket_rate": 0.16,
                "batter_avg": 32, "batter_sr": 140, "batter_pressure_resist": 0.75,
                "pitch_pace_mult": 0.8, "pitch_spin_mult": 0.7, "pitch_bounce_var": 0.2,
            }
        },
        {
            "label": "Jadeja vs Zampa on dry dusty pitch",
            "ctx": {
                "batter": "Jadeja", "bowler": "Zampa", "pitch_type": "dry",
                "venue": "Chepauk", "over": 12, "ball": 1, "phase": "middle",
                "consecutive_dots": 1, "wickets_fallen": 2, "runs_scored": 88,
                "target": 162, "runs_needed": 74, "balls_remaining": 47,
                "required_rate": 9.4, "current_rr": 7.3, "pressure_index": 0.45,
                "bowler_type": "spin", "bowler_economy": 7.0, "bowler_wicket_rate": 0.14,
                "batter_avg": 20, "batter_sr": 130, "batter_pressure_resist": 0.72,
                "pitch_pace_mult": 0.7, "pitch_spin_mult": 1.8, "pitch_bounce_var": 0.7,
            }
        },
    ]

    for s in scenarios:
        print(f"\n  Scenario: {s['label']}")
        print(f"  {'-'*50}")
        result = predictor.predict(s["ctx"])
        p = result["probabilities"]
        print(f"  Dot:      {p['dot']*100:5.1f}%  {'█'*int(p['dot']*30)}")
        print(f"  Single:   {p['single']*100:5.1f}%  {'█'*int(p['single']*30)}")
        print(f"  Boundary: {p['boundary']*100:5.1f}%  {'█'*int(p['boundary']*30)}")
        print(f"  Wicket:   {p['wicket']*100:5.1f}%  {'█'*int(p['wicket']*30)}")
        print(f"  Dominant: {result['dominant_outcome'].upper()} | Confidence: {result['confidence_pct']}%")
        print(f"  Commentary: {result['ai_commentary']}")


if __name__ == "__main__":
    train_all()
    demo_prediction()
