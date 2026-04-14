"""
CricAI - Feature Engineering Pipeline
Transforms raw ball-by-ball data into rich ML features for prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import os

CATEGORICAL_COLS = ["batter", "bowler", "pitch_type", "venue", "phase", "bowler_type"]
NUMERIC_COLS = [
    "over", "ball", "consecutive_dots", "wickets_fallen",
    "runs_scored", "runs_needed", "balls_remaining",
    "required_rate", "current_rr", "pressure_index",
    "bowler_economy", "bowler_wicket_rate",
    "batter_avg", "batter_sr", "batter_pressure_resist",
    "pitch_pace_mult", "pitch_spin_mult", "pitch_bounce_var",
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Core feature engineering transformer.
    Adds derived features that capture cricket context deeply.
    """

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names_ = []

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Pressure features
        df["pressure_x_dots"] = df["pressure_index"] * df["consecutive_dots"]
        df["pressure_x_wickets"] = df["pressure_index"] * df["wickets_fallen"]
        df["rr_gap"] = df["required_rate"] - df["current_rr"]
        df["rr_gap_x_pressure"] = df["rr_gap"] * df["pressure_index"]

        # Pitch-bowler interaction
        df["pace_bowler_on_green"] = (
            (df["bowler_type"] == "pace").astype(int) * df["pitch_pace_mult"]
        )
        df["spin_bowler_on_dry"] = (
            (df["bowler_type"] == "spin").astype(int) * df["pitch_spin_mult"]
        )
        df["bowler_pitch_advantage"] = np.where(
            df["bowler_type"] == "pace",
            df["pitch_pace_mult"],
            df["pitch_spin_mult"]
        )

        # Batter vulnerability on this pitch
        df["batter_pitch_risk"] = df["pressure_index"] * (1 - df["batter_pressure_resist"])
        df["batter_sr_normalized"] = df["batter_sr"] / 150.0

        # Phase encoding
        df["is_powerplay"] = (df["phase"] == "powerplay").astype(int)
        df["is_middle"] = (df["phase"] == "middle").astype(int)
        df["is_death"] = (df["phase"] == "death").astype(int)

        # Over progress (0 to 1)
        df["over_progress"] = (df["over"] * 6 + df["ball"]) / 120.0

        # Balls completed per over
        df["balls_in_over"] = df["ball"]

        # Run scoring pace features
        df["scoring_rate_vs_required"] = np.where(
            df["required_rate"] > 0,
            df["current_rr"] / (df["required_rate"] + 1e-5),
            1.0
        )
        df["balls_remaining_normalized"] = df["balls_remaining"] / 120.0
        df["wickets_remaining"] = 10 - df["wickets_fallen"]
        df["wickets_remaining_normalized"] = df["wickets_remaining"] / 10.0

        # Bounce variability risk (cracked pitch = high var)
        df["bounce_risk"] = df["pitch_bounce_var"] * df["pressure_index"]

        # Death over danger
        df["death_pressure"] = df["is_death"] * df["pressure_index"]
        df["death_bowler_skill"] = df["is_death"] * df["bowler_wicket_rate"]

        return df

    def fit(self, X: pd.DataFrame, y=None):
        df = self._add_derived_features(X)
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                le = LabelEncoder()
                le.fit(df[col].astype(str))
                self.label_encoders[col] = le
        # encode categoricals first so _get_numeric_features sees them
        for col, le in self.label_encoders.items():
            if col in df.columns:
                df[col + "_encoded"] = le.transform(df[col].astype(str))

        numeric_features = self._get_numeric_features(df)
        self.scaler.fit(df[numeric_features].fillna(0).values)
        self.feature_names_ = numeric_features
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = self._add_derived_features(X)

        for col, le in self.label_encoders.items():
            if col in df.columns:
                known = set(le.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in known else le.classes_[0]
                )
                df[col + "_encoded"] = le.transform(df[col])

        numeric_features = self._get_numeric_features(df)
        scaled = self.scaler.transform(df[numeric_features].fillna(0).values)
        df[numeric_features] = scaled
        self.feature_names_ = numeric_features
        return df

    def _get_numeric_features(self, df: pd.DataFrame) -> list:
        derived = [
            "pressure_x_dots", "pressure_x_wickets", "rr_gap", "rr_gap_x_pressure",
            "pace_bowler_on_green", "spin_bowler_on_dry", "bowler_pitch_advantage",
            "batter_pitch_risk", "batter_sr_normalized",
            "is_powerplay", "is_middle", "is_death",
            "over_progress", "balls_in_over",
            "scoring_rate_vs_required", "balls_remaining_normalized",
            "wickets_remaining", "wickets_remaining_normalized",
            "bounce_risk", "death_pressure", "death_bowler_skill",
        ]
        encoded = [col + "_encoded" for col in CATEGORICAL_COLS if col + "_encoded" in df.columns]
        all_numeric = NUMERIC_COLS + derived + encoded
        return [f for f in all_numeric if f in df.columns]

    def get_feature_names(self) -> list:
        return self.feature_names_

    def save(self, path: str):
        joblib.dump(self, path)
        print(f"Feature engineer saved → {path}")

    @staticmethod
    def load(path: str) -> "FeatureEngineer":
        return joblib.load(path)


def prepare_features(df: pd.DataFrame, fe: FeatureEngineer = None, fit: bool = False):
    """
    Prepare features from raw ball dataframe.
    Returns (X, fe) where X is the feature matrix.
    """
    if fe is None:
        fe = FeatureEngineer()

    if fit:
        fe.fit(df)

    df_transformed = fe.transform(df)
    feature_cols = fe.get_feature_names()
    X = df_transformed[feature_cols].fillna(0)
    return X, fe


if __name__ == "__main__":
    from data_generator import generate_dataset
    os.makedirs("/home/claude/cricai", exist_ok=True)

    print("Generating training data...")
    df = generate_dataset(n_matches=1000)

    print("\nRunning feature engineering...")
    X, fe = prepare_features(df, fit=True)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Total features: {len(fe.get_feature_names())}")
    print("\nFeature names:")
    for i, name in enumerate(fe.get_feature_names(), 1):
        print(f"  {i:2d}. {name}")

    fe.save("/home/claude/cricai/feature_engineer.pkl")
    df.to_csv("/home/claude/cricai/training_data.csv", index=False)
    print("\nDone! Training data and feature engineer saved.")
