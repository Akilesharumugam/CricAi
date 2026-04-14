"""
CricAI - Synthetic Ball-by-Ball Data Generator
Generates realistic cricket data for model training.
In production, replace with CricAPI / ESPNcricinfo data.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List

np.random.seed(42)

BATTERS = {
    "Rohit":   {"avg": 32, "sr": 140, "spin_vuln": 0.25, "pace_vuln": 0.20, "pressure_resist": 0.75},
    "Kohli":   {"avg": 36, "sr": 138, "spin_vuln": 0.18, "pace_vuln": 0.15, "pressure_resist": 0.88},
    "Gill":    {"avg": 28, "sr": 145, "spin_vuln": 0.30, "pace_vuln": 0.22, "pressure_resist": 0.65},
    "Hardik":  {"avg": 24, "sr": 155, "spin_vuln": 0.35, "pace_vuln": 0.18, "pressure_resist": 0.70},
    "Jadeja":  {"avg": 20, "sr": 130, "spin_vuln": 0.15, "pace_vuln": 0.28, "pressure_resist": 0.72},
    "Bumrah":  {"avg": 10, "sr": 110, "spin_vuln": 0.45, "pace_vuln": 0.40, "pressure_resist": 0.50},
}

BOWLERS = {
    "Starc":      {"economy": 8.2, "wicket_rate": 0.18, "type": "pace", "death_skill": 0.80},
    "Hazlewood": {"economy": 7.5, "wicket_rate": 0.16, "type": "pace", "death_skill": 0.85},
    "Cummins":   {"economy": 7.8, "wicket_rate": 0.17, "type": "pace", "death_skill": 0.82},
    "Zampa":     {"economy": 7.0, "wicket_rate": 0.14, "type": "spin", "death_skill": 0.65},
    "Ashwin":    {"economy": 6.8, "wicket_rate": 0.15, "type": "spin", "death_skill": 0.60},
    "Shami":     {"economy": 8.5, "wicket_rate": 0.19, "type": "pace", "death_skill": 0.88},
}

PITCH_PROFILES = {
    "green":   {"pace_mult": 1.5, "spin_mult": 0.4, "bounce_var": 0.8, "avg_score": 145},
    "dry":     {"pace_mult": 0.7, "spin_mult": 1.8, "bounce_var": 0.7, "avg_score": 155},
    "flat":    {"pace_mult": 0.8, "spin_mult": 0.7, "bounce_var": 0.2, "avg_score": 185},
    "damp":    {"pace_mult": 1.2, "spin_mult": 0.9, "bounce_var": 0.5, "avg_score": 148},
    "cracked": {"pace_mult": 1.3, "spin_mult": 2.0, "bounce_var": 1.0, "avg_score": 132},
}

VENUES = ["Wankhede", "Eden Gardens", "Chepauk", "Chinnaswamy", "MCG", "SCG", "Lords"]


def generate_ball(
    batter_name: str,
    bowler_name: str,
    pitch_type: str,
    over: int,
    ball_in_over: int,
    consecutive_dots: int,
    wickets_fallen: int,
    runs_scored: int,
    target: int,
    venue: str,
) -> dict:
    """Generate one ball of cricket with realistic outcome probabilities."""

    batter = BATTERS[batter_name]
    bowler = BOWLERS[bowler_name]
    pitch = PITCH_PROFILES[pitch_type]

    balls_remaining = (20 - over) * 6 - ball_in_over
    runs_needed = target - runs_scored if target > 0 else 0
    required_rate = (runs_needed / balls_remaining * 6) if balls_remaining > 0 else 0
    current_rr = (runs_scored / max(1, over * 6 + ball_in_over)) * 6

    pressure_index = min(1.0, (
        (consecutive_dots * 0.12) +
        (max(0, required_rate - current_rr) * 0.05) +
        (wickets_fallen * 0.08) +
        (max(0, over - 15) * 0.04)
    ))

    phase = "powerplay" if over < 6 else ("middle" if over < 15 else "death")

    bowler_effectiveness = 1.0
    if bowler["type"] == "pace":
        bowler_effectiveness = pitch["pace_mult"]
    else:
        bowler_effectiveness = pitch["spin_mult"]

    if phase == "death":
        bowler_effectiveness *= bowler["death_skill"]

    batter_vuln = batter["spin_vuln"] if bowler["type"] == "spin" else batter["pace_vuln"]
    batter_pressure_penalty = (1 - batter["pressure_resist"]) * pressure_index

    base_wicket_prob = bowler["wicket_rate"] * bowler_effectiveness * (1 + batter_vuln + batter_pressure_penalty)
    base_wicket_prob = min(0.40, base_wicket_prob)

    aggression = 1.0
    if required_rate > 12:
        aggression = 1.3
    elif required_rate > 9:
        aggression = 1.1
    elif phase == "powerplay":
        aggression = 1.15

    base_boundary_prob = (batter["sr"] / 600) * aggression * (1 / bowler_effectiveness) * pitch["avg_score"] / 160
    base_boundary_prob = min(0.35, max(0.05, base_boundary_prob))

    dot_prob = min(0.65, 0.35 + (consecutive_dots * 0.04) + (batter_vuln * 0.2) + (bowler_effectiveness * 0.1))
    single_prob = max(0.10, 0.30 - (pressure_index * 0.05))
    wicket_prob = base_wicket_prob
    boundary_prob = base_boundary_prob

    total = dot_prob + single_prob + wicket_prob + boundary_prob
    dot_prob /= total
    single_prob /= total
    wicket_prob /= total
    boundary_prob /= total

    rand = np.random.random()
    if rand < wicket_prob:
        outcome = "W"
        runs_this_ball = 0
    elif rand < wicket_prob + boundary_prob:
        outcome = "4" if np.random.random() < 0.75 else "6"
        runs_this_ball = 4 if outcome == "4" else 6
    elif rand < wicket_prob + boundary_prob + dot_prob:
        outcome = "0"
        runs_this_ball = 0
    else:
        outcome = str(np.random.choice([1, 2, 3], p=[0.70, 0.25, 0.05]))
        runs_this_ball = int(outcome)

    return {
        "batter": batter_name,
        "bowler": bowler_name,
        "pitch_type": pitch_type,
        "venue": venue,
        "over": over,
        "ball": ball_in_over,
        "phase": phase,
        "consecutive_dots": consecutive_dots,
        "wickets_fallen": wickets_fallen,
        "runs_scored": runs_scored,
        "target": target,
        "runs_needed": runs_needed,
        "balls_remaining": balls_remaining,
        "required_rate": round(required_rate, 2),
        "current_rr": round(current_rr, 2),
        "pressure_index": round(pressure_index, 3),
        "bowler_type": bowler["type"],
        "bowler_economy": bowler["economy"],
        "bowler_wicket_rate": bowler["wicket_rate"],
        "batter_avg": batter["avg"],
        "batter_sr": batter["sr"],
        "batter_pressure_resist": batter["pressure_resist"],
        "pitch_pace_mult": pitch["pace_mult"],
        "pitch_spin_mult": pitch["spin_mult"],
        "pitch_bounce_var": pitch["bounce_var"],
        "outcome": outcome,
        "runs_this_ball": runs_this_ball,
        "is_wicket": 1 if outcome == "W" else 0,
        "is_boundary": 1 if outcome in ["4", "6"] else 0,
        "is_dot": 1 if runs_this_ball == 0 and outcome != "W" else 0,
        "dot_prob_true": round(dot_prob, 3),
        "wicket_prob_true": round(wicket_prob, 3),
        "boundary_prob_true": round(boundary_prob, 3),
    }


def generate_match(target: int = 0) -> List[dict]:
    """Simulate a full T20 innings ball by ball."""
    pitch_type = np.random.choice(list(PITCH_PROFILES.keys()))
    venue = np.random.choice(VENUES)
    batter_names = list(BATTERS.keys())
    bowler_names = list(BOWLERS.keys())

    balls = []
    runs_scored = 0
    wickets = 0
    current_batter_idx = 0
    consecutive_dots = 0

    for over in range(20):
        bowler = np.random.choice(bowler_names)
        for ball in range(6):
            if wickets >= 10:
                break
            batter = batter_names[min(current_batter_idx, len(batter_names) - 1)]
            b = generate_ball(
                batter, bowler, pitch_type, over, ball,
                consecutive_dots, wickets, runs_scored, target, venue
            )
            balls.append(b)
            runs_scored += b["runs_this_ball"]
            if b["is_dot"]:
                consecutive_dots += 1
            else:
                consecutive_dots = 0
            if b["is_wicket"]:
                wickets += 1
                current_batter_idx += 1

    return balls


def generate_dataset(n_matches: int = 2000) -> pd.DataFrame:
    """Generate full training dataset from N simulated matches."""
    all_balls = []
    for i in range(n_matches):
        target = np.random.randint(140, 200) if i % 2 == 1 else 0
        all_balls.extend(generate_match(target))
    df = pd.DataFrame(all_balls)
    print(f"Generated {len(df):,} balls from {n_matches} matches")
    print(f"Wicket rate: {df['is_wicket'].mean():.3f} | Boundary rate: {df['is_boundary'].mean():.3f} | Dot rate: {df['is_dot'].mean():.3f}")
    return df


if __name__ == "__main__":
    df = generate_dataset(500)
    df.to_csv("/home/claude/cricai/training_data.csv", index=False)
    print("Saved to training_data.csv")
    print(df.head(3).to_string())
