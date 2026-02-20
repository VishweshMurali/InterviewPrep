"""
SkillMapper - ML Difficulty Adapter
A Sklearn classifier that learns from your answer history
to predict the right difficulty level for your next question.
This is the 'real ML model' in the project.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

HISTORY_FILE = "./logs/session_history.json"


class DifficultyAdapter:
    """
    Predicts optimal difficulty: Easy / Medium / Hard
    
    Features used:
    - avg_score on concept (0-10)
    - attempt_count on concept
    - recent_trend (last 3 scores average)
    - global session accuracy
    
    This is a real Sklearn model trained incrementally on your history.
    """

    DIFFICULTY_LABELS = ["Easy", "Medium", "Hard"]

    def __init__(self):
        self.model   = LogisticRegression(max_iter=1000, random_state=42)
        self.encoder = LabelEncoder()
        self.encoder.fit(self.DIFFICULTY_LABELS)
        self.history  = []    # List of (features, label) tuples
        self.trained  = False
        self.load_history()

    # ── Rule-Based Fallback (before enough data) ───────────────────────────────
    def rule_based_difficulty(self, avg_score: float, attempt_count: int) -> str:
        """
        Used when we don't have enough data to train the model yet.
        Simple rule: score < 4 → Easy, 4-7 → Medium, > 7 → Hard
        """
        if attempt_count == 0 or avg_score < 4.0:
            return "Easy"
        elif avg_score < 7.0:
            return "Medium"
        else:
            return "Hard"

    # ── ML Prediction ─────────────────────────────────────────────────────────
    def predict_difficulty(self, concept: str, avg_score: float,
                           attempt_count: int, score_history: list) -> str:
        """
        Predict difficulty for the next question on this concept.
        Falls back to rules if model not yet trained.
        """
        if not self.trained or len(self.history) < 10:
            return self.rule_based_difficulty(avg_score, attempt_count)

        features = self._extract_features(avg_score, attempt_count, score_history)
        try:
            pred = self.model.predict([features])[0]
            return self.DIFFICULTY_LABELS[pred]
        except Exception:
            return self.rule_based_difficulty(avg_score, attempt_count)

    # ── Training ───────────────────────────────────────────────────────────────
    def record_and_train(self, avg_score: float, attempt_count: int,
                         score_history: list, score_received: int):
        """
        After every answer, record it and retrain the model.
        Label is derived from the score received — this is our RL-style supervision.
        """
        # Determine correct difficulty in hindsight
        if score_received >= 8:
            label = "Hard"     # They got it right — it was the right difficulty or easier
        elif score_received >= 5:
            label = "Medium"
        else:
            label = "Easy"     # They struggled — should have started easier

        features = self._extract_features(avg_score, attempt_count, score_history)
        self.history.append({"features": features, "label": label})
        self.save_history()

        if len(self.history) >= 10:
            self._train_model()

    def _train_model(self):
        """Retrain Logistic Regression on all collected history."""
        X = np.array([h["features"] for h in self.history])
        y = self.encoder.transform([h["label"] for h in self.history])
        try:
            self.model.fit(X, y)
            self.trained = True
        except Exception as e:
            print(f"[WARNING] Model training error: {e}")

    # ── Feature Engineering ────────────────────────────────────────────────────
    def _extract_features(self, avg_score: float, attempt_count: int,
                          score_history: list) -> list:
        """
        Extract 4 numerical features for the classifier.
        """
        recent_avg = np.mean(score_history[-3:]) if len(score_history) >= 3 else avg_score
        trend      = recent_avg - (np.mean(score_history[:-3]) if len(score_history) > 3 else avg_score)

        return [
            avg_score / 10.0,          # Normalized avg score [0,1]
            min(attempt_count / 20.0, 1.0),  # Normalized attempt count [0,1]
            recent_avg / 10.0,         # Normalized recent average [0,1]
            (trend + 10) / 20.0,       # Normalized trend [-10,10] → [0,1]
        ]

    # ── Stats for UI ───────────────────────────────────────────────────────────
    def get_model_info(self) -> dict:
        return {
            "trained":       self.trained,
            "samples":       len(self.history),
            "samples_needed": max(0, 10 - len(self.history)),
            "mode":          "ML Model" if self.trained else "Rule-Based",
        }

    # ── Persistence ────────────────────────────────────────────────────────────
    def save_history(self):
        Path("./logs").mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "w") as f:
            json.dump(self.history, f, indent=2)

    def load_history(self):
        if not Path(HISTORY_FILE).exists():
            return
        try:
            with open(HISTORY_FILE) as f:
                self.history = json.load(f)
            if len(self.history) >= 10:
                self._train_model()
        except Exception as e:
            print(f"[WARNING] Could not load history: {e}")

    def reset(self):
        self.history  = []
        self.trained  = False
        self.model    = LogisticRegression(max_iter=1000, random_state=42)
        if Path(HISTORY_FILE).exists():
            Path(HISTORY_FILE).unlink()
