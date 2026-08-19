"""
Out-of-Distribution (OOD) & Novelty Detector.

Uses Isolation Forest on normalized feature vectors to flag genomes that are
statistically anomalous relative to training sequence distribution (e.g. potential 5th clade or recombinant).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class NoveltyDetector:
    """
    Fits an IsolationForest on training features to detect OOD samples.
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> "NoveltyDetector":
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True
        return self

    def predict_anomaly(self, X: np.ndarray) -> List[Dict[str, Any]]:
        if not self.is_fitted:
            return [{"is_ood": False, "anomaly_score": 0.0, "ood_status": "Normal"}] * len(X)

        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)  # 1 for inlier, -1 for outlier/OOD
        scores = self.model.score_samples(X_scaled)  # lower score = more anomalous

        results = []
        for i in range(len(X)):
            is_ood = bool(preds[i] == -1)
            score = float(scores[i])
            status = "Out-of-Distribution / Anomaly" if is_ood else "Normal"
            results.append(
                {
                    "is_ood": is_ood,
                    "anomaly_score": round(score, 4),
                    "ood_status": status,
                }
            )
        return results
