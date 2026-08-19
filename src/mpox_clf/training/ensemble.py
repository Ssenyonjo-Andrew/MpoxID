"""
Ensemble / Soft-Voting Classifier for Mpox Clade Prediction.

Combines predictions from:
  - XGBoost Classifier
  - Random Forest Classifier
  - Logistic Regression Classifier

surfaces model consensus:
  - "3/3 models agree" (High certainty)
  - "2/3 models agree" (Moderate certainty / Review recommended)
  - "Disagreement (1/3)" (Low certainty / Flagged for manual review)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class MpoxEnsemble(BaseEstimator, ClassifierMixin):
    """
    Soft-voting ensemble classifier with model consensus and variance tracking.
    """

    def __init__(self, models: Dict[str, Any], classes: List[str]):
        self.models = models  # dict: {"xgboost": model1, "random_forest": model2, "logistic_regression": model3}
        self.classes_ = np.array(classes)
        self.model_names = list(models.keys())

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MpoxEnsemble":
        # Models are assumed to be pre-fitted individually in train.py
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return average predicted probability matrix shape (n_samples, n_classes).
        """
        probas = []
        for name, model in self.models.items():
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X)
                # Align if needed
                if hasattr(model, "classes_") and list(model.classes_) != list(self.classes_):
                    aligned = np.zeros((len(X), len(self.classes_)))
                    for j, c in enumerate(model.classes_):
                        if c in self.classes_:
                            idx = list(self.classes_).index(c)
                            aligned[:, idx] = p[:, j]
                    p = aligned
                probas.append(p)
            else:
                preds = model.predict(X)
                p = np.zeros((len(X), len(self.classes_)))
                for i, pred_val in enumerate(preds):
                    if pred_val in self.classes_:
                        idx = list(self.classes_).index(pred_val)
                        p[i, idx] = 1.0
                probas.append(p)

        # Average probability across all ensemble models
        avg_proba = np.mean(probas, axis=0)
        return avg_proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]

    def predict_with_consensus(self, X: np.ndarray) -> List[Dict[str, Any]]:
        """
        Predict clade while evaluating individual model consensus & uncertainty.

        Returns list of dicts per sample:
          {
            "ensemble_pred": clade,
            "confidence": float,
            "consensus_ratio": "3/3 agree" or "2/3 agree",
            "model_preds": {"xgboost": clade1, "random_forest": clade2, ...},
            "prob_std": float (variance across models)
          }
        """
        individual_preds: Dict[str, List[str]] = {}

        for name, model in self.models.items():
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X)
                if hasattr(model, "classes_") and list(model.classes_) != list(self.classes_):
                    max_idx = np.argmax(p, axis=1)
                    cls_preds = [str(model.classes_[idx]) for idx in max_idx]
                else:
                    max_idx = np.argmax(p, axis=1)
                    cls_preds = [str(self.classes_[idx]) for idx in max_idx]
            else:
                cls_preds = [str(c) for c in model.predict(X)]
            individual_preds[name] = cls_preds

        avg_proba = self.predict_proba(X)
        n_samples = len(X)
        n_models = len(self.models)
        results = []

        for i in range(n_samples):
            votes = [individual_preds[m][i] for m in self.model_names]
            top_idx = np.argmax(avg_proba[i])
            ensemble_pred = str(self.classes_[top_idx])
            conf = float(avg_proba[i, top_idx])

            # Count votes matching ensemble pred
            agree_count = votes.count(ensemble_pred)
            consensus_str = f"{agree_count}/{n_models} agree"

            results.append(
                {
                    "ensemble_pred": ensemble_pred,
                    "confidence": round(conf, 4),
                    "consensus_ratio": consensus_str,
                    "agree_count": agree_count,
                    "total_models": n_models,
                    "model_preds": {m: individual_preds[m][i] for m in self.model_names},
                }
            )

        return results
