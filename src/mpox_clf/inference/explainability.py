"""
Feature Explainability Engine for Mpox Clade Classification.

Computes top directional feature contributions driving individual sequence clade calls.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def explain_prediction(
    feature_names: List[str],
    sample_row: np.ndarray,
    training_means: Dict[str, Dict[str, float]],
    predicted_clade: str,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """
    Computes per-prediction feature contributions relative to training clade means.

    Returns top features that differentiate this sample toward the predicted clade.
    """
    if predicted_clade not in training_means:
        return []

    clade_means = training_means[predicted_clade]
    contributions = []

    for i, name in enumerate(feature_names):
        val = float(sample_row[i])
        mean_val = clade_means.get(name, 0.0)
        diff = val - mean_val

        # Ignore uninformative zero features
        if abs(val) < 1e-6 and abs(mean_val) < 1e-6:
            continue

        contributions.append(
            {
                "feature": name,
                "value": round(val, 4),
                "clade_mean": round(mean_val, 4),
                "abs_diff": abs(diff),
                "direction": "higher" if diff > 0 else "lower",
            }
        )

    contributions.sort(key=lambda x: -x["abs_diff"])
    return contributions[:top_n]
