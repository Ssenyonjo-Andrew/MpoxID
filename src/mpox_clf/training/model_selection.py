"""Model ranking helpers for deployment selection."""

from __future__ import annotations

from typing import Dict, List, Tuple


def _priority_score(result: Dict, priority_clades: List[str]) -> Tuple[float, float]:
    per_class = result.get("per_class")
    recalls = {}
    if per_class is not None and hasattr(per_class, "iterrows"):
        for _, row in per_class.iterrows():
            recalls[str(row["clade"])] = float(row["recall"])
    priority_vals = [recalls.get(c, 0.0) for c in priority_clades]
    return (min(priority_vals) if priority_vals else 0.0, sum(priority_vals))


def choose_best_model(
    holdout_results: Dict[str, Dict],
    *,
    priority_clades: List[str],
    min_priority_recall_floor: float = 0.2,
) -> Tuple[str, Dict[str, Dict]]:
    annotated = {}
    for name, res in holdout_results.items():
        min_priority, sum_priority = _priority_score(res, priority_clades)
        annotated[name] = {
            **res,
            "priority_min_recall": min_priority,
            "priority_sum_recall": sum_priority,
            "priority_recall_floor_ok": bool(min_priority >= min_priority_recall_floor),
        }

    ranked = sorted(
        annotated.items(),
        key=lambda kv: (
            float(kv[1].get("macro_f1", 0.0)),
            1.0 if kv[1]["priority_recall_floor_ok"] else 0.0,
            float(kv[1]["priority_min_recall"]),
            float(kv[1]["priority_sum_recall"]),
            float(kv[1].get("weighted_f1", 0.0)),
            float(kv[1].get("accuracy", 0.0)),
        ),
        reverse=True,
    )
    return ranked[0][0], annotated
