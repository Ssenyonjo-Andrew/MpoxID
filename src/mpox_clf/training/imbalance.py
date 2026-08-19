"""
Class-imbalance utilities for MPXV clade training.

Real-world issue
----------------
Public MPXV genomes are heavily skewed toward Clade IIb (2022+ global outbreak),
with far fewer Ia / Ib / IIa genomes. A naive accuracy score will look excellent
while missing African clade Ia/Ib cases — exactly the ones UVRI/district labs
must detect.

Strategies (we support all three; defaults use weighting + stratification):

1. Stratified splits / stratified k-fold
   Pros: every fold sees all clades; metrics stay comparable.
   Cons: does not create new minority examples; very rare classes may still
   have tiny absolute counts per fold.

2. class_weight='balanced' (LR, RF) / scale_pos_weight-style balancing (XGB)
   Pros: no synthetic sequences; preserves true genome distribution; simple.
   Cons: can over-emphasize noisy minority labels; probability calibration may
   shift (we still report macro-F1 which treats classes equally).

3. Resampling (RandomOverSampler on the *feature matrix*, never raw FASTA)
   Pros: forces equal clade counts for tree learners that ignore sample_weight.
   Cons: oversampling duplicates rows → optimistic CV if not done inside each
   fold; undersampling throws away majority genomes. Prefer weighting first.

Recommendation for this project: stratified CV + balanced class weights;
use resampling only if a clade has < ~30 genomes after filtering.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight


def clade_counts(y: np.ndarray) -> Dict[str, int]:
    return dict(Counter(y.tolist()))


def balanced_class_weight_dict(y: np.ndarray) -> Dict[str, float]:
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {c: float(w) for c, w in zip(classes, weights)}


def sample_weights(y: np.ndarray) -> np.ndarray:
    """Per-row weights for estimators that accept sample_weight=..."""
    return compute_sample_weight(class_weight="balanced", y=y)


def random_oversample(
    X: np.ndarray,
    y: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Naive random oversampling to the size of the largest class.

    IMPORTANT: call only on a training fold, never on the full dataset before
    splitting, or CV scores will leak.
    """
    rng = rng or np.random.default_rng(42)
    counts = Counter(y.tolist())
    max_n = max(counts.values())
    X_parts = []
    y_parts = []
    for label, n in counts.items():
        idx = np.where(y == label)[0]
        if n < max_n:
            extra = rng.choice(idx, size=max_n - n, replace=True)
            idx = np.concatenate([idx, extra])
        X_parts.append(X[idx])
        y_parts.append(y[idx])
    return np.vstack(X_parts), np.concatenate(y_parts)


def imbalance_report(y: np.ndarray) -> str:
    counts = clade_counts(y)
    total = sum(counts.values()) or 1
    lines = ["Class imbalance summary:"]
    for clade, n in sorted(counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {clade}: {n} ({100.0 * n / total:.1f}%)")
    ratio = max(counts.values()) / max(1, min(counts.values()))
    lines.append(f"  majority/minority ratio: {ratio:.1f}x")
    if ratio > 5:
        lines.append(
            "  NOTE: strong imbalance — rely on macro-F1 + per-class recall; "
            "do not use accuracy alone for model selection."
        )
    return "\n".join(lines)
