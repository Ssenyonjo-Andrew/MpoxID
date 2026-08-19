"""Model evaluation and markdown report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    sns = None
    HAS_SEABORN = False
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

PathLike = Union[str, Path]


def evaluate_predictions(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Optional[Sequence[str]] = None,
) -> Dict:
    labels = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = pd.DataFrame(
        {
            "clade": labels,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support,
        }
    )
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": cm,
        "labels": list(labels),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, zero_division=0
        ),
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: Sequence[str],
    out_path: PathLike,
    title: str = "Confusion matrix",
) -> None:
    if not HAS_MATPLOTLIB or not HAS_SEABORN:
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_model_comparison(
    results: Dict[str, Dict],
    out_path: PathLike,
) -> None:
    """Bar chart of accuracy and macro-F1 across candidate models."""
    if not HAS_MATPLOTLIB:
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(results.keys())
    acc = [results[n]["accuracy"] for n in names]
    f1 = [results[n]["macro_f1"] for n in names]
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, acc, width, label="Accuracy")
    ax.bar(x + width / 2, f1, width, label="Macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison (hold-out or CV mean)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_evaluation_report(
    results: Dict[str, Dict],
    out_path: PathLike,
    *,
    deploy_model: str,
    imbalance_text: str = "",
    extra_notes: str = "",
) -> None:
    """Write a technician-readable markdown evaluation summary."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [
        "# Mpox Clade Classifier — Evaluation Report",
        "",
        f"**Deployed model:** `{deploy_model}`",
        "",
        "## Class balance",
        "```",
        imbalance_text or "(not provided)",
        "```",
        "",
        "## Summary metrics",
        "",
        "| Model | Accuracy | Macro-F1 |",
        "|-------|----------|----------|",
    ]
    for name, res in results.items():
        mark = " ← deploy" if name == deploy_model else ""
        lines.append(
            f"| {name} | {res['accuracy']:.4f} | {res['macro_f1']:.4f} |{mark}"
        )

    for name, res in results.items():
        lines.extend(
            [
                "",
                f"## {name}",
                "",
                "### Per-class precision / recall / F1",
                "",
                res["per_class"].to_markdown(index=False)
                if hasattr(res["per_class"], "to_markdown")
                else res["per_class"].to_string(index=False),
                "",
                "### sklearn classification report",
                "```",
                res["classification_report"],
                "```",
            ]
        )

    if extra_notes:
        lines.extend(["", "## Notes", "", extra_notes, ""])

    lines.extend(
        [
            "",
            "## Interpretation for lab supervisors",
            "",
            "- Prefer **macro-F1** over accuracy when clades are imbalanced.",
            "- Check per-class **recall for Ia and Ib** — missing those is a "
            "public-health failure mode even if overall accuracy is high.",
            "- Confusion matrices are saved under `reports/figures/`.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")
