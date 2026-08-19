"""
Active Learning Tech Correction Hook.

Logs lab-confirmed predictions and user clade overrides into an append-only CSV file
(data/corrections/active_learning_log.csv) for future model retraining.
"""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

PathLike = Union[str, Path]


def default_corrections_log_path() -> Path:
    from ..utils.config import project_root
    return project_root() / "data" / "corrections" / "active_learning_log.csv"


def log_user_correction(
    sequence_id: str,
    sequence: str,
    predicted_clade: str,
    corrected_clade: str,
    confidence: float,
    user_notes: str = "",
    log_path: Optional[PathLike] = None,
) -> Path:
    """
    Appends a lab technician override record to the active learning corrections CSV.
    """
    path = Path(log_path) if log_path else default_corrections_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    seq_hash = hashlib.sha256(sequence.encode("utf-8")).hexdigest()[:16]
    file_exists = path.exists() and path.stat().st_size > 0

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "timestamp",
                    "sequence_id",
                    "sequence_hash",
                    "predicted_clade",
                    "corrected_clade",
                    "confidence",
                    "user_notes",
                ]
            )
        writer.writerow(
            [
                datetime.now(timezone.utc).isoformat(),
                sequence_id,
                seq_hash,
                predicted_clade,
                corrected_clade,
                round(float(confidence), 4),
                user_notes,
            ]
        )
    return path
