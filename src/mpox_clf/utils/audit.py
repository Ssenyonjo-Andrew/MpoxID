"""
Audit Logging System for Mpox Clade Classifier.

Writes append-only audit log records to data/audit/audit_log.jsonl recording:
  - Timestamp (ISO 8601 UTC)
  - Sequence ID & SHA256 sequence hash
  - Model name & model version / schema hash
  - Predicted clade, confidence, quality flag, consensus ratio, OOD status
  - Action / User event
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

PathLike = Union[str, Path]


def default_audit_log_path() -> Path:
    from ..utils.config import project_root
    return project_root() / "data" / "audit" / "audit_log.jsonl"


def log_audit_event(
    sequence_id: str,
    sequence: str,
    predicted_clade: str,
    confidence: float,
    quality_flag: str,
    consensus_ratio: str = "1/1 agree",
    is_ood: bool = False,
    model_version: str = "1.0.0",
    action: str = "predict",
    log_path: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """
    Appends an audit log entry in JSONL format.
    """
    path = Path(log_path) if log_path else default_audit_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    seq_hash = hashlib.sha256(sequence.encode("utf-8")).hexdigest()

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "sequence_id": sequence_id,
        "sequence_sha256": seq_hash,
        "predicted_clade": predicted_clade,
        "confidence": round(float(confidence), 4),
        "quality_flag": quality_flag,
        "consensus_ratio": consensus_ratio,
        "is_ood": is_ood,
        "model_version": model_version,
    }

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")

    return event
