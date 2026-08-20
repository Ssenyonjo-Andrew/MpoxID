"""Online-first inference service helpers.

The default provider is `inprocess`, which keeps a warm predictor in memory inside
the running app/process. An optional HTTP provider can be used when a local or
remote service is available. Offline mode always remains available.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..preprocessing.fasta_io import SequenceRecord, load_fasta
from ..utils.config import load_config
from .predict import MpoxPredictor, default_bundle_path

_WARM_PREDICTOR: Optional[MpoxPredictor] = None


def get_warm_predictor(bundle_path: Optional[str | Path] = None) -> MpoxPredictor:
    global _WARM_PREDICTOR
    resolved = Path(bundle_path) if bundle_path else default_bundle_path()
    if _WARM_PREDICTOR is None or _WARM_PREDICTOR.bundle_path.resolve() != resolved.resolve():
        _WARM_PREDICTOR = MpoxPredictor(resolved)
    return _WARM_PREDICTOR


def predict_records_inprocess(records: List[SequenceRecord], bundle_path: Optional[str | Path] = None):
    predictor = get_warm_predictor(bundle_path)
    return predictor.predict_records(records)


def predict_records_http(records: List[SequenceRecord], *, url: str, timeout_seconds: int = 30):
    payload = {
        "records": [
            {
                "id": r.id,
                "sequence": r.sequence,
                "description": r.description,
                "source_file": r.source_file,
                "contigs": list(r.contigs),
            }
            for r in records
        ]
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    import pandas as pd

    return pd.DataFrame(body.get("predictions", []))


def predict_online(fasta_path_or_records, bundle_path: Optional[str | Path] = None):
    """Predict through the configured warm in-process or HTTP provider."""
    settings = resolve_inference_mode("online")
    if isinstance(fasta_path_or_records, (list, tuple)) and (
        not fasta_path_or_records or isinstance(fasta_path_or_records[0], SequenceRecord)
    ):
        records = list(fasta_path_or_records)
    else:
        records = load_fasta(fasta_path_or_records, merge_contigs="auto")
    if settings["provider"] == "http":
        return predict_records_http(
            records, url=settings["url"], timeout_seconds=settings["timeout_seconds"]
        )
    return predict_records_inprocess(records, bundle_path=bundle_path)


def resolve_inference_mode(explicit_mode: Optional[str] = None) -> Dict[str, Any]:
    cfg = load_config()
    infer_cfg = cfg.get("inference", {})
    mode = (explicit_mode or infer_cfg.get("inference_mode") or "online").lower()
    provider = str(infer_cfg.get("online_provider", "inprocess")).lower()
    url = str(infer_cfg.get("online_url", "http://127.0.0.1:8765/predict"))
    timeout_seconds = int(infer_cfg.get("request_timeout_seconds", 30))
    return {
        "mode": mode,
        "provider": provider,
        "url": url,
        "timeout_seconds": timeout_seconds,
    }
