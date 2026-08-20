"""
Offline inference for Mpox clade classification + quality metrics.

Usage
-----
from mpox_clf.inference import MpoxPredictor, predict

df = predict("path/to/genomes.fasta")          # or a folder of FASTA files
df = predict(["a.fasta", "b.fasta"])

No network calls. Bundle is loaded once and cached on the predictor instance.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd

from ..preprocessing.fasta_io import SequenceRecord, load_fasta
from ..utils.config import project_root

from ..training.ensemble import MpoxEnsemble
from ..training.ood import NoveltyDetector
from ..utils.audit import log_audit_event
from .explainability import explain_prediction

PathLike = Union[str, Path]


def default_bundle_path() -> Path:
    return project_root() / "models" / "deploy_bundle.joblib"


@lru_cache(maxsize=2)
def _load_bundle_cached(path_str: str):
    return joblib.load(path_str)


class MpoxPredictor:
    """
    Loads deploy_bundle.joblib once; call `.predict(...)` for single or batch FASTA.
    """

    def __init__(self, bundle_path: Optional[PathLike] = None, use_ensemble: bool = False):
        self.bundle_path = Path(bundle_path) if bundle_path else default_bundle_path()
        if not self.bundle_path.exists():
            raise FileNotFoundError(
                f"Model bundle not found at {self.bundle_path}. "
                "Train first (scripts/train_all.py) or copy models/ from a USB stick."
            )
        self.bundle = _load_bundle_cached(str(self.bundle_path.resolve()))
        self.model = self.bundle["model"]
        self.ensemble = self.bundle.get("ensemble")
        self.ood_detector = self.bundle.get("ood_detector")
        self.extractor = self.bundle["extractor"]
        self.label_encoder = self.bundle["label_encoder"]
        self.model_name = self.bundle.get("model_name", "unknown")
        self.use_ensemble = use_ensemble and self.ensemble is not None
        self.clades = list(self.bundle.get("clades", self.label_encoder.classes_))
        self.clade_means = self.bundle.get("clade_means", {})
        self.schema_hash = self.bundle.get("schema_hash", "legacy")
        self.last_feature_matrix_: Optional[np.ndarray] = None
        self._top_kmers_meta = self._load_top_kmers_meta()

    def _load_top_kmers_meta(self):
        meta_path = self.bundle_path.parent / "training_meta.json"
        if meta_path.exists():
            try:
                return json.loads(meta_path.read_text(encoding="utf-8")).get(
                    "top_kmers_per_clade", {}
                )
            except Exception:
                return {}
        return {}

    def predict_records(
        self,
        records: Sequence[SequenceRecord],
        enable_audit_log: bool = True,
        batch_size: Optional[int] = None,
        progress_callback: Optional[Callable[[pd.DataFrame, int, int], None]] = None,
    ) -> pd.DataFrame:
        if not records:
            return pd.DataFrame()

        records = list(records)
        if batch_size and batch_size > 0 and len(records) > batch_size:
            batches = []
            feature_batches = []
            total = len(records)
            for start in range(0, total, batch_size):
                batch = self.predict_records(
                    records[start : start + batch_size],
                    enable_audit_log=enable_audit_log,
                )
                batches.append(batch)
                if self.last_feature_matrix_ is not None:
                    feature_batches.append(self.last_feature_matrix_)
                if progress_callback is not None:
                    progress_callback(pd.concat(batches, ignore_index=True), min(start + batch_size, total), total)
            if feature_batches:
                self.last_feature_matrix_ = np.vstack(feature_batches)
            return pd.concat(batches, ignore_index=True)

        feat_df = self.extractor.transform_records(list(records))
        X = self.extractor.model_matrix(feat_df)
        self.last_feature_matrix_ = X

        # Ensemble predictions & consensus if available
        if self.use_ensemble:
            ensemble_results = self.ensemble.predict_with_consensus(X)
        else:
            proba = self.model.predict_proba(X) if hasattr(self.model, "predict_proba") else None
            if proba is not None:
                pred_idx = np.argmax(proba, axis=1)
                conf = proba[np.arange(len(pred_idx)), pred_idx]
                pred_labels = self.label_encoder.inverse_transform(pred_idx)
            else:
                raw_pred = self.model.predict(X)
                if np.issubdtype(np.asarray(raw_pred).dtype, np.number):
                    pred_labels = self.label_encoder.inverse_transform(
                        np.asarray(raw_pred, dtype=int)
                    )
                else:
                    pred_labels = raw_pred
                conf = np.ones(len(pred_labels))

            ensemble_results = [
                {
                    "ensemble_pred": str(pred_labels[i]),
                    "confidence": float(conf[i]),
                    "consensus_ratio": "1/1 agree",
                    "model_preds": {self.model_name: str(pred_labels[i])},
                }
                for i in range(len(records))
            ]

        # OOD Anomaly Detection
        if self.ood_detector is not None:
            ood_results = self.ood_detector.predict_anomaly(X)
        else:
            ood_results = [{"is_ood": False, "anomaly_score": 0.0, "ood_status": "Normal"}] * len(records)

        rows = []
        for i, rec in enumerate(records):
            ens_item = ensemble_results[i]
            ood_item = ood_results[i]
            clade = ens_item["ensemble_pred"]

            # Feature explainability
            top_explanations = explain_prediction(
                self.extractor.feature_names_,
                X[i],
                self.clade_means,
                clade,
                top_n=5,
            )
            explain_str = "; ".join(
                f"{e['feature']}={e['value']} ({e['direction']} than clade mean {e['clade_mean']})"
                for e in top_explanations
            )

            clade_kmers = self._top_kmers_meta.get(clade, [])
            clade_kmer_str = ",".join(
                f"{d['kmer']}:{d['mean_freq']:.4f}" for d in clade_kmers[:10]
            )

            quality_flag = feat_df.loc[i, "quality_flag"]

            # Audit Logging
            if enable_audit_log:
                log_audit_event(
                    sequence_id=rec.id,
                    sequence=rec.sequence,
                    predicted_clade=clade,
                    confidence=ens_item["confidence"],
                    quality_flag=quality_flag,
                    consensus_ratio=ens_item["consensus_ratio"],
                    is_ood=ood_item["is_ood"],
                    model_version=self.schema_hash,
                )

            row = {
                "sequence_id": rec.id,
                "source_file": rec.source_file,
                "predicted_clade": clade,
                "confidence": ens_item["confidence"],
                "consensus_ratio": ens_item["consensus_ratio"],
                "is_ood": ood_item["is_ood"],
                "ood_status": ood_item["ood_status"],
                "anomaly_score": ood_item["anomaly_score"],
                "quality_flag": quality_flag,
                "quality_explanation": feat_df.loc[i, "quality_explanation"],
                "length": int(feat_df.loc[i, "length"]),
                "n_count": int(feat_df.loc[i, "n_count"]),
                "n_pct": round(float(feat_df.loc[i, "n_pct"]), 3),
                "non_acgtn_count": int(feat_df.loc[i, "non_acgtn_count"]),
                "gc_pct": round(float(feat_df.loc[i, "gc_pct"]), 3),
                "apobec3_combined_score": round(float(feat_df.loc[i, "apobec3_combined_score"]), 4),
                "cai_score": round(float(feat_df.loc[i, "cai_score"]), 4),
                "gc_skew": round(float(feat_df.loc[i, "gc_skew"]), 4),
                "at_skew": round(float(feat_df.loc[i, "at_skew"]), 4),
                "premature_stop_count": int(feat_df.loc[i, "premature_stop_count"]),
                "frameshift_flag": bool(feat_df.loc[i, "frameshift_flag"]),
                "n_contigs": int(feat_df.loc[i, "n_contigs"]),
                "n50": int(feat_df.loc[i, "n50"]),
                "top_kmers_k4": feat_df.loc[i, "top_kmers_k4"],
                "clade_discriminative_kmers": clade_kmer_str,
                "explainability_summary": explain_str,
                "sequence_raw": rec.sequence,
                "model_name": self.model_name,
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def predict(self, fasta_path_or_records) -> pd.DataFrame:
        """
        Accept a FASTA path, directory, list of paths, or list of SequenceRecord.
        """
        if (
            isinstance(fasta_path_or_records, (list, tuple))
            and fasta_path_or_records
            and isinstance(fasta_path_or_records[0], SequenceRecord)
        ):
            records = list(fasta_path_or_records)
        else:
            records = load_fasta(fasta_path_or_records, merge_contigs="auto")
        return self.predict_records(records)


# Module-level singleton for simple scripts / Streamlit
_PREDICTOR: Optional[MpoxPredictor] = None


def get_predictor(bundle_path: Optional[PathLike] = None) -> MpoxPredictor:
    global _PREDICTOR
    if _PREDICTOR is None or (
        bundle_path and Path(bundle_path).resolve() != _PREDICTOR.bundle_path.resolve()
    ):
        _PREDICTOR = MpoxPredictor(bundle_path)
    return _PREDICTOR


def predict(fasta_path_or_records, bundle_path: Optional[PathLike] = None) -> pd.DataFrame:
    """
    Single well-documented entry point.

    >>> from mpox_clf.inference import predict
    >>> df = predict("data/raw/batch1/")
    """
    return get_predictor(bundle_path).predict(fasta_path_or_records)
