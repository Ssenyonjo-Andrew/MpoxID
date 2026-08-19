"""
Unified feature extractor — one tidy row per sequence for train and inference.

Produces:
  - identity / length / composition / GC
  - quality metrics + Good/Fair/Poor flag (deterministic)
  - canonical k-mer frequencies (k=2,3,4)
  - codon / RSCU features from best frame

The fitted vocabulary (k-mer order) is part of the persisted artifact so
inference always uses the same column order as training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..preprocessing.fasta_io import SequenceRecord, load_fasta
from .bio_features import compute_all_bio_features
from .codon import compute_codon_features
from .kmer import build_canonical_vocab, count_canonical_kmers, top_kmers_for_sequence
from .quality import compute_quality_metrics, quality_flag_from_metrics

PathLike = Union[str, Path]


class FeatureExtractor:
    """
    Stateful feature builder.

    Call `fit()` once on training sequences (builds k-mer vocabularies),
    then `transform()` for train/test/inference. Persist with joblib alongside
    the classifier.
    """

    def __init__(
        self,
        kmer_sizes: Sequence[int] = (2, 3, 4),
        use_canonical_kmers: bool = True,
        include_codon_usage: bool = True,
        quality_thresholds: Optional[Dict[str, Any]] = None,
        reference_length: Optional[int] = 197_209,  # ~MPXV reference size; optional
    ):
        self.kmer_sizes = list(kmer_sizes)
        self.use_canonical_kmers = use_canonical_kmers
        self.include_codon_usage = include_codon_usage
        self.quality_thresholds = quality_thresholds or {}
        self.reference_length = reference_length
        self.kmer_vocabs_: Dict[int, List[str]] = {}
        self.feature_names_: List[str] = []
        self.is_fitted_: bool = False

    def fit(self, sequences: Sequence[str]) -> "FeatureExtractor":
        """Build canonical k-mer vocabularies (sequence content unused beyond alphabet)."""
        self.kmer_vocabs_ = {
            k: build_canonical_vocab(k) for k in self.kmer_sizes
        }
        # Probe one empty transform structure to lock column order
        probe = self._vectorize_one("ACGT" * 100, contigs=None, seq_id="probe")
        # Exclude non-numeric / report-only fields from model matrix
        self.feature_names_ = [
            c for c in probe.keys()
            if c not in self._meta_columns()
            and not isinstance(probe[c], (list, dict))
            and c not in ("quality_explanation", "quality_reasons", "frameshift_reasons",
                          "top_kmers_k4", "sequence_id")
        ]
        self.is_fitted_ = True
        return self

    @staticmethod
    def _meta_columns() -> set:
        return {
            "sequence_id",
            "quality_flag",
            "quality_explanation",
            "quality_reasons",
            "frameshift_reasons",
            "top_kmers_k4",
            "frameshift_flag",  # bool kept as 0/1 in model features separately
        }

    def _vectorize_one(
        self,
        sequence: str,
        contigs: Optional[Sequence[str]],
        seq_id: str,
    ) -> Dict[str, Any]:
        q = compute_quality_metrics(
            sequence,
            contigs=contigs,
            reference_length=self.reference_length,
        )
        flag = quality_flag_from_metrics(q, **self.quality_thresholds)

        row: Dict[str, Any] = {
            "sequence_id": seq_id,
            **{k: v for k, v in q.items() if k != "frameshift_reasons"},
            "frameshift_flag": int(bool(q["frameshift_flag"])),
            "frameshift_reasons": q["frameshift_reasons"],
            **flag,
        }

        # Nucleotide composition already in q; add N proportion alias for models
        row["n_proportion"] = q["n_pct"] / 100.0

        # Biological & genomic features (APOBEC3, GC-skew, CAI, SNP anchors, deletion flags)
        row.update(compute_all_bio_features(sequence))

        for k, vocab in self.kmer_vocabs_.items():
            row.update(count_canonical_kmers(sequence, k, vocab))

        if 4 in self.kmer_vocabs_:
            top = top_kmers_for_sequence(sequence, 4, self.kmer_vocabs_[4], top_n=10)
            row["top_kmers_k4"] = ",".join(f"{km}:{val:.4f}" for km, val in top)

        if self.include_codon_usage:
            row.update(compute_codon_features(sequence))

        return row

    def transform_records(self, records: Sequence[SequenceRecord]) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("FeatureExtractor must be fit() before transform")
        rows = [
            self._vectorize_one(r.sequence, r.contigs or None, r.id)
            for r in records
        ]
        return pd.DataFrame(rows)

    def transform(
        self,
        sequences: Sequence[str],
        ids: Optional[Sequence[str]] = None,
        contigs_list: Optional[Sequence[Optional[Sequence[str]]]] = None,
    ) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("FeatureExtractor must be fit() before transform")
        ids = list(ids) if ids is not None else [f"seq_{i}" for i in range(len(sequences))]
        contigs_list = list(contigs_list) if contigs_list is not None else [None] * len(sequences)
        rows = [
            self._vectorize_one(seq, contigs, sid)
            for seq, sid, contigs in zip(sequences, ids, contigs_list)
        ]
        return pd.DataFrame(rows)

    def model_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Select and order numeric columns used by the classifier."""
        if not self.feature_names_:
            raise RuntimeError("feature_names_ empty — call fit() first")
        missing = [c for c in self.feature_names_ if c not in df.columns]
        if missing:
            raise KeyError(f"Missing feature columns: {missing[:10]}...")
        return df[self.feature_names_].astype(float).to_numpy()

    def fit_transform_records(self, records: Sequence[SequenceRecord]) -> pd.DataFrame:
        self.fit([r.sequence for r in records])
        return self.transform_records(records)


def extract_features_dataframe(
    fasta_path: PathLike,
    extractor: Optional[FeatureExtractor] = None,
    *,
    fit: bool = False,
) -> pd.DataFrame:
    """Convenience: FASTA path → feature DataFrame."""
    records = load_fasta(fasta_path)
    if extractor is None:
        extractor = FeatureExtractor()
        fit = True
    if fit or not extractor.is_fitted_:
        extractor.fit([r.sequence for r in records])
    return extractor.transform_records(records)
